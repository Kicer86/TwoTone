
import enum
import logging
import os
from math import isfinite
from typing import NamedTuple

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

from ..utils import generic_utils, image_utils, media_analysis, video_utils
from . import debug_routines, frame_comparison, matching_analysis
from .boundary_search import BoundarySearch
from .debug_routines import DebugRoutines
from .frame_comparison import FrameComparison, MatchingSide
from .matching_analysis import CoverageSummary, GlobalLinearDiscontinuityReport
from .melt_cache import MeltCache
from .melt_common import FrameInfo, FramesInfo, format_timestamp
from .phash_cache import PhashCache


DEFAULT_MAX_TEMPORARY_FRAME_OFFSET = 2.0


class MappingRelation(enum.Enum):
    """Detected relation between matching video frames."""

    GENERIC = "generic"
    # The two files are related by a single global linear frame map
    # ``rhs_frame ~= slope*lhs_frame + intercept`` (a constant frame offset is
    # the ``slope == 1`` special case).  The audio is placed with one global
    # time-scale derived from the fit over all matched pairs — which is a no-op
    # when the files play at the same speed and a stretch on e.g. a 25 fps PAL
    # speedup vs 24 fps.  The boundaries themselves come from the content-aware
    # walk, not from the fit.
    GLOBAL_LINEAR = "global_linear"


class GlobalLinearFit(NamedTuple):
    """A detected global linear frame relation ``rhs_frame ~= slope*lhs + intercept``.

    ``is_constant_offset`` is True for the ``slope == 1`` special case.
    ``time_scale = slope*lhs_fps/rhs_fps`` is the *nominal-fps* stretch factor,
    used only for plausibility checks and logging.  It must NOT be used to
    scale audio: when the declared fps differs from the real timestamp ratio it
    drifts audibly (~1.2 s across a film).  Audio scaling uses the matched-span
    timestamp ratio instead (see ``MeltPerformer.patch_audio_constant_offset``).
    """
    slope: float
    intercept: float
    is_constant_offset: bool
    time_scale: float


class SegmentsMappingResult(NamedTuple):
    mapping: list[tuple[int, int]]
    lhs_all_frames: FramesInfo
    rhs_all_frames: FramesInfo
    relation: MappingRelation
    lhs_fps: float
    rhs_fps: float
    frame_slope: float | None = None
    # A candidate frame relation is not audio certification when these exist.
    timing_discontinuities: tuple[tuple[int, int, int, int, int], ...] = ()
    # PairMatcher returns only content-continuous mappings, but keeping the
    # verdict explicit lets downstream execution defend that contract without
    # reclassifying the mapping.
    content_discontinuities: tuple[tuple[int, int, int, int, int], ...] = ()
    timing_discontinuity_debug_path: str | None = None


class PairMatcher:

    _MAX_BOUNDARY_ERROR_FRAMES = matching_analysis.MAX_BOUNDARY_ERROR_FRAMES
    # Minimum phash distance treated as "still the same frame".  The adaptive
    # cutoff (median + 2*std of the matched pairs) collapses to 0 when the
    # surviving matches are pixel-perfect; without a floor, any check keyed on it
    # would then demand pixel-perfect frames — impossible across a speed or
    # resolution difference — so it is floored to this value.
    _MIN_PHASH_CUTOFF = frame_comparison.MIN_PHASH_CUTOFF
    # Global-linear detection thresholds, shared with the relation-diagnostics
    # log so the quoted limits never drift from the real ones.
    _MAX_CONSTANT_OFFSET_STD = 1.0
    _MAX_DRIFT_SLOPE_DELTA = 0.05
    def has_identical_timeline_content(self) -> bool:
        """Quickly certify that both videos show the same content in-place.

        This is deliberately a conservative fast path.  It samples both
        boundaries and evenly-spaced interior frames at identical timestamps;
        a failed sample merely asks callers to use :meth:`create_segments_mapping`
        for the authoritative answer.  It never declares different-looking
        transfers incompatible by itself.
        """
        if self.lhs_duration_ms is None or self.rhs_duration_ms is None:
            return False
        if self.lhs_duration_ms != self.rhs_duration_ms:
            return False

        duration = self.lhs_duration_ms
        if duration <= 0:
            return False

        if self.lhs_fps == self.rhs_fps:
            lhs_scan = self._analysis_result_for(
                self.lhs_path,
                self.lhs_label,
                media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            )
            rhs_scan = self._analysis_result_for(
                self.rhs_path,
                self.rhs_label,
                media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            )
            return self._scans_have_identical_timeline_content(lhs_scan, rhs_scan)

        timestamps = list(media_analysis.identity_timestamps(duration, self.lhs_fps))
        self._probe_frames()
        lhs_timestamps = [self._nearest_frame_timestamp(self.lhs_all_frames, timestamp) for timestamp in timestamps]
        rhs_timestamps = [self._nearest_frame_timestamp(self.rhs_all_frames, timestamp) for timestamp in timestamps]
        identity_wd = os.path.join(self.debug_wd, "identity")
        lhs_frames = self._extract_identity_samples(
            self.lhs_path, self.lhs_all_frames, lhs_timestamps, os.path.join(identity_wd, "lhs"), self.lhs_label,
        )
        rhs_frames = self._extract_identity_samples(
            self.rhs_path, self.rhs_all_frames, rhs_timestamps, os.path.join(identity_wd, "rhs"), self.rhs_label,
        )
        if lhs_frames is None or rhs_frames is None:
            return False

        for timestamp, lhs_timestamp, rhs_timestamp in zip(timestamps, lhs_timestamps, rhs_timestamps):
            distance = abs(
                self.phash.get(self._extracted_path(lhs_frames[lhs_timestamp]))
                - self.phash.get(self._extracted_path(rhs_frames[rhs_timestamp]))
            )
            if distance > self._MIN_PHASH_CUTOFF:
                self.logger.debug(
                    "Equal-length identity check failed at %d ms (pHash distance %d).",
                    timestamp, distance,
                )
                return False

        self.logger.debug("Equal-length identity check passed (%d samples).", len(timestamps))
        return True

    def _scans_have_identical_timeline_content(
        self,
        lhs_scan: media_analysis.VideoScanResult,
        rhs_scan: media_analysis.VideoScanResult,
    ) -> bool:
        lhs_samples = lhs_scan.identity_samples
        rhs_samples = rhs_scan.identity_samples
        lhs_expected = len(media_analysis.identity_timestamps(self.lhs_duration_ms or 0, self.lhs_fps))
        rhs_expected = len(media_analysis.identity_timestamps(self.rhs_duration_ms or 0, self.rhs_fps))

        def has_complete_distinct_samples(
            samples: tuple[media_analysis.VideoSample, ...],
            expected: int,
        ) -> bool:
            return (
                len(samples) == expected
                and len({sample.timestamp_ms for sample in samples}) == expected
                and len({sample.path for sample in samples}) == expected
            )

        if (
            not has_complete_distinct_samples(lhs_samples, lhs_expected)
            or not has_complete_distinct_samples(rhs_samples, rhs_expected)
        ):
            self.logger.debug(
                "Equal-length identity check could not collect all distinct samples "
                "(%d/%d and %d/%d).",
                len(lhs_samples),
                lhs_expected,
                len(rhs_samples),
                rhs_expected,
            )
            return False

        identity_wd = os.path.join(self.debug_wd, "identity")
        lhs_frames = self._normalize_scan_samples(
            lhs_samples,
            os.path.join(identity_wd, "lhs"),
            self.lhs_label,
        )
        rhs_frames = self._normalize_scan_samples(
            rhs_samples,
            os.path.join(identity_wd, "rhs"),
            self.rhs_label,
        )

        for lhs_sample, rhs_sample in zip(lhs_samples, rhs_samples):
            distance = abs(
                self.phash.get(self._extracted_path(lhs_frames[lhs_sample.timestamp_ms]))
                - self.phash.get(self._extracted_path(rhs_frames[rhs_sample.timestamp_ms]))
            )
            if distance > self._MIN_PHASH_CUTOFF:
                self.logger.debug(
                    "Equal-length identity check failed near %d ms (pHash distance %d).",
                    lhs_sample.target_ms,
                    distance,
                )
                return False

        self.logger.debug("Equal-length identity check passed (%d samples).", len(lhs_samples))
        return True

    def _normalize_scan_samples(
        self,
        samples: tuple[media_analysis.VideoSample, ...],
        output_dir: str,
        label: str,
    ) -> FramesInfo:
        os.makedirs(output_dir, exist_ok=True)
        frames: FramesInfo = {
            sample.timestamp_ms: FrameInfo(
                frame_id=sample.frame_id,
                path=sample.path,
            )
            for sample in samples
        }
        return self._normalize_frames(
            frames,
            output_dir,
            desc=f"Normalizing identity samples: {label}",
        )

    @staticmethod
    def _nearest_frame_timestamp(frames: FramesInfo, timestamp: int) -> int:
        return min(frames, key=lambda candidate: abs(candidate - timestamp))

    def _extract_identity_samples(
        self,
        video_path: str,
        all_frames: FramesInfo,
        timestamps: list[int],
        identity_wd: str,
        label: str,
    ) -> FramesInfo | None:
        sample_frames = {timestamp: all_frames[timestamp] for timestamp in set(timestamps)}
        frame_ranges = [
            (int(info["frame_id"]), int(info["frame_id"]))
            for info in sample_frames.values()
        ]
        os.makedirs(identity_wd, exist_ok=True)
        video_utils.extract_frames_at_ranges(
            video_path,
            identity_wd,
            frame_ranges,
            sample_frames,
            scale=(960, -2),
            format="png",
            interruption=self.interruption,
            desc=f"Identity samples: {label}",
            logger=self.logger,
        )
        if any(info.get("path") is None for info in sample_frames.values()):
            self.logger.debug("Could not extract all identity-check frames from %s.", video_path)
            return None
        normalized_wd = os.path.join(identity_wd, "normalized")
        os.makedirs(normalized_wd, exist_ok=True)
        return self._normalize_frames(
            sample_frames,
            normalized_wd,
            desc=f"Normalizing identity samples: {label}",
        )

    def __init__(
        self,
        interruption: generic_utils.InterruptibleProcess,
        wd: str,
        lhs_path: str,
        rhs_path: str,
        logger: logging.Logger,
        lhs_label: str = "#1",
        rhs_label: str = "#2",
        cache: MeltCache | None = None,
        *,
        media_analysis_session: media_analysis.MediaAnalysisSession,
        max_temporary_frame_offset: float = DEFAULT_MAX_TEMPORARY_FRAME_OFFSET,
    ) -> None:
        if (
            not isfinite(max_temporary_frame_offset)
            or max_temporary_frame_offset < DEFAULT_MAX_TEMPORARY_FRAME_OFFSET
        ):
            raise ValueError(
                "Maximum temporary frame offset must be finite and at least "
                f"{DEFAULT_MAX_TEMPORARY_FRAME_OFFSET:g}."
            )
        self.interruption = interruption
        self.wd = os.path.join(wd, "pair_matcher")
        self.logger = logger
        self.cache = cache
        self.media_analysis = media_analysis_session
        self.max_temporary_frame_offset = max_temporary_frame_offset
        self.phash = PhashCache()
        lhs_video_data = video_utils.get_video_data(lhs_path, logger=self.logger)["video"][0]
        rhs_video_data = video_utils.get_video_data(rhs_path, logger=self.logger)["video"][0]
        lhs_wd = os.path.join(self.wd, "lhs")
        rhs_wd = os.path.join(self.wd, "rhs")
        self.lhs = MatchingSide.create(
            lhs_path,
            lhs_label,
            generic_utils.fps_str_to_float(lhs_video_data["fps"]),
            lhs_video_data.get("length"),
            lhs_wd,
        )
        self.rhs = MatchingSide.create(
            rhs_path,
            rhs_label,
            generic_utils.fps_str_to_float(rhs_video_data["fps"]),
            rhs_video_data.get("length"),
            rhs_wd,
        )
        self.debug_wd = os.path.join(self.wd, "debug")

        for directory in (
            lhs_wd,
            rhs_wd,
            *self.lhs.directories(),
            *self.rhs.directories(),
            self.debug_wd,
        ):
            os.makedirs(directory)

        self.frame_comparison = FrameComparison(
            self.interruption,
            self.logger,
            self.lhs,
            self.rhs,
        )
        self.boundary_search = BoundarySearch(
            self.logger,
            self.lhs,
            self.rhs,
            self.frame_comparison,
        )

    # Preserve the established PairMatcher attribute surface while one
    # MatchingSide now owns each side's state.
    lhs_path = property(
        lambda self: self.lhs.path,
        lambda self, value: setattr(self.lhs, "path", value),
    )
    rhs_path = property(
        lambda self: self.rhs.path,
        lambda self, value: setattr(self.rhs, "path", value),
    )
    lhs_label = property(
        lambda self: self.lhs.label,
        lambda self, value: setattr(self.lhs, "label", value),
    )
    rhs_label = property(
        lambda self: self.rhs.label,
        lambda self, value: setattr(self.rhs, "label", value),
    )
    lhs_fps = property(
        lambda self: self.lhs.fps,
        lambda self, value: setattr(self.lhs, "fps", value),
    )
    rhs_fps = property(
        lambda self: self.rhs.fps,
        lambda self, value: setattr(self.rhs, "fps", value),
    )
    lhs_duration_ms = property(
        lambda self: self.lhs.duration_ms,
        lambda self, value: setattr(self.lhs, "duration_ms", value),
    )
    rhs_duration_ms = property(
        lambda self: self.rhs.duration_ms,
        lambda self, value: setattr(self.rhs, "duration_ms", value),
    )

    @property
    def lhs_all_frames(self) -> FramesInfo:
        return self.lhs.all_frames

    @lhs_all_frames.setter
    def lhs_all_frames(self, frames: FramesInfo) -> None:
        self.lhs.all_frames = frames

    @property
    def rhs_all_frames(self) -> FramesInfo:
        return self.rhs.all_frames

    @rhs_all_frames.setter
    def rhs_all_frames(self, frames: FramesInfo) -> None:
        self.rhs.all_frames = frames

    def _analysis_result_for(
        self,
        path: str,
        label: str,
        features: media_analysis.MediaAnalysisFeature,
    ) -> media_analysis.VideoScanResult:
        request = media_analysis.MediaAnalysisRequest(
            path=path,
            label=label,
            features=features,
        )
        return self.media_analysis.fulfill(request, raise_on_error=True)

    @staticmethod
    def calculate_ratio(pairs: list[tuple[int, int]]) -> float:
        return matching_analysis.calculate_ratio(pairs)

    @staticmethod
    def _timeline_frame_ids(frames: FramesInfo, fps: float) -> dict[int, int]:
        return matching_analysis.timeline_frame_ids(frames, fps)

    # Maximum relative deviation between an observed pair ratio and the
    # expected one before a match is considered inconsistent.
    _MAX_RELATIVE_RATIO_ERROR = matching_analysis.MAX_RELATIVE_RATIO_ERROR

    @staticmethod
    def is_ratio_acceptable(ratio: float, perfect_ratio: float) -> bool:
        return matching_analysis.is_ratio_acceptable(ratio, perfect_ratio)

    # A gap between consecutive matched pairs may miss the mapping's median
    # local slope by at most this much (absolute, and relative for long gaps)
    # before it counts as a hole in the shared scene sequence.  Speed waviness
    # (a VHS-style source) oscillates around the median and stays well below;
    # a content cut (e.g. a commercial break) is one-sided and multi-second.
    _MAX_GAP_TIME_DEFICIT_MS = matching_analysis.MAX_GAP_TIME_DEFICIT_MS
    _MAX_GAP_TIME_DEFICIT_RATIO = matching_analysis.MAX_GAP_TIME_DEFICIT_RATIO
    # A GLOBAL_LINEAR mapping promises frame-accurate progression. A local
    # advance may differ from the fitted relation by at most two frames before
    # it requires refinement or an explicit temporary-excursion override.
    # Unlike the GENERIC time-domain tolerance below, this limit must not grow
    # with gap length: a content cut is a step in frame offset, not drift.
    _MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES = matching_analysis.MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES
    DEFAULT_MAX_TEMPORARY_FRAME_OFFSET = DEFAULT_MAX_TEMPORARY_FRAME_OFFSET
    _SCENE_THRESHOLD = 0.30

    @staticmethod
    def _flagged_gap_deficits(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return matching_analysis._flagged_gap_deficits(pairs)

    @staticmethod
    def find_content_discontinuities(
        mapping: list[tuple[int, int]],
    ) -> list[tuple[int, int, int, int, int]]:
        """Locate holes in the shared scene sequence of a matched pair.

        A gap whose rhs advance misses the median-slope expectation (see
        ``_flagged_gap_deficits``) marks content present on one side only.

        Returns ``(lhs_from, lhs_to, rhs_from, rhs_to, deficit_ms)`` per
        offending gap; a positive deficit means rhs is missing content there,
        a negative one means lhs is.
        """
        return matching_analysis.find_content_discontinuities(mapping)

    @staticmethod
    def find_global_linear_content_discontinuities(
        mapping: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
        *,
        frame_slope: float,
        lhs_fps: float,
        rhs_fps: float,
        require_decoded_confirmation: bool = True,
    ) -> list[tuple[int, int, int, int, int]]:
        """Locate absolute frame-step errors in a GLOBAL_LINEAR mapping.

        The fitted relation predicts ``rhs_advance = frame_slope *
        lhs_advance`` between every two matched anchors. A content removal or
        insertion changes that advance by a fixed number of frames; scaling
        the tolerance with the time between sparse anchors would hide exactly
        such edits. First and last intervals are bracketed by matched anchors
        too; boundary extension outside those anchors cannot verify their
        interior. Every suspicious interval needs refinement or content evidence. When
        ``require_decoded_confirmation`` is true, a timeline step must also be
        visible in decoded frame ordinals before it is reported as a content
        discontinuity. Otherwise, include inferred slot steps as refinement
        candidates even without an observed PTS step. These are suspicious
        intervals, not proof of an edit:
        a false visual match can also produce opposite steps. Sparse support
        or a later compensating step cannot clear them; callers must refine
        the observations or verify the suspected false match against content.
        The returned deficit remains milliseconds for error reporting.
        An empty content list does not certify audio timing; execution decisions
        must consume the full report from ``_classify_global_linear_discontinuities``,
        whose timing entries measure deviations from the observed timestamp slope.
        """
        return matching_analysis.find_global_linear_content_discontinuities(
            mapping,
            lhs_all_frames,
            rhs_all_frames,
            frame_slope=frame_slope,
            lhs_fps=lhs_fps,
            rhs_fps=rhs_fps,
            require_decoded_confirmation=require_decoded_confirmation,
        )

    @staticmethod
    def _classify_global_linear_discontinuities(
        mapping: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
        *,
        frame_slope: float,
        lhs_fps: float,
        rhs_fps: float,
    ) -> GlobalLinearDiscontinuityReport:
        return matching_analysis.classify_global_linear_discontinuities(
            mapping,
            lhs_all_frames,
            rhs_all_frames,
            frame_slope=frame_slope,
            lhs_fps=lhs_fps,
            rhs_fps=rhs_fps,
        )

    def _refine_scenes_in_region(
        self,
        gap: tuple[int, int, int, int, int],
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
    ) -> tuple[list[int], list[int], bool]:
        """Extend one scene list and return timestamps within the region."""
        lhs_from, lhs_to, rhs_from, rhs_to, _ = gap
        lhs_local = [ts for ts in lhs_scene_changes if lhs_from < ts < lhs_to]
        rhs_local = [ts for ts in rhs_scene_changes if rhs_from < ts < rhs_to]

        lhs_side = (
            self.lhs_path, self.lhs_label,
            lhs_from, lhs_to, self.lhs_all_frames, lhs_scene_changes,
        )
        rhs_side = (
            self.rhs_path, self.rhs_label,
            rhs_from, rhs_to, self.rhs_all_frames, rhs_scene_changes,
        )
        if len(lhs_local) <= len(rhs_local):
            sides = (lhs_side, rhs_side)
        else:
            sides = (rhs_side, lhs_side)

        self.logger.debug(
            "Refining suspicious region: %s %s-%s (%d scene(s)) "
            "↔ %s %s-%s (%d scene(s)); checking the sparser side first",
            self.lhs_label, generic_utils.ms_to_time(lhs_from),
            generic_utils.ms_to_time(lhs_to), len(lhs_local),
            self.rhs_label, generic_utils.ms_to_time(rhs_from),
            generic_utils.ms_to_time(rhs_to), len(rhs_local),
        )
        scenes_added = False
        for side_index, (path, label, start_ms, end_ms, all_frames, scenes) in enumerate(sides):
            analysis = self._analysis_result_for(
                path,
                label,
                media_analysis.MediaAnalysisFeature.SCENE_CHANGES,
            )
            frame_timestamps = sorted(all_frames)
            candidates: dict[int, float] = {}
            for candidate in analysis.scene_candidates:
                if not start_ms < candidate.timestamp_ms < end_ms:
                    continue
                timestamp = self._snap_to_nearest_frame(
                    frame_timestamps,
                    candidate.timestamp_ms,
                )
                if start_ms < timestamp < end_ms and timestamp not in scenes:
                    candidates[timestamp] = max(
                        candidates.get(timestamp, 0.0),
                        candidate.score,
                    )

            ranked_scores = sorted(candidates.values(), reverse=True)
            if not ranked_scores:
                self.logger.debug(
                    "Local scene refinement %s found no new candidates",
                    label,
                )
                continue

            # Several candidates can balance the initially sparser side. The
            # fallback side is already at least as dense, so consume only its
            # strongest candidate before rematching and narrowing the gap.
            desired_count = (
                max(1, abs(len(lhs_local) - len(rhs_local)))
                if side_index == 0
                else 1
            )
            cutoff = ranked_scores[min(desired_count, len(ranked_scores)) - 1]
            new_scenes = sorted(
                timestamp
                for timestamp, score in candidates.items()
                if score >= cutoff
            )
            scenes.extend(new_scenes)
            scenes_added = True
            self.logger.debug(
                "Local scene refinement %s selected %d scene(s) at score %.6f",
                label, len(new_scenes), cutoff,
            )
            break

        if not scenes_added:
            self.logger.debug(
                "Local scene refinement found no new candidates on either side; "
                "region remains unresolved",
            )

        return (
            sorted({lhs_from, lhs_to, *(ts for ts in lhs_scene_changes if lhs_from < ts < lhs_to)}),
            sorted({rhs_from, rhs_to, *(ts for ts in rhs_scene_changes if rhs_from < ts < rhs_to)}),
            scenes_added,
        )

    def _drop_pairs_breaking_local_linearity(
        self, pairs: list[tuple[int, int]],
        lhs_normalized_frames: FramesInfo,
        rhs_normalized_frames: FramesInfo,
        *, frame_slope: float | None = None,
    ) -> list[tuple[int, int]]:
        """Remove a cancelling excursion only after content disproves the pair.

        Opposite edits and a false match have the same timing signature. The
        existing gap detector nominates candidates; sequence verification must
        support the neighbours' line and contradict the candidate on both sides.
        Without that evidence the pair, and its suspicious gaps, survive.
        """
        result = sorted(pairs)
        while True:
            if frame_slope is not None:
                gaps = self.find_global_linear_content_discontinuities(
                    result, self.lhs_all_frames, self.rhs_all_frames,
                    frame_slope=frame_slope, lhs_fps=self.lhs_fps, rhs_fps=self.rhs_fps,
                )
                indices = {pair: i for i, pair in enumerate(result)}
                flagged = [(indices[(l1, r1)], deficit) for l1, _, r1, _, deficit in gaps]
            else:
                flagged = PairMatcher._flagged_gap_deficits(result)
            rogue: tuple[int, int, int] | None = None
            for (gap_a, deficit_a), (gap_b, deficit_b) in zip(flagged, flagged[1:]):
                if gap_b != gap_a + 1 or deficit_a * deficit_b >= 0:
                    continue
                merged_gap = result[gap_b + 1][0] - result[gap_a][0]
                threshold = (
                    self._MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES * 1000 / self.rhs_fps
                    if frame_slope is not None else max(
                        self._MAX_GAP_TIME_DEFICIT_MS,
                        self._MAX_GAP_TIME_DEFICIT_RATIO * merged_gap,
                    )
                )
                if abs(deficit_a + deficit_b) <= threshold and self._verify_false_pair(
                    result[gap_a], result[gap_b], result[gap_b + 1],
                    lhs_normalized_frames, rhs_normalized_frames,
                ):
                    rogue = (gap_a + 1, deficit_a, deficit_b)
                    break
            if rogue is None:
                return result
            rogue_index, deficit_before, deficit_after = rogue
            lhs_ts, rhs_ts = result.pop(rogue_index)
            self.logger.warning(
                "Dropping content-disproved pair %s ↔ %s (%+d ms before, %+d ms after)",
                generic_utils.ms_to_time(lhs_ts),
                generic_utils.ms_to_time(rhs_ts),
                deficit_before,
                deficit_after,
            )

    def _verify_false_pair(
        self, before: tuple[int, int], pair: tuple[int, int], after: tuple[int, int],
        lhs_normalized_frames: FramesInfo, rhs_normalized_frames: FramesInfo,
    ) -> bool:
        """Verify a replacement sequence before discarding an isolated anchor.

        Expand a sequence around the anchor, staying within half the distance
        to its neighbours on both timelines. Each temporal side needs two
        distinct rich replacement pairs and rich contradiction of the original.
        Unavailable or low-entropy images supply no evidence for that comparison;
        a rich replacement mismatch vetoes removal. Repeated/static content that
        also supports the original remains ambiguous. Comparisons reuse the
        normalized, shared-crop space and calibrated cutoff, extracting only
        the next small batch until there is enough evidence or the bound is hit.
        """
        if not self.lhs_all_frames or not self.rhs_all_frames:
            return False
        if any(l not in lhs_normalized_frames or r not in rhs_normalized_frames
               for l, r in (before, after)):
            return False
        lhs_ts, rhs_ts = pair
        if after[0] <= before[0] or after[1] <= before[1]:
            return False
        slope = (after[1] - before[1]) / (after[0] - before[0])
        radius = min(
            (lhs_ts - before[0]) // 2, (after[0] - lhs_ts) // 2,
            int((rhs_ts - before[1]) / (2 * slope)),
            int((after[1] - rhs_ts) / (2 * slope)),
        )
        if radius <= 0:
            return False
        ctx = self.frame_comparison.build_boundary_verify_context(
            [before, after], lhs_normalized_frames, rhs_normalized_frames,
            minimum_cutoff=self._MIN_PHASH_CUTOFF,
        )
        # These samples interpolate between neighbours. The boundary-gap
        # allowance for extrapolated crops/predictions can accept distinct
        # pictures here, hiding the evidence against a false anchor. Use the
        # local calibration (with the ordinary comparison floor) for both
        # replacement support and contradiction, retaining mismatch vetoes.
        supported: list[set[tuple[int, int]]] = [set(), set()]
        contradicted = [False, False]
        # Leave two batches inside even a short window: one unusable sample
        # must not exhaust the opportunities for two distinct rich pairs.
        step = max(1, min(250, radius // 4))
        for distance in range(step, radius + 1, 2 * step):
            self.interruption.check_for_stop()
            offsets = [-distance, distance]
            if distance + step <= radius:
                offsets.extend((-(distance + step), distance + step))
            if distance == step:
                offsets.insert(0, 0)
            samples = []
            for offset in offsets:
                lhs = self._nearest_frame_timestamp(self.lhs_all_frames, lhs_ts + offset)
                predicted = round(before[1] + (lhs - before[0]) * slope)
                original = round(rhs_ts + (lhs - lhs_ts) * slope)
                if not (before[1] < predicted < after[1] and before[1] < original < after[1]):
                    return False
                samples.append((
                    lhs,
                    self._nearest_frame_timestamp(self.rhs_all_frames, predicted),
                    self._nearest_frame_timestamp(self.rhs_all_frames, original),
                ))
            self.frame_comparison.prefetch(
                self.lhs,
                [lhs for lhs, _, _ in samples],
            )
            self.frame_comparison.prefetch(
                self.rhs,
                [ts for _, predicted, original in samples for ts in (predicted, original)],
            )
            for lhs, predicted, original in samples:
                lhs_img = self.frame_comparison.comparison_image(ctx.lhs, lhs)
                predicted_img = self.frame_comparison.comparison_image(
                    ctx.rhs,
                    predicted,
                )
                if not all(img is not None and self._is_rich(img)
                           for img in (lhs_img, predicted_img)):
                    continue
                if not self.frame_comparison.boundary_content_matches(
                    ctx,
                    lhs,
                    predicted,
                ):
                    return False
                if lhs == lhs_ts:
                    continue
                side = int(lhs > lhs_ts)
                # Snapping sparse timestamps must not count one image twice.
                if all(lhs != l and predicted != r for l, r in supported[side]):
                    supported[side].add((lhs, predicted))
                original_img = self.frame_comparison.comparison_image(
                    ctx.rhs,
                    original,
                )
                if original_img is not None and self._is_rich(original_img):
                    if not self.frame_comparison.boundary_content_matches(
                        ctx,
                        lhs,
                        original,
                    ):
                        contradicted[side] = True
            if all(len(pairs) >= 2 for pairs in supported) and all(contradicted):
                return True
        return False

    @staticmethod
    def coverage_summary(
        mappings: list[tuple[int, int]],
        lhs_duration_ms: int,
        rhs_duration_ms: int,
        *,
        lhs_fps: float,
        rhs_fps: float,
    ) -> CoverageSummary:
        """Compute a human-readable coverage summary for matched files.

        ``full_coverage`` is True when matched content reaches both video edges
        within at most two frames per input; the ``*_gap_s`` fields report the
        unmatched seconds at each edge and ``ratio`` the speed ratio between
        the two files.
        """
        return matching_analysis.coverage_summary(
            mappings,
            lhs_duration_ms,
            rhs_duration_ms,
            lhs_fps=lhs_fps,
            rhs_fps=rhs_fps,
        )

    @staticmethod
    def filter_phash_outliers(phash: PhashCache, pairs: list[tuple[int, int]], lhs_set: FramesInfo, rhs_set: FramesInfo) -> list[tuple[int, int]]:
        if len(pairs) <= 3:
            # Too few data points for reliable MAD-based outlier detection.
            # Let downstream filters (ORB, history check) handle quality control.
            return pairs

        dists_array = np.array([
            abs(phash.get(PairMatcher._extracted_path(lhs_set[l])) - phash.get(PairMatcher._extracted_path(rhs_set[r])))
            for l, r in pairs
        ], dtype=float)
        med = float(np.median(dists_array))
        mad = float(np.median(np.abs(dists_array - med)))
        threshold = med + 1.5 * mad
        return [pair for pair, dist in zip(pairs, dists_array) if dist <= threshold]

    @staticmethod
    def summarize_pairs(phash: PhashCache, pairs: list[tuple[int, int]], lhs: FramesInfo, rhs: FramesInfo, verbose: bool = False) -> str:
        return debug_routines.summarize_pairs(phash, pairs, lhs, rhs, verbose)

    @staticmethod
    def summarize_segments(pairs: list[tuple[int, int]], lhs_fps: float, rhs_fps: float, verbose: bool = True, lhs_label: str = "#1", rhs_label: str = "#2") -> str:
        return debug_routines.summarize_segments(
            pairs,
            lhs_fps,
            rhs_fps,
            verbose,
            lhs_label,
            rhs_label,
        )

    def look_for_boundaries(
        self,
        lhs: FramesInfo,
        rhs: FramesInfo,
        first: tuple[int, int],
        last: tuple[int, int],
        cutoff: float,
        max_gap_seconds: float = 15.0,
        extrapolate: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.boundary_search.look_for_boundaries(
            lhs,
            rhs,
            first,
            last,
            cutoff,
            max_gap_seconds,
            extrapolate,
        )

    def extrapolate_through_low_entropy(
        self,
        lhs: FramesInfo,
        rhs: FramesInfo,
        boundary: tuple[int, int],
        reference: tuple[int, int],
        ratio: float,
        direction: int,
        entered_low_entropy: bool,
    ) -> tuple[int, int]:
        return self.boundary_search.extrapolate_through_low_entropy(
            lhs,
            rhs,
            boundary,
            reference,
            ratio,
            direction,
            entered_low_entropy,
        )

    def _extrapolate_and_verify_global_linear(
        self,
        fit: GlobalLinearFit,
        matching_pairs: list[tuple[int, int]],
        lhs_normalized_frames: FramesInfo,
        rhs_normalized_frames: FramesInfo,
    ) -> list[tuple[int, int]]:
        return self.boundary_search.extrapolate_and_verify_global_linear(
            fit,
            matching_pairs,
            lhs_normalized_frames,
            rhs_normalized_frames,
        )

    def snap_to_edges(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
        snap_frames: int = 16,
    ) -> list[tuple[int, int]]:
        return self.boundary_search.snap_to_edges(
            matching_pairs,
            lhs_all_frames,
            rhs_all_frames,
            snap_frames,
        )

    def detect_global_linear(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
        *,
        max_slope_delta: float = _MAX_DRIFT_SLOPE_DELTA,
        max_median_residual_frames: float = 1.5,
        max_p95_residual_frames: float = 4.0,
        max_outlier_ratio: float = 0.25,
        max_audio_time_scale_delta: float = 0.30,
        max_constant_offset_std: float = _MAX_CONSTANT_OFFSET_STD,
        min_span_frames: int = 250,
    ) -> GlobalLinearFit | None:
        """Detect a single global linear frame relationship over the matched pairs.

        Both files are assumed related by ``rhs_frame ~= slope*lhs_frame +
        intercept`` in presentation-timeline frame coordinates.  Unlike decode
        ordinals, these coordinates retain empty AVI frame slots.  A constant
        frame offset is just the ``slope == 1`` special case, so it is tried
        first (it needs only two well-separated matches); when the offset is not
        constant the slope and intercept are fitted with RANSAC.  Either way the
        audio is later placed with one global time-scale, so a constant offset
        and a time-scaled drift share the same handling.

        ``time_scale = slope*lhs_fps/rhs_fps`` may differ from 1 (e.g. a 25 fps
        PAL speedup vs 24 fps); that only means the audio must be stretched.  It
        is rejected when larger than ``max_audio_time_scale_delta`` (an
        implausible match).

        This method only *detects* the relation and returns the fit; it does
        **not** decide the first/last common pair.  The boundaries are found by
        the content/entropy-aware boundary walk (shared with the GENERIC path),
        which uses the certified pairs' full observed timestamp correspondence as its predictor
        and stops at genuine divergence. Blindly
        extrapolating the line to the video edges would pair unrelated content
        across divergent intros/outros and is non-deterministic across ffmpeg
        builds, so it is deliberately avoided.
        """
        if len(matching_pairs) < 2:
            return None

        try:
            lhs_frame_ids_by_timestamp = self._timeline_frame_ids(lhs_all_frames, self.lhs_fps)
            rhs_frame_ids_by_timestamp = self._timeline_frame_ids(rhs_all_frames, self.rhs_fps)
            lhs_frame_ids = np.array(
                [lhs_frame_ids_by_timestamp[l] for l, _ in matching_pairs], dtype=float
            )
            rhs_frame_ids = np.array(
                [rhs_frame_ids_by_timestamp[r] for _, r in matching_pairs], dtype=float
            )
        except KeyError:
            return None

        fit = matching_analysis.fit_constant_offset(
            lhs_frame_ids,
            rhs_frame_ids,
            matching_pairs,
            lhs_fps=self.lhs_fps,
            rhs_fps=self.rhs_fps,
            max_std=max_constant_offset_std,
            min_two_pair_span=min_span_frames,
            logger=self.logger,
        )
        is_constant_offset = fit is not None
        if fit is None:
            fit = matching_analysis.fit_linear_drift(
                lhs_frame_ids,
                rhs_frame_ids,
                lhs_fps=self.lhs_fps,
                rhs_fps=self.rhs_fps,
                max_slope_delta=max_slope_delta,
                max_median_residual_frames=max_median_residual_frames,
                max_p95_residual_frames=max_p95_residual_frames,
                max_outlier_ratio=max_outlier_ratio,
                min_span_frames=min_span_frames,
                logger=self.logger,
            )
        if fit is None:
            return None
        slope, intercept = fit

        time_scale = slope * self.lhs_fps / self.rhs_fps
        if abs(time_scale - 1.0) > max_audio_time_scale_delta:
            self.logger.debug(
                f"Global-linear check: slope={slope:.6f} implies audio time scale "
                f"{time_scale:.5f}, beyond the realistic {max_audio_time_scale_delta:.3f} — skipping"
            )
            return None

        debug_routines.log_global_linear_relation(
            self.logger,
            lhs_label=self.lhs_label,
            rhs_label=self.rhs_label,
            slope=slope,
            intercept=intercept,
            time_scale=time_scale,
            is_constant_offset=is_constant_offset,
        )
        return GlobalLinearFit(slope=slope, intercept=intercept,
                               is_constant_offset=is_constant_offset, time_scale=time_scale)

    def _raise_for_unresolved_content_discontinuities(
        self,
        discontinuities: list[tuple[int, int, int, int, int]],
        debug_path: str | None = None,
    ) -> None:
        groups = self._group_content_discontinuities(discontinuities)
        suggested_temporary_offsets: list[int] = []
        for group in groups:
            excursion = self._temporary_frame_offset_excursion(group)
            if (
                excursion is not None
                and excursion[0] > self.max_temporary_frame_offset
            ):
                suggested_temporary_offsets.append(excursion[0])
        error = debug_routines.unresolved_content_discontinuities_error(
            self.logger,
            groups,
            self.lhs_all_frames,
            self.rhs_all_frames,
            lhs_label=self.lhs_label,
            rhs_label=self.rhs_label,
            rhs_fps=self.rhs_fps,
            compensation_tolerance_frames=self._MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES,
            suggested_temporary_offsets=suggested_temporary_offsets,
            debug_path=debug_path,
        )
        raise RuntimeError(error)

    @staticmethod
    def _group_content_discontinuities(
        discontinuities: list[tuple[int, int, int, int, int]],
    ) -> list[list[tuple[int, int, int, int, int]]]:
        return matching_analysis.group_content_discontinuities(discontinuities)

    def _filter_allowed_temporary_frame_offset_excursions(
        self,
        discontinuities: list[tuple[int, int, int, int, int]],
    ) -> list[tuple[int, int, int, int, int]]:
        """Allow one temporary offset excursion that returns at the next anchor."""
        remaining: list[tuple[int, int, int, int, int]] = []
        for group in self._group_content_discontinuities(discontinuities):
            excursion = self._temporary_frame_offset_excursion(group)
            allowed = (
                excursion is not None
                and self._MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES
                < excursion[0]
                <= self.max_temporary_frame_offset
            )
            if allowed:
                assert excursion is not None
                temporary_offset, outer_offset = excursion
                _, lhs_middle, _, rhs_middle, _ = group[0]
                lhs_frame_middle = int(self.lhs_all_frames[lhs_middle]["frame_id"])
                rhs_frame_middle = int(self.rhs_all_frames[rhs_middle]["frame_id"])
                self.logger.warning(
                    "Accepting temporary %d-frame alignment offset between "
                    "%s frame %d at %s and %s frame %d at %s; the next "
                    "matched frames return to the same outer offset of %+d frame(s).",
                    temporary_offset,
                    self.lhs_label,
                    lhs_frame_middle,
                    format_timestamp(lhs_middle),
                    self.rhs_label,
                    rhs_frame_middle,
                    format_timestamp(rhs_middle),
                    outer_offset,
                )
            else:
                remaining.extend(group)
        return remaining

    def _temporary_frame_offset_excursion(
        self,
        group: list[tuple[int, int, int, int, int]],
    ) -> tuple[int, int] | None:
        """Return the temporary shift and outer offset for one restored jump."""
        if len(group) != 2:
            return None

        lhs_from, lhs_middle, rhs_from, rhs_middle, _ = group[0]
        _, lhs_to, _, rhs_to, _ = group[1]
        lhs_frame_from = int(self.lhs_all_frames[lhs_from]["frame_id"])
        lhs_frame_middle = int(self.lhs_all_frames[lhs_middle]["frame_id"])
        lhs_frame_to = int(self.lhs_all_frames[lhs_to]["frame_id"])
        rhs_frame_from = int(self.rhs_all_frames[rhs_from]["frame_id"])
        rhs_frame_middle = int(self.rhs_all_frames[rhs_middle]["frame_id"])
        rhs_frame_to = int(self.rhs_all_frames[rhs_to]["frame_id"])
        outer_offset = lhs_frame_from - rhs_frame_from
        if lhs_frame_to - rhs_frame_to != outer_offset:
            return None

        temporary_offset = abs(
            (lhs_frame_middle - rhs_frame_middle) - outer_offset
        )
        return temporary_offset, outer_offset

    def _extract_discontinuity_frames(
        self,
        groups: list[list[tuple[int, int, int, int, int]]],
    ) -> None:
        """Extract every missing decoded frame in final error regions."""
        for video_path, target_dir, frames, label, side in (
            (
                self.lhs_path,
                self.lhs.all_dir,
                self.lhs_all_frames,
                self.lhs_label,
                "lhs",
            ),
            (
                self.rhs_path,
                self.rhs.all_dir,
                self.rhs_all_frames,
                self.rhs_label,
                "rhs",
            ),
        ):
            requested: set[int] = set()
            for group in groups:
                if side == "lhs":
                    start_ms, end_ms = group[0][0], group[-1][1]
                else:
                    start_ms, end_ms = group[0][2], group[-1][3]
                requested.update(
                    timestamp
                    for timestamp in frames
                    if start_ms <= timestamp <= end_ms
                )
            missing = sorted(
                timestamp
                for timestamp in requested
                if frames[timestamp].get("path") is None
            )
            if not missing:
                continue
            ranges = self._compute_frame_ranges(missing, frames, margin=0)
            self.logger.info(
                "Extracting %d missing diagnostic frame(s) for %s",
                len(missing),
                label,
            )
            video_utils.extract_frames_at_ranges(
                video_path,
                target_dir,
                ranges,
                frames,
                scale=(960, -2),
                format="png",
                interruption=self.interruption,
                desc=f"Extracting discontinuity diagnostics: {label}",
                logger=self.logger,
            )

    def create_segments_mapping(self) -> SegmentsMappingResult:

        lhs_analysis = self._analysis_result_for(
            self.lhs_path,
            self.lhs_label,
            media_analysis.MediaAnalysisFeature.MATCHING,
        )
        rhs_analysis = self._analysis_result_for(
            self.rhs_path,
            self.rhs_label,
            media_analysis.MediaAnalysisFeature.MATCHING,
        )
        lhs_scene_changes, rhs_scene_changes = self._detect_scenes(
            lhs_analysis,
            rhs_analysis,
        )
        self._probe_frames(lhs_analysis, rhs_analysis)
        lhs_scene_ranges, rhs_scene_ranges = self._extract_scene_frames(lhs_scene_changes, rhs_scene_changes)
        lhs_normalized_frames, rhs_normalized_frames, lhs_key_frames, rhs_key_frames = self._normalize_extracted(
            lhs_scene_changes, rhs_scene_changes,
        )

        debug = DebugRoutines(self.debug_wd, self.lhs_all_frames, self.rhs_all_frames)
        debug.dump_frames(lhs_key_frames, f"{self.lhs_label} key frames")
        debug.dump_frames(rhs_key_frames, f"{self.rhs_label} key frames")

        matching_pairs = self._match_key_frames(
            lhs_key_frames, rhs_key_frames,
            lhs_normalized_frames, rhs_normalized_frames,
            debug=debug,
        )

        # Diagnostic summary of the frame-space relationship.  Logged at INFO so
        # the chosen relation (and the audio strategy that follows) can be
        # understood without enabling --verbose.
        self._log_relation_diagnostics(matching_pairs)

        # Detect a candidate global linear frame relation (constant offset =
        # slope 1, or a fitted near-identity drift). If suspicious regions add
        # observations, those final pairs must certify the relation again
        # before it can determine boundary prediction or the audio strategy.
        global_linear_fit = self.detect_global_linear(
            matching_pairs, self.lhs_all_frames, self.rhs_all_frames,
        )
        timing_discontinuities: tuple[tuple[int, int, int, int, int], ...] = ()

        if global_linear_fit is not None:
            initial_frame_slope = global_linear_fit.slope
            discontinuity_report = self._classify_global_linear_discontinuities(
                matching_pairs,
                self.lhs_all_frames,
                self.rhs_all_frames,
                frame_slope=initial_frame_slope,
                lhs_fps=self.lhs_fps,
                rhs_fps=self.rhs_fps,
            )
            if discontinuity_report.content:
                cleaned_pairs = self._drop_pairs_breaking_local_linearity(
                    matching_pairs, lhs_normalized_frames, rhs_normalized_frames,
                    frame_slope=initial_frame_slope,
                )
                if cleaned_pairs != matching_pairs:
                    matching_pairs = cleaned_pairs
                    global_linear_fit = self.detect_global_linear(
                        matching_pairs, self.lhs_all_frames, self.rhs_all_frames,
                    )
                    discontinuity_report = self._classify_global_linear_discontinuities(
                        matching_pairs,
                        self.lhs_all_frames,
                        self.rhs_all_frames,
                        frame_slope=(
                            global_linear_fit.slope
                            if global_linear_fit is not None
                            else initial_frame_slope
                        ),
                        lhs_fps=self.lhs_fps,
                        rhs_fps=self.rhs_fps,
                    )

            if discontinuity_report.content:
                refinement_slope = (
                    global_linear_fit.slope
                    if global_linear_fit is not None
                    else initial_frame_slope
                )
                refined_pairs = self._refine_global_linear_discontinuities(
                    matching_pairs,
                    lhs_normalized_frames, rhs_normalized_frames,
                    list(lhs_scene_changes), list(rhs_scene_changes),
                    frame_slope=refinement_slope,
                )
                refined_pairs = self._drop_pairs_breaking_local_linearity(
                    refined_pairs, lhs_normalized_frames, rhs_normalized_frames,
                    frame_slope=refinement_slope,
                )
                refined_fit = self.detect_global_linear(
                    refined_pairs, self.lhs_all_frames, self.rhs_all_frames,
                )
                discontinuity_report = self._classify_global_linear_discontinuities(
                    refined_pairs,
                    self.lhs_all_frames,
                    self.rhs_all_frames,
                    frame_slope=(
                        refined_fit.slope
                        if refined_fit is not None
                        else refinement_slope
                    ),
                    lhs_fps=self.lhs_fps,
                    rhs_fps=self.rhs_fps,
                )

                if discontinuity_report.content:
                    discontinuity_report = discontinuity_report._replace(
                        content=self._filter_allowed_temporary_frame_offset_excursions(
                            discontinuity_report.content
                        )
                    )

                if discontinuity_report.content:
                    debug.dump_matches(refined_pairs, "localized discontinuities")
                    discontinuity_groups = self._group_content_discontinuities(
                        discontinuity_report.content
                    )
                    debug_path = debug.dump_content_discontinuities(
                        discontinuity_groups,
                        lhs_label=self.lhs_label,
                        rhs_label=self.rhs_label,
                        lhs_fps=self.lhs_fps,
                        rhs_fps=self.rhs_fps,
                        compensation_tolerance_frames=(
                            self._MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES
                        ),
                        prepare_frames=lambda: (
                            self._extract_discontinuity_frames(
                                discontinuity_groups
                            )
                        ),
                    )
                    self._raise_for_unresolved_content_discontinuities(
                        discontinuity_report.content,
                        debug_path,
                    )
                matching_pairs = refined_pairs
                global_linear_fit = refined_fit
                if global_linear_fit is not None and not discontinuity_report.timing:
                    self.logger.info(
                        "Local scene refinement resolved all suspected content discontinuities."
                    )
                elif global_linear_fit is None and not discontinuity_report.timing:
                    self.logger.info(
                        "Local scene refinement found no confirmed content discontinuity, "
                        "but the final observations do not support one global linear relation; "
                        "using the generic mapping path."
                    )
            timing_discontinuities = tuple(discontinuity_report.timing)

        if global_linear_fit is not None:
            relation = MappingRelation.GLOBAL_LINEAR
            frame_slope = global_linear_fit.slope
            # Use the certified pairs' observed span to predict shared content,
            # extend each boundary toward the video edge, and keep the
            # extension only where the predicted pair is content-verified (same
            # frame, or both low-entropy black).  A genuinely different intro/outro
            # fails verification and the boundary stays at the outermost real
            # match — the line is never blindly projected across divergent
            # content. No later timestamp edge-snap may bypass verification.
            if not timing_discontinuities:
                matching_pairs = self._extrapolate_and_verify_global_linear(
                    global_linear_fit, matching_pairs, lhs_normalized_frames, rhs_normalized_frames,
                )
                debug.dump_matches(matching_pairs, "after verified global-linear extrapolation")
        else:
            relation = MappingRelation.GENERIC
            frame_slope = None
            if not timing_discontinuities:
                # No global relation — fall back to the content/entropy-aware
                # iterative boundary search.
                matching_pairs = self._extract_and_refine_boundaries(
                    matching_pairs, lhs_scene_ranges, rhs_scene_ranges,
                    lhs_normalized_frames, rhs_normalized_frames, debug,
                )
                # The refined pairs can reveal a linear relation the sparse
                # initial matching could not certify.  A mapping that now passes
                # the full global-linear detection earns the same verified
                # boundary extension as a detected global-linear one: the fitted
                # line predicts the edge pairs, and every extension is
                # content-verified before being kept.  This must run before the
                # edge snap: snapped pairs are artificial timestamps with no
                # normalized frames behind them, while the fit and the
                # verification may only ever see real matched pairs.
                generic_fit = self.detect_global_linear(
                    matching_pairs, self.lhs_all_frames, self.rhs_all_frames,
                )
                if generic_fit is not None:
                    matching_pairs = self._extrapolate_and_verify_global_linear(
                        generic_fit, matching_pairs, lhs_normalized_frames, rhs_normalized_frames,
                    )
                    debug.dump_matches(matching_pairs, "after verified linear extrapolation")
                matching_pairs = self.snap_to_edges(matching_pairs, self.lhs_all_frames, self.rhs_all_frames)

        if timing_discontinuities:
            debug.dump_matches(matching_pairs, "timing discontinuities")
            timing_discontinuity_groups = self._group_content_discontinuities(
                list(timing_discontinuities)
            )
            timing_discontinuity_debug_path = debug.dump_timing_discontinuities(
                timing_discontinuity_groups,
                lhs_label=self.lhs_label,
                rhs_label=self.rhs_label,
                lhs_fps=self.lhs_fps,
                rhs_fps=self.rhs_fps,
                prepare_frames=lambda: self._extract_discontinuity_frames(
                    timing_discontinuity_groups
                ),
            )
            self.logger.warning(
                "Matched content has continuous decoded advances but discontinuous presentation timestamps; "
                "the candidate %s frame relation is unsafe for audio execution.", relation.value,
            )
        else:
            timing_discontinuity_debug_path = None
            self.logger.info("Mapping relation chosen: %s", relation.value)

        return SegmentsMappingResult(
            mapping=matching_pairs,
            lhs_all_frames=self.lhs_all_frames,
            rhs_all_frames=self.rhs_all_frames,
            relation=relation,
            lhs_fps=self.lhs_fps,
            rhs_fps=self.rhs_fps,
            frame_slope=frame_slope,
            timing_discontinuities=timing_discontinuities,
            timing_discontinuity_debug_path=timing_discontinuity_debug_path,
        )

    def _log_relation_diagnostics(self, matching_pairs: list[tuple[int, int]]) -> None:
        debug_routines.log_relation_diagnostics(
            self.logger,
            matching_pairs,
            self.lhs_all_frames,
            self.rhs_all_frames,
            lhs_label=self.lhs_label,
            rhs_label=self.rhs_label,
            lhs_fps=self.lhs_fps,
            rhs_fps=self.rhs_fps,
            max_constant_offset_std=self._MAX_CONSTANT_OFFSET_STD,
            max_drift_slope_delta=self._MAX_DRIFT_SLOPE_DELTA,
        )

    def _normalize_frames(
        self,
        frames: FramesInfo,
        output_dir: str,
        desc: str = "Normalizing frames",
        prefix: str = "",
    ) -> FramesInfo:
        return self.frame_comparison.normalize_frames(
            frames,
            output_dir,
            desc,
            prefix,
        )

    _is_rich = staticmethod(frame_comparison.is_rich)
    _extracted_path = staticmethod(frame_comparison.extracted_path)
    _filter_low_detailed = staticmethod(frame_comparison.filter_low_detail)
    _get_frames_for_timestamps = staticmethod(
        frame_comparison.frames_for_timestamps
    )

    def _three_before(self, timestamps: list[int], target: int) -> list[int]:
        timestamps = sorted(timestamps)
        idx = int(np.searchsorted(timestamps, target))
        return list(filter(lambda x: x in timestamps, timestamps[max(0, idx-3):idx]))

    def _nearest_three(self, timestamps: list[int], target: int) -> list[int]:
        timestamps = sorted(timestamps)
        idx = int(np.searchsorted(timestamps, target))
        return list(filter(lambda x: x in timestamps, timestamps[max(0, idx-1):idx+2]))

    @staticmethod
    def _snap_to_nearest_frame(keys: list[int], target: int) -> int:
        """Return the timestamp from *keys* closest to *target*."""
        idx = int(np.searchsorted(keys, target))
        candidates = []
        if idx > 0:
            candidates.append(keys[idx - 1])
        if idx < len(keys):
            candidates.append(keys[idx])
        return min(candidates, key=lambda k: abs(k - target))

    def _best_phash_match(
        self,
        lhs_ts: int,
        rhs_ts_guess: int,
        lhs_all_set: FramesInfo,
        rhs_all_set: FramesInfo,
        *,
        timeline_ratio: float = 1.0,
    ) -> tuple[int, int] | None:
        """Refine one scene anchor using its local frame sequence.

        A single perceptual hash cannot reliably distinguish adjacent frames
        in a mostly static shot.  Score each candidate alignment across the
        five-frame neighbourhood of the original lhs anchor instead.  This
        keeps a scene cut aligned even when one wrong adjacent-frame pair has
        a slightly smaller hash distance (or ties the correct pair).  Moving
        the scene anchor additionally requires the winning sequence to look
        like genuinely shared pictures.  With different picture geometry,
        every raw-frame hash can be poor and tiny decoder-dependent differences
        between those poor scores are not evidence of a one-frame correction.
        """
        lhs_keys = sorted(lhs_all_set)
        rhs_keys = sorted(rhs_all_set)
        lhs_near = self._nearest_three(lhs_keys, lhs_ts)
        rhs_near = self._nearest_three(rhs_keys, rhs_ts_guess)
        if not lhs_near or not rhs_near:
            return None

        if not isfinite(timeline_ratio) or timeline_ratio <= 0:
            timeline_ratio = 1.0

        lhs_anchor_frame = int(lhs_all_set[lhs_ts]["frame_id"])
        lhs_samples = sorted(
            timestamp
            for timestamp, info in lhs_all_set.items()
            if abs(int(info["frame_id"]) - lhs_anchor_frame) <= 2
        )
        max_rhs_snap_error = max(2.0, 750.0 / self.rhs_fps)

        sequence_candidates: list[tuple[float, float, int, int]] = []
        single_frame_candidates: list[tuple[float, float, int, int]] = []
        for lhs_candidate in lhs_near:
            for rhs_candidate in rhs_near:
                movement = (
                    abs(lhs_candidate - lhs_ts) * self.lhs_fps
                    + abs(rhs_candidate - rhs_ts_guess) * self.rhs_fps
                ) / 1000
                single_distance = abs(
                    self.phash.get(self._extracted_path(lhs_all_set[lhs_candidate]))
                    - self.phash.get(self._extracted_path(rhs_all_set[rhs_candidate]))
                )
                single_frame_candidates.append((
                    float(single_distance), movement,
                    lhs_candidate, rhs_candidate,
                ))

                distances = []
                for lhs_sample in lhs_samples:
                    predicted_rhs = (
                        rhs_candidate
                        + (lhs_sample - lhs_candidate) / timeline_ratio
                    )
                    rhs_sample = self._snap_to_nearest_frame(
                        rhs_keys, round(predicted_rhs),
                    )
                    if abs(rhs_sample - predicted_rhs) > max_rhs_snap_error:
                        break
                    distances.append(abs(
                        self.phash.get(self._extracted_path(lhs_all_set[lhs_sample]))
                        - self.phash.get(self._extracted_path(rhs_all_set[rhs_sample]))
                    ))

                if len(distances) == len(lhs_samples) and len(distances) >= 3:
                    sequence_candidates.append((
                        float(np.mean(distances)), movement,
                        lhs_candidate, rhs_candidate,
                    ))

        if sequence_candidates:
            best = min(sequence_candidates)
            if best[0] > self._MIN_PHASH_CUTOFF:
                return lhs_ts, rhs_ts_guess
        elif single_frame_candidates:
            best = min(single_frame_candidates)
        else:
            return None
        _, _, best_lhs, best_rhs = best
        return best_lhs, best_rhs

    def _build_matches(self, lhs: FramesInfo, rhs: FramesInfo) -> list[tuple[int, int, int]]:
        lhs_items = list(lhs.items())
        rhs_items = list(rhs.items())

        all_matches = []
        for lhs_ts, lhs_info in lhs_items:
            lhs_hash = self.phash.get(self._extracted_path(lhs_info))
            for rhs_ts, rhs_info in rhs_items:
                rhs_hash = self.phash.get(self._extracted_path(rhs_info))
                distance = abs(lhs_hash - rhs_hash)
                all_matches.append((distance, lhs_ts, rhs_ts))

        all_matches.sort()
        return all_matches

    def _build_initial_candidates(self, lhs: FramesInfo, rhs: FramesInfo) -> list[tuple[int, int]]:
        all_matches = self._build_matches(lhs, rhs)

        used_lhs = set()
        used_rhs = set()
        pairs = []

        for distance, lhs_ts, rhs_ts in all_matches:
            if lhs_ts not in used_lhs and rhs_ts not in used_rhs:
                pairs.append((lhs_ts, rhs_ts))
                used_lhs.add(lhs_ts)
                used_rhs.add(rhs_ts)

        return sorted(pairs)

    def _reject_outliers(self, pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(pairs) < 3:
            return pairs

        lhs_vals, rhs_vals = zip(*pairs)
        lhs_array = np.array(lhs_vals).reshape(-1, 1)
        rhs_array = np.array(rhs_vals)
        model = RANSACRegressor(LinearRegression(), residual_threshold=5000)
        model.fit(lhs_array, rhs_array)
        inliers = model.inlier_mask_
        return [p for p, keep in zip(pairs, inliers) if keep]

    def _check_history(self, pair: tuple[int, int], lhs_pool: FramesInfo, rhs_pool: FramesInfo, cutoff: float) -> bool:
        lhs_three = self._three_before(list(lhs_pool.keys()), pair[0])
        rhs_three = self._three_before(list(rhs_pool.keys()), pair[1])

        if len(lhs_three) < 3 and len(rhs_three) < 3:
            return True
        if not lhs_three or not rhs_three:
            return True

        # At least one available match before the current pair is required.
        # Sparse or irregular PTS can leave one side with only one or two
        # earlier frames even though the other side has the full three.
        lhs_frames = {l: lhs_pool[l] for l in lhs_three}
        rhs_frames = {r: rhs_pool[r] for r in rhs_three}
        matches = self._build_matches(lhs_frames, rhs_frames)

        if len(matches) > 0:
            best_match = matches[0][0]

            if best_match <= cutoff:
                return True

        return False

    def _extrapolate_matches(self, known_pairs: list[tuple[int, int]], lhs_pool: FramesInfo, rhs_pool: FramesInfo, phash: PhashCache) -> list[tuple[int, int]]:
        known_pairs.sort()
        lhs_used = {l for l, _ in known_pairs}
        rhs_used = {r for _, r in known_pairs}
        lhs_free = sorted(set(lhs_pool.keys()) - lhs_used)
        rhs_keys = sorted(rhs_pool.keys())

        if len(known_pairs) < 2:
            return known_pairs

        median_ratio = PairMatcher.calculate_ratio(known_pairs)
        first_known_pair = known_pairs[0]
        cutoff = self._calculate_cutoff(phash, known_pairs, lhs_pool, rhs_pool)

        new_pairs = []
        for l in lhs_free:
            expected_rhs = first_known_pair[1] + (l - first_known_pair[0]) / median_ratio
            nearest_rhs_candidates = self._nearest_three(rhs_keys, int(expected_rhs))

            for rhs_candidate in nearest_rhs_candidates:
                ratio = (l - first_known_pair[0]) / (rhs_candidate - first_known_pair[1]) if (rhs_candidate - first_known_pair[1]) != 0 else None
                if ratio and PairMatcher.is_ratio_acceptable(ratio, median_ratio):
                    if rhs_candidate not in rhs_used:
                        # make sure lhs and rhs_candidate are matching
                        lhs_path = self._extracted_path(lhs_pool[l])
                        rhs_path = self._extracted_path(rhs_pool[rhs_candidate])

                        pdiff = abs(phash.get(lhs_path) - phash.get(rhs_path))
                        phash_matching = pdiff < cutoff
                        matching = image_utils.are_images_similar(lhs_path, rhs_path)
                        if phash_matching and matching:
                            new_pairs.append((l, rhs_candidate))
                            rhs_used.add(rhs_candidate)
                            break

        return sorted(set(known_pairs + new_pairs))

    def _crop_both_sets(
        self,
        pairs_with_timestamps: list[tuple[int, int]],
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo,
        lhs_cropped_dir: str,
        rhs_cropped_dir: str,
    ) -> tuple[FramesInfo, FramesInfo]:
        return self.frame_comparison.crop_both(
            pairs_with_timestamps,
            lhs_frames,
            rhs_frames,
            lhs_cropped_dir,
            rhs_cropped_dir,
        )

    @staticmethod
    def _calculate_cutoff(
        phash: PhashCache,
        pairs: list[tuple[int, int]],
        lhs: FramesInfo,
        rhs: FramesInfo,
    ) -> float:
        return frame_comparison.calculate_cutoff(phash, pairs, lhs, rhs)

    @staticmethod
    def _floored_phash_cutoff(cutoff: float) -> float:
        return frame_comparison.floored_phash_cutoff(cutoff)

    def _make_pairs(self, lhs: FramesInfo, rhs: FramesInfo, lhs_all: FramesInfo, rhs_all: FramesInfo) -> list[tuple[int, int]]:
        # Pipeline
        lhs = PairMatcher._filter_low_detailed(lhs)
        rhs = PairMatcher._filter_low_detailed(rhs)

        if not lhs or not rhs:
            return []

        initial = self._build_initial_candidates(lhs, rhs)
        self.logger.debug(f"Initial candidates:        {PairMatcher.summarize_pairs(self.phash, initial, lhs_all, rhs_all)}")

        stable = self._reject_outliers(initial)
        self.logger.debug(f"After linear matching:     {PairMatcher.summarize_pairs(self.phash, stable, lhs_all, rhs_all)}")

        stable = PairMatcher.filter_phash_outliers(self.phash, stable, lhs_all, rhs_all)
        self.logger.debug(f"Phash outlier elimination: {PairMatcher.summarize_pairs(self.phash, stable, lhs_all, rhs_all)}")

        extrapolated = self._extrapolate_matches(stable, lhs, rhs, self.phash)
        self.logger.debug(f"Extrapolation:             {PairMatcher.summarize_pairs(self.phash, extrapolated, lhs_all, rhs_all)}")

        timeline_ratio = (
            PairMatcher.calculate_ratio(extrapolated)
            if len(extrapolated) >= 2 else 1.0
        )
        extrapolated_refined: list[tuple[int, int]] = []
        for l, r in extrapolated:
            best_match = self._best_phash_match(
                l, r, lhs_all, rhs_all,
                timeline_ratio=timeline_ratio,
            )
            if best_match is not None:
                extrapolated_refined.append(best_match)
        self.logger.debug(f"Frame adjustment:          {PairMatcher.summarize_pairs(self.phash, extrapolated_refined, lhs_all, rhs_all)}")

        outliers_eliminated = PairMatcher.filter_phash_outliers(self.phash, extrapolated_refined, lhs_all, rhs_all)
        self.logger.debug(f"Phash outlier elimination: {PairMatcher.summarize_pairs(self.phash, outliers_eliminated, lhs_all, rhs_all)}")

        orb_filtered = [
            (lhs_ts, rhs_ts) for lhs_ts, rhs_ts in outliers_eliminated
            if image_utils.are_images_similar(
                self._extracted_path(lhs_all[lhs_ts]), self._extracted_path(rhs_all[rhs_ts]))
        ]
        self.logger.debug(f"After ORB elimination:     {PairMatcher.summarize_pairs(self.phash, orb_filtered, lhs_all, rhs_all)}")

        raw_cutoff = self._calculate_cutoff(self.phash, orb_filtered, lhs_all, rhs_all)
        cutoff = self._floored_phash_cutoff(raw_cutoff)
        final = [pair for pair in orb_filtered if self._check_history(pair, lhs_all, rhs_all, cutoff)]
        self.logger.debug(f"After history analysis:    {PairMatcher.summarize_pairs(self.phash, final, lhs_all, rhs_all)}")

        unique_pairs = sorted(set(final))

        consistent = self._drop_pairs_breaking_local_linearity(unique_pairs, lhs_all, rhs_all)
        self.logger.debug(f"Local linearity filter:    {PairMatcher.summarize_pairs(self.phash, consistent, lhs_all, rhs_all)}")

        self.logger.debug(PairMatcher.summarize_segments(consistent, self.lhs_fps, self.rhs_fps, lhs_label=self.lhs_label, rhs_label=self.rhs_label))

        return consistent

    @staticmethod
    def _compute_frame_ranges(
        scene_timestamps: list[int],
        all_frames: FramesInfo,
        margin: int = 5,
    ) -> list[tuple[int, int]]:
        """Compute merged frame-number ranges around scene-change timestamps.

        For each *scene_timestamp*, looks up the ``frame_id`` in
        *all_frames* and creates a ``(frame_id − margin, frame_id + margin)``
        range.  Overlapping/adjacent ranges are merged and clamped to
        ``[0, max_frame_id]``.

        Returns a sorted list of ``(start, end)`` inclusive ranges.
        """
        max_frame = max(int(info["frame_id"]) for info in all_frames.values())
        frame_ids: list[int] = []
        for ts in scene_timestamps:
            if ts in all_frames:
                frame_ids.append(int(all_frames[ts]["frame_id"]))

        if not frame_ids:
            return [(0, max_frame)]

        raw = sorted(set(
            (max(0, fid - margin), min(max_frame, fid + margin))
            for fid in frame_ids
        ))

        merged: list[tuple[int, int]] = [raw[0]]
        for start, end in raw[1:]:
            if start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        return merged

    @staticmethod
    def _extracted_subset(all_frames: FramesInfo) -> FramesInfo:
        """Return only entries from *all_frames* that have been extracted (path is not None)."""
        return {ts: info for ts, info in all_frames.items() if info.get("path") is not None}

    @staticmethod
    def _complement_ranges(
        scene_ranges: list[tuple[int, int]], max_frame: int,
    ) -> list[tuple[int, int]]:
        """Return frame ranges NOT covered by *scene_ranges*."""
        merged = sorted(scene_ranges)
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for start, end in merged:
            if cursor < start:
                gaps.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
        if cursor <= max_frame:
            gaps.append((cursor, max_frame))
        return gaps

    def _detect_scenes(
        self,
        lhs_analysis: media_analysis.VideoScanResult,
        rhs_analysis: media_analysis.VideoScanResult,
    ) -> tuple[list[int], list[int]]:
        """Phase 1: Detect scene changes in both files."""
        lhs_scene_changes = self._detect_scenes_for(lhs_analysis, self.lhs_label)
        rhs_scene_changes = self._detect_scenes_for(rhs_analysis, self.rhs_label)

        if len(lhs_scene_changes) == 0 or len(rhs_scene_changes) == 0:
            raise RuntimeError("Not enough scene changes detected")

        return lhs_scene_changes, rhs_scene_changes

    def _detect_scenes_for(
        self,
        analysis: media_analysis.VideoScanResult,
        label: str,
    ) -> list[int]:
        scene_changes = analysis.scene_changes_at(self._SCENE_THRESHOLD)

        self.logger.info(
            "[1/6] Scene changes for %s selected from media scan at threshold "
            "%.6f (%d scenes)",
            label,
            self._SCENE_THRESHOLD,
            len(scene_changes),
        )
        return list(scene_changes)

    def _probe_frames(
        self,
        lhs_analysis: media_analysis.VideoScanResult | None = None,
        rhs_analysis: media_analysis.VideoScanResult | None = None,
    ) -> None:
        """Phase 2: Probe all frame timestamps (fast — no images written)."""
        if not self.lhs_all_frames:
            if lhs_analysis is None:
                lhs_analysis = self._analysis_result_for(
                    self.lhs_path,
                    self.lhs_label,
                    media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
                )
            self.lhs_all_frames = self._probe_frames_for(lhs_analysis, self.lhs_label)
        if not self.rhs_all_frames:
            if rhs_analysis is None:
                rhs_analysis = self._analysis_result_for(
                    self.rhs_path,
                    self.rhs_label,
                    media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
                )
            self.rhs_all_frames = self._probe_frames_for(rhs_analysis, self.rhs_label)

    def _probe_frames_for(
        self,
        analysis: media_analysis.VideoScanResult,
        label: str,
    ) -> FramesInfo:
        self.logger.info(
            "[2/6] Frame probes for %s restored from media scan (%d frames)",
            label,
            len(analysis.frames),
        )
        return {
            timestamp: FrameInfo(
                frame_id=int(info["frame_id"]),
                path=info["path"],
            )
            for timestamp, info in analysis.frames_copy().items()
        }

    def _extract_scene_frames(
        self,
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Phase 3: Extract frames around scene changes and log key frames."""
        scene_margin = 5
        lhs_scene_ranges = self._compute_frame_ranges(lhs_scene_changes, self.lhs_all_frames, margin=scene_margin)
        rhs_scene_ranges = self._compute_frame_ranges(rhs_scene_changes, self.rhs_all_frames, margin=scene_margin)

        lhs_scene_frame_count = sum(e - s + 1 for s, e in lhs_scene_ranges)
        rhs_scene_frame_count = sum(e - s + 1 for s, e in rhs_scene_ranges)
        self.logger.info(
            f"Selective extraction: {lhs_scene_frame_count}/{len(self.lhs_all_frames)} frames for {self.lhs_label}, "
            f"{rhs_scene_frame_count}/{len(self.rhs_all_frames)} frames for {self.rhs_label}"
        )

        self._extract_scene_frames_for(
            self.lhs_path, self.lhs.all_dir, lhs_scene_ranges, self.lhs_all_frames, self.lhs_label,
        )
        self._extract_scene_frames_for(
            self.rhs_path, self.rhs.all_dir, rhs_scene_ranges, self.rhs_all_frames, self.rhs_label,
        )

        lhs_key_frames_str = [str(self.lhs_all_frames[lhs]["frame_id"]) for lhs in lhs_scene_changes if lhs in self.lhs_all_frames]
        rhs_key_frames_str = [str(self.rhs_all_frames[rhs]["frame_id"]) for rhs in rhs_scene_changes if rhs in self.rhs_all_frames]

        self.logger.debug(f"{self.lhs_label} key frames: {' '.join(lhs_key_frames_str)}")
        self.logger.debug(f"{self.rhs_label} key frames: {' '.join(rhs_key_frames_str)}")

        return lhs_scene_ranges, rhs_scene_ranges

    def _extract_scene_frames_for(
        self,
        video_path: str,
        target_dir: str,
        scene_ranges: list[tuple[int, int]],
        probed_metadata: FramesInfo,
        label: str,
    ) -> None:
        """Restore initial cached frames, then extract only missing requested frames."""
        initial = not any(info.get("path") for info in probed_metadata.values())
        restored = False
        if initial and self.cache:
            restored = self.cache.load_scene_frames(video_path, target_dir, probed_metadata)
            if restored:
                self.logger.info("[3/6] Scene frames for %s restored from cache", label)

        requested = {frame for start, end in scene_ranges for frame in range(start, end + 1)}
        missing = [
            ts for ts, info in probed_metadata.items()
            if int(info["frame_id"]) in requested and not info.get("path")
        ]
        if missing:
            video_utils.extract_frames_at_ranges(
                video_path, target_dir,
                self._compute_frame_ranges(missing, probed_metadata, margin=0),
                probed_metadata,
                scale=(960, -2), format="png", interruption=self.interruption,
                desc=f"[3/6] Extracting scene frames: {label}", logger=self.logger,
            )

            # Restored frames may be symlinks into the cache; replacing that
            # cache would invalidate the files currently used by this run.
            if initial and not restored and self.cache:
                self.cache.save_scene_frames(video_path, target_dir, probed_metadata)

    def _normalize_extracted(
        self,
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
        lhs_normalized_frames: FramesInfo | None = None,
        rhs_normalized_frames: FramesInfo | None = None,
    ) -> tuple[FramesInfo, FramesInfo, FramesInfo, FramesInfo]:
        """Phase 4: Normalize missing frames in-place and identify key frames.

        Returns (lhs_normalized, rhs_normalized, lhs_key_frames, rhs_key_frames).
        """
        if lhs_normalized_frames is None:
            lhs_normalized_frames = {}
        if rhs_normalized_frames is None:
            rhs_normalized_frames = {}
        for all_frames, normalized, directory, label in (
            (self.lhs_all_frames, lhs_normalized_frames, self.lhs.normalized_dir, self.lhs_label),
            (self.rhs_all_frames, rhs_normalized_frames, self.rhs.normalized_dir, self.rhs_label),
        ):
            missing = {
                ts: info for ts, info in self._extracted_subset(all_frames).items()
                if ts not in normalized
            }
            if missing:
                normalized.update(self._normalize_frames(
                    missing, directory, desc=f"[4/6] Normalizing: {label}",
                ))

        lhs_key_frames = PairMatcher._get_frames_for_timestamps(lhs_scene_changes, lhs_normalized_frames)
        rhs_key_frames = PairMatcher._get_frames_for_timestamps(rhs_scene_changes, rhs_normalized_frames)

        return lhs_normalized_frames, rhs_normalized_frames, lhs_key_frames, rhs_key_frames

    def _match_key_frames(
        self,
        lhs_key_frames: FramesInfo,
        rhs_key_frames: FramesInfo,
        lhs_normalized_frames: FramesInfo,
        rhs_normalized_frames: FramesInfo,
        *,
        debug: DebugRoutines | None = None,
    ) -> list[tuple[int, int]]:
        """Match the initially selected scene frames."""
        self.logger.info("[5/6] Matching key frames")
        matching_pairs = self._make_pairs(
            lhs_key_frames,
            rhs_key_frames,
            lhs_normalized_frames,
            rhs_normalized_frames,
        )
        matching_pairs = sorted(set(matching_pairs))
        if not matching_pairs:
            raise RuntimeError("No matching pairs found between the two files")

        if debug is not None:
            lhs_extracted = self._extracted_subset(self.lhs_all_frames)
            rhs_extracted = self._extracted_subset(self.rhs_all_frames)
            debug.dump_matches(matching_pairs, "initial matching")
            self.logger.debug("Pairs summary after initial matching:")
            self.logger.debug(PairMatcher.summarize_pairs(
                self.phash,
                matching_pairs,
                lhs_extracted,
                rhs_extracted,
                verbose=True,
            ))
        return matching_pairs

    def _refine_global_linear_discontinuities(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_normalized_frames: FramesInfo,
        rhs_normalized_frames: FramesInfo,
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
        *,
        frame_slope: float,
    ) -> list[tuple[int, int]]:
        """Iteratively add scene evidence inside suspicious linear intervals."""
        matching_pairs = sorted(set(matching_pairs))

        def find_discontinuities(
            pairs: list[tuple[int, int]],
        ) -> list[tuple[int, int, int, int, int]]:
            return self.find_global_linear_content_discontinuities(
                pairs, self.lhs_all_frames, self.rhs_all_frames,
                frame_slope=frame_slope,
                lhs_fps=self.lhs_fps, rhs_fps=self.rhs_fps,
                require_decoded_confirmation=False,
            )

        discontinuities = find_discontinuities(matching_pairs)
        if not discontinuities:
            return matching_pairs

        self.logger.info(
            "Starting local scene refinement for %d suspicious timing region(s): "
            "%s ↔ %s",
            len(discontinuities),
            self.lhs_label,
            self.rhs_label,
        )

        refined = set(matching_pairs)
        pending = list(reversed(discontinuities))
        unresolved = []
        # Each rematch consumes at least one previously unselected cached
        # candidate. The finite candidate pool bounds retries even if no new
        # pairs are found; candidate consumption alone never resolves a gap.
        while pending:
            self.interruption.check_for_stop()
            gap = pending.pop()
            lhs_from, lhs_to, rhs_from, rhs_to, _ = gap
            lhs_scenes, rhs_scenes, scenes_added = (
                self._refine_scenes_in_region(
                    gap,
                    lhs_scene_changes, rhs_scene_changes,
                )
            )
            if scenes_added:
                self._extract_scene_frames(lhs_scenes, rhs_scenes)
                _, _, lhs_candidates, rhs_candidates = self._normalize_extracted(
                    lhs_scenes, rhs_scenes, lhs_normalized_frames, rhs_normalized_frames,
                )
                local_pairs = self._make_pairs(
                    lhs_candidates, rhs_candidates,
                    lhs_normalized_frames, rhs_normalized_frames,
                )
                local_pairs = sorted({
                    (lhs_from, rhs_from), (lhs_to, rhs_to),
                    *(pair for pair in local_pairs
                      if lhs_from < pair[0] < lhs_to
                      and rhs_from < pair[1] < rhs_to),
                })
                if set(local_pairs) - refined:
                    refined.update(local_pairs)
                    pending.extend(reversed(find_discontinuities(local_pairs)))
                else:
                    # Unchanged anchors imply the same suspected gap. Avoid
                    # rebuilding full-film coordinates to rediscover it.
                    pending.append(gap)
            else:
                unresolved.append(gap)

        refined_pairs = sorted(refined)
        self.logger.info(
            "Local scene refinement complete: %d additional matched pair(s), "
            "%d remaining suspicious region(s)",
            len(refined - set(matching_pairs)),
            len(unresolved),
        )
        return refined_pairs

    def _extract_and_refine_boundaries(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_scene_ranges: list[tuple[int, int]],
        rhs_scene_ranges: list[tuple[int, int]],
        lhs_normalized_frames: FramesInfo,
        rhs_normalized_frames: FramesInfo,
        debug: DebugRoutines,
    ) -> list[tuple[int, int]]:
        """Phase 6: Extract remaining frames and refine boundary pairs."""
        self.logger.info("[6/6] Refining boundaries")

        lhs_max_frame = max(int(info["frame_id"]) for info in self.lhs_all_frames.values())
        rhs_max_frame = max(int(info["frame_id"]) for info in self.rhs_all_frames.values())

        lhs_boundary_ranges = self._complement_ranges(lhs_scene_ranges, lhs_max_frame)
        rhs_boundary_ranges = self._complement_ranges(rhs_scene_ranges, rhs_max_frame)

        lhs_boundary_count = sum(e - s + 1 for s, e in lhs_boundary_ranges)
        rhs_boundary_count = sum(e - s + 1 for s, e in rhs_boundary_ranges)
        self.logger.info(
            f"Extracting boundary frames: {lhs_boundary_count} for {self.lhs_label}, "
            f"{rhs_boundary_count} for {self.rhs_label}"
        )

        video_utils.extract_frames_at_ranges(
            self.lhs_path, self.lhs.boundary_dir, lhs_boundary_ranges, self.lhs_all_frames,
            scale=(960, -2), format="png", interruption=self.interruption,
            desc=f"[6/6] Extracting boundary frames: {self.lhs_label}",
            logger=self.logger,
        )
        video_utils.extract_frames_at_ranges(
            self.rhs_path, self.rhs.boundary_dir, rhs_boundary_ranges, self.rhs_all_frames,
            scale=(960, -2), format="png", interruption=self.interruption,
            desc=f"[6/6] Extracting boundary frames: {self.rhs_label}",
            logger=self.logger,
        )

        # Normalize the newly extracted boundary frames and merge
        lhs_new = {ts: info for ts, info in self.lhs_all_frames.items()
                   if info.get("path") is not None and ts not in lhs_normalized_frames}
        rhs_new = {ts: info for ts, info in self.rhs_all_frames.items()
                   if info.get("path") is not None and ts not in rhs_normalized_frames}

        if lhs_new:
            lhs_boundary_normalized = self._normalize_frames(
                lhs_new, self.lhs.normalized_dir, prefix="b_",
                desc=f"[6/6] Normalizing boundary: {self.lhs_label}",
            )
            lhs_normalized_frames.update(lhs_boundary_normalized)

        if rhs_new:
            rhs_boundary_normalized = self._normalize_frames(
                rhs_new, self.rhs.normalized_dir, prefix="b_",
                desc=f"[6/6] Normalizing boundary: {self.rhs_label}",
            )
            rhs_normalized_frames.update(rhs_boundary_normalized)

        matching_pairs, _, _ = self._refine_boundary_pairs(
            matching_pairs, lhs_normalized_frames, rhs_normalized_frames, debug,
        )

        # Final boundary search on uncropped normalized frames.
        self.logger.debug("Final boundary search (uncropped frames, with extrapolation)")
        matching_pairs = self._final_uncropped_boundary_search(
            matching_pairs, lhs_normalized_frames, rhs_normalized_frames, debug,
        )

        return matching_pairs

    def _refine_boundary_pairs(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_normalized_frames: FramesInfo,
        rhs_normalized_frames: FramesInfo,
        debug: DebugRoutines,
    ) -> tuple[list[tuple[int, int]], FramesInfo, FramesInfo]:
        """Iteratively refine boundary pairs using cropped-frame matching.

        Crops both frame sets based on the current matching, searches for
        boundaries on the cropped frames, and repeats until convergence.

        Returns ``(matching_pairs, lhs_cropped, rhs_cropped)``.
        """
        prev_first, prev_last = None, None
        iteration = 0
        lhs_normalized_cropped_frames: FramesInfo = {}
        rhs_normalized_cropped_frames: FramesInfo = {}
        phash4normalized = PhashCache()

        while True:
            iteration += 1
            self.logger.debug(f"Boundary refinement iteration {iteration} (cropped frames)")
            self.interruption.check_for_stop()

            lhs_normalized_cropped_frames, rhs_normalized_cropped_frames = self._crop_both_sets(
                pairs_with_timestamps=matching_pairs,
                lhs_frames=lhs_normalized_frames,
                rhs_frames=rhs_normalized_frames,
                lhs_cropped_dir=self.lhs.normalized_cropped_dir,
                rhs_cropped_dir=self.rhs.normalized_cropped_dir,
            )

            first_lhs, first_rhs = matching_pairs[0]
            last_lhs, last_rhs = matching_pairs[-1]
            self.logger.debug(f"First pair: {lhs_normalized_cropped_frames[first_lhs]['path']} {rhs_normalized_cropped_frames[first_rhs]['path']}")
            self.logger.debug(f"Last pair:  {lhs_normalized_cropped_frames[last_lhs]['path']} {rhs_normalized_cropped_frames[last_rhs]['path']}")

            phash4normalized = PhashCache()
            self.logger.debug(f"Cropped and aligned:       {PairMatcher.summarize_pairs(phash4normalized, matching_pairs, lhs_normalized_cropped_frames, rhs_normalized_cropped_frames)}")

            cutoff = self._calculate_cutoff(phash4normalized, matching_pairs, lhs_normalized_cropped_frames, rhs_normalized_cropped_frames)

            first, last = self.look_for_boundaries(
                lhs_normalized_cropped_frames, rhs_normalized_cropped_frames,
                matching_pairs[0], matching_pairs[-1], cutoff, extrapolate=False,
            )

            if first == prev_first and last == prev_last:
                break

            if first != prev_first:
                matching_pairs = [first, *matching_pairs]
                prev_first = first
            if last != prev_last:
                matching_pairs = [*matching_pairs, last]
                prev_last = last

            debug.dump_matches(matching_pairs, "improving boundaries")

        self.logger.info(f"Boundary refinement converged after {iteration} iteration(s)")
        self.logger.debug("Status after boundaries lookup:\n")
        self.logger.debug(PairMatcher.summarize_segments(matching_pairs, self.lhs_fps, self.rhs_fps, lhs_label=self.lhs_label, rhs_label=self.rhs_label))
        self.logger.debug(PairMatcher.summarize_pairs(phash4normalized, matching_pairs, self.lhs_all_frames, self.rhs_all_frames, verbose=True))

        return matching_pairs, lhs_normalized_cropped_frames, rhs_normalized_cropped_frames

    def _final_uncropped_boundary_search(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_normalized_frames: FramesInfo,
        rhs_normalized_frames: FramesInfo,
        debug: DebugRoutines,
    ) -> list[tuple[int, int]]:
        """Run a final boundary search on uncropped normalized frames."""
        phash_uncropped = PhashCache()
        uncropped_cutoff = self._calculate_cutoff(phash_uncropped, matching_pairs, lhs_normalized_frames, rhs_normalized_frames)
        final_first, final_last = self.look_for_boundaries(
            lhs_normalized_frames, rhs_normalized_frames,
            matching_pairs[0], matching_pairs[-1],
            uncropped_cutoff,
        )
        if final_first != matching_pairs[0]:
            matching_pairs = [final_first, *matching_pairs]
            self.logger.debug(f"Uncropped search extended first boundary to {final_first}")
        if final_last != matching_pairs[-1]:
            matching_pairs = [*matching_pairs, final_last]
            self.logger.debug(f"Uncropped search extended last boundary to {final_last}")

        debug.dump_matches(matching_pairs, "after uncropped boundary search")
        self.logger.debug("Final status:\n")
        self.logger.debug(PairMatcher.summarize_segments(matching_pairs, self.lhs_fps, self.rhs_fps, lhs_label=self.lhs_label, rhs_label=self.rhs_label))
        self.logger.debug(PairMatcher.summarize_pairs(phash_uncropped, matching_pairs, self.lhs_all_frames, self.rhs_all_frames, verbose=True))

        return matching_pairs
