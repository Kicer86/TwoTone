
import enum
import logging
import os
from functools import partial
from math import isfinite
from typing import NamedTuple

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

from ..utils import generic_utils, image_utils, media_analysis
from . import debug_routines, frame_comparison, matching_analysis
from .boundary_search import BoundarySearch
from .debug_routines import DebugRoutines
from .frame_comparison import FrameComparison, MatchingSide
from .matching_analysis import Discontinuity, DiscontinuityGroup
from .melt_cache import MeltCache
from .melt_common import FrameInfo, FrameSpan, FramesInfo
from .phash_cache import PhashCache

GlobalLinearDiscontinuityReport = matching_analysis.GlobalLinearDiscontinuityReport



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
    used only for logging.  It must NOT be used to
    scale audio: when the declared fps differs from the real timestamp ratio it
    drifts audibly (~1.2 s across a film).  Audio scaling uses the matched-span
    timestamp ratio instead; ``AudioPatcher`` applies that global time scale.
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
    timing_discontinuities: tuple[Discontinuity, ...] = ()
    # Frame-based content gaps the global-linear classification could not
    # resolve with local scene evidence, including temporary offsets that
    # return at the next matched anchor.  Whether any is acceptable is decided
    # by ``mapping_validation``.
    content_discontinuities: tuple[Discontinuity, ...] = ()
    content_discontinuity_debug_path: str | None = None
    timing_discontinuity_debug_path: str | None = None
    # Gaps the time-domain detector finds in a GENERIC mapping; its
    # ``max(2000 ms, 2%)`` tolerance is far coarser than the frame-based
    # global-linear classification.
    generic_content_discontinuities: tuple[Discontinuity, ...] = ()

    # Track alignment needs only the extent of each decoded timeline; the
    # per-frame caches above serve diagnostics.
    @property
    def lhs_frame_span(self) -> FrameSpan | None:
        return FrameSpan.of(self.lhs_all_frames)

    @property
    def rhs_frame_span(self) -> FrameSpan | None:
        return FrameSpan.of(self.rhs_all_frames)


class PairMatcher:

    # Global-linear detection thresholds, shared with the relation-diagnostics
    # log so the quoted limits never drift from the real ones.
    _MAX_CONSTANT_OFFSET_STD = 1.0
    _MAX_DRIFT_SLOPE_DELTA = 0.05
    _MAX_MEDIAN_RESIDUAL_FRAMES = 1.5
    _MAX_P95_RESIDUAL_FRAMES = 4.0
    _MAX_OUTLIER_RATIO = 0.25
    _MIN_GLOBAL_LINEAR_SPAN_FRAMES = 250
    _SCENE_THRESHOLD = 0.30

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
    ) -> None:
        self.interruption = interruption
        self.wd = os.path.join(wd, "pair_matcher")
        self.logger = logger
        self.cache = cache
        self.media_analysis = media_analysis_session
        self.phash = PhashCache()
        lhs_video_data, rhs_video_data = (
            self._primary_video_data(path)
            for path in (lhs_path, rhs_path)
        )
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

    def _primary_video_data(self, path: str) -> dict:
        probe = self.media_analysis.probe(path)
        if probe.error is not None:
            raise RuntimeError(probe.error)
        return probe.normalized_data["video"][0]

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

    def has_identical_timeline_content(self) -> bool:
        """Quickly certify that both videos show the same content in-place.

        This is deliberately a conservative fast path.  It samples both
        boundaries and evenly-spaced interior frames at identical timestamps;
        a failed sample merely asks callers to use :meth:`create_segments_mapping`
        for the authoritative answer.  It never declares different-looking
        transfers incompatible by itself.
        """
        if self.lhs.duration_ms is None or self.rhs.duration_ms is None:
            return False
        if self.lhs.duration_ms != self.rhs.duration_ms:
            return False

        duration = self.lhs.duration_ms
        if duration <= 0:
            return False

        if self.lhs.fps == self.rhs.fps:
            lhs_scan = self._analysis_result_for(
                self.lhs.path,
                self.lhs.label,
                media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            )
            rhs_scan = self._analysis_result_for(
                self.rhs.path,
                self.rhs.label,
                media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            )
            return self._scans_have_identical_timeline_content(lhs_scan, rhs_scan)

        timestamps = list(media_analysis.identity_timestamps(duration, self.lhs.fps))
        self._probe_frames()
        lhs_timestamps = [self._nearest_frame_timestamp(self.lhs.all_frames, timestamp) for timestamp in timestamps]
        rhs_timestamps = [self._nearest_frame_timestamp(self.rhs.all_frames, timestamp) for timestamp in timestamps]
        identity_wd = os.path.join(self.debug_wd, "identity")
        lhs_frames = self._extract_identity_samples(self.lhs, lhs_timestamps, os.path.join(identity_wd, "lhs"))
        rhs_frames = self._extract_identity_samples(self.rhs, rhs_timestamps, os.path.join(identity_wd, "rhs"))
        if lhs_frames is None or rhs_frames is None:
            return False

        return self._identity_samples_match([
            (timestamp, lhs_frames[lhs_timestamp], rhs_frames[rhs_timestamp])
            for timestamp, lhs_timestamp, rhs_timestamp in zip(timestamps, lhs_timestamps, rhs_timestamps)
        ])

    def _extract_identity_samples(
        self,
        side: MatchingSide,
        timestamps: list[int],
        identity_wd: str,
    ) -> FramesInfo | None:
        sample_frames = {timestamp: side.all_frames[timestamp] for timestamp in set(timestamps)}
        frame_ranges = [
            (int(info["frame_id"]), int(info["frame_id"]))
            for info in sample_frames.values()
        ]
        os.makedirs(identity_wd, exist_ok=True)
        self.frame_comparison.extract_frames(
            side.path,
            identity_wd,
            frame_ranges,
            sample_frames,
            f"Identity samples: {side.label}",
        )
        if any(info.get("path") is None for info in sample_frames.values()):
            self.logger.debug("Could not extract all identity-check frames from %s.", side.path)
            return None
        normalized_wd = os.path.join(identity_wd, "normalized")
        os.makedirs(normalized_wd, exist_ok=True)
        return self.frame_comparison.normalize_frames(
            sample_frames,
            normalized_wd,
            desc=f"Normalizing identity samples: {side.label}",
        )

    @staticmethod
    def _nearest_frame_timestamp(frames: FramesInfo, timestamp: int) -> int:
        return min(frames, key=lambda candidate: abs(candidate - timestamp))

    def _scans_have_identical_timeline_content(
        self,
        lhs_scan: media_analysis.VideoScanResult,
        rhs_scan: media_analysis.VideoScanResult,
    ) -> bool:
        lhs_samples = lhs_scan.identity_samples
        rhs_samples = rhs_scan.identity_samples
        lhs_expected = len(media_analysis.identity_timestamps(self.lhs.duration_ms or 0, self.lhs.fps))
        rhs_expected = len(media_analysis.identity_timestamps(self.rhs.duration_ms or 0, self.rhs.fps))

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
            self.lhs.label,
        )
        rhs_frames = self._normalize_scan_samples(
            rhs_samples,
            os.path.join(identity_wd, "rhs"),
            self.rhs.label,
        )

        return self._identity_samples_match([
            (
                lhs_sample.target_ms,
                lhs_frames[lhs_sample.timestamp_ms],
                rhs_frames[rhs_sample.timestamp_ms],
            )
            for lhs_sample, rhs_sample in zip(lhs_samples, rhs_samples)
        ])

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
        return self.frame_comparison.normalize_frames(
            frames,
            output_dir,
            desc=f"Normalizing identity samples: {label}",
        )

    def _identity_samples_match(
        self,
        samples: list[tuple[int, FrameInfo, FrameInfo]],
    ) -> bool:
        """Compare identity samples given as ``(target_ms, lhs_frame, rhs_frame)``."""
        for target_ms, lhs_frame, rhs_frame in samples:
            distance = frame_comparison.phash_distance(self.phash, lhs_frame, rhs_frame)
            if distance > frame_comparison.MIN_PHASH_CUTOFF:
                self.logger.debug(
                    "Equal-length identity check failed near %d ms (pHash distance %d).",
                    target_ms,
                    distance,
                )
                return False

        self.logger.debug("Equal-length identity check passed (%d samples).", len(samples))
        return True

    def create_segments_mapping(self) -> SegmentsMappingResult:
        """Match both videos and map their shared content frame to frame."""
        lhs_analysis = self._analysis_result_for(
            self.lhs.path,
            self.lhs.label,
            media_analysis.MediaAnalysisFeature.MATCHING,
        )
        rhs_analysis = self._analysis_result_for(
            self.rhs.path,
            self.rhs.label,
            media_analysis.MediaAnalysisFeature.MATCHING,
        )
        lhs_scene_changes, rhs_scene_changes = self._detect_scenes(
            lhs_analysis,
            rhs_analysis,
        )
        self._probe_frames(lhs_analysis, rhs_analysis)
        lhs_scene_ranges, rhs_scene_ranges = self._extract_scene_frames(lhs_scene_changes, rhs_scene_changes)
        lhs_key_frames, rhs_key_frames = self._normalize_extracted(
            lhs_scene_changes, rhs_scene_changes,
        )

        debug = DebugRoutines(self.debug_wd, self.lhs.all_frames, self.rhs.all_frames)
        debug.dump_frames(lhs_key_frames, f"{self.lhs.label} key frames")
        debug.dump_frames(rhs_key_frames, f"{self.rhs.label} key frames")

        matching_pairs = self._match_key_frames(
            lhs_key_frames, rhs_key_frames,
            debug=debug,
        )

        # Diagnostic summary of the frame-space relationship.  Logged at INFO so
        # the chosen relation (and the audio strategy that follows) can be
        # understood without enabling --verbose.
        self._log_relation_diagnostics(matching_pairs)

        matching_pairs, global_linear_fit, timing_discontinuities, content_discontinuities = (
            self._certify_global_linear(
                matching_pairs, lhs_scene_changes, rhs_scene_changes,
            )
        )

        relation = (
            MappingRelation.GLOBAL_LINEAR
            if global_linear_fit is not None
            else MappingRelation.GENERIC
        )
        timing_discontinuity_debug_path: str | None = None
        content_discontinuity_debug_path: str | None = None
        generic_content_discontinuities: tuple[Discontinuity, ...] = ()
        if content_discontinuities:
            content_discontinuity_debug_path = self._report_content_discontinuities(
                matching_pairs, content_discontinuities, debug,
            )
        if timing_discontinuities:
            timing_discontinuity_debug_path = self._report_timing_discontinuities(
                matching_pairs, timing_discontinuities, relation, debug,
            )
        elif not self._has_content_gap(content_discontinuities):
            matching_pairs = self._extend_boundaries(
                matching_pairs,
                global_linear_fit,
                lhs_scene_ranges,
                rhs_scene_ranges,
                debug,
            )
            self.logger.info("Mapping relation chosen: %s", relation.value)
            if relation is MappingRelation.GENERIC:
                generic_content_discontinuities = tuple(
                    matching_analysis.find_content_discontinuities(matching_pairs)
                )

        return SegmentsMappingResult(
            mapping=matching_pairs,
            lhs_all_frames=self.lhs.all_frames,
            rhs_all_frames=self.rhs.all_frames,
            relation=relation,
            lhs_fps=self.lhs.fps,
            rhs_fps=self.rhs.fps,
            frame_slope=global_linear_fit.slope if global_linear_fit is not None else None,
            timing_discontinuities=timing_discontinuities,
            content_discontinuities=content_discontinuities,
            content_discontinuity_debug_path=content_discontinuity_debug_path,
            timing_discontinuity_debug_path=timing_discontinuity_debug_path,
            generic_content_discontinuities=generic_content_discontinuities,
        )

    def _log_relation_diagnostics(self, matching_pairs: list[tuple[int, int]]) -> None:
        debug_routines.log_relation_diagnostics(
            self.logger,
            matching_pairs,
            self.lhs.all_frames,
            self.rhs.all_frames,
            lhs_label=self.lhs.label,
            rhs_label=self.rhs.label,
            lhs_fps=self.lhs.fps,
            rhs_fps=self.rhs.fps,
            max_constant_offset_std=self._MAX_CONSTANT_OFFSET_STD,
            max_drift_slope_delta=self._MAX_DRIFT_SLOPE_DELTA,
        )

    def _detect_scenes(
        self,
        lhs_analysis: media_analysis.VideoScanResult,
        rhs_analysis: media_analysis.VideoScanResult,
    ) -> tuple[list[int], list[int]]:
        """Phase 1: Detect scene changes in both files."""
        lhs_scene_changes = self._detect_scenes_for(lhs_analysis, self.lhs.label)
        rhs_scene_changes = self._detect_scenes_for(rhs_analysis, self.rhs.label)

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
        for side, analysis in ((self.lhs, lhs_analysis), (self.rhs, rhs_analysis)):
            if side.all_frames:
                continue
            if analysis is None:
                analysis = self._analysis_result_for(
                    side.path,
                    side.label,
                    media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
                )
            side.all_frames = self._probe_frames_for(analysis, side.label)

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
        lhs_scene_ranges = self._compute_frame_ranges(lhs_scene_changes, self.lhs.all_frames, margin=scene_margin)
        rhs_scene_ranges = self._compute_frame_ranges(rhs_scene_changes, self.rhs.all_frames, margin=scene_margin)

        self.logger.info(
            "Selective extraction: %d/%d frames for %s, %d/%d frames for %s",
            sum(e - s + 1 for s, e in lhs_scene_ranges), len(self.lhs.all_frames), self.lhs.label,
            sum(e - s + 1 for s, e in rhs_scene_ranges), len(self.rhs.all_frames), self.rhs.label,
        )

        sides = (
            (self.lhs, lhs_scene_changes, lhs_scene_ranges),
            (self.rhs, rhs_scene_changes, rhs_scene_ranges),
        )
        for side, _, scene_ranges in sides:
            self._extract_scene_frames_for(side, scene_ranges)
        for side, scene_changes, _ in sides:
            self.logger.debug(
                "%s key frames: %s",
                side.label,
                " ".join(
                    str(side.all_frames[timestamp]["frame_id"])
                    for timestamp in scene_changes
                    if timestamp in side.all_frames
                ),
            )

        return lhs_scene_ranges, rhs_scene_ranges

    def _extract_scene_frames_for(
        self,
        side: MatchingSide,
        scene_ranges: list[tuple[int, int]],
    ) -> None:
        """Restore initial cached frames, then extract only missing requested frames."""
        initial = not any(info.get("path") for info in side.all_frames.values())
        restored = False
        if initial and self.cache:
            restored = self.cache.load_scene_frames(side.path, side.all_dir, side.all_frames)
            if restored:
                self.logger.info("[3/6] Scene frames for %s restored from cache", side.label)

        requested = {frame for start, end in scene_ranges for frame in range(start, end + 1)}
        missing = [
            ts for ts, info in side.all_frames.items()
            if int(info["frame_id"]) in requested and not info.get("path")
        ]
        if missing:
            self.frame_comparison.extract_frames(
                side.path, side.all_dir,
                self._compute_frame_ranges(missing, side.all_frames, margin=0),
                side.all_frames,
                f"[3/6] Extracting scene frames: {side.label}",
            )

            # Restored frames may be symlinks into the cache; replacing that
            # cache would invalidate the files currently used by this run.
            if initial and not restored and self.cache:
                self.cache.save_scene_frames(side.path, side.all_dir, side.all_frames)

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

    def _normalize_extracted(
        self,
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
    ) -> tuple[FramesInfo, FramesInfo]:
        """Phase 4: Normalize newly extracted frames and identify key frames.

        Returns (lhs_key_frames, rhs_key_frames).
        """
        for side in (self.lhs, self.rhs):
            missing = {
                ts: info for ts, info in self._extracted_subset(side.all_frames).items()
                if ts not in side.normalized_frames
            }
            if missing:
                side.normalized_frames.update(self.frame_comparison.normalize_frames(
                    missing, side.normalized_dir, desc=f"[4/6] Normalizing: {side.label}",
                ))

        return (
            frame_comparison.frames_for_timestamps(lhs_scene_changes, self.lhs.normalized_frames),
            frame_comparison.frames_for_timestamps(rhs_scene_changes, self.rhs.normalized_frames),
        )

    @staticmethod
    def _extracted_subset(all_frames: FramesInfo) -> FramesInfo:
        """Return only entries from *all_frames* that have been extracted (path is not None)."""
        return {ts: info for ts, info in all_frames.items() if info.get("path") is not None}

    def _match_key_frames(
        self,
        lhs_key_frames: FramesInfo,
        rhs_key_frames: FramesInfo,
        *,
        debug: DebugRoutines | None = None,
    ) -> list[tuple[int, int]]:
        """Match the initially selected scene frames."""
        self.logger.info("[5/6] Matching key frames")
        matching_pairs = self._make_pairs(lhs_key_frames, rhs_key_frames)
        matching_pairs = sorted(set(matching_pairs))
        if not matching_pairs:
            raise RuntimeError("No matching pairs found between the two files")

        if debug is not None:
            debug.dump_matches(matching_pairs, "initial matching")
            if self.logger.isEnabledFor(logging.DEBUG):
                lhs_extracted = self._extracted_subset(self.lhs.all_frames)
                rhs_extracted = self._extracted_subset(self.rhs.all_frames)
                self.logger.debug("Pairs summary after initial matching:")
                self.logger.debug(debug_routines.summarize_pairs(
                    self.phash, matching_pairs, lhs_extracted, rhs_extracted,
                    verbose=True,
                ))
        return matching_pairs

    def _make_pairs(self, lhs: FramesInfo, rhs: FramesInfo) -> list[tuple[int, int]]:
        # Pipeline
        lhs_all = self.lhs.normalized_frames
        rhs_all = self.rhs.normalized_frames
        lhs = frame_comparison.filter_low_detail(lhs)
        rhs = frame_comparison.filter_low_detail(rhs)

        if not lhs or not rhs:
            return []

        log_pairs = partial(
            debug_routines.log_pairs_summary,
            self.logger,
            self.phash,
            lhs_all,
            rhs_all,
        )
        initial = self._build_initial_candidates(lhs, rhs)
        log_pairs("Initial candidates:        ", initial)

        stable = self._reject_outliers(initial)
        log_pairs("After linear matching:     ", stable)

        stable = PairMatcher.filter_phash_outliers(self.phash, stable, lhs_all, rhs_all)
        log_pairs("Phash outlier elimination: ", stable)

        extrapolated = self._extrapolate_matches(stable, lhs, rhs, self.phash)
        log_pairs("Extrapolation:             ", extrapolated)

        timeline_ratio = (
            matching_analysis.calculate_ratio(extrapolated)
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
        log_pairs("Frame adjustment:          ", extrapolated_refined)

        outliers_eliminated = PairMatcher.filter_phash_outliers(self.phash, extrapolated_refined, lhs_all, rhs_all)
        log_pairs("Phash outlier elimination: ", outliers_eliminated)

        orb_filtered = [
            (lhs_ts, rhs_ts) for lhs_ts, rhs_ts in outliers_eliminated
            if image_utils.are_images_similar(
                frame_comparison.extracted_path(lhs_all[lhs_ts]), frame_comparison.extracted_path(rhs_all[rhs_ts]))
        ]
        log_pairs("After ORB elimination:     ", orb_filtered)

        raw_cutoff = frame_comparison.calculate_cutoff(self.phash, orb_filtered, lhs_all, rhs_all)
        cutoff = frame_comparison.floored_phash_cutoff(raw_cutoff)
        final = [pair for pair in orb_filtered if self._check_history(pair, lhs_all, rhs_all, cutoff)]
        log_pairs("After history analysis:    ", final)

        unique_pairs = sorted(set(final))

        consistent = self._drop_pairs_breaking_local_linearity(unique_pairs)
        log_pairs("Local linearity filter:    ", consistent)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(debug_routines.summarize_segments(
                consistent, self.lhs.fps, self.rhs.fps,
                lhs_label=self.lhs.label, rhs_label=self.rhs.label,
            ))

        return consistent

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

    def _build_matches(self, lhs: FramesInfo, rhs: FramesInfo) -> list[tuple[int, int, int]]:
        lhs_items = list(lhs.items())
        rhs_items = list(rhs.items())

        all_matches = []
        for lhs_ts, lhs_info in lhs_items:
            lhs_hash = self.phash.get(frame_comparison.extracted_path(lhs_info))
            for rhs_ts, rhs_info in rhs_items:
                rhs_hash = self.phash.get(frame_comparison.extracted_path(rhs_info))
                distance = abs(lhs_hash - rhs_hash)
                all_matches.append((distance, lhs_ts, rhs_ts))

        all_matches.sort()
        return all_matches

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
        lhs_near = matching_analysis.nearest_three(lhs_keys, lhs_ts)
        rhs_near = matching_analysis.nearest_three(rhs_keys, rhs_ts_guess)
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
        max_rhs_snap_error = max(2.0, 750.0 / self.rhs.fps)

        sequence_candidates: list[tuple[float, float, int, int]] = []
        single_frame_candidates: list[tuple[float, float, int, int]] = []
        for lhs_candidate in lhs_near:
            for rhs_candidate in rhs_near:
                movement = (
                    abs(lhs_candidate - lhs_ts) * self.lhs.fps
                    + abs(rhs_candidate - rhs_ts_guess) * self.rhs.fps
                ) / 1000
                single_distance = frame_comparison.phash_distance(
                    self.phash, lhs_all_set[lhs_candidate], rhs_all_set[rhs_candidate],
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
                    rhs_sample = matching_analysis.snap_to_nearest_frame(
                        rhs_keys, round(predicted_rhs),
                    )
                    if abs(rhs_sample - predicted_rhs) > max_rhs_snap_error:
                        break
                    distances.append(frame_comparison.phash_distance(
                        self.phash, lhs_all_set[lhs_sample], rhs_all_set[rhs_sample],
                    ))

                if len(distances) == len(lhs_samples) and len(distances) >= 3:
                    sequence_candidates.append((
                        float(np.mean(distances)), movement,
                        lhs_candidate, rhs_candidate,
                    ))

        if sequence_candidates:
            best = min(sequence_candidates)
            if best[0] > frame_comparison.MIN_PHASH_CUTOFF:
                return lhs_ts, rhs_ts_guess
        elif single_frame_candidates:
            best = min(single_frame_candidates)
        else:
            return None
        _, _, best_lhs, best_rhs = best
        return best_lhs, best_rhs

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

    def _three_before(self, timestamps: list[int], target: int) -> list[int]:
        timestamps = sorted(timestamps)
        idx = int(np.searchsorted(timestamps, target))
        return timestamps[max(0, idx-3):idx]

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

    def _extrapolate_matches(self, known_pairs: list[tuple[int, int]], lhs_pool: FramesInfo, rhs_pool: FramesInfo, phash: PhashCache) -> list[tuple[int, int]]:
        known_pairs.sort()
        lhs_used = {l for l, _ in known_pairs}
        rhs_used = {r for _, r in known_pairs}
        lhs_free = sorted(set(lhs_pool.keys()) - lhs_used)
        rhs_keys = sorted(rhs_pool.keys())

        if len(known_pairs) < 2:
            return known_pairs

        median_ratio = matching_analysis.calculate_ratio(known_pairs)
        first_known_pair = known_pairs[0]
        cutoff = frame_comparison.calculate_cutoff(phash, known_pairs, lhs_pool, rhs_pool)

        new_pairs = []
        for l in lhs_free:
            expected_rhs = first_known_pair[1] + (l - first_known_pair[0]) / median_ratio
            nearest_rhs_candidates = matching_analysis.nearest_three(rhs_keys, int(expected_rhs))

            for rhs_candidate in nearest_rhs_candidates:
                ratio = (l - first_known_pair[0]) / (rhs_candidate - first_known_pair[1]) if (rhs_candidate - first_known_pair[1]) != 0 else None
                if ratio and matching_analysis.is_ratio_acceptable(ratio, median_ratio):
                    if rhs_candidate not in rhs_used:
                        # make sure lhs and rhs_candidate are matching
                        lhs_path = frame_comparison.extracted_path(lhs_pool[l])
                        rhs_path = frame_comparison.extracted_path(rhs_pool[rhs_candidate])

                        pdiff = abs(phash.get(lhs_path) - phash.get(rhs_path))
                        phash_matching = pdiff < cutoff
                        matching = image_utils.are_images_similar(lhs_path, rhs_path)
                        if phash_matching and matching:
                            new_pairs.append((l, rhs_candidate))
                            rhs_used.add(rhs_candidate)
                            break

        return sorted(set(known_pairs + new_pairs))

    @staticmethod
    def filter_phash_outliers(phash: PhashCache, pairs: list[tuple[int, int]], lhs_set: FramesInfo, rhs_set: FramesInfo) -> list[tuple[int, int]]:
        if len(pairs) <= 3:
            # Too few data points for reliable MAD-based outlier detection.
            # Let downstream filters (ORB, history check) handle quality control.
            return pairs

        dists_array = np.array([
            frame_comparison.phash_distance(phash, lhs_set[l], rhs_set[r])
            for l, r in pairs
        ], dtype=float)
        med = float(np.median(dists_array))
        mad = float(np.median(np.abs(dists_array - med)))
        threshold = med + 1.5 * mad
        return [pair for pair, dist in zip(pairs, dists_array) if dist <= threshold]

    def _drop_pairs_breaking_local_linearity(
        self, pairs: list[tuple[int, int]],
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
                gaps = matching_analysis.find_global_linear_content_discontinuities(
                    result, self.lhs.all_frames, self.rhs.all_frames,
                    frame_slope=frame_slope, lhs_fps=self.lhs.fps, rhs_fps=self.rhs.fps,
                )
                indices = {pair: i for i, pair in enumerate(result)}
                flagged = [(indices[(l1, r1)], deficit) for l1, _, r1, _, deficit in gaps]
            else:
                flagged = matching_analysis._flagged_gap_deficits(result)
            rogue: tuple[int, int, int] | None = None
            for (gap_a, deficit_a), (gap_b, deficit_b) in zip(flagged, flagged[1:]):
                if gap_b != gap_a + 1 or deficit_a * deficit_b >= 0:
                    continue
                merged_gap = result[gap_b + 1][0] - result[gap_a][0]
                threshold = (
                    matching_analysis.MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES * 1000 / self.rhs.fps
                    if frame_slope is not None else max(
                        matching_analysis.MAX_GAP_TIME_DEFICIT_MS,
                        matching_analysis.MAX_GAP_TIME_DEFICIT_RATIO * merged_gap,
                    )
                )
                if abs(deficit_a + deficit_b) <= threshold and self._verify_false_pair(
                    result[gap_a], result[gap_b], result[gap_b + 1],
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
        if not self.lhs.all_frames or not self.rhs.all_frames:
            return False
        if any(l not in self.lhs.normalized_frames or r not in self.rhs.normalized_frames
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
            [before, after],
            minimum_cutoff=frame_comparison.MIN_PHASH_CUTOFF,
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
                lhs = self._nearest_frame_timestamp(self.lhs.all_frames, lhs_ts + offset)
                predicted = round(before[1] + (lhs - before[0]) * slope)
                original = round(rhs_ts + (lhs - lhs_ts) * slope)
                if not (before[1] < predicted < after[1] and before[1] < original < after[1]):
                    return False
                samples.append((
                    lhs,
                    self._nearest_frame_timestamp(self.rhs.all_frames, predicted),
                    self._nearest_frame_timestamp(self.rhs.all_frames, original),
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
                if not all(img is not None and frame_comparison.is_rich(img)
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
                if original_img is not None and frame_comparison.is_rich(original_img):
                    if not self.frame_comparison.boundary_content_matches(
                        ctx,
                        lhs,
                        original,
                    ):
                        contradicted[side] = True
            if all(len(pairs) >= 2 for pairs in supported) and all(contradicted):
                return True
        return False

    def _certify_global_linear(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
    ) -> tuple[
        list[tuple[int, int]],
        GlobalLinearFit | None,
        tuple[Discontinuity, ...],
        tuple[Discontinuity, ...],
    ]:
        """Certify one global linear frame relation for the matched pairs.

        A candidate relation is a constant offset (slope 1) or a fitted
        near-identity drift.  Suspected content discontinuities first drop
        pairs that break local linearity, then gather scene evidence inside the
        suspicious regions.  Whenever that changes the pairs, the final pairs
        must certify the relation again before it can determine boundary
        prediction or the audio strategy.

        Returns the final pairs, the certified fit (``None`` for the generic
        path), the timing discontinuities and the content discontinuities
        local scene evidence could not resolve.
        """
        fit = self.detect_global_linear(
            matching_pairs, self.lhs.all_frames, self.rhs.all_frames,
        )
        if fit is None:
            return matching_pairs, None, (), ()

        initial_frame_slope = fit.slope
        report = self._classify_discontinuities(matching_pairs, initial_frame_slope)
        if report.content:
            cleaned_pairs = self._drop_pairs_breaking_local_linearity(
                matching_pairs,
                frame_slope=initial_frame_slope,
            )
            if cleaned_pairs != matching_pairs:
                matching_pairs = cleaned_pairs
                fit = self.detect_global_linear(
                    matching_pairs, self.lhs.all_frames, self.rhs.all_frames,
                )
                report = self._classify_discontinuities(
                    matching_pairs,
                    fit.slope if fit is not None else initial_frame_slope,
                )

        if report.content:
            matching_pairs, fit, report = self._refine_content_discontinuities(
                matching_pairs,
                lhs_scene_changes,
                rhs_scene_changes,
                frame_slope=fit.slope if fit is not None else initial_frame_slope,
            )
        return matching_pairs, fit, tuple(report.timing), tuple(report.content)

    def detect_global_linear(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
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
        PAL speedup vs 24 fps); that only means the audio must be stretched.
        Whether the implied speed change is plausible is decided by
        ``mapping_validation``, not here.

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
            lhs_frame_ids_by_timestamp = matching_analysis.timeline_frame_ids(lhs_all_frames, self.lhs.fps)
            rhs_frame_ids_by_timestamp = matching_analysis.timeline_frame_ids(rhs_all_frames, self.rhs.fps)
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
            lhs_fps=self.lhs.fps,
            rhs_fps=self.rhs.fps,
            max_std=self._MAX_CONSTANT_OFFSET_STD,
            min_two_pair_span=self._MIN_GLOBAL_LINEAR_SPAN_FRAMES,
            logger=self.logger,
        )
        is_constant_offset = fit is not None
        if fit is None:
            fit = matching_analysis.fit_linear_drift(
                lhs_frame_ids,
                rhs_frame_ids,
                lhs_fps=self.lhs.fps,
                rhs_fps=self.rhs.fps,
                max_slope_delta=self._MAX_DRIFT_SLOPE_DELTA,
                max_median_residual_frames=self._MAX_MEDIAN_RESIDUAL_FRAMES,
                max_p95_residual_frames=self._MAX_P95_RESIDUAL_FRAMES,
                max_outlier_ratio=self._MAX_OUTLIER_RATIO,
                min_span_frames=self._MIN_GLOBAL_LINEAR_SPAN_FRAMES,
                logger=self.logger,
            )
        if fit is None:
            return None
        slope, intercept = fit

        time_scale = slope * self.lhs.fps / self.rhs.fps
        debug_routines.log_global_linear_relation(
            self.logger,
            lhs_label=self.lhs.label,
            rhs_label=self.rhs.label,
            slope=slope,
            intercept=intercept,
            time_scale=time_scale,
            is_constant_offset=is_constant_offset,
        )
        return GlobalLinearFit(slope=slope, intercept=intercept,
                               is_constant_offset=is_constant_offset, time_scale=time_scale)

    def _classify_discontinuities(
        self,
        matching_pairs: list[tuple[int, int]],
        frame_slope: float,
    ) -> matching_analysis.GlobalLinearDiscontinuityReport:
        return matching_analysis.classify_global_linear_discontinuities(
            matching_pairs,
            self.lhs.all_frames,
            self.rhs.all_frames,
            frame_slope=frame_slope,
            lhs_fps=self.lhs.fps,
            rhs_fps=self.rhs.fps,
        )

    def _refine_content_discontinuities(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
        *,
        frame_slope: float,
    ) -> tuple[
        list[tuple[int, int]],
        GlobalLinearFit | None,
        matching_analysis.GlobalLinearDiscontinuityReport,
    ]:
        """Resolve suspected content discontinuities with local scene evidence."""
        refined_pairs = self._refine_global_linear_discontinuities(
            matching_pairs,
            list(lhs_scene_changes), list(rhs_scene_changes),
            frame_slope=frame_slope,
        )
        refined_pairs = self._drop_pairs_breaking_local_linearity(
            refined_pairs,
            frame_slope=frame_slope,
        )
        refined_fit = self.detect_global_linear(
            refined_pairs, self.lhs.all_frames, self.rhs.all_frames,
        )
        report = self._classify_discontinuities(
            refined_pairs,
            refined_fit.slope if refined_fit is not None else frame_slope,
        )

        if not report.content and not report.timing:
            if refined_fit is not None:
                self.logger.info(
                    "Local scene refinement resolved all suspected content discontinuities."
                )
            else:
                self.logger.info(
                    "Local scene refinement found no confirmed content discontinuity, "
                    "but the final observations do not support one global linear relation; "
                    "using the generic mapping path."
                )
        return refined_pairs, refined_fit, report

    def _refine_global_linear_discontinuities(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
        *,
        frame_slope: float,
    ) -> list[tuple[int, int]]:
        """Iteratively add scene evidence inside suspicious linear intervals."""
        matching_pairs = sorted(set(matching_pairs))

        def find_discontinuities(
            pairs: list[tuple[int, int]],
        ) -> list[Discontinuity]:
            return matching_analysis.find_global_linear_content_discontinuities(
                pairs, self.lhs.all_frames, self.rhs.all_frames,
                frame_slope=frame_slope,
                lhs_fps=self.lhs.fps, rhs_fps=self.rhs.fps,
                require_decoded_confirmation=False,
            )

        discontinuities = find_discontinuities(matching_pairs)
        if not discontinuities:
            return matching_pairs

        self.logger.info(
            "Starting local scene refinement for %d suspicious timing region(s): "
            "%s ↔ %s",
            len(discontinuities),
            self.lhs.label,
            self.rhs.label,
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
                lhs_candidates, rhs_candidates = self._normalize_extracted(
                    lhs_scenes, rhs_scenes,
                )
                local_pairs = self._make_pairs(lhs_candidates, rhs_candidates)
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

    def _refine_scenes_in_region(
        self,
        gap: Discontinuity,
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
    ) -> tuple[list[int], list[int], bool]:
        """Extend one scene list and return timestamps within the region."""
        lhs_from, lhs_to, rhs_from, rhs_to, _ = gap
        lhs_local = [ts for ts in lhs_scene_changes if lhs_from < ts < lhs_to]
        rhs_local = [ts for ts in rhs_scene_changes if rhs_from < ts < rhs_to]

        lhs_side = (self.lhs, lhs_from, lhs_to, lhs_scene_changes)
        rhs_side = (self.rhs, rhs_from, rhs_to, rhs_scene_changes)
        if len(lhs_local) <= len(rhs_local):
            sides = (lhs_side, rhs_side)
        else:
            sides = (rhs_side, lhs_side)

        self.logger.debug(
            "Refining suspicious region: %s %s-%s (%d scene(s)) "
            "↔ %s %s-%s (%d scene(s)); checking the sparser side first",
            self.lhs.label, generic_utils.ms_to_time(lhs_from),
            generic_utils.ms_to_time(lhs_to), len(lhs_local),
            self.rhs.label, generic_utils.ms_to_time(rhs_from),
            generic_utils.ms_to_time(rhs_to), len(rhs_local),
        )
        scenes_added = False
        for side_index, (side, start_ms, end_ms, scenes) in enumerate(sides):
            analysis = self._analysis_result_for(
                side.path,
                side.label,
                media_analysis.MediaAnalysisFeature.SCENE_CHANGES,
            )
            frame_timestamps = sorted(side.all_frames)
            candidates: dict[int, float] = {}
            for candidate in analysis.scene_candidates:
                if not start_ms < candidate.timestamp_ms < end_ms:
                    continue
                timestamp = matching_analysis.snap_to_nearest_frame(
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
                    side.label,
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
                side.label, len(new_scenes), cutoff,
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

    def _has_content_gap(self, content_discontinuities: tuple[Discontinuity, ...]) -> bool:
        """Whether a discontinuity interrupts the shared content.

        A temporary offset that returns at the next matched anchor keeps the
        shared content continuous; any other discontinuity leaves no single
        mapping whose boundaries can be extended.
        """
        return any(
            matching_analysis.temporary_frame_offset_excursion(
                group, self.lhs.all_frames, self.rhs.all_frames,
            ) is None
            for group in matching_analysis.group_content_discontinuities(
                list(content_discontinuities)
            )
        )

    def _report_content_discontinuities(
        self,
        matching_pairs: list[tuple[int, int]],
        content_discontinuities: tuple[Discontinuity, ...],
        debug: DebugRoutines,
    ) -> str | None:
        """Dump diagnostics for unresolved content steps and return their path."""
        debug.dump_matches(matching_pairs, "localized discontinuities")
        groups = matching_analysis.group_content_discontinuities(
            list(content_discontinuities)
        )
        return debug.dump_content_discontinuities(
            groups,
            lhs_label=self.lhs.label,
            rhs_label=self.rhs.label,
            lhs_fps=self.lhs.fps,
            rhs_fps=self.rhs.fps,
            prepare_frames=lambda: self._extract_discontinuity_frames(groups),
        )

    def _extract_discontinuity_frames(
        self,
        groups: list[DiscontinuityGroup],
    ) -> None:
        """Extract every missing decoded frame in final error regions."""
        # Discontinuity fields come as (from, to) pairs: lhs first, then rhs.
        for side, from_field in ((self.lhs, 0), (self.rhs, 2)):
            frames = side.all_frames
            requested: set[int] = set()
            for group in groups:
                start_ms, end_ms = group[0][from_field], group[-1][from_field + 1]
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
                side.label,
            )
            self.frame_comparison.extract_frames(
                side.path,
                side.all_dir,
                ranges,
                frames,
                f"Extracting discontinuity diagnostics: {side.label}",
            )

    def _report_timing_discontinuities(
        self,
        matching_pairs: list[tuple[int, int]],
        timing_discontinuities: tuple[Discontinuity, ...],
        relation: MappingRelation,
        debug: DebugRoutines,
    ) -> str | None:
        """Dump diagnostics for unsafe timestamp steps and return their path."""
        debug.dump_matches(matching_pairs, "timing discontinuities")
        groups = matching_analysis.group_content_discontinuities(
            list(timing_discontinuities)
        )
        debug_path = debug.dump_timing_discontinuities(
            groups,
            lhs_label=self.lhs.label,
            rhs_label=self.rhs.label,
            lhs_fps=self.lhs.fps,
            rhs_fps=self.rhs.fps,
            prepare_frames=lambda: self._extract_discontinuity_frames(groups),
        )
        self.logger.warning(
            "Matched content has continuous decoded advances but discontinuous presentation timestamps; "
            "the candidate %s frame relation is unsafe for audio execution.", relation.value,
        )
        return debug_path

    def _extend_boundaries(
        self,
        matching_pairs: list[tuple[int, int]],
        global_linear_fit: GlobalLinearFit | None,
        lhs_scene_ranges: list[tuple[int, int]],
        rhs_scene_ranges: list[tuple[int, int]],
        debug: DebugRoutines,
    ) -> list[tuple[int, int]]:
        """Move the mapping's first and last pairs toward the video edges."""
        if global_linear_fit is not None:
            # Use the certified pairs' observed span to predict shared content,
            # extend each boundary toward the video edge, and keep the
            # extension only where the predicted pair is content-verified (same
            # frame, or both low-entropy black).  A genuinely different intro/outro
            # fails verification and the boundary stays at the outermost real
            # match — the line is never blindly projected across divergent
            # content. No later timestamp edge-snap may bypass verification.
            matching_pairs = self.boundary_search.extrapolate_and_verify_global_linear(
                matching_pairs,
            )
            debug.dump_matches(matching_pairs, "after verified global-linear extrapolation")
            return matching_pairs

        # No global relation — fall back to the content/entropy-aware
        # iterative boundary search.
        matching_pairs = self._extract_and_refine_boundaries(
            matching_pairs, lhs_scene_ranges, rhs_scene_ranges, debug,
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
            matching_pairs, self.lhs.all_frames, self.rhs.all_frames,
        )
        if generic_fit is not None:
            matching_pairs = self.boundary_search.extrapolate_and_verify_global_linear(
                matching_pairs,
            )
            debug.dump_matches(matching_pairs, "after verified linear extrapolation")
        return self.boundary_search.snap_to_edges(matching_pairs)

    def _extract_and_refine_boundaries(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_scene_ranges: list[tuple[int, int]],
        rhs_scene_ranges: list[tuple[int, int]],
        debug: DebugRoutines,
    ) -> list[tuple[int, int]]:
        """Phase 6: Extract remaining frames and refine boundary pairs."""
        self.logger.info("[6/6] Refining boundaries")

        sides = (self.lhs, self.rhs)
        lhs_boundary_ranges, rhs_boundary_ranges = (
            self._complement_ranges(
                scene_ranges,
                max(int(info["frame_id"]) for info in side.all_frames.values()),
            )
            for side, scene_ranges in zip(sides, (lhs_scene_ranges, rhs_scene_ranges))
        )
        self.logger.info(
            "Extracting boundary frames: %d for %s, %d for %s",
            sum(e - s + 1 for s, e in lhs_boundary_ranges), self.lhs.label,
            sum(e - s + 1 for s, e in rhs_boundary_ranges), self.rhs.label,
        )

        for side, boundary_ranges in zip(sides, (lhs_boundary_ranges, rhs_boundary_ranges)):
            self.frame_comparison.extract_frames(
                side.path, side.boundary_dir, boundary_ranges, side.all_frames,
                f"[6/6] Extracting boundary frames: {side.label}",
            )

        # Normalize the newly extracted boundary frames and merge.
        for side in sides:
            new_frames = {
                timestamp: info
                for timestamp, info in side.all_frames.items()
                if info.get("path") is not None
                and timestamp not in side.normalized_frames
            }
            if new_frames:
                side.normalized_frames.update(self.frame_comparison.normalize_frames(
                    new_frames, side.normalized_dir, prefix="b_",
                    desc=f"[6/6] Normalizing boundary: {side.label}",
                ))

        return self.boundary_search.refine_generic_boundaries(matching_pairs, debug)

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
