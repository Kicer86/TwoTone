
import cv2 as cv
import enum
import logging
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import RANSACRegressor, LinearRegression
from tqdm import tqdm
from typing import Callable, NamedTuple, TypedDict

from .debug_routines import DebugRoutines
from .melt_cache import MeltCache
from .melt_common import FramesInfo
from .phash_cache import PhashCache, compute_phash
from ..utils import files_utils, generic_utils, image_utils, video_utils


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


class CoverageSummary(TypedDict):
    """Human-readable coverage report for two matched files."""

    full_coverage: bool
    lhs_start_gap_s: float
    rhs_start_gap_s: float
    lhs_end_gap_s: float
    rhs_end_gap_s: float
    ratio: float


class SegmentsMappingResult(NamedTuple):
    mapping: list[tuple[int, int]]
    lhs_all_frames: FramesInfo
    rhs_all_frames: FramesInfo
    relation: MappingRelation


class PairMatcher:

    _MAX_BOUNDARY_ERROR_FRAMES = 2
    # Minimum phash distance treated as "still the same frame".  The adaptive
    # cutoff (median + 2*std of the matched pairs) collapses to 0 when the
    # surviving matches are pixel-perfect; without a floor, any check keyed on it
    # would then demand pixel-perfect frames — impossible across a speed or
    # resolution difference — so it is floored to this value.
    _MIN_PHASH_CUTOFF = 16
    # Maximum phash distance (bits, of hash_size 16 → 256) between two
    # boundary-extrapolation frames still considered the same picture; see
    # _boundary_content_matches for the measured separation behind the value.
    _MAX_BOUNDARY_PHASH_DISTANCE = 96
    # Global-linear detection thresholds, shared with the relation-diagnostics
    # log so the quoted limits never drift from the real ones.
    _MAX_CONSTANT_OFFSET_STD = 1.0
    _MAX_DRIFT_SLOPE_DELTA = 0.05

    def __init__(self, interruption: generic_utils.InterruptibleProcess, wd: str, lhs_path: str, rhs_path: str, logger: logging.Logger, lhs_label: str = "#1", rhs_label: str = "#2", cache: MeltCache | None = None) -> None:
        self.interruption = interruption
        self.wd = os.path.join(wd, "pair_matcher")
        self.lhs_path = lhs_path
        self.rhs_path = rhs_path
        self.lhs_label = lhs_label
        self.rhs_label = rhs_label
        self.logger = logger
        self.cache = cache
        self.phash = PhashCache()
        lhs_video_data = video_utils.get_video_data(lhs_path, logger=self.logger)["video"][0]
        rhs_video_data = video_utils.get_video_data(rhs_path, logger=self.logger)["video"][0]
        self.lhs_fps = generic_utils.fps_str_to_float(lhs_video_data["fps"])
        self.rhs_fps = generic_utils.fps_str_to_float(rhs_video_data["fps"])
        self.lhs_duration_ms: int | None = lhs_video_data.get("length")
        self.rhs_duration_ms: int | None = rhs_video_data.get("length")

        lhs_wd = os.path.join(self.wd, "lhs")
        rhs_wd = os.path.join(self.wd, "rhs")

        self.lhs_all_wd = os.path.join(lhs_wd, "all")
        self.rhs_all_wd = os.path.join(rhs_wd, "all")
        self.lhs_boundary_wd = os.path.join(lhs_wd, "boundary")
        self.rhs_boundary_wd = os.path.join(rhs_wd, "boundary")
        self.lhs_normalized_wd = os.path.join(lhs_wd, "norm")
        self.rhs_normalized_wd = os.path.join(rhs_wd, "norm")
        self.lhs_normalized_cropped_wd = os.path.join(lhs_wd, "norm_cropped")
        self.rhs_normalized_cropped_wd = os.path.join(rhs_wd, "norm_cropped")
        self.debug_wd = os.path.join(self.wd, "debug")

        for d in [lhs_wd,
                  rhs_wd,
                  self.lhs_all_wd,
                  self.rhs_all_wd,
                  self.lhs_boundary_wd,
                  self.rhs_boundary_wd,
                  self.lhs_normalized_wd,
                  self.rhs_normalized_wd,
                  self.lhs_normalized_cropped_wd,
                  self.rhs_normalized_cropped_wd,
                  self.debug_wd,
        ]:
            os.makedirs(d)

    @staticmethod
    def calculate_ratio(pairs: list[tuple[int, int]]) -> float:
        ratios = [(r[0] - l[0]) / (r[1] - l[1]) for l, r in zip(pairs[:-1], pairs[1:]) if (r[1] - l[1]) != 0]
        if not ratios:
            return float("nan")
        median_ratio = np.median(ratios)
        return float(median_ratio)

    @staticmethod
    def is_ratio_acceptable(ratio: float, perfect_ratio: float) -> bool:
        return abs(ratio - perfect_ratio) < 0.05 * perfect_ratio

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
        first = mappings[0]
        last = mappings[-1]

        lhs_start_gap = first[0]
        rhs_start_gap = first[1]
        lhs_end_gap = max(0, lhs_duration_ms - last[0])
        rhs_end_gap = max(0, rhs_duration_ms - last[1])

        if lhs_fps <= 0 or rhs_fps <= 0:
            raise ValueError("FPS values must be positive")

        lhs_frame_duration_ms = 1000 / lhs_fps
        rhs_frame_duration_ms = 1000 / rhs_fps
        lhs_edge_tolerance_ms = (
            PairMatcher._MAX_BOUNDARY_ERROR_FRAMES * lhs_frame_duration_ms
        )
        rhs_edge_tolerance_ms = (
            PairMatcher._MAX_BOUNDARY_ERROR_FRAMES * rhs_frame_duration_ms
        )

        # A frame timestamp marks the start of the frame. The expected timestamp
        # of the final frame is therefore one frame duration before the stream end.
        lhs_end_error = max(0, lhs_end_gap - lhs_frame_duration_ms)
        rhs_end_error = max(0, rhs_end_gap - rhs_frame_duration_ms)
        full_coverage = (
            lhs_start_gap <= lhs_edge_tolerance_ms
            and lhs_end_error <= lhs_edge_tolerance_ms
            and rhs_start_gap <= rhs_edge_tolerance_ms
            and rhs_end_error <= rhs_edge_tolerance_ms
        )

        return {
            "full_coverage": full_coverage,
            "lhs_start_gap_s": lhs_start_gap / 1000,
            "rhs_start_gap_s": rhs_start_gap / 1000,
            "lhs_end_gap_s": lhs_end_gap / 1000,
            "rhs_end_gap_s": rhs_end_gap / 1000,
            "ratio": PairMatcher.calculate_ratio(mappings),
        }

    @staticmethod
    def filter_phash_outliers(phash: PhashCache, pairs: list[tuple[int, int]], lhs_set: FramesInfo, rhs_set: FramesInfo) -> list[tuple[int, int]]:
        if len(pairs) <= 3:
            # Too few data points for reliable MAD-based outlier detection.
            # Let downstream filters (ORB, history check) handle quality control.
            return pairs

        dists_array = np.array([abs(phash.get(lhs_set[l]["path"]) - phash.get(rhs_set[r]["path"])) for l, r in pairs], dtype=float)
        med = float(np.median(dists_array))
        mad = float(np.median(np.abs(dists_array - med)))
        threshold = med + 1.5 * mad
        return [pair for pair, dist in zip(pairs, dists_array) if dist <= threshold]

    @staticmethod
    def summarize_pairs(phash: PhashCache, pairs: list[tuple[int, int]], lhs: FramesInfo, rhs: FramesInfo, verbose: bool = False) -> str:
        if not pairs:
            return "Pairs: 0"

        distances = []
        for lhs_ts, rhs_ts in pairs:
            lhs_path = lhs.get(lhs_ts, {}).get("path") if lhs_ts in lhs else None
            rhs_path = rhs.get(rhs_ts, {}).get("path") if rhs_ts in rhs else None
            if lhs_path is None or rhs_path is None:
                continue
            d = abs(phash.get(lhs_path) - phash.get(rhs_path))
            distances.append((d, lhs_ts, rhs_ts))

        if not distances:
            return f"Pairs: {len(pairs)} (no extracted paths for summary)"

        arr = np.array([d[0] for d in distances])
        median = np.median(arr)
        mean = np.mean(arr)
        std = np.std(arr)
        max_val = np.max(arr)
        min_val = np.min(arr)

        # Identify the max pair
        max_entry = max(distances, key=lambda x: x[0])
        max_lhs_path = lhs.get(max_entry[1], {}).get("path", "?")
        max_rhs_path = rhs.get(max_entry[2], {}).get("path", "?")

        summary = (
            f"Pairs: {len(pairs)} | "
            f"Median: {median:.2f} | "
            f"Mean: {mean:.2f} | "
            f"Std Dev: {std:.2f} | "
            f"Min: {min_val} | "
            f"Max: {max_val} | "
            f"Max Pair: {max_lhs_path} <-> {max_rhs_path}"
        )

        if verbose:
            details = []
            for dist, lhs_ts, rhs_ts in distances:
                lp = lhs.get(lhs_ts, {}).get("path", "?")
                rp = rhs.get(rhs_ts, {}).get("path", "?")
                details.append(f"  {lp} <-> {rp} | Diff: {dist}")
            summary += "\nDetailed pairs:" + "\n" + "\n".join(details)

        return summary

    @staticmethod
    def summarize_segments(pairs: list[tuple[int, int]], lhs_fps: float, rhs_fps: float, verbose: bool = True, lhs_label: str = "#1", rhs_label: str = "#2") -> str:
        if len(pairs) < 2:
            return "Not enough pairs to build segments."

        pairs_sorted = sorted(pairs)
        segments = []

        lhs_frame_step = 1000.0 / lhs_fps
        rhs_frame_step = 1000.0 / rhs_fps
        max_quant_error = max(lhs_frame_step, rhs_frame_step)

        for (lhs1, rhs1), (lhs2, rhs2) in zip(pairs_sorted[:-1], pairs_sorted[1:]):
            lhs_delta = lhs2 - lhs1
            rhs_delta = rhs2 - rhs1
            if rhs_delta <= 0:
                continue
            ratio = lhs_delta / rhs_delta

            # Estimate ratio uncertainty
            min_delta = min(lhs_delta, rhs_delta)
            ratio_error = (2 * max_quant_error) / min_delta if min_delta > 0 else float("inf")
            confidence = "LOW" if ratio_error > 0.1 else "OK"

            segments.append((lhs1, lhs2, rhs1, rhs2, lhs_delta, rhs_delta, ratio, ratio_error, confidence))

        if not segments:
            return "No valid segments."

        ratios = np.array([s[6] for s in segments])
        out = [
            f"Segments: {len(segments)} | "
            f"Median ratio: {np.median(ratios):.4f} | "
            f"Mean ratio: {np.mean(ratios):.4f} | "
            f"Std Dev: {np.std(ratios):.4f} | "
            f"Min: {np.min(ratios):.4f} | Max: {np.max(ratios):.4f}"
        ]

        if verbose:
            out.append("\nDetailed segments:")
            for lhs1, lhs2, rhs1, rhs2, ldelta, rdelta, ratio, err, conf in segments:
                out.append(
                    f"  {lhs_label} {lhs1}->{lhs2} ({ldelta:4} ms), "
                    f"{rhs_label} {rhs1}->{rhs2} ({rdelta:4} ms), "
                    f"Ratio: {ratio:.4f}, "
                    f"Error~{err:.2%}, Confidence: {conf}"
                )

        return '\n'.join(out)

    def look_for_boundaries(self, lhs: FramesInfo, rhs: FramesInfo, first: tuple[int, int], last: tuple[int, int], cutoff: float, max_gap_seconds: float = 15.0, extrapolate: bool = True):
        """Find the first and last common frame pair by walking outward from known matches.

        Uses the linear time mapping derived from *first* and *last* to predict
        where each LHS frame should appear in RHS, then walks from the current
        boundaries toward the edges of the video.  Allows configurable gaps of
        non-matching frames before giving up, instead of stopping on first miss.

        When the search enters a low-entropy region (dark frames, end credits),
        phash matching becomes unreliable.  In that case the boundary is
        extrapolated linearly from well-matched pairs, but only if both files
        show a consistent entropy transition at the predicted position.
        """
        PairMatcher._assert_frames_extracted(lhs, "look_for_boundaries(lhs)")
        PairMatcher._assert_frames_extracted(rhs, "look_for_boundaries(rhs)")

        self.logger.debug("Improving boundaries")
        self.logger.debug(f"Current first: {first} and last: {last}")
        phash = PhashCache()
        ratio = PairMatcher.calculate_ratio([first, last])

        # When first == last (single pair) the ratio is NaN;
        # fall back to fps-based ratio.
        if np.isnan(ratio):
            ratio = self.lhs_fps / self.rhs_fps

        cutoff = self._boundary_cutoff(cutoff)

        # --- Fast edge pre-check ---
        # Before the iterative search, check whether both videos share content
        # at each edge.  When they do, extend the gap budget for find_boundary
        # so it does not give up before reaching the video edge.  The anchor
        # and prediction logic are left untouched for maximum precision.
        lhs_keys = sorted(lhs.keys())
        rhs_keys = sorted(rhs.keys())

        first_gap_seconds = max_gap_seconds
        last_gap_seconds = max_gap_seconds

        for direction in [-1, 1]:
            label = "start" if direction == -1 else "end"
            matches = self._edge_content_matches(
                lhs, rhs, lhs_keys, rhs_keys, phash,
                first, last, ratio, cutoff, direction,
            )
            if not matches:
                self.logger.debug(
                    f"Edge pre-check ({label}): edges do NOT visually match "
                    f"— using default gap budget {max_gap_seconds}s"
                )
                continue

            anchor = first if direction == -1 else last
            edge_lhs = lhs_keys[0] if direction == -1 else lhs_keys[-1]
            distance_s = abs(anchor[0] - edge_lhs) / 1000.0

            if direction == -1:
                first_gap_seconds = max(max_gap_seconds, distance_s + 2.0)
                self.logger.debug(
                    f"Edge pre-check ({label}): edges match, "
                    f"extending gap budget to {first_gap_seconds:.1f}s "
                    f"(anchor={anchor[0]}ms, edge={edge_lhs}ms, distance={distance_s:.1f}s)"
                )
            else:
                last_gap_seconds = max(max_gap_seconds, distance_s + 2.0)
                self.logger.debug(
                    f"Edge pre-check ({label}): edges match, "
                    f"extending gap budget to {last_gap_seconds:.1f}s "
                    f"(anchor={anchor[0]}ms, edge={edge_lhs}ms, distance={distance_s:.1f}s)"
                )

        refined_first, first_low_entropy = self._find_boundary(
            lhs, rhs, first, last, -1, first_gap_seconds, ratio, cutoff, phash,
        )
        self.logger.debug(
            f"Boundary start: walked {first[0]}ms → {refined_first[0]}ms "
            f"({'entered low-entropy zone' if first_low_entropy else 'gap budget or edge reached'}, "
            f"budget={first_gap_seconds:.1f}s)"
        )

        refined_last, last_low_entropy = self._find_boundary(
            lhs, rhs, last, first, 1, last_gap_seconds, ratio, cutoff, phash,
        )
        self.logger.debug(
            f"Boundary end: walked {last[0]}ms → {refined_last[0]}ms "
            f"({'entered low-entropy zone' if last_low_entropy else 'gap budget or edge reached'}, "
            f"budget={last_gap_seconds:.1f}s)"
        )

        if extrapolate:
            # Extrapolate through low-entropy regions when possible
            refined_first = self.extrapolate_through_low_entropy(
                lhs, rhs, refined_first, refined_last, ratio, direction=-1,
                entered_low_entropy=first_low_entropy,
            )
            refined_last = self.extrapolate_through_low_entropy(
                lhs, rhs, refined_last, refined_first, ratio, direction=1,
                entered_low_entropy=last_low_entropy,
            )

        return refined_first, refined_last

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
        """Extend *boundary* through a low-entropy zone to the video edge.

        When the boundary search stops at a low-entropy zone (black frames,
        end credits, etc.), this method checks whether the region between the
        current boundary and the video edge is consistently low-entropy in
        *both* files.  If so, the boundary is linearly extrapolated to the
        edge; otherwise it is kept unchanged (the files likely have different
        intro/outro content).

        The coarse step used by ``find_boundary`` may overshoot the exact
        content→dark transition by a few frames.  To compensate, this method
        walks from the boundary toward the edge, skips any high-entropy frames
        near the boundary (at most ``step * 3``), and then verifies that the
        remaining zone to the edge is entirely low-entropy.  The tolerance is
        ``step * 3`` rather than ``step`` because the RHS boundary is predicted
        via the linear ratio and may be off by more than the walk stride.
        """
        if not entered_low_entropy:
            return boundary

        label = "start" if direction == -1 else "end"
        lhs_keys = sorted(lhs.keys())
        rhs_keys = sorted(rhs.keys())
        step = max(1, int(self.lhs_fps * 0.5))
        max_skip = step * 3

        # Determine LHS edge and gap frames between boundary and edge
        if direction == -1:
            edge_lhs = lhs_keys[0]
            lhs_gap = [k for k in lhs_keys if k < boundary[0]]
        else:
            edge_lhs = lhs_keys[-1]
            lhs_gap = [k for k in lhs_keys if k > boundary[0]]

        if not lhs_gap:
            return boundary

        # Verify LHS gap is low-entropy
        if not self._verify_gap_is_low_entropy(lhs, lhs_gap, max_skip, direction, label, self.lhs_label):
            return boundary

        # Predict RHS edge position and clamp to valid range
        predicted_rhs = int(boundary[1] + (edge_lhs - boundary[0]) / ratio)
        clamped_rhs = max(rhs_keys[0], min(rhs_keys[-1], predicted_rhs))

        # Determine RHS gap between boundary and predicted edge
        if direction == -1:
            rhs_gap = [k for k in rhs_keys if k < boundary[1] and k >= clamped_rhs]
        else:
            rhs_gap = [k for k in rhs_keys if k > boundary[1] and k <= clamped_rhs]

        # Verify RHS gap is low-entropy
        if rhs_gap and not self._verify_gap_is_low_entropy(rhs, rhs_gap, max_skip, direction, label, self.rhs_label):
            return boundary

        # Both files have consistently low-entropy gaps — extrapolate to edge
        new_rhs = PairMatcher._snap_to_nearest_frame(rhs_keys, clamped_rhs)

        self.logger.debug(
            f"Boundary {label}: extrapolating through low-entropy zone from "
            f"({boundary[0]}, {boundary[1]}) to ({edge_lhs}, {new_rhs})."
        )
        return (edge_lhs, new_rhs)

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
        intercept`` (read from ``frame_id`` in *FramesInfo*).  A constant frame
        offset is just the ``slope == 1`` special case, so it is tried first (it
        needs only two well-separated matches); when the offset is not constant
        the slope and intercept are fitted with RANSAC.  Either way the audio is
        later placed with one global time-scale, so a constant offset and a
        time-scaled drift share the same handling.

        ``time_scale = slope*lhs_fps/rhs_fps`` may differ from 1 (e.g. a 25 fps
        PAL speedup vs 24 fps); that only means the audio must be stretched.  It
        is rejected when larger than ``max_audio_time_scale_delta`` (an
        implausible match).

        This method only *detects* the relation and returns the fit; it does
        **not** decide the first/last common pair.  The boundaries are found by
        the content/entropy-aware boundary walk (shared with the GENERIC path),
        which uses this fit only as a precise predictor of where identical frames
        should be and stops at the first genuine divergence.  Blindly
        extrapolating the line to the video edges would pair unrelated content
        across divergent intros/outros and is non-deterministic across ffmpeg
        builds, so it is deliberately avoided.
        """
        if len(matching_pairs) < 2:
            return None

        try:
            lhs_frame_ids = np.array(
                [int(lhs_all_frames[l]["frame_id"]) for l, _ in matching_pairs], dtype=float
            )
            rhs_frame_ids = np.array(
                [int(rhs_all_frames[r]["frame_id"]) for _, r in matching_pairs], dtype=float
            )
        except KeyError:
            return None

        fit = self._fit_constant_offset(
            lhs_frame_ids, rhs_frame_ids, matching_pairs,
            max_std=max_constant_offset_std, min_two_pair_span=min_span_frames,
        )
        is_constant_offset = fit is not None
        if fit is None:
            fit = self._fit_linear_drift(
                lhs_frame_ids, rhs_frame_ids,
                max_slope_delta=max_slope_delta,
                max_median_residual_frames=max_median_residual_frames,
                max_p95_residual_frames=max_p95_residual_frames,
                max_outlier_ratio=max_outlier_ratio,
                min_span_frames=min_span_frames,
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

        self._log_global_linear(slope, intercept, time_scale, is_constant_offset)
        return GlobalLinearFit(slope=slope, intercept=intercept,
                               is_constant_offset=is_constant_offset, time_scale=time_scale)

    def _fit_constant_offset(
        self,
        lhs_frame_ids: np.ndarray,
        rhs_frame_ids: np.ndarray,
        matching_pairs: list[tuple[int, int]],
        *,
        max_std: float,
        min_two_pair_span: int,
    ) -> tuple[float, float] | None:
        """Fit a constant frame offset (slope == 1).

        Returns ``(slope=1.0, intercept)`` where ``rhs_frame ~= lhs_frame +
        intercept``, or ``None`` when the offset is not sufficiently constant.
        """
        if len(matching_pairs) == 2:
            span = min(int(np.ptp(lhs_frame_ids)), int(np.ptp(rhs_frame_ids)))
            if span < min_two_pair_span:
                self.logger.debug(
                    f"Constant-offset check: only two pairs spanning {span} frames — skipping"
                )
                return None

        offsets = lhs_frame_ids - rhs_frame_ids
        std_offset = float(np.std(offsets))
        if std_offset > max_std:
            self.logger.debug(
                f"Constant-offset check: frame-number std={std_offset:.2f} exceeds {max_std:.1f} — skipping"
            )
            return None

        ratio = PairMatcher.calculate_ratio(matching_pairs)
        expected_ratio = self.rhs_fps / self.lhs_fps
        if not PairMatcher.is_ratio_acceptable(ratio, expected_ratio):
            self.logger.debug(
                f"Constant-offset check: ratio={ratio:.4f} too far from expected {expected_ratio:.4f} — skipping"
            )
            return None

        k = round(float(np.median(offsets)))  # lhs_frame - rhs_frame
        self.logger.debug(
            "Constant offset detected: %d frame(s) (median=%.1f, std=%.2f).",
            k, float(np.median(offsets)), std_offset,
        )
        return (1.0, float(-k))

    def _fit_linear_drift(
        self,
        lhs_frame_ids: np.ndarray,
        rhs_frame_ids: np.ndarray,
        *,
        max_slope_delta: float,
        max_median_residual_frames: float,
        max_p95_residual_frames: float,
        max_outlier_ratio: float,
        min_span_frames: int,
    ) -> tuple[float, float] | None:
        """Fit a near-identity drift line with RANSAC.

        Returns ``(slope, intercept)`` for ``rhs_frame ~= slope*lhs_frame +
        intercept`` with ``slope`` close to 1, or ``None`` when the matches do
        not fit a clean line.
        """
        if len(lhs_frame_ids) < 4:
            return None

        lhs_span = float(np.max(lhs_frame_ids) - np.min(lhs_frame_ids))
        if lhs_span < min_span_frames:
            self.logger.debug(
                f"Linear-drift check: matched frame span {lhs_span:.0f} is below {min_span_frames} frames — skipping"
            )
            return None

        ransac = RANSACRegressor(
            LinearRegression(), residual_threshold=max_p95_residual_frames, random_state=0,
        )
        ransac.fit(lhs_frame_ids.reshape(-1, 1), rhs_frame_ids)
        inliers = ransac.inlier_mask_
        if inliers is None:
            inliers = np.ones(len(lhs_frame_ids), dtype=bool)

        inlier_count = int(np.sum(inliers))
        if inlier_count < 4:
            self.logger.debug(
                f"Linear-drift check: only {inlier_count}/{len(lhs_frame_ids)} pairs fit the model — skipping"
            )
            return None

        outlier_ratio = 1.0 - inlier_count / len(lhs_frame_ids)
        if outlier_ratio > max_outlier_ratio:
            self.logger.debug(
                f"Linear-drift check: outlier ratio {outlier_ratio:.1%} exceeds {max_outlier_ratio:.1%} — skipping"
            )
            return None

        model = LinearRegression()
        model.fit(lhs_frame_ids[inliers].reshape(-1, 1), rhs_frame_ids[inliers])
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)

        if slope <= 0:
            self.logger.debug(f"Linear-drift check: non-positive slope {slope:.6f} — skipping")
            return None

        slope_delta = abs(slope - 1.0)
        if slope_delta > max_slope_delta:
            self.logger.debug(
                f"Linear-drift check: slope={slope:.6f} differs from 1.0 by {slope_delta:.6f}, "
                f"max {max_slope_delta:.6f} — skipping"
            )
            return None

        predicted_rhs = model.predict(lhs_frame_ids[inliers].reshape(-1, 1))
        residuals = np.abs(rhs_frame_ids[inliers] - predicted_rhs)
        median_residual = float(np.median(residuals))
        p95_residual = float(np.percentile(residuals, 95))
        if median_residual > max_median_residual_frames or p95_residual > max_p95_residual_frames:
            self.logger.debug(
                f"Linear-drift check: residuals too high "
                f"(median={median_residual:.2f}, p95={p95_residual:.2f}) — skipping"
            )
            return None

        return (slope, intercept)

    def _log_global_linear(
        self,
        slope: float,
        intercept: float,
        time_scale: float,
        is_constant_offset: bool,
    ) -> None:
        """Log the detected global linear relation (the fit, not the boundaries)."""
        lhs_ref = f"{self.lhs_label} ({os.path.basename(self.lhs_path)})"
        rhs_ref = f"{self.rhs_label} ({os.path.basename(self.rhs_path)})"

        if is_constant_offset:
            k = int(round(-intercept))  # lhs_frame - rhs_frame
            if k > 0:
                relation_text = (
                    f"a small constant frame offset of {k} frame(s); "
                    f"{self.lhs_label} starts the shared content later than {self.rhs_label}"
                )
            elif k < 0:
                relation_text = (
                    f"a small constant frame offset of {-k} frame(s); "
                    f"{self.rhs_label} starts the shared content later than {self.lhs_label}"
                )
            else:
                relation_text = "no frame offset"
            same_content_phrase = "have the same content"
        else:
            relation_text = f"a linear frame drift (slope={slope:.6f}, intercept={intercept:+.1f})"
            same_content_phrase = "share content"

        if abs(time_scale - 1.0) > 0.005:
            relation_text += f", audio time-scaled by {time_scale:.5f}"

        self.logger.info("Files %s and %s %s with %s.", lhs_ref, rhs_ref, same_content_phrase, relation_text)

    def _extrapolate_and_verify_global_linear(
        self,
        fit: GlobalLinearFit,
        matching_pairs: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Extend the first/last common pair along the fitted line through verified content.

        The fit predicts, for any lhs frame, exactly which rhs frame should hold
        the same content.  Each boundary is projected to the overlap of the full
        frame ranges and then **content-verified across the whole gap** between
        the outermost match and the projected edge: the extension is kept only
        when every sampled predicted pair in that gap is visually the same frame
        or both low-entropy (a shared black lead-in/out).  A single endpoint check
        is not enough — a divergent outro can still fade to black on its last
        frame — so the gap is sampled.  When it does not verify, the boundary is
        left at the outermost real match; the line is never blindly projected
        across divergent content.
        """
        slope, intercept = fit.slope, fit.intercept
        if slope == 0:
            return sorted(matching_pairs)

        lhs_by_frame = {int(info["frame_id"]): ts for ts, info in self.lhs_all_frames.items()}
        rhs_by_frame = {int(info["frame_id"]): ts for ts, info in self.rhs_all_frames.items()}
        lhs_min_frame, lhs_max_frame = min(lhs_by_frame), max(lhs_by_frame)
        rhs_min_frame, rhs_max_frame = min(rhs_by_frame), max(rhs_by_frame)

        result = sorted(matching_pairs)
        eps = 1e-6
        # Snap a boundary whose projection lands within a few frames of BOTH
        # video edges onto the edges themselves — the fit's slope error
        # accumulates over a long extrapolation, so an equal-length shared
        # lead-in/out can project a couple frames short of the edge (and would
        # otherwise stay there, off by 2-3 frames per ffmpeg build).  A
        # genuinely offset lead-in/out (e.g. 2s vs 6s black) is far past the
        # tolerance and keeps its predicted position.  The snapped target is
        # still content-verified by the walk below, never blindly accepted.
        snap_tol = 4

        first_lhs_frame = max(lhs_min_frame, int(np.ceil((rhs_min_frame - intercept) / slope - eps)))
        first_rhs_frame = int(round(slope * first_lhs_frame + intercept))
        if (
            first_lhs_frame - lhs_min_frame <= snap_tol
            and abs(int(round(slope * lhs_min_frame + intercept)) - rhs_min_frame) <= snap_tol
        ):
            first_lhs_frame = lhs_min_frame
            first_rhs_frame = rhs_min_frame
        first_rhs_frame = max(rhs_min_frame, min(rhs_max_frame, first_rhs_frame))
        self._maybe_insert_verified_boundary(
            result, "start", slope, intercept, first_lhs_frame, first_rhs_frame, lhs_by_frame, rhs_by_frame,
        )

        last_lhs_frame = min(lhs_max_frame, int(np.floor((rhs_max_frame - intercept) / slope + eps)))
        last_rhs_frame = int(round(slope * last_lhs_frame + intercept))
        if (
            lhs_max_frame - last_lhs_frame <= snap_tol
            and abs(int(round(slope * lhs_max_frame + intercept)) - rhs_max_frame) <= snap_tol
        ):
            last_lhs_frame = lhs_max_frame
            last_rhs_frame = rhs_max_frame
        last_rhs_frame = max(rhs_min_frame, min(rhs_max_frame, last_rhs_frame))
        self._maybe_insert_verified_boundary(
            result, "end", slope, intercept, last_lhs_frame, last_rhs_frame, lhs_by_frame, rhs_by_frame,
        )

        return result

    def _maybe_insert_verified_boundary(
        self,
        result: list[tuple[int, int]],
        side: str,
        slope: float,
        intercept: float,
        lhs_frame: int,
        rhs_frame: int,
        lhs_by_frame: dict[int, int],
        rhs_by_frame: dict[int, int],
    ) -> None:
        """Extend the boundary toward the projected edge as far as content verifies.

        Walks from the outermost match toward the projected boundary and keeps the
        furthest predicted pair that is still content-verified.  A gap that is
        shared all the way (black lead-out, or frame-identical tail) reaches the
        edge; a gap that starts shared but then diverges (shared body then a
        different outro) stops at the divergence; a gap that diverges immediately
        (different intro) does not move the boundary at all.
        """
        lhs_ts = lhs_by_frame.get(lhs_frame)
        rhs_ts = rhs_by_frame.get(rhs_frame)
        if lhs_ts is None or rhs_ts is None:
            return

        anchor = result[0] if side == "start" else result[-1]
        if side == "start":
            beyond = lhs_ts < anchor[0] or rhs_ts < anchor[1]
        else:
            beyond = lhs_ts > anchor[0] or rhs_ts > anchor[1]
        if not beyond or (lhs_ts, rhs_ts) == anchor:
            return

        anchor_info = self.lhs_all_frames.get(anchor[0])
        if anchor_info is None:
            return
        anchor_lhs_frame = int(anchor_info["frame_id"])

        verified = self._walk_shared_boundary(
            slope, intercept, anchor_lhs_frame, lhs_frame, rhs_frame, lhs_by_frame, rhs_by_frame,
        )
        if verified is None or verified == anchor:
            self.logger.debug(
                "Global-linear %s gap diverges before any shared frame — keeping the "
                "boundary at the outermost match", side,
            )
            return

        if side == "start":
            result.insert(0, verified)
        else:
            result.append(verified)

    def _walk_shared_boundary(
        self,
        slope: float,
        intercept: float,
        anchor_lhs_frame: int,
        boundary_lhs_frame: int,
        boundary_rhs_frame: int,
        lhs_by_frame: dict[int, int],
        rhs_by_frame: dict[int, int],
    ) -> tuple[int, int] | None:
        """Walk ~0.5s-spaced predicted pairs from the match toward the projected
        boundary; return the furthest ``(lhs_ts, rhs_ts)`` that is still shared.

        Sampling the whole gap (not just its endpoint) distinguishes a shared black
        lead-out from a divergent outro that merely fades to black on its last
        frame, and stopping at the first divergence keeps the shared body that
        precedes a different tail.  Returns ``None`` when the very first step
        diverges.
        """
        direction = 1 if boundary_lhs_frame >= anchor_lhs_frame else -1
        step = max(1, int(self.lhs_fps * 0.5))

        # Ordered (lhs_frame, rhs_frame) samples from just past the match to the
        # projected boundary (inclusive); the boundary uses its snapped rhs.
        samples: list[tuple[int, int]] = []
        f = anchor_lhs_frame + direction * step
        while (direction == 1 and f < boundary_lhs_frame) or (direction == -1 and f > boundary_lhs_frame):
            rhs_f = int(round(slope * f + intercept))
            if rhs_f not in rhs_by_frame and rhs_by_frame:
                rhs_f = min(rhs_by_frame, key=lambda k: abs(k - rhs_f))
            samples.append((f, rhs_f))
            f += direction * step
        samples.append((boundary_lhs_frame, boundary_rhs_frame))

        # Bulk-extract every sampled frame up front (two ffmpeg calls) so the
        # per-sample verification below only reads images.  The rhs fetch also
        # covers the ±2 neighbours used by the prediction-jitter retry — a
        # later single-frame extraction would wipe the prefetched files
        # (extract_frames_at_ranges cleans its target directory).
        self._prefetch_boundary_images(self.lhs_path, self.lhs_boundary_wd, self.lhs_all_frames, [lf for lf, _ in samples])
        rhs_with_neighbours = sorted({rf + d for _, rf in samples for d in (-2, -1, 0, 1, 2)})
        self._prefetch_boundary_images(self.rhs_path, self.rhs_boundary_wd, self.rhs_all_frames, rhs_with_neighbours)

        best: tuple[int, int] | None = None
        for lhs_f, rhs_f in samples:
            lhs_ts = lhs_by_frame.get(lhs_f)
            rhs_ts = rhs_by_frame.get(rhs_f)
            if lhs_ts is None or rhs_ts is None:
                break
            if not self._shared_content_at_prediction(lhs_f, rhs_f, lhs_ts, rhs_by_frame):
                break
            best = (lhs_ts, rhs_ts)
        return best

    def _shared_content_at_prediction(
        self,
        lhs_frame: int,
        rhs_frame: int,
        lhs_ts: int,
        rhs_by_frame: dict[int, int],
    ) -> bool:
        """Content-check the predicted pair, tolerating ±2 frames of rhs jitter.

        The fit's rhs prediction can be a frame or two off (rounding, and
        frame-probe differences between ffmpeg builds); in fast motion that is
        enough to push the exact predicted pair past the phash gate even for
        genuinely shared content.  Retry against the rhs neighbours before
        declaring divergence — a divergent intro/outro fails for every
        neighbour, so the tolerance does not weaken divergence rejection, and
        the returned boundary pair stays the predicted one (on the fitted
        line), keeping the boundary error within the allowed 1-2 frames.
        """
        for delta in (0, -1, 1, -2, 2):
            candidate_frame = rhs_frame + delta
            candidate_ts = rhs_by_frame.get(candidate_frame)
            if candidate_ts is None:
                continue
            if self._boundary_content_matches(lhs_frame, candidate_frame, lhs_ts, candidate_ts):
                return True
        return False

    def _prefetch_boundary_images(
        self, video_path: str, out_dir: str, frames: FramesInfo, frame_ids: list[int],
    ) -> None:
        """Extract any not-yet-extracted *frame_ids* in a single ffmpeg pass."""
        frame_id_to_ts = {int(info["frame_id"]): ts for ts, info in frames.items()}
        missing = sorted({
            fid for fid in frame_ids
            if frame_id_to_ts.get(fid) is not None and not frames[frame_id_to_ts[fid]].get("path")
        })
        if not missing:
            return
        try:
            video_utils.extract_frames_at_ranges(
                video_path, out_dir, [(fid, fid) for fid in missing], frames,
                scale=(960, -2), format="png", interruption=self.interruption,
                desc="Verifying boundary gap", logger=self.logger,
            )
        except Exception as e:  # pragma: no cover - extraction failure is non-fatal
            self.logger.debug("Boundary gap extraction failed: %s", e)

    def _boundary_content_matches(self, lhs_frame: int, rhs_frame: int, lhs_ts: int, rhs_ts: int) -> bool:
        """Verify that the extrapolated boundary frames actually share content.

        True when both frames are low-entropy (a shared black lead-in/out) or the
        two frames are visually the same picture.  A divergent intro/outro (e.g.
        grass vs atoms) is neither, so the extrapolation is rejected there.

        ORB match count alone cannot make that call: two unrelated textured
        frames (grass vs atoms) can yield dozens of cross-checked ORB matches,
        just with large descriptor distances.  The fit predicts the *same*
        frame, so the perceptual hashes must also be close.  Measured on the
        960px boundary frames (256-bit hash): the same frame across a
        quality/speed change is ≲ 22 bits apart, a shared-content sample whose
        prediction is a frame or two off in fast motion reaches ~66, and
        divergent intros are ≳ 124 — the 96-bit gate keeps ~30 bits of margin
        to both sides.

        The hashes are computed fresh (not via ``self.phash``): boundary
        extraction re-uses ``frame_%08d`` file names across invocations, so a
        path-keyed cache could return the hash of a previously extracted frame.
        """
        lhs_path = self._ensure_boundary_image(self.lhs_path, self.lhs_boundary_wd, self.lhs_all_frames, lhs_frame, lhs_ts)
        rhs_path = self._ensure_boundary_image(self.rhs_path, self.rhs_boundary_wd, self.rhs_all_frames, rhs_frame, rhs_ts)
        if lhs_path is None or rhs_path is None:
            return False

        lhs_black = not PairMatcher._is_rich(lhs_path)
        rhs_black = not PairMatcher._is_rich(rhs_path)
        if lhs_black and rhs_black:
            return True
        if lhs_black != rhs_black:
            return False
        phash_distance = abs(compute_phash(lhs_path) - compute_phash(rhs_path))
        if phash_distance > self._MAX_BOUNDARY_PHASH_DISTANCE:
            return False
        return image_utils.are_images_similar(lhs_path, rhs_path)

    def _ensure_boundary_image(
        self, video_path: str, out_dir: str, frames: FramesInfo, frame_id: int, ts: int,
    ) -> str | None:
        """Return an on-disk image path for *frame_id*, extracting it on demand.

        The frame is already probed (so its timestamp entry exists) but may lack
        an extracted image; this extracts just that single frame for boundary
        verification, far cheaper than the full boundary refinement.
        """
        info = frames.get(ts)
        if info is None:
            return None
        if info.get("path"):
            return info["path"]
        try:
            video_utils.extract_frames_at_ranges(
                video_path, out_dir, [(frame_id, frame_id)], frames,
                scale=(960, -2), format="png", interruption=self.interruption,
                desc="Verifying boundary frame", logger=self.logger,
            )
        except Exception as e:  # pragma: no cover - extraction failure is non-fatal
            self.logger.debug("Boundary frame extraction failed for frame %d: %s", frame_id, e)
            return None
        return frames.get(ts, {}).get("path")

    def snap_to_edges(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
        snap_frames: int = 16,
    ) -> list[tuple[int, int]]:
        """Snap first/last pair timestamps to video edges when within a few frames.

        When the first or last mapping pair is very close to a video edge
        (within *snap_frames* frames), keeping a tiny head or tail audio
        segment is pointless and may cause artifacts.  This method extends
        those pairs to the edge (timestamp 0 for start, video duration for
        end) so downstream audio patching skips the trivial segments.

        Synthetic entries are added to *lhs_all_frames* / *rhs_all_frames*
        for any newly created timestamps (pointing to the nearest real frame
        file) so debug routines remain functional.
        """
        lhs_threshold_ms = snap_frames * 1000 / self.lhs_fps
        rhs_threshold_ms = snap_frames * 1000 / self.rhs_fps

        lhs_duration = video_utils.get_video_duration(self.lhs_path, logger=self.logger)
        rhs_duration = video_utils.get_video_duration(self.rhs_path, logger=self.logger)

        lhs_keys = sorted(lhs_all_frames.keys())
        rhs_keys = sorted(rhs_all_frames.keys())

        first_l, first_r = matching_pairs[0]
        last_l, last_r = matching_pairs[-1]

        # --- Start edge ---
        new_first_l = 0 if first_l <= lhs_threshold_ms else first_l
        new_first_r = 0 if first_r <= rhs_threshold_ms else first_r

        if (new_first_l, new_first_r) != (first_l, first_r):
            self.logger.debug(
                f"Edge snap: first pair ({first_l}, {first_r}) → "
                f"({new_first_l}, {new_first_r})"
            )
            matching_pairs[0] = (new_first_l, new_first_r)
            if new_first_l not in lhs_all_frames:
                lhs_all_frames[new_first_l] = lhs_all_frames[lhs_keys[0]].copy()
            if new_first_r not in rhs_all_frames:
                rhs_all_frames[new_first_r] = rhs_all_frames[rhs_keys[0]].copy()

        # --- End edge ---
        new_last_l = lhs_duration if (lhs_duration - last_l) <= lhs_threshold_ms else last_l
        new_last_r = rhs_duration if (rhs_duration - last_r) <= rhs_threshold_ms else last_r

        if (new_last_l, new_last_r) != (last_l, last_r):
            self.logger.debug(
                f"Edge snap: last pair ({last_l}, {last_r}) → "
                f"({new_last_l}, {new_last_r})"
            )
            matching_pairs[-1] = (new_last_l, new_last_r)
            if new_last_l not in lhs_all_frames:
                lhs_all_frames[new_last_l] = lhs_all_frames[lhs_keys[-1]].copy()
            if new_last_r not in rhs_all_frames:
                rhs_all_frames[new_last_r] = rhs_all_frames[rhs_keys[-1]].copy()

        return matching_pairs

    def create_segments_mapping(self) -> SegmentsMappingResult:

        lhs_scene_changes, rhs_scene_changes = self._detect_scenes()
        self._probe_frames()
        lhs_scene_ranges, rhs_scene_ranges = self._extract_scene_frames(lhs_scene_changes, rhs_scene_changes)
        lhs_normalized_frames, rhs_normalized_frames, lhs_key_frames, rhs_key_frames = self._normalize_extracted(
            lhs_scene_changes, rhs_scene_changes,
        )

        debug = DebugRoutines(self.debug_wd, self.lhs_all_frames, self.rhs_all_frames)
        debug.dump_frames(lhs_key_frames, f"{self.lhs_label} key frames")
        debug.dump_frames(rhs_key_frames, f"{self.rhs_label} key frames")

        matching_pairs = self._match_key_frames(
            lhs_key_frames, rhs_key_frames, lhs_normalized_frames, rhs_normalized_frames, debug,
        )

        # Diagnostic summary of the frame-space relationship.  Logged at INFO so
        # the chosen relation (and the audio strategy that follows) can be
        # understood without enabling --verbose.
        self._log_relation_diagnostics(matching_pairs)

        # Detect a single global linear frame relation (constant offset = slope
        # 1, or a fitted near-identity drift).  This decides the *relation* (and
        # thus the audio strategy) and yields the precise time-scale for the
        # audio.
        global_linear_fit = self.detect_global_linear(
            matching_pairs, self.lhs_all_frames, self.rhs_all_frames,
        )

        if global_linear_fit is not None:
            relation = MappingRelation.GLOBAL_LINEAR
            # Use the fit as a precise predictor of where identical frames should
            # be, extend each boundary toward the video edge, and keep the
            # extension only where the predicted pair is content-verified (same
            # frame, or both low-entropy black).  A genuinely different intro/outro
            # fails verification and the boundary stays at the outermost real
            # match — the line is never blindly projected across divergent
            # content.  No timestamp edge-snap here: the extension is already
            # frame-exact on the fitted line.
            matching_pairs = self._extrapolate_and_verify_global_linear(global_linear_fit, matching_pairs)
            debug.dump_matches(matching_pairs, "after verified global-linear extrapolation")
        else:
            relation = MappingRelation.GENERIC
            # No global relation — fall back to the content/entropy-aware
            # iterative boundary search.
            matching_pairs = self._extract_and_refine_boundaries(
                matching_pairs, lhs_scene_ranges, rhs_scene_ranges,
                lhs_normalized_frames, rhs_normalized_frames, debug,
            )
            matching_pairs = self.snap_to_edges(matching_pairs, self.lhs_all_frames, self.rhs_all_frames)

        self.logger.info("Mapping relation chosen: %s", relation.value)

        return SegmentsMappingResult(
            mapping=matching_pairs,
            lhs_all_frames=self.lhs_all_frames,
            rhs_all_frames=self.rhs_all_frames,
            relation=relation,
        )

    def _log_relation_diagnostics(self, matching_pairs: list[tuple[int, int]]) -> None:
        """Log the frame-space metrics that drive the mapping-relation choice.

        Emitted at INFO so the constant-offset / linear-drift / generic decision
        (and the audio strategy that follows from it) is visible without
        --verbose.  The individual ``try_*`` methods still log their precise
        rejection reasons at DEBUG.
        """
        expected_ratio = self.rhs_fps / self.lhs_fps if self.lhs_fps else float("nan")
        self.logger.info(
            "Relation diagnostics: %s fps=%.4f, %s fps=%.4f, "
            "expected lhs/rhs time ratio=%.4f, matched pairs=%d",
            self.lhs_label, self.lhs_fps, self.rhs_label, self.rhs_fps,
            expected_ratio, len(matching_pairs),
        )

        if len(matching_pairs) < 2:
            self.logger.info("  too few matched pairs for offset/drift analysis")
            return

        try:
            lhs_frame_ids = np.array(
                [int(self.lhs_all_frames[l]["frame_id"]) for l, _ in matching_pairs],
                dtype=float,
            )
            rhs_frame_ids = np.array(
                [int(self.rhs_all_frames[r]["frame_id"]) for _, r in matching_pairs],
                dtype=float,
            )
        except KeyError:
            self.logger.info("  matched frames lack frame_id; skipping offset/drift analysis")
            return

        frame_offsets = lhs_frame_ids - rhs_frame_ids
        median_offset = float(np.median(frame_offsets))
        std_offset = float(np.std(frame_offsets))
        lhs_span = float(np.ptp(lhs_frame_ids))
        ratio = PairMatcher.calculate_ratio(matching_pairs)

        if lhs_span > 0:
            slope = float(np.polyfit(lhs_frame_ids, rhs_frame_ids, 1)[0])
        else:
            slope = float("nan")
        time_scale = slope * self.lhs_fps / self.rhs_fps if self.rhs_fps else float("nan")

        self.logger.info(
            "  frame-offset median=%.2f std=%.2f (constant offset needs std<=%.1f, "
            "else RANSAC drift with slope within %.2f of 1.0); "
            "matched frame span=%.0f; observed time ratio=%.4f; "
            "frame slope=%.5f -> audio time scale=%.5f (audio is globally "
            "time-scaled by this factor)",
            median_offset, std_offset, self._MAX_CONSTANT_OFFSET_STD,
            self._MAX_DRIFT_SLOPE_DELTA, lhs_span, ratio, slope, time_scale,
        )

    def _normalize_frames(self, frames_info: FramesInfo, wd: str, desc: str = "Normalizing frames", prefix: str = "") -> FramesInfo:
        PairMatcher._assert_frames_extracted(frames_info, f"_normalize_frames({desc})")

        def crop_5_percent(image: cv.typing.MatLike) -> cv.typing.MatLike:
            height, width = image.shape
            dx = int(width * 0.05)
            dy = int(height * 0.05)

            image_cropped = image[dy:height - dy, dx:width - dx]

            return image_cropped

        def process_frame(item):
            timestamp, info = item
            self.interruption.check_for_stop()
            path = info["path"]
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read frame from {path}")
            img = crop_5_percent(img)
            img = cv.resize(img, (256, 256), interpolation=cv.INTER_AREA)
            _, file, ext = files_utils.split_path(path)
            new_path = os.path.join(wd, prefix + file + "." + ext)
            cv.imwrite(new_path, img)

            return timestamp, PairMatcher._get_new_info(info, new_path)

        with ThreadPoolExecutor() as executor:
            results_iter = executor.map(process_frame, frames_info.items())
            results = []
            with tqdm(total=len(frames_info), desc=desc, unit="frame",
                      **generic_utils.get_tqdm_defaults()) as pbar:
                for result in results_iter:
                    results.append(result)
                    pbar.update(1)

        return dict(results)

    @staticmethod
    def _is_rich(frame_path: str) -> bool:
        return image_utils.image_entropy(frame_path) > 3.5

    @staticmethod
    def _get_new_info(info: dict[str, str], path: str) -> dict[str, str]:
        new_info = info.copy()
        new_info["path"] = path
        return new_info

    @staticmethod
    def _assert_frames_extracted(frames: FramesInfo, context: str = "") -> None:
        """Validate that every entry in *frames* has a non-None path on disk.
        Raises ``AssertionError`` with a detailed message when a frame
        was probed but never extracted — this indicates a bug in the
        extraction-range computation.
        """
        for ts, info in frames.items():
            path = info.get("path")
            if path is None:
                frame_id = info.get("frame_id", "?")
                raise AssertionError(
                    f"Frame at {ts}ms (frame_id={frame_id}) has not been "
                    f"extracted to disk.  This is a bug — the frame should "
                    f"have been included in the extraction range.  "
                    f"Context: {context}"
                )

    @staticmethod
    def _filter_low_detailed(scenes: FramesInfo) -> FramesInfo:
        valuable_scenes = {timestamp: info for timestamp, info in scenes.items() if PairMatcher._is_rich(info["path"])}
        return valuable_scenes

    @staticmethod
    def _get_frames_for_timestamps(timestamps: list[int], frames_info: FramesInfo) -> FramesInfo:
        frame_files = {timestamp: info for timestamp, info in frames_info.items() if timestamp in timestamps}
        return frame_files

    @staticmethod
    def _compute_overlap(lhs_img: cv.typing.MatLike, rhs_img: cv.typing.MatLike, h) -> tuple[int, int, int, int]:
        # Expect images to be in the grayscale
        assert len(lhs_img.shape) == 2
        assert len(rhs_img.shape) == 2

        # Warp second image onto first
        warped_im2 = cv.warpPerspective(rhs_img, h, (lhs_img.shape[1], lhs_img.shape[0]))

        # Find overlapping region mask
        gray1 = lhs_img
        gray2 = warped_im2

        mask1 = (gray1 > 0).astype(np.uint8)
        mask2 = (gray2 > 0).astype(np.uint8)
        overlap_mask = cv.bitwise_and(mask1, mask2)

        # Find bounding rectangle of overlapping mask
        x, y, w, h = cv.boundingRect(overlap_mask)
        return (x, y, w, h)

    @staticmethod
    def _interpolate_crop_rects(timestamps_list: list[int], rects_list: list[tuple[int, int, int, int]]) -> Callable[[int], tuple[int, int, int, int]]:
        """
        Given a list of timestamps and matching crop rects, return a function that interpolates
        a crop for any timestamp between and extrapolates outside the range.
        rect = (x, y, w, h)
        """

        timestamps = np.array(timestamps_list)
        rects = np.array(rects_list)

        def interpolate(t):
            if t <= timestamps[0]:
                return tuple(rects[0])
            elif t >= timestamps[-1]:
                return tuple(rects[-1])
            else:
                x = np.interp(t, timestamps, rects[:, 0])
                y = np.interp(t, timestamps, rects[:, 1])
                w = np.interp(t, timestamps, rects[:, 2])
                h = np.interp(t, timestamps, rects[:, 3])
                return int(round(x)), int(round(y)), int(round(w)), int(round(h))

        return interpolate

    @staticmethod
    def _find_interpolated_crop(pairs_with_timestamps: list[tuple[int, int]], lhs_frames: FramesInfo, rhs_frames: FramesInfo) -> tuple[Callable[[int], tuple[int, int, int, int]], Callable[[int], tuple[int, int, int, int]]]:
        timestamps_lhs = []
        timestamps_rhs = []
        lhs_crops = []
        rhs_crops = []

        for lhs_t, rhs_t in pairs_with_timestamps:
            lhs_info = lhs_frames[lhs_t]
            rhs_info = rhs_frames[rhs_t]
            lhs_img = cv.imread(lhs_info["path"], cv.IMREAD_GRAYSCALE)
            rhs_img = cv.imread(rhs_info["path"], cv.IMREAD_GRAYSCALE)
            if lhs_img is None or rhs_img is None:
                continue

            orb = cv.ORB_create(1000)  # type: ignore[attr-defined]
            kp1, des1 = orb.detectAndCompute(lhs_img, None)
            kp2, des2 = orb.detectAndCompute(rhs_img, None)
            if des1 is None or des2 is None:
                continue

            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            if len(matches) < 3:
                continue

            matches = sorted(matches, key=lambda x: x.distance)
            pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

            h_matrix, inliers = cv.estimateAffinePartial2D(pts2, pts1, method=cv.RANSAC)
            if h_matrix is None:
                continue

            lhs_overlap = PairMatcher._compute_overlap(lhs_img, rhs_img, np.vstack([h_matrix, [0, 0, 1]]))
            rhs_overlap = PairMatcher._compute_overlap(rhs_img, lhs_img, np.vstack([cv.invertAffineTransform(h_matrix), [0, 0, 1]]))

            timestamps_lhs.append(lhs_t)
            timestamps_rhs.append(rhs_t)
            lhs_crops.append(lhs_overlap)
            rhs_crops.append(rhs_overlap)

        # Return interpolators
        return PairMatcher._interpolate_crop_rects(timestamps_lhs, lhs_crops), PairMatcher._interpolate_crop_rects(timestamps_rhs, rhs_crops)

    def _apply_crop_interpolated(self, frames: FramesInfo, dst_dir: str, crop_fn: Callable[[int], tuple[int, int, int, int]]) -> FramesInfo:
        def _process_frame(item):
            timestamp, info = item
            self.interruption.check_for_stop()
            path = info["path"]
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            x, y, w, h = crop_fn(timestamp)
            cropped = img[y:y+h, x:x+w]
            cropped = cv.resize(cropped, (128, 128))
            dst_path = os.path.join(dst_dir, os.path.basename(path))
            cv.imwrite(dst_path, cropped)
            return timestamp, PairMatcher._get_new_info(info, dst_path)

        with ThreadPoolExecutor() as executor:
            results = executor.map(_process_frame, frames.items())

        return dict(results)

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

    def _best_phash_match(self, lhs_ts: int, rhs_ts_guess: int, lhs_all_set: FramesInfo, rhs_all_set: FramesInfo) -> tuple[int, int] | None:
        lhs_near = self._nearest_three(list(lhs_all_set.keys()), lhs_ts)
        rhs_near = self._nearest_three(list(rhs_all_set.keys()), rhs_ts_guess)
        best = None
        best_dist = float("inf")
        for l in lhs_near:
            for r in rhs_near:
                if l in lhs_all_set and r in rhs_all_set:
                    d = abs(self.phash.get(lhs_all_set[l]["path"]) - self.phash.get(rhs_all_set[r]["path"]))
                    if d < best_dist:
                        best = (l, r)
                        best_dist = d
        return best

    def _build_matches(self, lhs: FramesInfo, rhs: FramesInfo) -> list[tuple[int, int, int]]:
        lhs_items = list(lhs.items())
        rhs_items = list(rhs.items())

        all_matches = []
        for lhs_ts, lhs_info in lhs_items:
            lhs_hash = self.phash.get(lhs_info["path"])
            for rhs_ts, rhs_info in rhs_items:
                rhs_hash = self.phash.get(rhs_info["path"])
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
        elif len(lhs_three) < 3 or len(rhs_three) < 3:
            raise NotImplementedError("Asymmetric history check not implemented")

        # at least one match before current pair is required
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
                        lhs_path = lhs_pool[l]["path"]
                        rhs_path = rhs_pool[rhs_candidate]["path"]

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
        rhs_cropped_dir: str
    ) -> tuple[FramesInfo, FramesInfo]:
        PairMatcher._assert_frames_extracted(lhs_frames, "_crop_both_sets(lhs)")
        PairMatcher._assert_frames_extracted(rhs_frames, "_crop_both_sets(rhs)")

        # Step 1: Get interpolated crop functions for both sets
        lhs_crop_fn, rhs_crop_fn = PairMatcher._find_interpolated_crop(pairs_with_timestamps, lhs_frames, rhs_frames)

        # Step 2: Apply interpolated cropping to each frame
        lhs_cropped = self._apply_crop_interpolated(lhs_frames, lhs_cropped_dir, lhs_crop_fn)
        rhs_cropped = self._apply_crop_interpolated(rhs_frames, rhs_cropped_dir, rhs_crop_fn)

        return lhs_cropped, rhs_cropped

    def _calculate_cutoff(
        self,
        phash: PhashCache,
        pairs: list[tuple[int, int]],
        lhs: FramesInfo,
        rhs: FramesInfo
    ) -> int:
        if not pairs:
            return 16  # sensible default when no pairs available

        distances = [abs(phash.get(lhs[lhs_ts]["path"]) - phash.get(rhs[rhs_ts]["path"])) for lhs_ts, rhs_ts in pairs]

        arr = np.array(distances)
        median = np.median(arr)
        std = np.std(arr)

        return median + std * 2

    # The two policies flooring an adaptive cutoff with _MIN_PHASH_CUTOFF.
    # They differ on purpose: the boundary walk always needs headroom, while
    # the temporal-consistency check must stay strict unless degenerate.

    @classmethod
    def _boundary_cutoff(cls, cutoff: float) -> float:
        """Cutoff for the boundary walk — never tighter than the floor.

        With few calibration pairs or cropped frames the computed cutoff can
        be as low as 4, which makes the search unable to extend even a single
        frame.
        """
        return max(cutoff, cls._MIN_PHASH_CUTOFF)

    @classmethod
    def _history_cutoff(cls, raw_cutoff: float) -> float:
        """Cutoff for the temporal-consistency check — floored only when degenerate.

        The adaptive cutoff (median + 2*std) collapses to ~0 only when the
        surviving matches are pixel-perfect (identical content — e.g. re-muxed
        or container-offset variants of the same source).  In that degenerate
        case the history check would demand pixel-perfect *neighbours* too —
        impossible across a speed/resolution difference — and would collapse a
        dozen good matches to one, dropping the pair into the GENERIC path.
        A normal (degraded) cutoff is left untouched so genuine mismatches
        stay rejected.
        """
        return cls._MIN_PHASH_CUTOFF if raw_cutoff < 1.0 else raw_cutoff

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

        extrapolated_refined: list[tuple[int, int]] = []
        for l, r in extrapolated:
            best_match = self._best_phash_match(l, r, lhs_all, rhs_all)
            if best_match is not None:
                extrapolated_refined.append(best_match)
        self.logger.debug(f"Frame adjustment:          {PairMatcher.summarize_pairs(self.phash, extrapolated_refined, lhs_all, rhs_all)}")

        outliers_eliminated = PairMatcher.filter_phash_outliers(self.phash, extrapolated_refined, lhs_all, rhs_all)
        self.logger.debug(f"Phash outlier elimination: {PairMatcher.summarize_pairs(self.phash, outliers_eliminated, lhs_all, rhs_all)}")

        orb_filtered = [
            (lhs_ts, rhs_ts) for lhs_ts, rhs_ts in outliers_eliminated
            if image_utils.are_images_similar(lhs_all[lhs_ts]["path"], rhs_all[rhs_ts]["path"])
        ]
        self.logger.debug(f"After ORB elimination:     {PairMatcher.summarize_pairs(self.phash, orb_filtered, lhs_all, rhs_all)}")

        raw_cutoff = self._calculate_cutoff(self.phash, orb_filtered, lhs_all, rhs_all)
        cutoff = self._history_cutoff(raw_cutoff)
        final = [pair for pair in orb_filtered if self._check_history(pair, lhs_all, rhs_all, cutoff)]
        self.logger.debug(f"After history analysis:    {PairMatcher.summarize_pairs(self.phash, final, lhs_all, rhs_all)}")

        unique_pairs = sorted(set(final))

        self.logger.debug(PairMatcher.summarize_segments(unique_pairs, self.lhs_fps, self.rhs_fps, lhs_label=self.lhs_label, rhs_label=self.rhs_label))

        return unique_pairs

    def _edge_content_matches(
        self,
        lhs: FramesInfo,
        rhs: FramesInfo,
        lhs_keys: list[int],
        rhs_keys: list[int],
        phash: "PhashCache",
        first: tuple[int, int],
        last: tuple[int, int],
        ratio: float,
        cutoff: float,
        direction: int,
    ) -> bool:
        """Check if both videos share the same content at one edge.

        *direction* = -1 checks the video start, +1 checks the video end.

        Compares the edge frame of LHS against the edge frame of RHS (not
        a prediction-based frame).  Validates with pHash, ratio consistency,
        and ORB.  Returns ``True`` when the edge pair looks like a valid match,
        meaning ``find_boundary`` should be able to reach the edge.
        """
        edge_lhs_ts = lhs_keys[0] if direction == -1 else lhs_keys[-1]
        edge_rhs_ts = rhs_keys[0] if direction == -1 else rhs_keys[-1]

        if edge_lhs_ts not in lhs or edge_rhs_ts not in rhs:
            return False

        d = abs(phash.get(lhs[edge_lhs_ts]["path"]) - phash.get(rhs[edge_rhs_ts]["path"]))
        if d > cutoff:
            return False

        # Ratio validation against the farther known pair
        reference = last if direction == -1 else first
        edge_pair = (edge_lhs_ts, edge_rhs_ts)
        if reference != edge_pair:
            cand_ratio = PairMatcher.calculate_ratio([edge_pair, reference])
            if not PairMatcher.is_ratio_acceptable(cand_ratio, ratio):
                return False

        # ORB verification on the half-resolution extracted frames
        lhs_path = self.lhs_all_frames[edge_lhs_ts]["path"] if edge_lhs_ts in self.lhs_all_frames else lhs[edge_lhs_ts]["path"]
        rhs_path = self.rhs_all_frames[edge_rhs_ts]["path"] if edge_rhs_ts in self.rhs_all_frames else rhs[edge_rhs_ts]["path"]
        return image_utils.are_images_similar(lhs_path, rhs_path)

    def _is_sustained_dark_zone(
        self,
        lhs: FramesInfo,
        lhs_keys: list[int],
        idx: int,
        direction: int,
    ) -> bool:
        """Check whether *idx* starts a sustained low-entropy zone (>= 1.5 s)."""
        look_ahead = max(3, int(1.5 * self.lhs_fps))
        for la in range(1, look_ahead + 1):
            la_idx = idx + direction * la
            if 0 <= la_idx < len(lhs_keys):
                if PairMatcher._is_rich(lhs[lhs_keys[la_idx]]["path"]):
                    return False
            # past the edge — treat as sustained
        return True

    def _try_jump_past_dark_zone(
        self,
        lhs: FramesInfo,
        rhs: FramesInfo,
        lhs_keys: list[int],
        rhs_keys: list[int],
        idx: int,
        direction: int,
        anchor: tuple[int, int],
        reference: tuple[int, int],
        ratio: float,
        cutoff: float,
        phash: "PhashCache",
    ) -> tuple[tuple[int, int] | None, int]:
        """Attempt to jump past a sustained dark zone.

        Scans from *idx* in *direction* until the first rich frame is found,
        then tries a phash + ratio match.  Returns ``(jumped_pair, jump_idx)``
        where *jumped_pair* is ``None`` when the jump failed.
        """
        jump_idx = idx + direction
        while 0 <= jump_idx < len(lhs_keys):
            if PairMatcher._is_rich(lhs[lhs_keys[jump_idx]]["path"]):
                jump_ts = lhs_keys[jump_idx]
                predicted_rhs = int(anchor[1] + (jump_ts - anchor[0]) / ratio)
                rhs_near = self._nearest_three(rhs_keys, predicted_rhs)
                best_dist = float('inf')
                best_pair: tuple[int, int] | None = None
                for rts in rhs_near:
                    if rts not in rhs or jump_ts not in lhs:
                        continue
                    d = abs(phash.get(lhs[jump_ts]["path"]) - phash.get(rhs[rts]["path"]))
                    if d < best_dist:
                        best_dist = d
                        best_pair = (jump_ts, rts)
                if best_pair is not None and best_dist <= cutoff:
                    accept = True
                    if reference != best_pair:
                        cand_ratio = PairMatcher.calculate_ratio([best_pair, reference])
                        accept = PairMatcher.is_ratio_acceptable(cand_ratio, ratio)
                    if accept:
                        return best_pair, jump_idx
                break
            jump_idx += direction
        return None, jump_idx

    def _try_match_edge_frame(
        self,
        lhs: FramesInfo,
        rhs: FramesInfo,
        lhs_keys: list[int],
        rhs_keys: list[int],
        anchor: tuple[int, int],
        current_best: tuple[int, int],
        reference: tuple[int, int],
        direction: int,
        ratio: float,
        cutoff: float,
        phash: "PhashCache",
    ) -> tuple[int, int] | None:
        """Try matching the actual video edge frame when the walk got close.

        The coarse step size may have skipped the edge frame; this method
        checks it explicitly.  Returns the matched pair or ``None``.
        """
        edge_idx = 0 if direction == -1 else len(lhs_keys) - 1
        edge_ts = lhs_keys[edge_idx]
        if edge_ts == current_best[0] or edge_ts not in lhs:
            return None

        predicted_rhs = int(anchor[1] + (edge_ts - anchor[0]) / ratio)
        clamped = max(rhs_keys[0], min(rhs_keys[-1], predicted_rhs))
        rhs_near = self._nearest_three(rhs_keys, clamped)

        best_dist = float('inf')
        best_pair: tuple[int, int] | None = None
        for rts in rhs_near:
            if rts not in rhs:
                continue
            d = abs(phash.get(lhs[edge_ts]["path"]) - phash.get(rhs[rts]["path"]))
            if d < best_dist:
                best_dist = d
                best_pair = (edge_ts, rts)

        if best_pair is not None and best_dist <= cutoff:
            accept = True
            if reference != best_pair:
                cand_ratio = PairMatcher.calculate_ratio([best_pair, reference])
                accept = PairMatcher.is_ratio_acceptable(cand_ratio, ratio)
            if accept:
                return best_pair
        return None

    def _find_boundary(
        self,
        lhs: FramesInfo,
        rhs: FramesInfo,
        anchor: tuple[int, int],
        reference: tuple[int, int],
        direction: int,
        gap_seconds: float,
        ratio: float,
        cutoff: float,
        phash: "PhashCache",
    ) -> tuple[tuple[int, int], bool]:
        """Walk from *anchor* in *direction*, using linear prediction to find matches.

        Returns ``(best_pair, entered_low_entropy)`` where the flag indicates
        that the search stopped because it reached a low-entropy zone rather
        than exhausting the gap budget on high-entropy mismatches.
        """
        lhs_keys = sorted(lhs.keys())
        rhs_keys = sorted(rhs.keys())

        current_best = anchor
        max_gap = int(gap_seconds * self.lhs_fps)
        consecutive_misses = 0
        entered_low_entropy = False

        step = max(1, int(self.lhs_fps * 0.5))
        start_idx = lhs_keys.index(anchor[0])
        i = 0

        while True:
            i += step
            idx = start_idx + direction * i
            if idx < 0 or idx >= len(lhs_keys):
                # The coarse step overshot the edge.  Switch to frame-by-frame
                # walking from current_best so we don't miss near-edge matches.
                if step > 1:
                    best_idx = lhs_keys.index(current_best[0])
                    i = direction * (best_idx - start_idx)
                    step = 1
                    continue
                break

            lhs_ts = lhs_keys[idx]

            # Dark frame — check if it's a sustained dark zone
            if not PairMatcher._is_rich(lhs[lhs_ts]["path"]):
                if self._is_sustained_dark_zone(lhs, lhs_keys, idx, direction):
                    jumped_pair, jump_idx = self._try_jump_past_dark_zone(
                        lhs, rhs, lhs_keys, rhs_keys, idx, direction,
                        anchor, reference, ratio, cutoff, phash,
                    )
                    if jumped_pair is not None:
                        dark_start = lhs_ts
                        dark_end = lhs_keys[jump_idx]
                        self.logger.debug(
                            f"Jumped over dark zone {dark_start}ms → {dark_end}ms "
                            f"(pair: {jumped_pair})"
                        )
                        current_best = jumped_pair
                        consecutive_misses = 0
                        i = direction * (jump_idx - start_idx)
                        continue
                    else:
                        entered_low_entropy = True
                        break
                # Transient dark frame — treat as a miss
                consecutive_misses += step
                if consecutive_misses >= max_gap:
                    break
                continue

            # Predict the corresponding RHS timestamp using the linear mapping
            predicted_rhs = int(anchor[1] + (lhs_ts - anchor[0]) / ratio)
            rhs_near = self._nearest_three(rhs_keys, predicted_rhs)

            best_dist = float('inf')
            best_pair: tuple[int, int] | None = None
            for rhs_ts in rhs_near:
                if rhs_ts not in rhs or lhs_ts not in lhs:
                    continue
                d = abs(phash.get(lhs[lhs_ts]["path"]) - phash.get(rhs[rhs_ts]["path"]))
                if d < best_dist:
                    best_dist = d
                    best_pair = (lhs_ts, rhs_ts)

            if best_pair is not None and best_dist <= cutoff:
                if reference != best_pair:
                    cand_ratio = PairMatcher.calculate_ratio([best_pair, reference])
                    if not PairMatcher.is_ratio_acceptable(cand_ratio, ratio):
                        consecutive_misses += step
                        if consecutive_misses >= max_gap:
                            break
                        continue

                current_best = best_pair
                consecutive_misses = 0
            else:
                consecutive_misses += step
                if consecutive_misses >= max_gap:
                    break

        # Try matching the actual edge frame
        if not entered_low_entropy and consecutive_misses < max_gap:
            edge_match = self._try_match_edge_frame(
                lhs, rhs, lhs_keys, rhs_keys, anchor, current_best,
                reference, direction, ratio, cutoff, phash,
            )
            if edge_match is not None:
                current_best = edge_match

        return current_best, entered_low_entropy

    def _verify_gap_is_low_entropy(
        self,
        frames: FramesInfo,
        gap_keys: list[int],
        max_skip: int,
        direction: int,
        label: str,
        side: str,
    ) -> bool:
        """Check that a gap between boundary and edge is mostly low-entropy.

        Walks *gap_keys* (sorted from boundary toward edge) and:
        1. Finds the first low-entropy frame,
        2. Rejects if too many high-entropy frames precede it (> *max_skip*),
        3. Rejects if the remaining portion has > 5% high-entropy frames.

        Returns ``True`` when the gap can be extrapolated through.
        """
        from_boundary = sorted(gap_keys, reverse=(direction == -1))
        le_start: int | None = None
        for i, ts in enumerate(from_boundary):
            if not PairMatcher._is_rich(frames[ts]["path"]):
                le_start = i
                break

        if le_start is None:
            self.logger.warning(
                f"Boundary {label}: gap in {side} is all high-entropy. "
                f"Cannot extrapolate."
            )
            return False

        if le_start > max_skip:
            self.logger.warning(
                f"Boundary {label}: {le_start} high-entropy {side} frames "
                f"between boundary and low-entropy zone (max expected: {max_skip}). "
                f"Cannot extrapolate."
            )
            return False

        remaining = from_boundary[le_start:]
        non_le = [k for k in remaining if PairMatcher._is_rich(frames[k]["path"])]
        noise_ratio = len(non_le) / len(remaining) if remaining else 1.0
        if noise_ratio > 0.05:
            self.logger.warning(
                f"Boundary {label}: {side} zone has {len(non_le)}/{len(remaining)} "
                f"({noise_ratio:.1%}) high-entropy frames — not contiguous. "
                f"Cannot extrapolate."
            )
            return False
        elif non_le:
            self.logger.debug(
                f"Boundary {label}: tolerating {len(non_le)}/{len(remaining)} "
                f"({noise_ratio:.1%}) sparse high-entropy frames in {side} zone."
            )

        return True

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

    def _detect_scenes(self) -> tuple[list[int], list[int]]:
        """Phase 1: Detect scene changes in both files."""
        lhs_scene_changes = self._detect_scenes_for(self.lhs_path, self.lhs_label)
        rhs_scene_changes = self._detect_scenes_for(self.rhs_path, self.rhs_label)

        if len(lhs_scene_changes) == 0 or len(rhs_scene_changes) == 0:
            raise RuntimeError("Not enough scene changes detected")

        return lhs_scene_changes, rhs_scene_changes

    def _detect_scenes_for(self, video_path: str, label: str) -> list[int]:
        if self.cache:
            cached = self.cache.load_scene_changes(video_path)
            if cached is not None:
                self.logger.info("[1/6] Scene changes for %s restored from cache (%d scenes)", label, len(cached))
                return cached

        result = video_utils.detect_scene_changes(
            video_path, threshold=0.3, logger=self.logger,
            interruption=self.interruption, desc=f"[1/6] Detecting scenes: {label}",
        )

        if self.cache:
            self.cache.save_scene_changes(video_path, result)

        return result

    def _probe_frames(self) -> None:
        """Phase 2: Probe all frame timestamps (fast — no images written)."""
        self.lhs_all_frames = self._probe_frames_for(self.lhs_path, self.lhs_label)
        self.rhs_all_frames = self._probe_frames_for(self.rhs_path, self.rhs_label)

    def _probe_frames_for(self, video_path: str, label: str) -> dict[int, dict]:
        if self.cache:
            cached = self.cache.load_frame_probes(video_path)
            if cached is not None:
                self.logger.info("[2/6] Frame probes for %s restored from cache (%d frames)", label, len(cached))
                return cached

        result = video_utils.probe_frame_timestamps(
            video_path, interruption=self.interruption,
            desc=f"[2/6] Probing frames: {label}",
            logger=self.logger,
        )

        if self.cache:
            self.cache.save_frame_probes(video_path, result)

        return result

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
            self.lhs_path, self.lhs_all_wd, lhs_scene_ranges, self.lhs_all_frames, self.lhs_label,
        )
        self._extract_scene_frames_for(
            self.rhs_path, self.rhs_all_wd, rhs_scene_ranges, self.rhs_all_frames, self.rhs_label,
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
        probed_metadata: dict[int, dict],
        label: str,
    ) -> None:
        if self.cache and self.cache.load_scene_frames(video_path, target_dir, probed_metadata):
            self.logger.info("[3/6] Scene frames for %s restored from cache", label)
            return

        video_utils.extract_frames_at_ranges(
            video_path, target_dir, scene_ranges, probed_metadata,
            scale=(960, -2), format="png", interruption=self.interruption,
            desc=f"[3/6] Extracting scene frames: {label}",
            logger=self.logger,
        )

        if self.cache:
            self.cache.save_scene_frames(video_path, target_dir, probed_metadata)

    def _normalize_extracted(
        self,
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
    ) -> tuple[FramesInfo, FramesInfo, FramesInfo, FramesInfo]:
        """Phase 4: Normalize extracted frames and identify key frames.

        Returns (lhs_normalized, rhs_normalized, lhs_key_frames, rhs_key_frames).
        """
        lhs_extracted = self._extracted_subset(self.lhs_all_frames)
        rhs_extracted = self._extracted_subset(self.rhs_all_frames)

        lhs_normalized_frames = self._normalize_frames(
            lhs_extracted, self.lhs_normalized_wd,
            desc=f"[4/6] Normalizing: {self.lhs_label}",
        )
        rhs_normalized_frames = self._normalize_frames(
            rhs_extracted, self.rhs_normalized_wd,
            desc=f"[4/6] Normalizing: {self.rhs_label}",
        )

        lhs_key_frames = PairMatcher._get_frames_for_timestamps(lhs_scene_changes, lhs_normalized_frames)
        rhs_key_frames = PairMatcher._get_frames_for_timestamps(rhs_scene_changes, rhs_normalized_frames)

        return lhs_normalized_frames, rhs_normalized_frames, lhs_key_frames, rhs_key_frames

    def _match_key_frames(
        self,
        lhs_key_frames: FramesInfo,
        rhs_key_frames: FramesInfo,
        lhs_normalized_frames: FramesInfo,
        rhs_normalized_frames: FramesInfo,
        debug: DebugRoutines,
    ) -> list[tuple[int, int]]:
        """Phase 5: Match key frames between the two files."""
        lhs_extracted = self._extracted_subset(self.lhs_all_frames)
        rhs_extracted = self._extracted_subset(self.rhs_all_frames)

        self.logger.info("[5/6] Matching key frames")
        matching_pairs = self._make_pairs(lhs_key_frames, rhs_key_frames, lhs_normalized_frames, rhs_normalized_frames)
        debug.dump_matches(matching_pairs, "initial matching")
        self.logger.debug("Pairs summary after initial matching:")
        self.logger.debug(PairMatcher.summarize_pairs(self.phash, matching_pairs, lhs_extracted, rhs_extracted, verbose = True))

        if not matching_pairs:
            raise RuntimeError("No matching pairs found between the two files")

        return matching_pairs

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
            self.lhs_path, self.lhs_boundary_wd, lhs_boundary_ranges, self.lhs_all_frames,
            scale=(960, -2), format="png", interruption=self.interruption,
            desc=f"[6/6] Extracting boundary frames: {self.lhs_label}",
            logger=self.logger,
        )
        video_utils.extract_frames_at_ranges(
            self.rhs_path, self.rhs_boundary_wd, rhs_boundary_ranges, self.rhs_all_frames,
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
                lhs_new, self.lhs_normalized_wd, prefix="b_",
                desc=f"[6/6] Normalizing boundary: {self.lhs_label}",
            )
            lhs_normalized_frames.update(lhs_boundary_normalized)

        if rhs_new:
            rhs_boundary_normalized = self._normalize_frames(
                rhs_new, self.rhs_normalized_wd, prefix="b_",
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
                lhs_cropped_dir=self.lhs_normalized_cropped_wd,
                rhs_cropped_dir=self.rhs_normalized_cropped_wd,
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
