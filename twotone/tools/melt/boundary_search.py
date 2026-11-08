"""Content-aware boundary algorithms for melt frame mappings."""

import logging
from bisect import bisect_right
from collections.abc import Callable
from math import ceil, floor
from typing import NamedTuple, Protocol

import numpy as np

from ..utils import image_utils, video_utils
from . import frame_comparison, matching_analysis
from .frame_comparison import BoundaryVerifyContext, FrameComparison, MatchingSide
from .melt_common import FramesInfo
from .phash_cache import PhashCache


# One isolated miss may be pHash noise; two consecutive misses establish a
# divergent intro/outro.  A boundary still lands only on a verified sample.
MAX_BOUNDARY_GAP_MISSES = 1
# Fitted predictions can land a frame or two away across ffmpeg builds.
BOUNDARY_PREDICTION_JITTER_FRAMES = 2


class GlobalLinearFitLike(Protocol):
    @property
    def slope(self) -> float: ...


class BoundarySearchContext(NamedTuple):
    """Inputs for one directional GENERIC boundary walk."""

    lhs: FramesInfo
    rhs: FramesInfo
    lhs_keys: list[int]
    rhs_keys: list[int]
    phash: PhashCache
    anchor: tuple[int, int]
    reference: tuple[int, int]
    ratio: float
    cutoff: float
    direction: int


class BoundarySearch:
    """Find common-content boundaries without access to the matcher object."""

    def __init__(
        self,
        logger: logging.Logger,
        lhs: MatchingSide,
        rhs: MatchingSide,
        comparison: FrameComparison,
    ) -> None:
        self.logger = logger
        self.lhs = lhs
        self.rhs = rhs
        self.comparison = comparison

    def look_for_boundaries(
        self,
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo,
        first: tuple[int, int],
        last: tuple[int, int],
        cutoff: float,
        max_gap_seconds: float = 15.0,
        extrapolate: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Walk outward from known matches to find first/last common frames."""
        frame_comparison.assert_frames_extracted(
            lhs_frames,
            "look_for_boundaries(lhs)",
        )
        frame_comparison.assert_frames_extracted(
            rhs_frames,
            "look_for_boundaries(rhs)",
        )

        self.logger.debug("Improving boundaries")
        self.logger.debug("Current first: %s and last: %s", first, last)
        phash = PhashCache()
        ratio = matching_analysis.calculate_ratio([first, last])
        if np.isnan(ratio):
            ratio = self.lhs.fps / self.rhs.fps
        cutoff = frame_comparison.floored_phash_cutoff(cutoff)

        lhs_keys = sorted(lhs_frames)
        rhs_keys = sorted(rhs_frames)
        start_context = BoundarySearchContext(
            lhs=lhs_frames,
            rhs=rhs_frames,
            lhs_keys=lhs_keys,
            rhs_keys=rhs_keys,
            phash=phash,
            anchor=first,
            reference=last,
            ratio=ratio,
            cutoff=cutoff,
            direction=-1,
        )
        end_context = start_context._replace(
            anchor=last,
            reference=first,
            direction=1,
        )

        gap_budgets = {-1: max_gap_seconds, 1: max_gap_seconds}
        for context in (start_context, end_context):
            label = "start" if context.direction == -1 else "end"
            if self._edge_content_matches(context):
                edge_lhs = (
                    lhs_keys[0]
                    if context.direction == -1
                    else lhs_keys[-1]
                )
                distance_seconds = abs(context.anchor[0] - edge_lhs) / 1000.0
                gap_budgets[context.direction] = max(
                    max_gap_seconds,
                    distance_seconds + 2.0,
                )
                self.logger.debug(
                    "Edge pre-check (%s): edges match, extending gap budget "
                    "to %.1fs (anchor=%dms, edge=%dms, distance=%.1fs)",
                    label,
                    gap_budgets[context.direction],
                    context.anchor[0],
                    edge_lhs,
                    distance_seconds,
                )
            else:
                self.logger.debug(
                    "Edge pre-check (%s): edges do NOT visually match — "
                    "using default gap budget %.1fs",
                    label,
                    max_gap_seconds,
                )

        refined_first, first_low_entropy = self._find_boundary(
            start_context,
            gap_budgets[-1],
        )
        self.logger.debug(
            "Boundary start: walked %dms → %dms (%s, budget=%.1fs)",
            first[0],
            refined_first[0],
            (
                "entered low-entropy zone"
                if first_low_entropy
                else "gap budget or edge reached"
            ),
            gap_budgets[-1],
        )
        refined_last, last_low_entropy = self._find_boundary(
            end_context,
            gap_budgets[1],
        )
        self.logger.debug(
            "Boundary end: walked %dms → %dms (%s, budget=%.1fs)",
            last[0],
            refined_last[0],
            (
                "entered low-entropy zone"
                if last_low_entropy
                else "gap budget or edge reached"
            ),
            gap_budgets[1],
        )
        if extrapolate:
            refined_first = self.extrapolate_through_low_entropy(
                lhs_frames,
                rhs_frames,
                refined_first,
                refined_last,
                ratio,
                direction=-1,
                entered_low_entropy=first_low_entropy,
            )
            refined_last = self.extrapolate_through_low_entropy(
                lhs_frames,
                rhs_frames,
                refined_last,
                refined_first,
                ratio,
                direction=1,
                entered_low_entropy=last_low_entropy,
            )
        return refined_first, refined_last

    def extrapolate_through_low_entropy(
        self,
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo,
        boundary: tuple[int, int],
        reference: tuple[int, int],
        ratio: float,
        direction: int,
        entered_low_entropy: bool,
    ) -> tuple[int, int]:
        del reference
        if not entered_low_entropy:
            return boundary

        label = "start" if direction == -1 else "end"
        lhs_keys = sorted(lhs_frames)
        rhs_keys = sorted(rhs_frames)
        step = max(1, int(self.lhs.fps * 0.5))
        max_skip = step * 3

        if direction == -1:
            edge_lhs = lhs_keys[0]
            lhs_gap = [key for key in lhs_keys if key < boundary[0]]
        else:
            edge_lhs = lhs_keys[-1]
            lhs_gap = [key for key in lhs_keys if key > boundary[0]]
        if not lhs_gap:
            return boundary
        if not self._verify_gap_is_low_entropy(
            lhs_frames,
            lhs_gap,
            max_skip,
            direction,
            label,
            self.lhs.label,
        ):
            return boundary

        predicted_rhs = int(boundary[1] + (edge_lhs - boundary[0]) / ratio)
        clamped_rhs = max(rhs_keys[0], min(rhs_keys[-1], predicted_rhs))
        if direction == -1:
            rhs_gap = [
                key
                for key in rhs_keys
                if clamped_rhs <= key < boundary[1]
            ]
        else:
            rhs_gap = [
                key
                for key in rhs_keys
                if boundary[1] < key <= clamped_rhs
            ]
        if rhs_gap and not self._verify_gap_is_low_entropy(
            rhs_frames,
            rhs_gap,
            max_skip,
            direction,
            label,
            self.rhs.label,
        ):
            return boundary

        new_rhs = self._snap_to_nearest_frame(rhs_keys, clamped_rhs)
        self.logger.debug(
            "Boundary %s: extrapolating through low-entropy zone from "
            "(%d, %d) to (%d, %d).",
            label,
            boundary[0],
            boundary[1],
            edge_lhs,
            new_rhs,
        )
        return edge_lhs, new_rhs

    def extrapolate_and_verify_global_linear(
        self,
        fit: GlobalLinearFitLike,
        matching_pairs: list[tuple[int, int]],
        lhs_normalized_frames: FramesInfo,
        rhs_normalized_frames: FramesInfo,
    ) -> list[tuple[int, int]]:
        """Extend a certified linear relation only through verified content."""
        result = sorted(matching_pairs)
        if fit.slope <= 0 or len(result) < 2:
            return result

        verify_context = self.comparison.build_boundary_verify_context(
            matching_pairs,
            lhs_normalized_frames,
            rhs_normalized_frames,
        )
        observed_pairs = np.asarray(result, dtype=float)
        pair_slopes = [
            (rhs2 - rhs1) / (lhs2 - lhs1)
            for index, (lhs1, rhs1) in enumerate(observed_pairs)
            for lhs2, rhs2 in observed_pairs[index + 1:]
            if lhs2 != lhs1
        ]
        if not pair_slopes:
            return result
        slope = float(np.median(pair_slopes))
        intercept = float(np.median(
            observed_pairs[:, 1] - slope * observed_pairs[:, 0]
        ))

        lhs_timestamps = sorted(self.lhs.all_frames)
        rhs_timestamps = sorted(self.rhs.all_frames)
        lhs_min_time, lhs_max_time = lhs_timestamps[0], lhs_timestamps[-1]
        rhs_min_time, rhs_max_time = rhs_timestamps[0], rhs_timestamps[-1]
        lhs_snap_tolerance_ms = 4 * 1000 / self.lhs.fps
        rhs_snap_tolerance_ms = 4 * 1000 / self.rhs.fps

        first_lhs_time = max(lhs_min_time, (rhs_min_time - intercept) / slope)
        first_rhs_time = slope * first_lhs_time + intercept
        if (
            first_lhs_time - lhs_min_time <= lhs_snap_tolerance_ms
            and abs(slope * lhs_min_time + intercept - rhs_min_time)
            <= rhs_snap_tolerance_ms
        ):
            first_lhs_time = lhs_min_time
            first_rhs_time = rhs_min_time
        first_rhs_time = max(rhs_min_time, min(rhs_max_time, first_rhs_time))
        self._maybe_insert_verified_boundary(
            result,
            "start",
            slope,
            intercept,
            first_lhs_time,
            first_rhs_time,
            lhs_timestamps,
            rhs_timestamps,
            verify_context,
        )

        last_lhs_time = min(lhs_max_time, (rhs_max_time - intercept) / slope)
        last_rhs_time = slope * last_lhs_time + intercept
        if (
            lhs_max_time - last_lhs_time <= lhs_snap_tolerance_ms
            and abs(slope * lhs_max_time + intercept - rhs_max_time)
            <= rhs_snap_tolerance_ms
        ):
            last_lhs_time = lhs_max_time
            last_rhs_time = rhs_max_time
        last_rhs_time = max(rhs_min_time, min(rhs_max_time, last_rhs_time))
        self._maybe_insert_verified_boundary(
            result,
            "end",
            slope,
            intercept,
            last_lhs_time,
            last_rhs_time,
            lhs_timestamps,
            rhs_timestamps,
            verify_context,
        )
        return result

    def snap_to_edges(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
        snap_frames: int = 16,
    ) -> list[tuple[int, int]]:
        lhs_threshold_ms = snap_frames * 1000 / self.lhs.fps
        rhs_threshold_ms = snap_frames * 1000 / self.rhs.fps
        lhs_duration = video_utils.get_video_duration(
            self.lhs.path,
            logger=self.logger,
        )
        rhs_duration = video_utils.get_video_duration(
            self.rhs.path,
            logger=self.logger,
        )
        lhs_keys = sorted(lhs_all_frames)
        rhs_keys = sorted(rhs_all_frames)

        first_lhs, first_rhs = matching_pairs[0]
        last_lhs, last_rhs = matching_pairs[-1]
        new_first_lhs = 0 if first_lhs <= lhs_threshold_ms else first_lhs
        new_first_rhs = 0 if first_rhs <= rhs_threshold_ms else first_rhs
        if (new_first_lhs, new_first_rhs) != (first_lhs, first_rhs):
            self.logger.debug(
                "Edge snap: first pair (%d, %d) → (%d, %d)",
                first_lhs,
                first_rhs,
                new_first_lhs,
                new_first_rhs,
            )
            matching_pairs[0] = new_first_lhs, new_first_rhs
            if new_first_lhs not in lhs_all_frames:
                lhs_all_frames[new_first_lhs] = lhs_all_frames[lhs_keys[0]].copy()
            if new_first_rhs not in rhs_all_frames:
                rhs_all_frames[new_first_rhs] = rhs_all_frames[rhs_keys[0]].copy()

        new_last_lhs = (
            lhs_duration
            if lhs_duration - last_lhs <= lhs_threshold_ms
            else last_lhs
        )
        new_last_rhs = (
            rhs_duration
            if rhs_duration - last_rhs <= rhs_threshold_ms
            else last_rhs
        )
        if (new_last_lhs, new_last_rhs) != (last_lhs, last_rhs):
            self.logger.debug(
                "Edge snap: last pair (%d, %d) → (%d, %d)",
                last_lhs,
                last_rhs,
                new_last_lhs,
                new_last_rhs,
            )
            matching_pairs[-1] = new_last_lhs, new_last_rhs
            if new_last_lhs not in lhs_all_frames:
                lhs_all_frames[new_last_lhs] = lhs_all_frames[lhs_keys[-1]].copy()
            if new_last_rhs not in rhs_all_frames:
                rhs_all_frames[new_last_rhs] = rhs_all_frames[rhs_keys[-1]].copy()
        return matching_pairs

    @staticmethod
    def _resolve_timestamp_samples(
        timestamps: list[int],
        sample_times: list[float],
        round_sample: Callable[[float], int] = round,
    ) -> dict[float, tuple[int, int]]:
        samples: dict[float, tuple[int, int]] = {}
        for sample_time in set(sample_times):
            index = bisect_right(timestamps, sample_time)
            if index > 0 and timestamps[index - 1] == sample_time:
                timestamp = timestamps[index - 1]
                samples[sample_time] = timestamp, timestamp
            elif 0 < index < len(timestamps):
                previous, following = timestamps[index - 1], timestamps[index]
                image_timestamp = previous
                sample_timestamp = round_sample(sample_time)
                if sample_timestamp == following:
                    image_timestamp = sample_timestamp
                samples[sample_time] = sample_timestamp, image_timestamp
        return samples

    def _maybe_insert_verified_boundary(
        self,
        result: list[tuple[int, int]],
        side: str,
        slope: float,
        intercept: float,
        lhs_time: float,
        rhs_time: float,
        lhs_timestamps: list[int],
        rhs_timestamps: list[int],
        verify_context: BoundaryVerifyContext,
    ) -> None:
        lhs_sample = self._resolve_timestamp_samples(
            lhs_timestamps,
            [lhs_time],
        ).get(lhs_time)
        rhs_sample = self._resolve_timestamp_samples(
            rhs_timestamps,
            [rhs_time],
        ).get(rhs_time)
        if lhs_sample is None or rhs_sample is None:
            return
        lhs_timestamp, rhs_timestamp = lhs_sample[0], rhs_sample[0]
        anchor = result[0] if side == "start" else result[-1]
        if side == "start":
            beyond = lhs_timestamp < anchor[0] or rhs_timestamp < anchor[1]
        else:
            beyond = lhs_timestamp > anchor[0] or rhs_timestamp > anchor[1]
        if not beyond or (lhs_timestamp, rhs_timestamp) == anchor:
            return

        verified = self._walk_shared_boundary(
            slope,
            intercept,
            anchor[0],
            lhs_time,
            rhs_time,
            lhs_timestamps,
            rhs_timestamps,
            verify_context,
        )
        if verified is None or verified == anchor:
            self.logger.debug(
                "Global-linear %s gap diverges before any shared frame — "
                "keeping the boundary at the outermost match",
                side,
            )
        elif side == "start":
            result.insert(0, verified)
        else:
            result.append(verified)

    def _walk_shared_boundary(
        self,
        slope: float,
        intercept: float,
        anchor_lhs_time: float,
        boundary_lhs_time: float,
        boundary_rhs_time: float,
        lhs_timestamps: list[int],
        rhs_timestamps: list[int],
        verify_context: BoundaryVerifyContext,
    ) -> tuple[int, int] | None:
        step_ms = max(1, int(self.lhs.fps * 0.5)) * 1000 / self.lhs.fps
        distance = boundary_lhs_time - anchor_lhs_time
        direction = 1 if distance >= 0 else -1
        lhs_sample_times = [
            anchor_lhs_time + direction * step_ms * index
            for index in range(1, int(np.ceil(abs(distance) / step_ms)))
        ]
        jitter = BOUNDARY_PREDICTION_JITTER_FRAMES
        rhs_frame_ms = 1000 / self.rhs.fps
        if (
            lhs_sample_times
            and abs(slope * (boundary_lhs_time - lhs_sample_times[-1]))
            <= jitter * rhs_frame_ms
        ):
            previous = (
                lhs_sample_times[-2]
                if len(lhs_sample_times) > 1
                else anchor_lhs_time
            )
            lhs_sample_times[-1] = round((previous + boundary_lhs_time) / 2)
        samples = [
            (lhs_time, slope * lhs_time + intercept)
            for lhs_time in lhs_sample_times
        ]
        samples.append((boundary_lhs_time, boundary_rhs_time))

        lhs_samples = self._resolve_timestamp_samples(
            lhs_timestamps,
            [lhs_time for lhs_time, _ in samples],
        )
        rhs_samples = self._resolve_timestamp_samples(
            rhs_timestamps,
            [
                rhs_time + delta * rhs_frame_ms
                for _, rhs_time in samples
                for delta in range(-jitter, jitter + 1)
            ],
        )
        boundary_round = floor if direction < 0 else ceil
        lhs_samples.update(self._resolve_timestamp_samples(
            lhs_timestamps,
            [boundary_lhs_time],
            boundary_round,
        ))
        rhs_samples.update(self._resolve_timestamp_samples(
            rhs_timestamps,
            [boundary_rhs_time],
            boundary_round,
        ))
        self.comparison.prefetch(
            self.lhs,
            [image_timestamp for _, image_timestamp in lhs_samples.values()],
        )
        self.comparison.prefetch(
            self.rhs,
            [image_timestamp for _, image_timestamp in rhs_samples.values()],
        )

        best: tuple[int, int] | None = None
        consecutive_misses = 0
        for lhs_time, rhs_time in samples:
            lhs_sample = lhs_samples.get(lhs_time)
            rhs_sample = rhs_samples.get(rhs_time)
            if lhs_sample is None or rhs_sample is None:
                break
            if self._shared_content_at_prediction(
                verify_context,
                rhs_time,
                lhs_sample[1],
                rhs_samples,
            ):
                best = lhs_sample[0], rhs_sample[0]
                consecutive_misses = 0
            else:
                consecutive_misses += 1
                if consecutive_misses > MAX_BOUNDARY_GAP_MISSES:
                    break
        return best

    def _shared_content_at_prediction(
        self,
        verify_context: BoundaryVerifyContext,
        rhs_time: float,
        lhs_timestamp: int,
        rhs_samples: dict[float, tuple[int, int]],
    ) -> bool:
        jitter = BOUNDARY_PREDICTION_JITTER_FRAMES
        rhs_frame_ms = 1000 / self.rhs.fps
        for delta in sorted(range(-jitter, jitter + 1), key=abs):
            candidate = rhs_samples.get(rhs_time + delta * rhs_frame_ms)
            if candidate is not None and self.comparison.boundary_content_matches(
                verify_context,
                lhs_timestamp,
                candidate[1],
            ):
                return True
        return False

    def _matched_predicted_pair(
        self,
        context: BoundarySearchContext,
        lhs_timestamp: int,
        *,
        clamp: bool = False,
    ) -> tuple[int, int] | None:
        predicted_rhs = int(
            context.anchor[1]
            + (lhs_timestamp - context.anchor[0]) / context.ratio
        )
        if clamp:
            predicted_rhs = max(
                context.rhs_keys[0],
                min(context.rhs_keys[-1], predicted_rhs),
            )
        rhs_near = self._nearest_three(context.rhs_keys, predicted_rhs)

        best_distance = float("inf")
        best_pair: tuple[int, int] | None = None
        for rhs_timestamp in rhs_near:
            if rhs_timestamp not in context.rhs or lhs_timestamp not in context.lhs:
                continue
            distance = abs(
                context.phash.get(frame_comparison.extracted_path(
                    context.lhs[lhs_timestamp]
                ))
                - context.phash.get(frame_comparison.extracted_path(
                    context.rhs[rhs_timestamp]
                ))
            )
            if distance < best_distance:
                best_distance = distance
                best_pair = lhs_timestamp, rhs_timestamp

        if best_pair is None or best_distance > context.cutoff:
            return None
        if context.reference != best_pair:
            candidate_ratio = matching_analysis.calculate_ratio([
                best_pair,
                context.reference,
            ])
            if not matching_analysis.is_ratio_acceptable(
                candidate_ratio,
                context.ratio,
            ):
                return None
        return best_pair

    def _edge_content_matches(self, context: BoundarySearchContext) -> bool:
        edge_lhs = (
            context.lhs_keys[0]
            if context.direction == -1
            else context.lhs_keys[-1]
        )
        edge_rhs = (
            context.rhs_keys[0]
            if context.direction == -1
            else context.rhs_keys[-1]
        )
        if edge_lhs not in context.lhs or edge_rhs not in context.rhs:
            return False
        distance = abs(
            context.phash.get(frame_comparison.extracted_path(
                context.lhs[edge_lhs]
            ))
            - context.phash.get(frame_comparison.extracted_path(
                context.rhs[edge_rhs]
            ))
        )
        if distance > context.cutoff:
            return False

        edge_pair = edge_lhs, edge_rhs
        if context.reference != edge_pair:
            candidate_ratio = matching_analysis.calculate_ratio([
                edge_pair,
                context.reference,
            ])
            if not matching_analysis.is_ratio_acceptable(
                candidate_ratio,
                context.ratio,
            ):
                return False

        lhs_path = (
            frame_comparison.path_or_none(self.lhs.all_frames, edge_lhs)
            or frame_comparison.extracted_path(context.lhs[edge_lhs])
        )
        rhs_path = (
            frame_comparison.path_or_none(self.rhs.all_frames, edge_rhs)
            or frame_comparison.extracted_path(context.rhs[edge_rhs])
        )
        return image_utils.are_images_similar(lhs_path, rhs_path)

    def _is_sustained_dark_zone(
        self,
        context: BoundarySearchContext,
        index: int,
    ) -> bool:
        look_ahead = max(3, int(1.5 * self.lhs.fps))
        for offset in range(1, look_ahead + 1):
            candidate_index = index + context.direction * offset
            if 0 <= candidate_index < len(context.lhs_keys):
                timestamp = context.lhs_keys[candidate_index]
                if frame_comparison.is_rich(
                    frame_comparison.extracted_path(context.lhs[timestamp])
                ):
                    return False
        return True

    def _try_jump_past_dark_zone(
        self,
        context: BoundarySearchContext,
        index: int,
    ) -> tuple[tuple[int, int] | None, int]:
        jump_index = index + context.direction
        while 0 <= jump_index < len(context.lhs_keys):
            timestamp = context.lhs_keys[jump_index]
            if frame_comparison.is_rich(
                frame_comparison.extracted_path(context.lhs[timestamp])
            ):
                return (
                    self._matched_predicted_pair(context, timestamp),
                    jump_index,
                )
            jump_index += context.direction
        return None, jump_index

    def _try_match_edge_frame(
        self,
        context: BoundarySearchContext,
        current_best: tuple[int, int],
    ) -> tuple[int, int] | None:
        edge_index = 0 if context.direction == -1 else len(context.lhs_keys) - 1
        edge_timestamp = context.lhs_keys[edge_index]
        if edge_timestamp == current_best[0] or edge_timestamp not in context.lhs:
            return None
        return self._matched_predicted_pair(context, edge_timestamp, clamp=True)

    def _find_boundary(
        self,
        context: BoundarySearchContext,
        gap_seconds: float,
    ) -> tuple[tuple[int, int], bool]:
        current_best = context.anchor
        max_gap = int(gap_seconds * self.lhs.fps)
        consecutive_misses = 0
        entered_low_entropy = False
        step = max(1, int(self.lhs.fps * 0.5))
        start_index = context.lhs_keys.index(context.anchor[0])
        distance = 0

        while True:
            distance += step
            index = start_index + context.direction * distance
            if index < 0 or index >= len(context.lhs_keys):
                if step > 1:
                    best_index = context.lhs_keys.index(current_best[0])
                    distance = context.direction * (best_index - start_index)
                    step = 1
                    continue
                break

            lhs_timestamp = context.lhs_keys[index]
            if not frame_comparison.is_rich(
                frame_comparison.extracted_path(context.lhs[lhs_timestamp])
            ):
                if self._is_sustained_dark_zone(context, index):
                    jumped_pair, jump_index = self._try_jump_past_dark_zone(
                        context,
                        index,
                    )
                    if jumped_pair is not None:
                        self.logger.debug(
                            "Jumped over dark zone %dms → %dms (pair: %s)",
                            lhs_timestamp,
                            context.lhs_keys[jump_index],
                            jumped_pair,
                        )
                        current_best = jumped_pair
                        consecutive_misses = 0
                        distance = context.direction * (jump_index - start_index)
                        continue
                    entered_low_entropy = True
                    break
                consecutive_misses += step
                if consecutive_misses >= max_gap:
                    break
                continue

            best_pair = self._matched_predicted_pair(context, lhs_timestamp)
            if best_pair is not None:
                current_best = best_pair
                consecutive_misses = 0
            else:
                consecutive_misses += step
                if consecutive_misses >= max_gap:
                    break

        if not entered_low_entropy and consecutive_misses < max_gap:
            edge_match = self._try_match_edge_frame(context, current_best)
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
        side_label: str,
    ) -> bool:
        from_boundary = sorted(gap_keys, reverse=direction == -1)
        low_entropy_start: int | None = None
        for index, timestamp in enumerate(from_boundary):
            if not frame_comparison.is_rich(
                frame_comparison.extracted_path(frames[timestamp])
            ):
                low_entropy_start = index
                break
        if low_entropy_start is None:
            self.logger.warning(
                "Boundary %s: gap in %s is all high-entropy. Cannot extrapolate.",
                label,
                side_label,
            )
            return False
        if low_entropy_start > max_skip:
            self.logger.warning(
                "Boundary %s: %d high-entropy %s frames between boundary and "
                "low-entropy zone (max expected: %d). Cannot extrapolate.",
                label,
                low_entropy_start,
                side_label,
                max_skip,
            )
            return False

        remaining = from_boundary[low_entropy_start:]
        high_entropy = [
            timestamp
            for timestamp in remaining
            if frame_comparison.is_rich(
                frame_comparison.extracted_path(frames[timestamp])
            )
        ]
        noise_ratio = len(high_entropy) / len(remaining) if remaining else 1.0
        if noise_ratio > 0.05:
            self.logger.warning(
                "Boundary %s: %s zone has %d/%d (%.1f%%) high-entropy "
                "frames — not contiguous. Cannot extrapolate.",
                label,
                side_label,
                len(high_entropy),
                len(remaining),
                noise_ratio * 100,
            )
            return False
        if high_entropy:
            self.logger.debug(
                "Boundary %s: tolerating %d/%d (%.1f%%) sparse high-entropy "
                "frames in %s zone.",
                label,
                len(high_entropy),
                len(remaining),
                noise_ratio * 100,
                side_label,
            )
        return True

    @staticmethod
    def _nearest_three(timestamps: list[int], target: int) -> list[int]:
        timestamps = sorted(timestamps)
        index = int(np.searchsorted(timestamps, target))
        return timestamps[max(0, index - 1):index + 2]

    @staticmethod
    def _snap_to_nearest_frame(keys: list[int], target: int) -> int:
        index = int(np.searchsorted(keys, target))
        candidates = []
        if index > 0:
            candidates.append(keys[index - 1])
        if index < len(keys):
            candidates.append(keys[index])
        return min(candidates, key=lambda key: abs(key - target))
