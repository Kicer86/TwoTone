"""Pure calculations over matched video timestamps and frame coordinates."""

import logging
from typing import NamedTuple, TypedDict

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

from .melt_common import FramesInfo

MAX_BOUNDARY_ERROR_FRAMES = 2
MAX_RELATIVE_RATIO_ERROR = 0.05
MAX_GAP_TIME_DEFICIT_MS = 2000
MAX_GAP_TIME_DEFICIT_RATIO = 0.02
MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES = 2.0
MAX_LINEAR_RESIDUAL_MS = 1000


class GlobalLinearDiscontinuityReport(NamedTuple):
    """Suspected content cuts and timestamp steps with continuous decode order.

    Content entries remain unresolved until verified against images. Timing
    entries also prohibit audio execution: equal decoded advances do not make
    the source PTS representable by one supported audio transform. Inferred
    frame steps nominate refinement regions but alone prove neither verdict.
    """

    content: list[tuple[int, int, int, int, int]]
    timing: list[tuple[int, int, int, int, int]]
    frame_steps: tuple[tuple[int, int, int, int, int], ...] = ()


class CoverageSummary(TypedDict):
    """Human-readable coverage report for two matched files."""

    full_coverage: bool
    lhs_start_gap_s: float
    rhs_start_gap_s: float
    lhs_end_gap_s: float
    rhs_end_gap_s: float
    ratio: float


class RelationDiagnostics(NamedTuple):
    """Frame-space metrics used to explain a mapping-relation decision."""

    expected_ratio: float
    matched_pairs: int
    coordinates_available: bool
    median_offset: float | None = None
    std_offset: float | None = None
    lhs_span: float | None = None
    observed_ratio: float | None = None
    frame_slope: float | None = None
    time_scale: float | None = None


class DiscontinuityRegionDescription(NamedTuple):
    """Shared calculated facts for one discontinuity diagnostic region."""

    group: tuple[tuple[int, int, int, int, int], ...]
    lhs_range: tuple[int, int]
    rhs_range: tuple[int, int]
    lhs_frame_range: tuple[int, int]
    rhs_frame_range: tuple[int, int]
    step_frame_ranges: tuple[tuple[int, int, int, int], ...]
    frame_errors: tuple[float, ...]
    compensating: bool


def nearest_three(timestamps: list[int], target: int) -> list[int]:
    """Return the timestamp immediately around *target*, plus its neighbors."""
    timestamps = sorted(timestamps)
    index = int(np.searchsorted(timestamps, target))
    return timestamps[max(0, index - 1):index + 2]


def snap_to_nearest_frame(keys: list[int], target: int) -> int:
    """Return the sorted timestamp closest to *target*, preferring the lower tie."""
    index = int(np.searchsorted(keys, target))
    candidates = []
    if index > 0:
        candidates.append(keys[index - 1])
    if index < len(keys):
        candidates.append(keys[index])
    return min(candidates, key=lambda key: abs(key - target))


def describe_discontinuity_regions(
    groups: list[list[tuple[int, int, int, int, int]]],
    lhs_all_frames: FramesInfo,
    rhs_all_frames: FramesInfo,
    *,
    rhs_fps: float,
    compensation_tolerance_frames: float,
) -> tuple[DiscontinuityRegionDescription, ...]:
    """Calculate facts shared by user-facing and debug diagnostics."""
    descriptions = []
    for group in groups:
        lhs_from, _, rhs_from, _, _ = group[0]
        _, lhs_to, _, rhs_to, _ = group[-1]
        step_frame_ranges = tuple(
            (
                int(lhs_all_frames[step_lhs_from]["frame_id"]),
                int(lhs_all_frames[step_lhs_to]["frame_id"]),
                int(rhs_all_frames[step_rhs_from]["frame_id"]),
                int(rhs_all_frames[step_rhs_to]["frame_id"]),
            )
            for step_lhs_from, step_lhs_to, step_rhs_from, step_rhs_to, _ in group
        )
        frame_errors = tuple(
            deficit_ms * rhs_fps / 1000
            for *_, deficit_ms in group
        )
        descriptions.append(DiscontinuityRegionDescription(
            group=tuple(group),
            lhs_range=(lhs_from, lhs_to),
            rhs_range=(rhs_from, rhs_to),
            lhs_frame_range=(step_frame_ranges[0][0], step_frame_ranges[-1][1]),
            rhs_frame_range=(step_frame_ranges[0][2], step_frame_ranges[-1][3]),
            step_frame_ranges=step_frame_ranges,
            frame_errors=frame_errors,
            compensating=(
                len(group) > 1
                and min(frame_errors) < 0 < max(frame_errors)
                and abs(sum(frame_errors)) <= compensation_tolerance_frames
            ),
        ))
    return tuple(descriptions)


def calculate_ratio(pairs: list[tuple[int, int]]) -> float:
    ratios = [
        (right[0] - left[0]) / (right[1] - left[1])
        for left, right in zip(pairs[:-1], pairs[1:])
        if right[1] - left[1] != 0
    ]
    if not ratios:
        return float("nan")
    return float(np.median(ratios))


def timeline_frame_ids(frames: FramesInfo, fps: float) -> dict[int, int]:
    """Map timestamps to decode ordinals augmented with empty timeline slots.

    A nominal FPS can be slightly inaccurate, so using ``timestamp * fps``
    as an absolute coordinate accumulates a false drift across long videos.
    Instead, retain decoder ordinals and infer empty presentation slots from
    local PTS gaps. A slot difference alone cannot establish a timing
    discontinuity; that requires observed timestamp evidence.
    """
    timestamps = sorted(frames)
    if not timestamps:
        return {}

    first_timestamp = timestamps[0]
    coordinate = int(frames[first_timestamp]["frame_id"])
    result = {first_timestamp: coordinate}
    previous_timestamp = first_timestamp
    previous_decode_id = coordinate

    for timestamp in timestamps[1:]:
        decode_id = int(frames[timestamp]["frame_id"])
        decode_advance = max(1, decode_id - previous_decode_id)
        timeline_advance = max(1, round((timestamp - previous_timestamp) * fps / 1000))
        coordinate += max(decode_advance, timeline_advance)
        result[timestamp] = coordinate
        previous_timestamp = timestamp
        previous_decode_id = decode_id

    return result


def is_ratio_acceptable(ratio: float, perfect_ratio: float) -> bool:
    return abs(ratio - perfect_ratio) < MAX_RELATIVE_RATIO_ERROR * perfect_ratio


def _flagged_gap_deficits(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return ``(gap_index, deficit_ms)`` for gaps that defy the median slope."""
    if len(pairs) < 3:
        return []

    slopes = [
        (rhs2 - rhs1) / (lhs2 - lhs1)
        for (lhs1, rhs1), (lhs2, rhs2) in zip(pairs[:-1], pairs[1:])
        if lhs2 > lhs1
    ]
    if not slopes:
        return []
    median_slope = float(np.median(slopes))

    flagged: list[tuple[int, int]] = []
    for index, ((lhs1, rhs1), (lhs2, rhs2)) in enumerate(zip(pairs[:-1], pairs[1:])):
        gap = lhs2 - lhs1
        if gap <= 0:
            continue
        deficit = round(gap * median_slope - (rhs2 - rhs1))
        threshold = max(MAX_GAP_TIME_DEFICIT_MS, MAX_GAP_TIME_DEFICIT_RATIO * gap)
        if abs(deficit) > threshold:
            flagged.append((index, deficit))
    return flagged


def find_content_discontinuities(
    mapping: list[tuple[int, int]],
) -> list[tuple[int, int, int, int, int]]:
    """Locate holes in the shared scene sequence of a matched pair."""
    pairs = sorted(mapping)
    return [
        (pairs[index][0], pairs[index + 1][0], pairs[index][1], pairs[index + 1][1], deficit)
        for index, deficit in _flagged_gap_deficits(pairs)
    ]


def classify_global_linear_discontinuities(
    mapping: list[tuple[int, int]],
    lhs_all_frames: FramesInfo,
    rhs_all_frames: FramesInfo,
    *,
    frame_slope: float,
    lhs_fps: float,
    rhs_fps: float,
) -> GlobalLinearDiscontinuityReport:
    pairs = sorted(mapping)
    if len(pairs) < 2 or frame_slope <= 0 or lhs_fps <= 0 or rhs_fps <= 0:
        return GlobalLinearDiscontinuityReport([], [])

    lhs_ids = timeline_frame_ids(lhs_all_frames, lhs_fps)
    rhs_ids = timeline_frame_ids(rhs_all_frames, rhs_fps)
    try:
        frame_pairs = [(lhs_ids[lhs_ts], rhs_ids[rhs_ts]) for lhs_ts, rhs_ts in pairs]
        decoded_frame_pairs = [
            (
                int(lhs_all_frames[lhs_ts]["frame_id"]),
                int(rhs_all_frames[rhs_ts]["frame_id"]),
            )
            for lhs_ts, rhs_ts in pairs
        ]
    except KeyError:
        return GlobalLinearDiscontinuityReport(find_content_discontinuities(pairs), [])

    decoded_slopes = [
        (rhs2 - rhs1) / (lhs2 - lhs1)
        for (lhs1, rhs1), (lhs2, rhs2) in zip(decoded_frame_pairs, decoded_frame_pairs[1:])
        if lhs2 > lhs1 and rhs2 > rhs1
    ]
    decoded_slope = float(np.median(decoded_slopes)) if decoded_slopes else None
    timestamp_slopes = [
        (rhs2 - rhs1) / (lhs2 - lhs1)
        for (lhs1, rhs1), (lhs2, rhs2) in zip(pairs, pairs[1:])
        if lhs2 > lhs1 and rhs2 > rhs1
    ]
    timestamp_slope = float(np.median(timestamp_slopes)) if timestamp_slopes else None

    content_result: list[tuple[int, int, int, int, int]] = []
    timing_result: list[tuple[int, int, int, int, int]] = []
    frame_steps: list[tuple[int, int, int, int, int]] = []
    for index, (
        (lhs1, rhs1),
        (lhs2, rhs2),
        (lhs_decoded1, rhs_decoded1),
        (lhs_decoded2, rhs_decoded2),
    ) in enumerate(
        zip(
            frame_pairs,
            frame_pairs[1:],
            decoded_frame_pairs,
            decoded_frame_pairs[1:],
        )
    ):
        lhs_advance = lhs2 - lhs1
        rhs_advance = rhs2 - rhs1
        if lhs_advance <= 0 or rhs_advance <= 0:
            continue
        deficit_frames = lhs_advance * frame_slope - rhs_advance
        if abs(deficit_frames) > MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES:
            frame_steps.append((
                pairs[index][0],
                pairs[index + 1][0],
                pairs[index][1],
                pairs[index + 1][1],
                round(deficit_frames * 1000 / rhs_fps),
            ))

        decoded_deficit = None
        if decoded_slope is not None:
            decoded_lhs_advance = lhs_decoded2 - lhs_decoded1
            decoded_rhs_advance = rhs_decoded2 - rhs_decoded1
            if decoded_lhs_advance > 0 and decoded_rhs_advance > 0:
                decoded_deficit = decoded_lhs_advance * decoded_slope - decoded_rhs_advance

        if (
            decoded_deficit is not None
            and abs(decoded_deficit) <= MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES
        ):
            if timestamp_slope is None:
                continue
            lhs_from, rhs_from = pairs[index]
            lhs_to, rhs_to = pairs[index + 1]
            deficit_ms = (lhs_to - lhs_from) * timestamp_slope - (rhs_to - rhs_from)
            deficit_frames = deficit_ms * rhs_fps / 1000
            result = timing_result
        else:
            deficit_ms = deficit_frames * 1000 / rhs_fps
            result = content_result

        if abs(deficit_frames) > MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES:
            result.append((
                pairs[index][0],
                pairs[index + 1][0],
                pairs[index][1],
                pairs[index + 1][1],
                round(deficit_ms),
            ))

    return GlobalLinearDiscontinuityReport(content_result, timing_result, tuple(frame_steps))


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
    report = classify_global_linear_discontinuities(
        mapping,
        lhs_all_frames,
        rhs_all_frames,
        frame_slope=frame_slope,
        lhs_fps=lhs_fps,
        rhs_fps=rhs_fps,
    )
    if require_decoded_confirmation:
        return report.content
    gaps = {gap[:4]: gap for gap in (*report.frame_steps, *report.content, *report.timing)}
    return sorted(gaps.values())


def linear_mapping_slope(pairs: list[tuple[int, int]]) -> float | None:
    """Return the endpoint slope when every pair lies close to that line."""
    if len(pairs) < 2:
        return None

    first_lhs, first_rhs = pairs[0]
    last_lhs, last_rhs = pairs[-1]
    lhs_span = last_lhs - first_lhs
    if lhs_span <= 0:
        return None

    slope = (last_rhs - first_rhs) / lhs_span
    if slope <= 0:
        return None

    for lhs_time, rhs_time in pairs[1:-1]:
        predicted_rhs = first_rhs + (lhs_time - first_lhs) * slope
        if abs(predicted_rhs - rhs_time) > MAX_LINEAR_RESIDUAL_MS:
            return None
    return slope


def coverage_summary(
    mappings: list[tuple[int, int]],
    lhs_duration_ms: int,
    rhs_duration_ms: int,
    *,
    lhs_fps: float,
    rhs_fps: float,
) -> CoverageSummary:
    """Compute the exact unmatched edges and full-coverage verdict."""
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
    lhs_edge_tolerance_ms = MAX_BOUNDARY_ERROR_FRAMES * lhs_frame_duration_ms
    rhs_edge_tolerance_ms = MAX_BOUNDARY_ERROR_FRAMES * rhs_frame_duration_ms

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
        "ratio": calculate_ratio(mappings),
    }


def fit_constant_offset(
    lhs_frame_ids: np.ndarray,
    rhs_frame_ids: np.ndarray,
    matching_pairs: list[tuple[int, int]],
    *,
    lhs_fps: float,
    rhs_fps: float,
    max_std: float,
    min_two_pair_span: int,
    logger: logging.Logger,
) -> tuple[float, float] | None:
    """Fit ``rhs_frame = lhs_frame + intercept`` when the offset is stable."""
    if len(matching_pairs) == 2:
        span = min(int(np.ptp(lhs_frame_ids)), int(np.ptp(rhs_frame_ids)))
        if span < min_two_pair_span:
            logger.debug(
                "Constant-offset check: only two pairs spanning %d frames — skipping",
                span,
            )
            return None

    offsets = lhs_frame_ids - rhs_frame_ids
    std_offset = float(np.std(offsets))
    if std_offset > max_std:
        logger.debug(
            "Constant-offset check: frame-number std=%.2f exceeds %.1f — skipping",
            std_offset,
            max_std,
        )
        return None

    ratio = calculate_ratio(matching_pairs)
    expected_ratio = rhs_fps / lhs_fps
    if not is_ratio_acceptable(ratio, expected_ratio):
        logger.debug(
            "Constant-offset check: ratio=%.4f too far from expected %.4f — skipping",
            ratio,
            expected_ratio,
        )
        return None

    offset = round(float(np.median(offsets)))
    logger.debug(
        "Constant offset detected: %d frame(s) (median=%.1f, std=%.2f).",
        offset,
        float(np.median(offsets)),
        std_offset,
    )
    return (1.0, float(-offset))


def fit_linear_drift(
    lhs_frame_ids: np.ndarray,
    rhs_frame_ids: np.ndarray,
    *,
    lhs_fps: float,
    rhs_fps: float,
    max_slope_delta: float,
    max_median_residual_frames: float,
    max_p95_residual_frames: float,
    max_outlier_ratio: float,
    min_span_frames: int,
    logger: logging.Logger,
) -> tuple[float, float] | None:
    """Fit a near-identity or nominal-FPS drift line with RANSAC."""
    if len(lhs_frame_ids) < 4:
        return None

    lhs_span = float(np.max(lhs_frame_ids) - np.min(lhs_frame_ids))
    if lhs_span < min_span_frames:
        logger.debug(
            "Linear-drift check: matched frame span %.0f is below %d frames — skipping",
            lhs_span,
            min_span_frames,
        )
        return None

    ransac = RANSACRegressor(
        LinearRegression(),
        residual_threshold=max_p95_residual_frames,
        random_state=0,
    )
    ransac.fit(lhs_frame_ids.reshape(-1, 1), rhs_frame_ids)
    inliers = ransac.inlier_mask_
    if inliers is None:
        inliers = np.ones(len(lhs_frame_ids), dtype=bool)

    inlier_count = int(np.sum(inliers))
    if inlier_count < 4:
        logger.debug(
            "Linear-drift check: only %d/%d pairs fit the model — skipping",
            inlier_count,
            len(lhs_frame_ids),
        )
        return None

    outlier_ratio = 1.0 - inlier_count / len(lhs_frame_ids)
    if outlier_ratio > max_outlier_ratio:
        logger.debug(
            "Linear-drift check: outlier ratio %.1f%% exceeds %.1f%% — skipping",
            outlier_ratio * 100,
            max_outlier_ratio * 100,
        )
        return None

    model = LinearRegression()
    model.fit(lhs_frame_ids[inliers].reshape(-1, 1), rhs_frame_ids[inliers])
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    if slope <= 0:
        logger.debug("Linear-drift check: non-positive slope %.6f — skipping", slope)
        return None

    nominal_fps_ratio = rhs_fps / lhs_fps if lhs_fps else 1.0
    slope_delta = min(abs(slope / anchor - 1.0) for anchor in (1.0, nominal_fps_ratio))
    if slope_delta > max_slope_delta:
        logger.debug(
            "Linear-drift check: slope=%.6f differs from both 1.0 and the fps ratio "
            "%.6f by more than %.6f — skipping",
            slope,
            nominal_fps_ratio,
            max_slope_delta,
        )
        return None

    predicted_rhs = model.predict(lhs_frame_ids[inliers].reshape(-1, 1))
    residuals = np.abs(rhs_frame_ids[inliers] - predicted_rhs)
    median_residual = float(np.median(residuals))
    p95_residual = float(np.percentile(residuals, 95))
    if median_residual > max_median_residual_frames or p95_residual > max_p95_residual_frames:
        logger.debug(
            "Linear-drift check: residuals too high (median=%.2f, p95=%.2f) — skipping",
            median_residual,
            p95_residual,
        )
        return None

    return (slope, intercept)


def relation_diagnostics(
    matching_pairs: list[tuple[int, int]],
    lhs_all_frames: FramesInfo,
    rhs_all_frames: FramesInfo,
    *,
    lhs_fps: float,
    rhs_fps: float,
) -> RelationDiagnostics:
    expected_ratio = rhs_fps / lhs_fps if lhs_fps else float("nan")
    if len(matching_pairs) < 2:
        return RelationDiagnostics(expected_ratio, len(matching_pairs), True)

    lhs_coordinates = timeline_frame_ids(lhs_all_frames, lhs_fps)
    rhs_coordinates = timeline_frame_ids(rhs_all_frames, rhs_fps)
    try:
        lhs_frame_ids = np.array(
            [lhs_coordinates[lhs] for lhs, _ in matching_pairs],
            dtype=float,
        )
        rhs_frame_ids = np.array(
            [rhs_coordinates[rhs] for _, rhs in matching_pairs],
            dtype=float,
        )
    except KeyError:
        return RelationDiagnostics(expected_ratio, len(matching_pairs), False)

    frame_offsets = lhs_frame_ids - rhs_frame_ids
    lhs_span = float(np.ptp(lhs_frame_ids))
    if lhs_span > 0:
        slope = float(np.polyfit(lhs_frame_ids, rhs_frame_ids, 1)[0])
    else:
        slope = float("nan")
    time_scale = slope * lhs_fps / rhs_fps if rhs_fps else float("nan")
    return RelationDiagnostics(
        expected_ratio=expected_ratio,
        matched_pairs=len(matching_pairs),
        coordinates_available=True,
        median_offset=float(np.median(frame_offsets)),
        std_offset=float(np.std(frame_offsets)),
        lhs_span=lhs_span,
        observed_ratio=calculate_ratio(matching_pairs),
        frame_slope=slope,
        time_scale=time_scale,
    )


def group_content_discontinuities(
    discontinuities: list[tuple[int, int, int, int, int]],
) -> list[list[tuple[int, int, int, int, int]]]:
    """Group adjacent errors that share their middle matched anchor."""
    groups: list[list[tuple[int, int, int, int, int]]] = []
    for discontinuity in sorted(discontinuities):
        if (
            groups
            and groups[-1][-1][1] == discontinuity[0]
            and groups[-1][-1][3] == discontinuity[2]
        ):
            groups[-1].append(discontinuity)
        else:
            groups.append([discontinuity])
    return groups


def strict_mapping(mapping: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse duplicate or non-monotonic matches before retiming tracks."""
    if len(mapping) < 2:
        return mapping

    result: list[tuple[int, int]] = []
    for lhs_time, rhs_time in sorted(mapping):
        if result and rhs_time == result[-1][1]:
            result[-1] = (lhs_time, rhs_time)
            continue
        if result and (lhs_time <= result[-1][0] or rhs_time < result[-1][1]):
            continue
        result.append((lhs_time, rhs_time))

    return result if result else mapping
