
import logging
import os
import shutil

from collections.abc import Callable
from typing import Literal

import numpy as np

from ..utils import generic_utils
from . import matching_analysis
from .matching_analysis import CoverageSummary
from .melt_common import FramesInfo, format_timestamp_range
from .phash_cache import PhashCache


Discontinuity = tuple[int, int, int, int, int]
DiscontinuityGroup = list[Discontinuity]


def _frame_path(frames: FramesInfo, timestamp: int) -> str | None:
    info = frames.get(timestamp)
    if info is None:
        return None
    return info.get("path")


def summarize_pairs(
    phash: PhashCache,
    pairs: list[tuple[int, int]],
    lhs: FramesInfo,
    rhs: FramesInfo,
    verbose: bool = False,
) -> str:
    if not pairs:
        return "Pairs: 0"

    distances = []
    for lhs_ts, rhs_ts in pairs:
        lhs_path = _frame_path(lhs, lhs_ts)
        rhs_path = _frame_path(rhs, rhs_ts)
        if lhs_path is None or rhs_path is None:
            continue
        distance = abs(phash.get(lhs_path) - phash.get(rhs_path))
        distances.append((distance, lhs_ts, rhs_ts))

    if not distances:
        return f"Pairs: {len(pairs)} (no extracted paths for summary)"

    values = np.array([distance for distance, _, _ in distances])
    max_entry = max(distances, key=lambda entry: entry[0])
    max_lhs_path = _frame_path(lhs, max_entry[1]) or "?"
    max_rhs_path = _frame_path(rhs, max_entry[2]) or "?"
    summary = (
        f"Pairs: {len(pairs)} | "
        f"Median: {np.median(values):.2f} | "
        f"Mean: {np.mean(values):.2f} | "
        f"Std Dev: {np.std(values):.2f} | "
        f"Min: {np.min(values)} | "
        f"Max: {np.max(values)} | "
        f"Max Pair: {max_lhs_path} <-> {max_rhs_path}"
    )

    if verbose:
        details = []
        for distance, lhs_ts, rhs_ts in distances:
            lhs_path = _frame_path(lhs, lhs_ts) or "?"
            rhs_path = _frame_path(rhs, rhs_ts) or "?"
            details.append(f"  {lhs_path} <-> {rhs_path} | Diff: {distance}")
        summary += "\nDetailed pairs:\n" + "\n".join(details)

    return summary


def log_pairs_summary(
    logger: logging.Logger,
    phash: PhashCache,
    lhs: FramesInfo,
    rhs: FramesInfo,
    prefix: str,
    pairs: list[tuple[int, int]],
    *,
    verbose: bool = False,
) -> None:
    """Log a pair summary without calculating image hashes above DEBUG."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "%s%s",
            prefix,
            summarize_pairs(phash, pairs, lhs, rhs, verbose=verbose),
        )


def summarize_segments(
    pairs: list[tuple[int, int]],
    lhs_fps: float,
    rhs_fps: float,
    verbose: bool = True,
    lhs_label: str = "#1",
    rhs_label: str = "#2",
) -> str:
    if len(pairs) < 2:
        return "Not enough pairs to build segments."

    pairs_sorted = sorted(pairs)
    segments = []
    max_quant_error = max(1000.0 / lhs_fps, 1000.0 / rhs_fps)
    for (lhs1, rhs1), (lhs2, rhs2) in zip(pairs_sorted[:-1], pairs_sorted[1:]):
        lhs_delta = lhs2 - lhs1
        rhs_delta = rhs2 - rhs1
        if rhs_delta <= 0:
            continue
        ratio = lhs_delta / rhs_delta
        min_delta = min(lhs_delta, rhs_delta)
        ratio_error = (2 * max_quant_error) / min_delta if min_delta > 0 else float("inf")
        confidence = "LOW" if ratio_error > 0.1 else "OK"
        segments.append((
            lhs1,
            lhs2,
            rhs1,
            rhs2,
            lhs_delta,
            rhs_delta,
            ratio,
            ratio_error,
            confidence,
        ))

    if not segments:
        return "No valid segments."

    ratios = np.array([segment[6] for segment in segments])
    output = [
        f"Segments: {len(segments)} | "
        f"Median ratio: {np.median(ratios):.4f} | "
        f"Mean ratio: {np.mean(ratios):.4f} | "
        f"Std Dev: {np.std(ratios):.4f} | "
        f"Min: {np.min(ratios):.4f} | Max: {np.max(ratios):.4f}"
    ]

    if verbose:
        output.append("\nDetailed segments:")
        for lhs1, lhs2, rhs1, rhs2, lhs_delta, rhs_delta, ratio, error, confidence in segments:
            output.append(
                f"  {lhs_label} {lhs1}->{lhs2} ({lhs_delta:4} ms), "
                f"{rhs_label} {rhs1}->{rhs2} ({rhs_delta:4} ms), "
                f"Ratio: {ratio:.4f}, "
                f"Error~{error:.2%}, Confidence: {confidence}"
            )

    return "\n".join(output)


def format_time(seconds: float) -> str:
    """Format seconds as a compact human-readable time string."""
    if seconds < 0:
        return f"-{format_time(-seconds)}"
    hours = int(seconds // 3600)
    remainder = seconds - hours * 3600
    minutes = int(remainder // 60)
    seconds = remainder - minutes * 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:04.1f}"
    if minutes > 0:
        return f"{minutes}:{seconds:04.1f}"
    return f"{seconds:.1f}s"


def render_overlap_diagram(
    lhs_id: int,
    rhs_id: int,
    lhs_duration_ms: int,
    rhs_duration_ms: int,
    summary: CoverageSummary,
) -> list[str]:
    """Build an ASCII diagram of two files projected onto the lhs timeline."""
    lhs_duration = lhs_duration_ms / 1000
    rhs_duration = rhs_duration_ms / 1000
    lhs_start_gap = summary["lhs_start_gap_s"]
    rhs_start_gap = summary["rhs_start_gap_s"]
    lhs_end_gap = summary["lhs_end_gap_s"]
    rhs_end_gap = summary["rhs_end_gap_s"]

    shared_lhs_duration = lhs_duration - lhs_start_gap - lhs_end_gap
    shared_rhs_duration = rhs_duration - rhs_start_gap - rhs_end_gap
    speed = shared_lhs_duration / shared_rhs_duration if shared_rhs_duration > 0 else 1.0

    lhs_start, lhs_end = 0.0, lhs_duration
    rhs_start = lhs_start_gap - rhs_start_gap * speed
    rhs_end = (lhs_duration - lhs_end_gap) + rhs_end_gap * speed
    timeline_start = min(lhs_start, rhs_start)
    timeline_end = max(lhs_end, rhs_end)
    if timeline_end - timeline_start <= 0:
        return []

    width = 70
    minimum_gap_width = 6
    has_left_gap = abs(lhs_start - rhs_start) > 0.0
    has_right_gap = abs(lhs_end - rhs_end) > 0.0
    left_width = minimum_gap_width if has_left_gap else 0
    right_width = minimum_gap_width if has_right_gap else 0
    middle_width = width - left_width - right_width

    lhs_column_start = 0 if lhs_start <= rhs_start or not has_left_gap else left_width
    rhs_column_start = 0 if rhs_start <= lhs_start or not has_left_gap else left_width
    lhs_column_end = width if lhs_end >= rhs_end or not has_right_gap else left_width + middle_width
    rhs_column_end = width if rhs_end >= lhs_end or not has_right_gap else left_width + middle_width

    def make_bar(column_start: int, column_end: int, label: str) -> str:
        bar_width = max(column_end - column_start, len(label) + 2)
        inner_width = bar_width - 2
        left_padding = (inner_width - len(label)) // 2
        right_padding = inner_width - len(label) - left_padding
        return " " * column_start + "|" + "_" * left_padding + label + "_" * right_padding + "|"

    lines = [
        make_bar(lhs_column_start, lhs_column_end, f"#{lhs_id}"),
        make_bar(rhs_column_start, rhs_column_end, f"#{rhs_id}"),
    ]
    edge_list = [
        (lhs_column_start, lhs_start - timeline_start),
        (rhs_column_start, rhs_start - timeline_start),
        (lhs_column_end, lhs_end - timeline_start),
        (rhs_column_end, rhs_end - timeline_start),
    ]
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, str]] = []
    for column, timestamp in sorted(edge_list):
        key = (column, round(timestamp * 10))
        if key not in seen:
            seen.add(key)
            edges.append((column, format_time(timestamp)))

    rows = [[" "] * (width + 20), [" "] * (width + 20)]
    row_end = [-1, -1]
    for column, label in edges:
        for row_index in range(2):
            if column >= row_end[row_index] and column + len(label) <= len(rows[row_index]):
                for offset, character in enumerate(label):
                    rows[row_index][column + offset] = character
                row_end[row_index] = column + len(label) + 1
                break

    for row in rows:
        label_row = "".join(row).rstrip()
        if label_row:
            lines.append(label_row)
    if abs(speed - 1.0) >= 0.001:
        lines.append(
            f"(#{rhs_id} plays at a different speed; its bar and times are projected "
            f"onto #{lhs_id}'s timeline — {format_time(rhs_duration)} of #{rhs_id} "
            f"spans {format_time(rhs_end - rhs_start)} there)"
        )
    return lines


def log_coverage(
    logger: logging.Logger,
    mappings: list[tuple[int, int]],
    lhs_duration_ms: int,
    rhs_duration_ms: int,
    lhs_fps: float,
    rhs_fps: float,
    lhs_id: int,
    rhs_id: int,
) -> None:
    """Log the exact coverage report for one compared file pair."""
    summary = matching_analysis.coverage_summary(
        mappings,
        lhs_duration_ms,
        rhs_duration_ms,
        lhs_fps=lhs_fps,
        rhs_fps=rhs_fps,
    )
    ratio = summary["ratio"]
    same_speed = abs(ratio - 1.0) < 0.001
    same_length = lhs_duration_ms == rhs_duration_ms

    if same_length and same_speed and summary["full_coverage"]:
        logger.info("Files are 100%% visually identical: #%d ↔ #%d", lhs_id, rhs_id)
        return

    lhs_length = generic_utils.ms_to_time(lhs_duration_ms)
    rhs_length = generic_utils.ms_to_time(rhs_duration_ms)
    if summary["full_coverage"]:
        logger.info(
            "Files share all content but play at different speeds "
            "(same frames, different frame rate; speed ratio %.5f): "
            "#%d (%s) ↔ #%d (%s)",
            ratio,
            lhs_id,
            lhs_length,
            rhs_id,
            rhs_length,
        )
    else:
        logger.info(
            "Files are NOT fully identical: #%d (%s) ↔ #%d (%s)",
            lhs_id,
            lhs_length,
            rhs_id,
            rhs_length,
        )
        if not same_speed:
            logger.info("  speed ratio: %.5f", ratio)
    logger.info(
        "  shared content starts at %.3f s in #%d and at %.3f s in #%d",
        summary["lhs_start_gap_s"],
        lhs_id,
        summary["rhs_start_gap_s"],
        rhs_id,
    )
    logger.info(
        "  shared content ends %.3f s before end of #%d and %.3f s before end of #%d",
        summary["lhs_end_gap_s"],
        lhs_id,
        summary["rhs_end_gap_s"],
        rhs_id,
    )

    diagram = render_overlap_diagram(
        lhs_id,
        rhs_id,
        lhs_duration_ms,
        rhs_duration_ms,
        summary,
    )
    if diagram:
        logger.info("Overlap diagram:\n%s", "\n".join(diagram))


def log_relation_diagnostics(
    logger: logging.Logger,
    matching_pairs: list[tuple[int, int]],
    lhs_all_frames: FramesInfo,
    rhs_all_frames: FramesInfo,
    *,
    lhs_label: str,
    rhs_label: str,
    lhs_fps: float,
    rhs_fps: float,
    max_constant_offset_std: float,
    max_drift_slope_delta: float,
) -> None:
    """Log the frame-space metrics that explain the relation choice."""
    diagnostics = matching_analysis.relation_diagnostics(
        matching_pairs,
        lhs_all_frames,
        rhs_all_frames,
        lhs_fps=lhs_fps,
        rhs_fps=rhs_fps,
    )
    logger.debug(
        "Relation diagnostics: %s fps=%.4f, %s fps=%.4f, "
        "expected lhs/rhs time ratio=%.4f, matched pairs=%d",
        lhs_label,
        lhs_fps,
        rhs_label,
        rhs_fps,
        diagnostics.expected_ratio,
        diagnostics.matched_pairs,
    )
    if diagnostics.matched_pairs < 2:
        logger.debug("  too few matched pairs for offset/drift analysis")
        return
    if not diagnostics.coordinates_available:
        logger.debug("  matched frames lack frame coordinates; skipping offset/drift analysis")
        return

    assert diagnostics.median_offset is not None
    assert diagnostics.std_offset is not None
    assert diagnostics.lhs_span is not None
    assert diagnostics.observed_ratio is not None
    assert diagnostics.frame_slope is not None
    assert diagnostics.time_scale is not None
    logger.debug(
        "  frame-offset median=%.2f std=%.2f (constant offset needs std<=%.1f, "
        "else RANSAC drift with slope within %.2f of 1.0 or of the fps ratio); "
        "matched frame span=%.0f; observed time ratio=%.4f; "
        "frame slope=%.5f -> audio time scale=%.5f (audio is globally "
        "time-scaled by this factor)",
        diagnostics.median_offset,
        diagnostics.std_offset,
        max_constant_offset_std,
        max_drift_slope_delta,
        diagnostics.lhs_span,
        diagnostics.observed_ratio,
        diagnostics.frame_slope,
        diagnostics.time_scale,
    )


def log_global_linear_relation(
    logger: logging.Logger,
    *,
    lhs_label: str,
    rhs_label: str,
    slope: float,
    intercept: float,
    time_scale: float,
    is_constant_offset: bool,
) -> None:
    """Log a detected global-linear fit without deciding its boundaries."""
    if is_constant_offset:
        offset = int(round(-intercept))
        if offset > 0:
            relation_text = (
                f"a small constant frame offset of {offset} frame(s); "
                f"{lhs_label} starts the shared content later than {rhs_label}"
            )
        elif offset < 0:
            relation_text = (
                f"a small constant frame offset of {-offset} frame(s); "
                f"{rhs_label} starts the shared content later than {lhs_label}"
            )
        else:
            relation_text = "no frame offset"
    else:
        relation_text = (
            f"a linear frame drift (slope={slope:.6f}, intercept={intercept:+.1f})"
        )

    if abs(time_scale - 1.0) > 0.005:
        relation_text += f", audio time-scaled by {time_scale:.5f}"

    logger.info(
        "Files %s and %s share content with %s.",
        lhs_label,
        rhs_label,
        relation_text,
    )


def unresolved_content_discontinuities_error(
    logger: logging.Logger,
    groups: list[DiscontinuityGroup],
    lhs_all_frames: FramesInfo,
    rhs_all_frames: FramesInfo,
    *,
    lhs_label: str,
    rhs_label: str,
    rhs_fps: float,
    compensation_tolerance_frames: float,
    suggested_temporary_offsets: list[int],
    debug_path: str | None,
) -> str:
    """Log unresolved regions and build the corresponding user-facing error."""
    descriptions = matching_analysis.describe_discontinuity_regions(
        groups,
        lhs_all_frames,
        rhs_all_frames,
        rhs_fps=rhs_fps,
        compensation_tolerance_frames=compensation_tolerance_frames,
    )
    for description in descriptions:
        group = description.group
        lhs_from, lhs_to = description.lhs_range
        rhs_from, rhs_to = description.rhs_range
        lhs_frame_from, lhs_frame_to = description.lhs_frame_range
        rhs_frame_from, rhs_frame_to = description.rhs_frame_range
        outer_start_offset = lhs_frame_from - rhs_frame_from
        outer_end_offset = lhs_frame_to - rhs_frame_to
        if description.compensating and outer_start_offset == outer_end_offset:
            anchor_offsets = [outer_start_offset] + [
                lhs_frame_to - rhs_frame_to
                for _, lhs_frame_to, _, rhs_frame_to in description.step_frame_ranges
            ]
            inner_shift = max(
                abs(offset - outer_start_offset)
                for offset in anchor_offsets[1:-1]
            )
            if outer_start_offset > 0:
                offset_relation = (
                    f" ({lhs_label} = {rhs_label} + {outer_start_offset})"
                )
            elif outer_start_offset < 0:
                offset_relation = (
                    f" ({lhs_label} = {rhs_label} - {abs(outer_start_offset)})"
                )
            else:
                offset_relation = ""
            logger.error(
                "Local frame-sequence difference: the first and last matched "
                "frames use the same %d-frame offset%s, but inside this range "
                "the match shifts by %d frame%s.\n"
                "  %s: frames %d-%d; %s\n"
                "  %s: frames %d-%d; %s",
                abs(outer_start_offset),
                offset_relation,
                inner_shift,
                "" if inner_shift == 1 else "s",
                lhs_label,
                lhs_frame_from,
                lhs_frame_to,
                format_timestamp_range(lhs_from, lhs_to),
                rhs_label,
                rhs_frame_from,
                rhs_frame_to,
                format_timestamp_range(rhs_from, rhs_to),
            )
        else:
            logger.error(
                "Local frame-sequence difference: matched-frame alignment "
                "changes by up to %.1f frames inside this range.\n"
                "  %s: frames %d-%d; %s\n"
                "  %s: frames %d-%d; %s",
                max(abs(error) for error in description.frame_errors),
                lhs_label,
                lhs_frame_from,
                lhs_frame_to,
                format_timestamp_range(lhs_from, lhs_to),
                rhs_label,
                rhs_frame_from,
                rhs_frame_to,
                format_timestamp_range(rhs_from, rhs_to),
            )

        for step, frame_range, frame_error in zip(
            group,
            description.step_frame_ranges,
            description.frame_errors,
        ):
            lhs_from, lhs_to, rhs_from, rhs_to, deficit_ms = step
            lhs_frame_from, lhs_frame_to, rhs_frame_from, rhs_frame_to = frame_range
            logger.debug(
                "Suspected content discontinuity between matched anchors: %s "
                "(decoded frames %d-%d) in %s ↔ %s "
                "(decoded frames %d-%d) in %s (%+d ms, %+.1f frames)",
                format_timestamp_range(lhs_from, lhs_to),
                lhs_frame_from,
                lhs_frame_to,
                lhs_label,
                format_timestamp_range(rhs_from, rhs_to),
                rhs_frame_from,
                rhs_frame_to,
                rhs_label,
                deficit_ms,
                frame_error,
            )

    if debug_path is not None:
        details_path = os.path.join(debug_path, "details.txt")
        logger.error("Full frame diagnostics: %s", details_path)
        investigation_target = details_path
        report_material = debug_path
    else:
        investigation_target = "the frame ranges logged above"
        report_material = "the complete melt log"

    region_count = len(groups)
    region_word = "region" if region_count == 1 else "regions"
    investigation_scope = (
        "the reported region"
        if region_count == 1
        else f"all {region_count} reported regions"
    )
    error = (
        "Matched videos contain a local frame-sequence difference in "
        f"{region_count} unresolved {region_word}. Inspect {investigation_scope} "
        f"in {investigation_target}."
    )
    if suggested_temporary_offsets:
        error += (
            " If the frames confirm that each listed compensating temporary "
            "shift is a real, acceptable difference between the videos, rerun with "
            f"--max-temporary-frame-offset {max(suggested_temporary_offsets)}."
        )
        if len(suggested_temporary_offsets) < region_count:
            error += (
                " This option applies only to those compensating shifts; it "
                "will not suppress the other reported regions."
            )
    error += (
        " If the extracted frames do not confirm the reported sequence "
        "difference, report a PairMatcher bug and include "
        f"{report_material}."
    )
    return error


def log_timing_discontinuities(
    logger: logging.Logger,
    discontinuities: tuple[Discontinuity, ...],
    *,
    consumer: str,
    lhs_id: int,
    rhs_id: int,
    debug_path: str | None,
) -> None:
    for lhs_from, lhs_to, rhs_from, rhs_to, deficit in discontinuities:
        logger.error(
            "  Unsupported %s timeline discontinuity: %s in #%d ↔ %s in #%d "
            "(%+d ms)",
            consumer,
            format_timestamp_range(lhs_from, lhs_to),
            lhs_id,
            format_timestamp_range(rhs_from, rhs_to),
            rhs_id,
            deficit,
        )
    if debug_path is not None:
        logger.error(
            "Full frame diagnostics: %s",
            os.path.join(debug_path, "details.txt"),
        )


def log_content_discontinuities(
    logger: logging.Logger,
    discontinuities: tuple[Discontinuity, ...],
    *,
    lhs_id: int,
    rhs_id: int,
    rhs_fps: float,
    include_frame_deficit: bool,
) -> None:
    for lhs_from, lhs_to, rhs_from, rhs_to, deficit in discontinuities:
        suffix = (
            f", {deficit * rhs_fps / 1000:+.1f} frames"
            if include_frame_deficit
            else ""
        )
        logger.error(
            "  Suspected content discontinuity: %s in #%d ↔ %s in #%d "
            "(%+d ms%s)",
            format_timestamp_range(lhs_from, lhs_to),
            lhs_id,
            format_timestamp_range(rhs_from, rhs_to),
            rhs_id,
            deficit,
            suffix,
        )


class DebugRoutines:
    def __init__(self, debug_dir: str, lhs_all_frames: FramesInfo, rhs_all_frames: FramesInfo) -> None:
        self.it = 0
        self.debug_dir = debug_dir
        self.lhs_all_frames = lhs_all_frames
        self.rhs_all_frames = rhs_all_frames

    def dump_frames(self, matches: FramesInfo, phase: str) -> None:
        target_dir = os.path.join(self.debug_dir, f"#{self.it} {phase}")
        self.it += 1

        os.makedirs(target_dir)

        for i, (ts, info) in enumerate(matches.items()):
            path = info.get("path")
            if path:
                self._link_file(
                    path,
                    os.path.join(target_dir, f"{i:06d}_lhs_{ts:08d}"),
                )

    def dump_matches(self, matches: list[tuple[int, int]], phase: str) -> None:
        target_dir = os.path.join(self.debug_dir, f"#{self.it} {phase}")
        self.it += 1

        os.makedirs(target_dir)

        for i, (lhs_ts, rhs_ts) in enumerate(matches):
            lhs_info = self.lhs_all_frames.get(lhs_ts)
            rhs_info = self.rhs_all_frames.get(rhs_ts)
            lhs_path = lhs_info.get("path") if lhs_info else None
            rhs_path = rhs_info.get("path") if rhs_info else None
            if lhs_path:
                self._link_file(
                    lhs_path,
                    os.path.join(target_dir, f"{i:06d}_lhs_{lhs_ts:08d}"),
                )
            if rhs_path:
                self._link_file(
                    rhs_path,
                    os.path.join(target_dir, f"{i:06d}_rhs_{rhs_ts:08d}"),
                )

    def dump_pairs(self, matches: list[tuple[int, int, int, int]]) -> None:
        target_dir = os.path.join(self.debug_dir, f"#{self.it} subsegments")
        self.it += 1

        os.makedirs(target_dir)

        for i, (lhs_ts_b, lhs_ts_e, rhs_ts_b, rhs_ts_e) in enumerate(matches):
            for ts, side, tag in [
                (lhs_ts_b, self.lhs_all_frames, f"lhs_b_{lhs_ts_b:08d}"),
                (lhs_ts_e, self.lhs_all_frames, f"lhs_e_{lhs_ts_e:08d}"),
                (rhs_ts_b, self.rhs_all_frames, f"rhs_b_{rhs_ts_b:08d}"),
                (rhs_ts_e, self.rhs_all_frames, f"rhs_e_{rhs_ts_e:08d}"),
            ]:
                info = side.get(ts)
                path = info.get("path") if info else None
                if path:
                    self._link_file(
                        path,
                        os.path.join(target_dir, f"{i:06d}_{tag}"),
                    )

    def dump_content_discontinuities(
        self,
        groups: list[DiscontinuityGroup],
        *,
        lhs_label: str,
        rhs_label: str,
        lhs_fps: float,
        rhs_fps: float,
        prepare_frames: Callable[[], None],
        compensation_tolerance_frames: float = 4.0,
    ) -> str | None:
        """Write final error details and complete frame spans for every region."""
        return self._dump_discontinuities(
            groups,
            lhs_label=lhs_label,
            rhs_label=rhs_label,
            lhs_fps=lhs_fps,
            rhs_fps=rhs_fps,
            prepare_frames=prepare_frames,
            diagnostic_type="content",
            compensation_tolerance_frames=compensation_tolerance_frames,
        )

    def dump_timing_discontinuities(
        self,
        groups: list[DiscontinuityGroup],
        *,
        lhs_label: str,
        rhs_label: str,
        lhs_fps: float,
        rhs_fps: float,
        prepare_frames: Callable[[], None],
    ) -> str | None:
        """Write complete frame spans for unsupported matched-timeline steps."""
        return self._dump_discontinuities(
            groups,
            lhs_label=lhs_label,
            rhs_label=rhs_label,
            lhs_fps=lhs_fps,
            rhs_fps=rhs_fps,
            prepare_frames=prepare_frames,
            diagnostic_type="timing",
        )

    def _dump_discontinuities(
        self,
        groups: list[DiscontinuityGroup],
        *,
        lhs_label: str,
        rhs_label: str,
        lhs_fps: float,
        rhs_fps: float,
        prepare_frames: Callable[[], None],
        diagnostic_type: Literal["content", "timing"],
        compensation_tolerance_frames: float = 4.0,
    ) -> str | None:
        prepare_frames()
        target_dir = os.path.join(
            self.debug_dir,
            f"#{self.it} {diagnostic_type} discontinuity error",
        )
        self.it += 1
        os.makedirs(target_dir)

        if diagnostic_type == "content":
            details = [
                "Content discontinuity diagnostics",
                "",
                "Every decoded frame in each inclusive region is linked below.",
                "A compensating region restores the outer alignment but still means",
                "that the videos contain different local frame sequences.",
            ]
        else:
            details = [
                "Timing discontinuity diagnostics",
                "",
                "Every decoded frame in each inclusive region is linked below.",
                "These matched regions change the observed timeline relation and",
                "cannot be represented by one supported audio transform.",
            ]
        details.extend([
            f"{lhs_label} nominal frame rate: {lhs_fps:.6f} fps",
            f"{rhs_label} nominal frame rate: {rhs_fps:.6f} fps",
            "",
        ])
        descriptions = matching_analysis.describe_discontinuity_regions(
            groups,
            self.lhs_all_frames,
            self.rhs_all_frames,
            rhs_fps=rhs_fps,
            compensation_tolerance_frames=compensation_tolerance_frames,
        )
        for region_index, description in enumerate(descriptions):
            lhs_from, lhs_to = description.lhs_range
            rhs_from, rhs_to = description.rhs_range
            if diagnostic_type == "timing":
                kind = "unresolved matched-timeline step"
            else:
                kind = (
                    "compensating adjacent frame-step errors"
                    if description.compensating
                    else "unresolved frame-step error"
                )
            lhs_first_id, lhs_last_id = description.lhs_frame_range
            rhs_first_id, rhs_last_id = description.rhs_frame_range

            details.extend([
                f"Region {region_index}: {kind}",
                (
                    f"  {lhs_label}: {format_timestamp_range(lhs_from, lhs_to)}; "
                    "decoded frames "
                    f"{lhs_first_id} -> {lhs_last_id}"
                ),
                (
                    f"  {rhs_label}: {format_timestamp_range(rhs_from, rhs_to)}; "
                    "decoded frames "
                    f"{rhs_first_id} -> {rhs_last_id}"
                ),
                "  Steps:",
            ])
            for step_index, (step, frame_range, frame_error) in enumerate(zip(
                description.group,
                description.step_frame_ranges,
                description.frame_errors,
            )):
                step_lhs_from, step_lhs_to, step_rhs_from, step_rhs_to, deficit_ms = step
                step_lhs_first, step_lhs_last, step_rhs_first, step_rhs_last = frame_range
                details.append(
                    f"    {step_index + 1}: {lhs_label} {step_lhs_first}->{step_lhs_last} "
                    f"at {format_timestamp_range(step_lhs_from, step_lhs_to)} <-> "
                    f"{rhs_label} {step_rhs_first}->{step_rhs_last} "
                    f"at {format_timestamp_range(step_rhs_from, step_rhs_to)}; "
                    f"error {deficit_ms:+d} ms "
                    f"({frame_error:+.1f} rhs frames)"
                )
            details.append("")

            region_dir = os.path.join(target_dir, f"region_{region_index:03d}")
            self._link_frame_span(
                os.path.join(region_dir, "lhs"),
                self.lhs_all_frames,
                lhs_from,
                lhs_to,
            )
            self._link_frame_span(
                os.path.join(region_dir, "rhs"),
                self.rhs_all_frames,
                rhs_from,
                rhs_to,
            )

        with open(os.path.join(target_dir, "details.txt"), "w", encoding="utf-8") as details_file:
            details_file.write("\n".join(details))
        return target_dir

    @classmethod
    def _link_frame_span(
        cls,
        target_dir: str,
        frames: FramesInfo,
        start_ms: int,
        end_ms: int,
    ) -> None:
        os.makedirs(target_dir)
        for timestamp, info in sorted(frames.items()):
            if not start_ms <= timestamp <= end_ms:
                continue
            path = info.get("path")
            if path is None:
                continue
            extension = os.path.splitext(path)[1]
            name = (
                f"frame_{int(info['frame_id']):06d}_"
                f"{timestamp:010d}{extension}"
            )
            cls._link_file(path, os.path.join(target_dir, name))

    @staticmethod
    def _link_file(source: str, target: str) -> None:
        """Link a debug frame, with portable fallbacks for Windows hosts."""
        try:
            os.symlink(source, target)
        except (NotImplementedError, OSError):
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
