"""Decide whether a melt video mapping may drive track alignment.

``PairMatcher`` reports what it observed about two videos: the matched
pairs, their relation and any discontinuities.  This module compares those
observations with what the user permits and what melt can execute, and
refuses a mapping that must not reach audio or subtitle alignment.  It is the
single place such a refusal is raised.
"""

import logging

from . import debug_routines, matching_analysis
from .matching_analysis import Discontinuity
from .melt_common import format_timestamp
from .pair_matcher import MappingRelation, SegmentsMappingResult

# Smallest accepted ``--max-temporary-frame-offset``: a temporary shift of at
# most this many frames is already within the global-linear step tolerance.
DEFAULT_MAX_TEMPORARY_FRAME_OFFSET = 2.0
# Two transfers of one film never differ in playback speed by more than this
# fraction; a mapping implying more is an implausible match.
MAX_PLAYBACK_SPEED_DELTA = 0.30


def validate_mapping(
    matching: SegmentsMappingResult,
    *,
    consumer: str,
    lhs_id: int,
    rhs_id: int,
    max_temporary_frame_offset: float,
    logger: logging.Logger,
) -> None:
    """Raise ``RuntimeError`` unless *matching* is safe to align *consumer*.

    *consumer* names the track type being aligned (``"audio"`` or
    ``"subtitle"``) in the error.  A temporary frame offset that returns at
    the next matched anchor is accepted up to *max_temporary_frame_offset*
    frames; every other content discontinuity is refused.
    """
    lhs_label, rhs_label = f"#{lhs_id}", f"#{rhs_id}"
    speed_ratio = playback_speed_ratio(matching.mapping)
    if speed_ratio is not None and abs(speed_ratio - 1.0) > MAX_PLAYBACK_SPEED_DELTA:
        raise RuntimeError(
            f"Cannot align {consumer}: the matched video content of {rhs_label} plays "
            f"at {speed_ratio:.3f}x the speed of {lhs_label}, beyond the plausible "
            f"±{MAX_PLAYBACK_SPEED_DELTA:.0%}; the videos were most likely matched incorrectly."
        )

    unresolved = _without_accepted_temporary_offsets(
        matching,
        max_temporary_frame_offset,
        lhs_label=lhs_label,
        rhs_label=rhs_label,
        logger=logger,
    )
    if unresolved:
        _refuse_content_discontinuities(
            matching,
            unresolved,
            max_temporary_frame_offset,
            lhs_label=lhs_label,
            rhs_label=rhs_label,
            logger=logger,
        )

    if matching.timing_discontinuities:
        debug_routines.log_timing_discontinuities(
            logger,
            matching.timing_discontinuities,
            consumer=consumer,
            lhs_id=lhs_id,
            rhs_id=rhs_id,
            debug_path=matching.timing_discontinuity_debug_path,
        )
        raise RuntimeError(
            f"Unsupported {consumer} timeline discontinuity: the matched PTS mapping "
            "has a PTS step despite continuous decoded video"
        )

    if matching.generic_content_discontinuities:
        debug_routines.log_content_discontinuities(
            logger,
            matching.generic_content_discontinuities,
            lhs_id=lhs_id,
            rhs_id=rhs_id,
            rhs_fps=matching.rhs_fps,
            include_frame_deficit=(
                matching.relation is MappingRelation.GLOBAL_LINEAR
            ),
        )
        raise RuntimeError(
            "Unresolved suspected content discontinuities: insufficient evidence to rule out "
            "holes in the scene sequence (e.g. commercial-break cuts) — patching "
            f"{consumer} across such discontinuities is not supported yet"
        )

    if not matching.mapping:
        raise RuntimeError(
            f"Cannot align {consumer}: no shared video timeline mapping was found."
        )


def playback_speed_ratio(mapping: list[tuple[int, int]]) -> float | None:
    """Source-over-base duration of the matched span, or ``None`` if undefined.

    This is the speed change the audio would have to be scaled by, measured
    from matched timestamps rather than declared frame rates.
    """
    if len(mapping) < 2:
        return None
    lhs_timestamps, rhs_timestamps = zip(*mapping)
    lhs_span = max(lhs_timestamps) - min(lhs_timestamps)
    if lhs_span <= 0:
        return None
    return (max(rhs_timestamps) - min(rhs_timestamps)) / lhs_span


def _without_accepted_temporary_offsets(
    matching: SegmentsMappingResult,
    max_temporary_frame_offset: float,
    *,
    lhs_label: str,
    rhs_label: str,
    logger: logging.Logger,
) -> list[Discontinuity]:
    """Drop temporary offset excursions the user tolerates; keep the rest."""
    remaining: list[Discontinuity] = []
    for group in matching_analysis.group_content_discontinuities(
        list(matching.content_discontinuities)
    ):
        excursion = matching_analysis.temporary_frame_offset_excursion(
            group, matching.lhs_all_frames, matching.rhs_all_frames,
        )
        if (
            excursion is None
            or not matching_analysis.MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES
            < excursion[0]
            <= max_temporary_frame_offset
        ):
            remaining.extend(group)
            continue

        temporary_offset, outer_offset = excursion
        _, lhs_middle, _, rhs_middle, _ = group[0]
        logger.warning(
            "Accepting temporary %d-frame alignment offset between "
            "%s frame %d at %s and %s frame %d at %s; the next "
            "matched frames return to the same outer offset of %+d frame(s).",
            temporary_offset,
            lhs_label,
            int(matching.lhs_all_frames[lhs_middle]["frame_id"]),
            format_timestamp(lhs_middle),
            rhs_label,
            int(matching.rhs_all_frames[rhs_middle]["frame_id"]),
            format_timestamp(rhs_middle),
            outer_offset,
        )
    return remaining


def _refuse_content_discontinuities(
    matching: SegmentsMappingResult,
    discontinuities: list[Discontinuity],
    max_temporary_frame_offset: float,
    *,
    lhs_label: str,
    rhs_label: str,
    logger: logging.Logger,
) -> None:
    """Dump and log the unresolved regions, then raise the user-facing error."""
    groups = matching_analysis.group_content_discontinuities(discontinuities)
    suggested_temporary_offsets: list[int] = []
    for group in groups:
        excursion = matching_analysis.temporary_frame_offset_excursion(
            group, matching.lhs_all_frames, matching.rhs_all_frames,
        )
        if excursion is not None and excursion[0] > max_temporary_frame_offset:
            suggested_temporary_offsets.append(excursion[0])
    debug_path = (
        matching.dump_content_discontinuities(discontinuities)
        if matching.dump_content_discontinuities is not None
        else None
    )
    raise RuntimeError(debug_routines.unresolved_content_discontinuities_error(
        logger,
        groups,
        matching.lhs_all_frames,
        matching.rhs_all_frames,
        lhs_label=lhs_label,
        rhs_label=rhs_label,
        rhs_fps=matching.rhs_fps,
        compensation_tolerance_frames=matching_analysis.MAX_GLOBAL_LINEAR_STEP_ERROR_FRAMES,
        suggested_temporary_offsets=suggested_temporary_offsets,
        debug_path=debug_path,
    ))
