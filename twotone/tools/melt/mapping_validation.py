"""Decide whether a melt video mapping may drive track alignment.

``PairMatcher`` reports what it observed about two videos: the matched
pairs, their relation and any discontinuities.  This module compares those
observations with what melt can execute and refuses a mapping that must not
reach audio or subtitle alignment.  It is the single place such a refusal is
raised.
"""

import logging

from . import debug_routines
from .pair_matcher import MappingRelation, SegmentsMappingResult


def validate_mapping(
    matching: SegmentsMappingResult,
    *,
    consumer: str,
    lhs_id: int,
    rhs_id: int,
    logger: logging.Logger,
) -> None:
    """Raise ``RuntimeError`` unless *matching* is safe to align *consumer*.

    *consumer* names the track type being aligned (``"audio"`` or
    ``"subtitle"``) in the error.
    """
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

    discontinuities = (
        matching.content_discontinuities
        or matching.generic_content_discontinuities
    )
    if discontinuities:
        debug_routines.log_content_discontinuities(
            logger,
            discontinuities,
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
