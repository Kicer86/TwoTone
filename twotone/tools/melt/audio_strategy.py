"""Audio strategy selection for melt timeline mappings.

Pure decisions only: given a validated ``SegmentsMappingResult`` and the
probed facts it cannot carry itself (the effective-fps tempo ratio and the
mapping-to-stream bias), choose how the source audio is fitted to the base
video and build the mapping that strategy is executed with.  Probing stays
with the caller; execution lives in ``audio_patcher``.
"""

import math
import statistics
from dataclasses import dataclass

from . import matching_analysis
from .audio_patcher import AudioStrategy, segment_range
from .melt_common import FrameSpan
from .pair_matcher import MappingRelation, SegmentsMappingResult

FPS_RATIO_TOLERANCE = 0.001
# Effective-fps ratios closer to 1.0 than this are treated as the same
# playback speed (probe noise), not a real tempo difference.
MIN_TEMPO_RATIO_DELTA = 0.005
# A mapping counts as pure frame-rate drift only when the matched spans
# agree in wall-clock duration within this fraction.
MAX_SAME_SPEED_SPAN_DELTA = 0.01
# An effective-fps tempo ratio may replace the matched pairs' time mapping
# only when both agree within this fraction; a larger disagreement means
# the files are not frame-for-frame and the fps ratio says nothing about
# the time mapping.
MAX_FPS_VS_OBSERVED_RATIO_ERROR = 0.05
# Collapsing a linear mapping to one global time-scale requires the
# matched span to cover at least this fraction of the base video;
# projecting a global tempo from a small matched window is guesswork.
MIN_LINEAR_REBUILD_COVERAGE = 0.80
# Number of subsegments the per-scene audio patcher splits the mapped
# range into.
SUBSEGMENT_COUNT = 20


@dataclass(frozen=True)
class AudioPlan:
    """The chosen strategy and everything it is executed with."""

    strategy: AudioStrategy
    mapping: list[tuple[int, int]]
    unscaled_shift_ms: int = 0
    collapsed_linear_mapping: bool = False

    @property
    def segment_count(self) -> int:
        return SUBSEGMENT_COUNT if self.strategy == AudioStrategy.SUBSEGMENT else 1


def plan_audio(
    matching: SegmentsMappingResult,
    *,
    tempo_ratio: float | None,
    stream_bias_ms: int,
) -> AudioPlan:
    """Choose the strategy and build its mapping and placement shift.

    *tempo_ratio* is the base-over-source effective-fps ratio (``None`` when
    unknown); *stream_bias_ms* is the base-minus-source mapping-to-stream
    bias used to place an unscaled track (see ``unscaled_timeline_shift_ms``).
    """
    strategy = choose_strategy(matching, tempo_ratio)
    if strategy == AudioStrategy.UNSCALED:
        return AudioPlan(
            strategy,
            matching.mapping,
            unscaled_shift_ms=unscaled_timeline_shift_ms(
                matching.mapping,
                matching.lhs_fps,
                stream_bias_ms=stream_bias_ms,
                rhs_fps=matching.rhs_fps,
            ),
        )

    mapping = matching_analysis.strict_mapping(matching.mapping)
    if strategy == AudioStrategy.GLOBAL_TIME_SCALE:
        return AudioPlan(strategy, mapping)

    collapsed = try_collapse_linear_mapping(mapping, matching.lhs_frame_span, tempo_ratio)
    if collapsed is not None:
        return AudioPlan(strategy, collapsed, collapsed_linear_mapping=True)
    return AudioPlan(strategy, mapping)


def choose_strategy(
    matching: SegmentsMappingResult,
    tempo_ratio: float | None,
) -> AudioStrategy:
    """Pick the strategy for fitting the source audio to the base video.

    Decision table (mapping relation × observed drift):

    - GLOBAL_LINEAR, same speed          → UNSCALED
    - GLOBAL_LINEAR, real speed change   → GLOBAL_TIME_SCALE
    - GENERIC, pure frame-rate drift     → UNSCALED
    - GENERIC otherwise                  → SUBSEGMENT

    UNSCALED applies when no global time-scale is needed: a same-speed
    global-linear relation (a constant offset, or a frame-rate conversion
    that preserves playback time — different fps but same speed), or a
    generic match that is a pure frame-rate drift.  Only a genuine
    playback-speed difference (e.g. a 25 fps PAL transfer vs 24 fps) is
    time-scaled.  UNSCALED still normalizes through the FLAC-domain flow
    because mismatched inputs need deterministic trimming and placement; a
    constant timeline shift carried by the mapping (shared content genuinely
    starting on different wall-clock positions, with no container offset
    recording it) is applied at placement time — see
    ``unscaled_timeline_shift_ms``.

    The mapping the chosen strategy must be applied with is built by
    ``plan_audio``: UNSCALED keeps the raw pairs (they only derive a
    placement shift); GLOBAL_TIME_SCALE needs the strict cleanup for its trim
    ranges; SUBSEGMENT may additionally collapse an effectively linear match
    to one clean line (``try_collapse_linear_mapping``) so the subsegment
    patcher scales correctly — container metadata may sharpen the line's
    slope there but never override what the pairs observe.  Mapping edges
    are not touched here — boundary decisions belong to ``PairMatcher``.
    """
    mapping = matching.mapping

    seg_raw = segment_range(mapping)
    raw_source_span = seg_raw.rhs_end - seg_raw.rhs_start
    raw_base_span = seg_raw.lhs_end - seg_raw.lhs_start
    raw_source_per_base = raw_source_span / raw_base_span if raw_base_span else 1.0
    frame_rate_only_drift = is_frame_rate_only_drift(
        mapping,
        tempo_ratio,
        matching.lhs_fps,
        matching.rhs_fps,
    )

    if matching.relation == MappingRelation.GLOBAL_LINEAR:
        # Scale only when the span ratio is a real speed change and not
        # merely a same-speed frame-rate conversion.
        needs_scaling = needs_fps_scaling(raw_source_per_base) and not frame_rate_only_drift
        return AudioStrategy.GLOBAL_TIME_SCALE if needs_scaling else AudioStrategy.UNSCALED

    if frame_rate_only_drift:
        return AudioStrategy.UNSCALED

    return AudioStrategy.SUBSEGMENT


def needs_fps_scaling(ratio: float) -> bool:
    """Classify a global span as a real playback-speed change.

    This is a strategy-selection policy, not a decision to skip a
    physical transform for an individual audio part.  Per-part scaling is
    instead determined by the sample-derived duration budget in
    :meth:`AudioPatcher._scale_audio_part`.
    """
    return (
        ratio >= 1.0 + FPS_RATIO_TOLERANCE
        or ratio <= 1.0 - FPS_RATIO_TOLERANCE
    )


def is_frame_rate_only_drift(
    mapping: list[tuple[int, int]],
    tempo_ratio: float | None,
    lhs_nominal_fps: float | None,
    rhs_nominal_fps: float | None,
) -> bool:
    """Whether the pairs show a frame-rate change at unchanged playback speed."""
    if len(mapping) < 2 or lhs_nominal_fps is None or rhs_nominal_fps is None or rhs_nominal_fps <= 0:
        return False
    if tempo_ratio is None:
        return False
    nominal_ratio = lhs_nominal_fps / rhs_nominal_fps
    if abs(tempo_ratio - 1.0) < MIN_TEMPO_RATIO_DELTA:
        return False
    if abs(tempo_ratio / nominal_ratio - 1.0) >= MIN_TEMPO_RATIO_DELTA:
        return False

    observed_pairs = sorted(mapping)
    if len(observed_pairs) >= 4:
        observed_pairs = observed_pairs[1:-1]

    seg = segment_range(observed_pairs)
    target_duration = seg.lhs_end - seg.lhs_start
    source_duration = seg.rhs_end - seg.rhs_start
    if target_duration <= 0 or source_duration <= 0:
        return False

    observed_ratio = source_duration / target_duration
    return abs(observed_ratio - 1.0) < MAX_SAME_SPEED_SPAN_DELTA


def try_collapse_linear_mapping(
    mapping: list[tuple[int, int]],
    lhs_frame_span: FrameSpan | None,
    tempo_ratio: float | None,
) -> list[tuple[int, int]] | None:
    """Collapse a linear mapping to the two endpoints of a tempo-built line.

    The pairs are frame-quantized and can carry a corrupted anchor from a
    low-entropy zone; *tempo_ratio* (the effective-fps tempo from
    container metadata) is free of both defects, so when it agrees with
    the pairs' slope within ``MAX_FPS_VS_OBSERVED_RATIO_ERROR``,
    describes a real speed change, and the matched span covers
    ≥ ``MIN_LINEAR_REBUILD_COVERAGE`` of the base, the mapping is
    replaced with a two-point line at that tempo — one global time-scale.
    """
    if lhs_frame_span is None or tempo_ratio is None or tempo_ratio <= 0:
        return None

    pairs = sorted(mapping)
    slope = matching_analysis.linear_mapping_slope(pairs)
    if slope is None:
        return None
    if abs(tempo_ratio / slope - 1.0) >= MAX_FPS_VS_OBSERVED_RATIO_ERROR:
        return None
    if abs(tempo_ratio - 1.0) < MIN_TEMPO_RATIO_DELTA:
        return None

    first_lhs, first_rhs = pairs[0]
    last_lhs, _ = pairs[-1]
    lhs_span = last_lhs - first_lhs
    base_span = lhs_frame_span.duration_ms
    if base_span <= 0 or lhs_span < base_span * MIN_LINEAR_REBUILD_COVERAGE:
        return None

    return [
        (first_lhs, first_rhs),
        (last_lhs, first_rhs + round(lhs_span * tempo_ratio)),
    ]


def unscaled_timeline_shift_ms(
    mapping: list[tuple[int, int]],
    lhs_fps: float | None,
    stream_bias_ms: int = 0,
    rhs_fps: float | None = None,
) -> int:
    """Base-minus-source timeline shift carried by an unscaled mapping.

    UNSCALED plays the source audio without changing speed, so the only degree of
    freedom left is where the track starts on the base timeline.  The
    matched pairs yield that shift as the median of ``lhs - rhs``,
    corrected by ``stream_bias_ms`` — the difference of the two sides'
    mapping-to-stream biases (see ``MeltPerformer._mapping_to_video_ms``)
    — so the shift lands in stream space, the space tracks are placed in.
    There a container start offset cancels out: content restored to its
    canonical position by such an offset yields a zero shift and keeps being
    placed by its content start alone.

    A delta close to the base's whole-frame grid is the unambiguous case
    (for example, dropped leading frames) and stays snapped to that grid.
    Small off-grid deltas are timestamp phase noise: two independently
    quantized frame timestamps can differ by up to half a frame from each
    side.  A delta larger than that ambiguity can nevertheless be real when
    the inputs use different frame rates; accept its millisecond median
    when both its robust spread and its beginning-to-end drift remain
    within the same quantization budget.
    """
    if not mapping or not lhs_fps or lhs_fps <= 0:
        return 0

    lhs_frame_ms = 1000 / lhs_fps
    ordered_deltas = [
        lhs - rhs + stream_bias_ms
        for lhs, rhs in sorted(mapping)
    ]
    median_delta = statistics.median(ordered_deltas)
    frames = round(median_delta / lhs_frame_ms)
    snapped_delta = frames * lhs_frame_ms
    if abs(median_delta - snapped_delta) <= lhs_frame_ms / 4:
        return round(snapped_delta)

    rhs_frame_ms = (
        1000 / rhs_fps
        if rhs_fps is not None and rhs_fps > 0
        else lhs_frame_ms
    )
    quantization_budget_ms = (lhs_frame_ms + rhs_frame_ms) / 2
    if abs(median_delta) <= quantization_budget_ms:
        return 0

    deviations = sorted(abs(delta - median_delta) for delta in ordered_deltas)
    p95_index = math.ceil(0.95 * len(deviations)) - 1
    p95_deviation = deviations[p95_index]
    edge_count = max(1, len(ordered_deltas) // 5)
    edge_drift = abs(
        statistics.median(ordered_deltas[:edge_count])
        - statistics.median(ordered_deltas[-edge_count:])
    )
    if (
        p95_deviation > quantization_budget_ms
        or edge_drift > quantization_budget_ms
    ):
        return 0

    return round(median_delta)
