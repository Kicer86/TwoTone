"""Audio execution for melt timeline mappings.

Strategy selection and mapping recovery intentionally remain in
``MeltPerformer``.  This module only executes an already selected request.
"""

import enum
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple

from ..utils import generic_utils, process_utils, video_utils
from .debug_routines import DebugRoutines
from .melt_common import AudioStreamRef, FramesInfo
from .track_timeline import TrackTimeline


class SegmentRange(NamedTuple):
    lhs_start: int
    lhs_end: int
    rhs_start: int
    rhs_end: int


def segment_range(pairs: Sequence[tuple[int, int]]) -> SegmentRange:
    """Return the bounding lhs/rhs range from a list of matched pairs."""
    left, right = zip(*pairs)
    return SegmentRange(min(left), max(left), min(right), max(right))


class AudioStrategy(enum.Enum):
    """How a mismatched source audio track is fitted to the base video."""

    UNSCALED = "unscaled_normalized"
    GLOBAL_TIME_SCALE = "constant_offset"
    SUBSEGMENT = "subsegment"


@dataclass(frozen=True)
class TimelineInterval:
    """A closed-open interval on a named millisecond timeline."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError("Timeline interval end precedes its start.")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class VideoToAudioTimeline:
    """Translate PairMatcher video timestamps to one selected audio stream."""

    mapping_to_video_ms: int
    audio_content_start_ms: int
    audio_content_end_ms: int | None = None

    def video_to_stream_ms(self, timestamp_ms: int) -> int:
        return timestamp_ms + self.mapping_to_video_ms

    def video_to_rebased_audio_ms(self, timestamp_ms: int) -> int:
        return self.video_to_stream_ms(timestamp_ms) - self.audio_content_start_ms

    def rebased_audio_to_video_ms(self, timestamp_ms: int) -> int:
        return timestamp_ms - self.mapping_to_video_ms + self.audio_content_start_ms


@dataclass(frozen=True)
class AudioPatchRequest:
    """All immutable inputs shared by melt audio patch strategies."""

    working_dir: str
    output_path: str
    source_audio: AudioStreamRef
    base_audio: AudioStreamRef | None
    mapping: tuple[tuple[int, int], ...]
    target_interval: TimelineInterval
    output_interval: TimelineInterval
    source_timeline: VideoToAudioTimeline
    base_timeline: VideoToAudioTimeline | None
    fill_gaps_from_base: bool
    lhs_frames: FramesInfo
    rhs_frames: FramesInfo
    output_mapping_to_video_ms: int = 0


@dataclass(frozen=True)
class AudioPatchResult:
    """A validated patched audio track and its output-timeline placement."""

    stream: AudioStreamRef
    timeline_start_ms: int
    duration_ms: int
    transformation: str


@dataclass(frozen=True)
class _AudioMappingSegment:
    target: TimelineInterval
    source: TimelineInterval


@dataclass(frozen=True)
class AudioSourceWindow:
    """A requested source window and the physical samples available in it."""

    requested: TimelineInterval
    available: TimelineInterval

    def __post_init__(self) -> None:
        if self.available.start_ms < self.requested.start_ms \
                or self.available.end_ms > self.requested.end_ms:
            raise ValueError("Available audio must be contained in its requested window.")

    @property
    def missing_prefix_ms(self) -> int:
        return self.available.start_ms - self.requested.start_ms

    @property
    def missing_suffix_ms(self) -> int:
        return self.requested.end_ms - self.available.end_ms


@dataclass(frozen=True)
class _AudioPart:
    path: str
    duration_ms: int
    source_window: AudioSourceWindow | None = None


class AudioPatcher:
    """Decode, transform, concatenate, encode, and validate audio requests."""

    _SIGNIFICANT_START_CORRECTION_MS = 50
    _ATEMPO_TAIL_SAMPLES = 1024
    _PREFERRED_RATIONAL_SCALE_RATE = 192000
    _MAX_RATIONAL_SCALE_RATE = 2147483647
    _AAC_STANDARD_LAYOUTS = frozenset({
        "mono", "stereo", "3.0", "4.0", "5.0", "5.1", "6.1", "7.1",
    })

    def __init__(
        self,
        logger: logging.Logger,
        timeline: TrackTimeline,
    ) -> None:
        self.logger = logger
        self.timeline = timeline
        self.workspace = timeline.workspace
        self._normalized_audio_cache: dict[
            tuple[str, int, int | None, int | None], str
        ] = {}

    def reset_cache(self) -> None:
        self._normalized_audio_cache.clear()

    def execute(
        self,
        request: AudioPatchRequest,
        strategy: AudioStrategy,
        *,
        segment_count: int = 1,
        min_subsegment_duration: float = 30.0,
        unscaled_shift_ms: int = 0,
    ) -> AudioPatchResult:
        """Execute one fully specified audio patch request."""
        working_dir = os.path.join(request.working_dir, "audio_extraction")
        os.makedirs(working_dir, exist_ok=True)

        source_params = self._get_audio_params(request.source_audio)
        normalized_params = (
            source_params[0],
            source_params[1],
            self._flac_safe_fmt(source_params[2]),
        )
        segments = self._build_strategy_audio_segments(
            request,
            strategy,
            segment_count=segment_count,
            min_subsegment_duration=min_subsegment_duration,
            unscaled_shift_ms=unscaled_shift_ms,
        )
        covered_target_interval = TimelineInterval(
            segments[0].target.start_ms,
            segments[-1].target.end_ms,
        )

        if strategy == AudioStrategy.SUBSEGMENT and request.lhs_frames and request.rhs_frames:
            debug_dir = os.path.join(working_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            DebugRoutines(debug_dir, request.lhs_frames, request.rhs_frames).dump_pairs([
                (
                    segment.target.start_ms,
                    segment.target.end_ms,
                    segment.source.start_ms,
                    segment.source.end_ms,
                )
                for segment in segments
            ])

        self.logger.info(
            "Audio patch for %s (%s): base=[%d…%d] ms, source=[%d…%d] ms, segments=%d",
            self.timeline._audio_source_description(request.source_audio),
            strategy.value,
            request.target_interval.start_ms,
            request.target_interval.end_ms,
            segments[0].source.start_ms,
            segments[-1].source.end_ms,
            len(segments),
        )

        source_parts: list[_AudioPart] = []
        for index, segment in enumerate(segments):
            raw_label = (
                "source_trimmed"
                if strategy == AudioStrategy.GLOBAL_TIME_SCALE
                else "source_unscaled"
                if strategy == AudioStrategy.UNSCALED
                else f"source_segment_{index}"
            )
            raw_part = self._decode_audio_window(
                request.source_audio,
                request.source_timeline,
                segment.source,
                os.path.join(working_dir, f"{raw_label}.flac"),
                normalized_params,
                label=f"{raw_label} audio",
            )
            source_per_base = segment.source.duration_ms / segment.target.duration_ms
            scaled_label = (
                "source_scaled"
                if strategy == AudioStrategy.GLOBAL_TIME_SCALE
                else "source_unscaled_normalized"
                if strategy == AudioStrategy.UNSCALED
                else f"source_segment_{index}_scaled"
            )
            source_parts.append(self._scale_audio_part(
                raw_part,
                source_per_base,
                os.path.join(working_dir, f"{scaled_label}.flac"),
                source_params[1],
                strategy,
                label=f"{scaled_label} audio",
                progress_source=self.timeline._audio_source_description(request.source_audio),
            ))

        physical_source_interval = self._source_patch_target_interval(
            segments,
            source_parts,
        )
        source_timeline_start_ms = physical_source_interval.start_ms
        start_correction_ms = source_timeline_start_ms - covered_target_interval.start_ms
        if request.fill_gaps_from_base and start_correction_ms > self._SIGNIFICANT_START_CORRECTION_MS:
            raise RuntimeError(
                f"Audio start correction of {start_correction_ms} ms detected "
                "in fill-audio-gaps mode. The source container's audio starts "
                "later than its video, which cannot be compensated when head/tail "
                "are filled from the base file. Use default (silence) mode instead."
            )

        head_part, tail_part = self._prepare_audio_gap_parts(
            request,
            physical_source_interval,
            normalized_params,
            working_dir,
        )
        timeline_start_ms = request.output_mapping_to_video_ms + source_timeline_start_ms
        if head_part is not None:
            assert request.base_timeline is not None
            head_start_on_base = max(0, request.base_timeline.video_to_rebased_audio_ms(0))
            timeline_start_ms = (
                request.output_mapping_to_video_ms
                + request.base_timeline.rebased_audio_to_video_ms(head_start_on_base)
            )

        ordered_parts = (
            ([head_part.path] if head_part is not None else [])
            + [part.path for part in source_parts]
            + ([tail_part.path] if tail_part is not None else [])
        )
        self._concat_and_encode(
            ordered_parts,
            os.path.join(working_dir, "concat.txt"),
            request.output_path,
            channel_layout=self._get_audio_channel_layout(request.source_audio),
            progress_description=(
                "Encoding aligned audio from "
                f"{self.timeline._audio_source_description(request.source_audio)}: FLAC -> AAC"
            ),
            logger=self.logger,
        )

        expected_duration_ms = sum(part.duration_ms for part in source_parts)
        if head_part is not None:
            expected_duration_ms += head_part.duration_ms
        if tail_part is not None:
            expected_duration_ms += tail_part.duration_ms
        actual_duration_ms = video_utils.get_video_duration(request.output_path, logger=self.logger)
        if actual_duration_ms is None:
            raise RuntimeError(f"Could not determine patched audio duration for {request.output_path}.")
        self._validate_audio_duration(
            actual_duration_ms,
            expected_duration_ms,
            "final patched audio",
            sample_rate=source_params[1],
            segment_count=len(source_parts),
            encoded=True,
        )
        expected_end_ms = (
            request.output_mapping_to_video_ms + request.output_interval.end_ms
            if head_part is not None or tail_part is not None
            else request.output_mapping_to_video_ms + physical_source_interval.end_ms
        )
        self._validate_audio_placement(
            timeline_start_ms,
            actual_duration_ms,
            expected_end_ms,
            source_params[1],
            len(source_parts),
        )

        transformation = (
            "unscaled-normalization"
            if strategy == AudioStrategy.UNSCALED
            else "global-time-scale"
            if strategy == AudioStrategy.GLOBAL_TIME_SCALE
            else f"subsegment-atempo ({len(source_parts)} segments)"
        )
        return AudioPatchResult(
            stream=AudioStreamRef(request.output_path, 0, 0, request.source_audio.language),
            timeline_start_ms=timeline_start_ms,
            duration_ms=actual_duration_ms,
            transformation=transformation,
        )

    def _build_strategy_audio_segments(
        self,
        request: AudioPatchRequest,
        strategy: AudioStrategy,
        *,
        segment_count: int,
        min_subsegment_duration: float,
        unscaled_shift_ms: int,
    ) -> list[_AudioMappingSegment]:
        if strategy == AudioStrategy.UNSCALED:
            source_start_ms = (
                -unscaled_shift_ms
                + request.output_mapping_to_video_ms
                - request.source_timeline.mapping_to_video_ms
            )
            return [_AudioMappingSegment(
                request.output_interval,
                TimelineInterval(
                    source_start_ms,
                    source_start_ms + request.output_interval.duration_ms,
                ),
            )]
        if strategy == AudioStrategy.SUBSEGMENT:
            return self._build_audio_subsegments(
                request.mapping,
                segment_count,
                min_subsegment_duration,
            )

        source_range = segment_range(request.mapping)
        return [_AudioMappingSegment(
            request.target_interval,
            TimelineInterval(source_range.rhs_start, source_range.rhs_end),
        )]

    @staticmethod
    def _build_audio_subsegments(
        mapping: Sequence[tuple[int, int]],
        segment_count: int,
        min_subsegment_duration: float,
    ) -> list[_AudioMappingSegment]:
        segment = segment_range(mapping)
        target_duration = segment.lhs_end - segment.lhs_start
        if target_duration <= 0:
            raise RuntimeError("Audio mapping has no positive target duration.")

        count = max(1, segment_count)
        targets = [segment.lhs_start + index * target_duration // count for index in range(count + 1)]

        def closest_pair(value: int) -> tuple[int, int]:
            return min(mapping, key=lambda pair: abs(pair[0] - value))

        selected_pairs: list[tuple[int, int]] = []
        for target in targets:
            pair = closest_pair(target)
            if not selected_pairs or pair != selected_pairs[-1]:
                selected_pairs.append(pair)

        raw_segments = [
            [
                selected_pairs[index][0],
                selected_pairs[index + 1][0],
                selected_pairs[index][1],
                selected_pairs[index + 1][1],
            ]
            for index in range(len(selected_pairs) - 1)
            if selected_pairs[index + 1][0] > selected_pairs[index][0]
            and selected_pairs[index + 1][1] > selected_pairs[index][1]
        ]
        if not raw_segments:
            raise RuntimeError("Audio mapping cannot form a positive subsegment.")

        minimum_ms = round(min_subsegment_duration * 1000)
        while len(raw_segments) > 1:
            short_index = next((
                index for index, part in enumerate(raw_segments)
                if part[1] - part[0] < minimum_ms or part[3] - part[2] < minimum_ms
            ), None)
            if short_index is None:
                break
            if short_index == 0:
                first, second = raw_segments[0], raw_segments[1]
                raw_segments[1] = [first[0], second[1], first[2], second[3]]
                del raw_segments[0]
            else:
                previous, current = raw_segments[short_index - 1], raw_segments[short_index]
                raw_segments[short_index - 1] = [previous[0], current[1], previous[2], current[3]]
                del raw_segments[short_index]

        return [
            _AudioMappingSegment(
                TimelineInterval(part[0], part[1]),
                TimelineInterval(part[2], part[3]),
            )
            for part in raw_segments
        ]

    def _decode_audio_window(
        self,
        stream: AudioStreamRef,
        timeline: VideoToAudioTimeline,
        video_interval: TimelineInterval,
        output_path: str,
        normalize_to: tuple[int, int, str],
        *,
        label: str,
    ) -> _AudioPart:
        stream_start_ms = timeline.video_to_stream_ms(video_interval.start_ms)
        stream_end_ms = timeline.video_to_stream_ms(video_interval.end_ms)
        requested_window = TimelineInterval(stream_start_ms, stream_end_ms)
        available_start_ms = max(0, requested_window.start_ms, timeline.audio_content_start_ms)
        available_end_ms = requested_window.end_ms
        if timeline.audio_content_end_ms is not None:
            available_end_ms = min(available_end_ms, timeline.audio_content_end_ms)
        available_window = TimelineInterval(
            available_start_ms,
            max(available_start_ms, available_end_ms),
        )
        planned_source_window = AudioSourceWindow(requested_window, available_window)
        if planned_source_window.available.duration_ms <= 0:
            raise RuntimeError(f"{label} lies outside the selected audio stream.")

        self.timeline._decode_audio_stream_to_flac(
            stream,
            output_path,
            trim_start_ms=max(0, planned_source_window.available.start_ms),
            trim_end_ms=max(0, planned_source_window.available.end_ms),
            normalize_to=normalize_to,
            logger=self.logger,
        )
        actual_duration_ms = video_utils.get_video_duration(output_path, logger=self.logger)
        if actual_duration_ms is None:
            raise RuntimeError(f"Could not determine duration for {label}.")
        if actual_duration_ms > planned_source_window.available.duration_ms:
            raise RuntimeError(
                f"Decoded {label} exceeds its requested source window: got "
                f"{actual_duration_ms} ms, expected at most "
                f"{planned_source_window.available.duration_ms} ms."
            )
        if actual_duration_ms == planned_source_window.available.duration_ms:
            self._validate_audio_duration(
                actual_duration_ms,
                planned_source_window.available.duration_ms,
                label,
                sample_rate=normalize_to[1],
            )

        actual_available_window = TimelineInterval(
            planned_source_window.available.start_ms,
            planned_source_window.available.start_ms + actual_duration_ms,
        )
        source_window = AudioSourceWindow(requested_window, actual_available_window)
        return _AudioPart(output_path, actual_duration_ms, source_window)

    def _scale_audio_part(
        self,
        source_part: _AudioPart,
        source_per_base: float,
        output_path: str,
        sample_rate: int,
        strategy: AudioStrategy,
        *,
        label: str,
        progress_source: str | None = None,
    ) -> _AudioPart:
        if source_per_base > 0:
            target_duration_ms = round(source_part.duration_ms / source_per_base)
            no_filter_tolerance_ms = self._audio_duration_tolerance_ms(sample_rate)
            duration_deviation_ms = abs(source_part.duration_ms - target_duration_ms)
            needs_time_scale = (
                strategy != AudioStrategy.UNSCALED
                and duration_deviation_ms > no_filter_tolerance_ms
            )
        else:
            raise RuntimeError("Audio mapping has no positive source duration.")

        if needs_time_scale:
            if strategy == AudioStrategy.GLOBAL_TIME_SCALE:
                filter_arg = self._global_time_scale_filter(
                    source_part.duration_ms,
                    target_duration_ms,
                    sample_rate,
                )
            else:
                filter_arg = f"atempo={source_per_base:.12g}"
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-i", source_part.path,
                    "-filter:a", filter_arg,
                    "-sample_fmt", "s32", "-c:a", "flac",
                    output_path,
                ], show_progress=True,
                    progress_description=(
                        f"Time-scaling {progress_source}: FLAC -> FLAC"
                        if progress_source is not None
                        else "Time-scaling audio"
                    ),
                    logger=self.logger)
            )
            actual_duration_ms = video_utils.get_video_duration(output_path, logger=self.logger)
            if actual_duration_ms is not None:
                self._validate_audio_duration(
                    actual_duration_ms,
                    target_duration_ms,
                    label,
                    sample_rate=sample_rate,
                    filter_tail_samples=(
                        self._ATEMPO_TAIL_SAMPLES
                        if strategy == AudioStrategy.SUBSEGMENT
                        else 0
                    ),
                )
                scaled_part = _AudioPart(output_path, actual_duration_ms, source_part.source_window)
            else:
                raise RuntimeError(f"Could not determine duration for {label}.")
        else:
            scaled_part = source_part
        return scaled_part

    @classmethod
    def _global_time_scale_filter(
        cls,
        source_duration_ms: int,
        target_duration_ms: int,
        sample_rate: int,
    ) -> str:
        if source_duration_ms > 0 and target_duration_ms > 0:
            requested_ratio = Fraction(source_duration_ms, target_duration_ms)
        else:
            raise ValueError("Global audio scaling requires positive source and target durations.")

        if 0 < sample_rate <= cls._MAX_RATIONAL_SCALE_RATE:
            source_samples = round(Fraction(source_duration_ms * sample_rate, 1000))
            target_samples = round(Fraction(target_duration_ms * sample_rate, 1000))
            duration_tolerance_ms = cls._audio_duration_tolerance_ms(sample_rate)
        else:
            raise ValueError(
                "Global rational audio scaling supports sample rates from 1 to "
                f"{cls._MAX_RATIONAL_SCALE_RATE} Hz."
            )

        selected_rates: tuple[int, int] | None = None
        direct_virtual_sample_rate = round(Fraction(
            source_duration_ms * sample_rate,
            target_duration_ms,
        ))
        if 0 < direct_virtual_sample_rate <= cls._MAX_RATIONAL_SCALE_RATE:
            direct_scaled_samples = round(Fraction(
                source_samples * sample_rate,
                direct_virtual_sample_rate,
            ))
            direct_sample_error = abs(direct_scaled_samples - target_samples)
            if direct_sample_error * 1000 <= duration_tolerance_ms * sample_rate:
                selected_rates = (direct_virtual_sample_rate, sample_rate)

        rate_limit = max(sample_rate, cls._PREFERRED_RATIONAL_SCALE_RATE)
        while rate_limit <= cls._MAX_RATIONAL_SCALE_RATE and selected_rates is None:
            ratio = requested_ratio.limit_denominator(rate_limit)
            largest_ratio_component = max(ratio.numerator, ratio.denominator)
            maximum_multiplier = cls._MAX_RATIONAL_SCALE_RATE // largest_ratio_component
            if maximum_multiplier > 0:
                preferred_multiplier = max(1, round(sample_rate / ratio.denominator))
                rate_multiplier = min(preferred_multiplier, maximum_multiplier)
                virtual_sample_rate = ratio.numerator * rate_multiplier
                resample_rate = ratio.denominator * rate_multiplier
                if (
                    0 < virtual_sample_rate <= cls._MAX_RATIONAL_SCALE_RATE
                    and 0 < resample_rate <= cls._MAX_RATIONAL_SCALE_RATE
                ):
                    scaled_samples = round(Fraction(
                        source_samples * resample_rate,
                        virtual_sample_rate,
                    ))
                    sample_error = abs(scaled_samples - target_samples)
                    if sample_error * 1000 <= duration_tolerance_ms * sample_rate:
                        selected_rates = (virtual_sample_rate, resample_rate)

            if selected_rates is None and rate_limit < cls._MAX_RATIONAL_SCALE_RATE:
                rate_limit = min(cls._MAX_RATIONAL_SCALE_RATE, rate_limit * 2)
            else:
                break

        if selected_rates is not None:
            virtual_sample_rate, resample_rate = selected_rates
            filters = [
                f"asetrate={virtual_sample_rate}",
                f"aresample={resample_rate}",
            ]
            if resample_rate != sample_rate:
                filters.append(f"asetrate={sample_rate}")
            return ",".join(filters)

        raise RuntimeError(
            "Cannot represent global audio scaling from "
            f"{source_duration_ms} ms to {target_duration_ms} ms within the "
            f"{duration_tolerance_ms} ms duration budget using FFmpeg rates up to "
            f"{cls._MAX_RATIONAL_SCALE_RATE} Hz."
        )

    def _source_patch_start_ms(
        self,
        first_segment: _AudioMappingSegment,
        first_part: _AudioPart,
    ) -> int:
        if first_part.source_window is None:
            raise RuntimeError("Decoded audio part is missing its source window.")
        source_per_base = first_segment.source.duration_ms / first_segment.target.duration_ms
        return first_segment.target.start_ms + round(
            first_part.source_window.missing_prefix_ms / source_per_base
        )

    def _source_patch_target_interval(
        self,
        segments: Sequence[_AudioMappingSegment],
        source_parts: Sequence[_AudioPart],
    ) -> TimelineInterval:
        if not segments or len(segments) != len(source_parts):
            raise RuntimeError("Audio patch has no one-to-one segment and part layout.")
        first_part = source_parts[0]
        last_part = source_parts[-1]
        if first_part.source_window is None or last_part.source_window is None:
            raise RuntimeError("Decoded audio part is missing its source window.")

        start_ms = self._source_patch_start_ms(segments[0], first_part)
        last_segment = segments[-1]
        source_per_base = last_segment.source.duration_ms / last_segment.target.duration_ms
        end_ms = last_segment.target.end_ms - round(
            last_part.source_window.missing_suffix_ms / source_per_base
        )
        return TimelineInterval(start_ms, max(start_ms, end_ms))

    def _prepare_audio_gap_parts(
        self,
        request: AudioPatchRequest,
        physical_source_interval: TimelineInterval,
        normalize_to: tuple[int, int, str],
        working_dir: str,
    ) -> tuple[_AudioPart | None, _AudioPart | None]:
        if not request.fill_gaps_from_base:
            return None, None
        if request.base_audio is None or request.base_timeline is None:
            raise RuntimeError("Cannot fill audio gaps without an explicitly selected base audio stream.")

        head_part = None
        tail_part = None
        if physical_source_interval.start_ms > request.output_interval.start_ms:
            head_part = self._decode_audio_window(
                request.base_audio,
                request.base_timeline,
                TimelineInterval(request.output_interval.start_ms, physical_source_interval.start_ms),
                os.path.join(working_dir, "head.flac"),
                normalize_to,
                label="base-audio head",
            )
        if physical_source_interval.end_ms < request.output_interval.end_ms:
            tail_part = self._decode_audio_window(
                request.base_audio,
                request.base_timeline,
                TimelineInterval(physical_source_interval.end_ms, request.output_interval.end_ms),
                os.path.join(working_dir, "tail.flac"),
                normalize_to,
                label="base-audio tail",
            )
        return head_part, tail_part

    def _get_audio_params(self, audio_stream: AudioStreamRef) -> tuple[int, int, str]:
        stream = self.timeline._audio_stream_info(audio_stream)
        if stream is None:
            raise RuntimeError(
                f"ffprobe audio stream #{audio_stream.ffprobe_stream_index} "
                f"not found in {audio_stream.path}."
            )
        return int(stream["channels"]), int(stream["sample_rate"]), stream["sample_fmt"]

    def _get_audio_channel_layout(self, audio_stream: AudioStreamRef) -> str | None:
        stream = self.timeline._audio_stream_info(audio_stream)
        if stream is None:
            raise RuntimeError(
                f"ffprobe audio stream #{audio_stream.ffprobe_stream_index} "
                f"not found in {audio_stream.path}."
            )
        return stream.get("channel_layout") or None

    @staticmethod
    def _audio_duration_tolerance_ms(
        sample_rate: int,
        *,
        segment_count: int = 1,
        encoded: bool = False,
        filter_tail_samples: int = 0,
    ) -> int:
        if sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive.")
        sample_rounding_ms = math.ceil(max(1, segment_count) * 1000 / sample_rate)
        aac_frame_ms = math.ceil(1024 * 1000 / sample_rate) if encoded else 0
        filter_tail_ms = math.ceil(max(0, filter_tail_samples) * 1000 / sample_rate)
        probe_rounding_ms = 2
        return sample_rounding_ms + aac_frame_ms + filter_tail_ms + probe_rounding_ms

    def _validate_audio_duration(
        self,
        actual_ms: int,
        expected_ms: int,
        label: str,
        *,
        sample_rate: int = 48000,
        segment_count: int = 1,
        encoded: bool = False,
        filter_tail_samples: int = 0,
    ) -> None:
        tolerance_ms = self._audio_duration_tolerance_ms(
            sample_rate,
            segment_count=segment_count,
            encoded=encoded,
            filter_tail_samples=filter_tail_samples,
        )
        deviation_ms = abs(actual_ms - expected_ms)
        if deviation_ms > tolerance_ms:
            raise RuntimeError(
                f"Audio duration mismatch in {label}: got {actual_ms} ms, "
                f"expected {expected_ms} ms (difference: {deviation_ms} ms, "
                f"max allowed: {tolerance_ms} ms)."
            )

    def _validate_audio_placement(
        self,
        start_ms: int,
        duration_ms: int,
        expected_end_ms: int,
        sample_rate: int,
        segment_count: int,
    ) -> None:
        actual_end_ms = start_ms + duration_ms
        tolerance_ms = self._audio_duration_tolerance_ms(
            sample_rate,
            segment_count=segment_count,
            encoded=True,
        )
        if abs(actual_end_ms - expected_end_ms) > tolerance_ms:
            raise RuntimeError(
                "Patched audio placement mismatch: "
                f"track ends at {actual_end_ms} ms, expected {expected_end_ms} ms "
                f"(max allowed: {tolerance_ms} ms)."
            )

    @staticmethod
    def _flac_safe_fmt(sample_fmt: str) -> str:
        base = sample_fmt.removesuffix("p")
        if base in ("flt", "dbl"):
            return "s32"
        return sample_fmt

    @classmethod
    def _concat_and_encode(
        cls,
        parts: list[str],
        concat_list_path: str,
        output_path: str,
        channel_layout: str | None = None,
        progress_description: str = "Encoding audio to AAC",
        logger: logging.Logger | None = None,
    ) -> None:
        def _esc(path: str) -> str:
            return path.replace("'", "'\\''")

        with open(concat_list_path, "w", encoding="utf-8") as concat_file:
            concat_file.writelines(f"file '{_esc(part)}'\n" for part in parts)

        needs_layout_fix = (
            channel_layout is not None
            and channel_layout not in cls._AAC_STANDARD_LAYOUTS
        )
        layout_args: list[str] = []
        if needs_layout_fix:
            allowed = "|".join(sorted(cls._AAC_STANDARD_LAYOUTS))
            layout_args = ["-af", f"aformat=channel_layouts={allowed}"]

        process_utils.raise_on_error(
            process_utils.start_process("ffmpeg", [
                "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                *layout_args,
                "-c:a", "aac",
                output_path,
            ], show_progress=True, progress_description=progress_description, logger=logger)
        )

    def temporary_audio_path(self, label: str, ffprobe_stream_index: int) -> str:
        return self.workspace.unique_file(f"{label}_{ffprobe_stream_index}", "mka")

    def prepare_normalized_unscaled_audio(
        self,
        source_stream: AudioStreamRef,
        desired_end_ms: int | None = None,
        desired_start_ms: int | None = None,
    ) -> AudioStreamRef:
        """Normalize one stream without changing its playback speed."""
        source_path = source_stream.path
        ffprobe_stream_index = source_stream.ffprobe_stream_index
        cache_key = (source_path, ffprobe_stream_index, desired_end_ms, desired_start_ms)
        cached_path = self._normalized_audio_cache.get(cache_key)
        if cached_path is not None:
            return AudioStreamRef(cached_path, 0, 0, source_stream.language)

        flac_path = self.temporary_audio_path("normalized_unscaled_flac", ffprobe_stream_index)
        self.timeline._decode_audio_stream_to_flac(
            source_stream,
            flac_path,
            sample_fmt="s32",
            logger=self.logger,
        )

        prepared_flac = flac_path
        if desired_end_ms is not None:
            if desired_start_ms is None:
                desired_start_ms = self.timeline._source_stream_start_offset_ms(
                    flac_path,
                    "audio",
                    0,
                )
            window_start_ms = max(0, -desired_start_ms)
            window_end_ms = max(window_start_ms, desired_end_ms - desired_start_ms)
            prepared_flac = self.temporary_audio_path("normalized_unscaled_trim", ffprobe_stream_index)
            trim_filter = (
                f"atrim=start={window_start_ms / 1000:.6f}:end={window_end_ms / 1000:.6f},"
                "asetpts=PTS-STARTPTS"
            )
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-i", flac_path, "-map", "0:a:0",
                    "-filter:a", trim_filter,
                    "-sample_fmt", "s32", "-c:a", "flac", prepared_flac,
                ], show_progress=True,
                    progress_description=(
                        f"Trimming FLAC from {self.timeline._audio_source_description(source_stream)} "
                        f"to {generic_utils.ms_to_time(window_start_ms)}-"
                        f"{generic_utils.ms_to_time(window_end_ms)}"
                    ),
                    logger=self.logger)
            )

        output_path = self.temporary_audio_path("normalized_unscaled_audio", ffprobe_stream_index)
        channel_layout = self._get_audio_channel_layout(source_stream)
        self._concat_and_encode(
            [prepared_flac],
            self.workspace.unique_file(
                f"normalized_unscaled_concat_{ffprobe_stream_index}",
                "txt",
            ),
            output_path,
            channel_layout=channel_layout,
            progress_description=(
                "Encoding normalized audio from "
                f"{self.timeline._audio_source_description(source_stream)}: FLAC -> AAC"
            ),
            logger=self.logger,
        )

        self._normalized_audio_cache[cache_key] = output_path
        return AudioStreamRef(output_path, 0, 0, source_stream.language)
