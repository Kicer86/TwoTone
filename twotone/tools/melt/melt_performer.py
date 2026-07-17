import enum
import logging
import math
import os
import statistics

from dataclasses import dataclass
from typing import Any, Iterable, NamedTuple, Sequence
from tqdm import tqdm

from ..utils import files_utils, generic_utils, language_utils, process_utils, video_utils
from .debug_routines import DebugRoutines
from .melt_cache import MeltCache
from .pair_matcher import CoverageSummary, MappingRelation, PairMatcher, SegmentsMappingResult
from .melt_common import (
    AttachmentRef,
    AudioStreamRef,
    FramesInfo,
    StreamType,
    SubtitleStreamRef,
    VideoStreamRef,
    _is_length_mismatch,
)
from .track_timeline import TrackTimelineMixin


class _SegmentRange(NamedTuple):
    lhs_start: int
    lhs_end: int
    rhs_start: int
    rhs_end: int


class _PairMatchResult(NamedTuple):
    """A cached PairMatcher outcome for one (base, source) video pair."""
    matching: SegmentsMappingResult
    source_duration: int


class _StreamEntry(NamedTuple):
    stream_type: StreamType
    tid: int
    path: str
    language: str | None
    sync_offset_ms: int | None = None


class _AudioStrategy(enum.Enum):
    """How a mismatched source audio track is fitted to the base video."""

    # No time-scale needed — normalize the source audio without changing its
    # speed, then position it by its content start plus the matched-pair
    # timeline shift via mkvmerge --sync (or a head trim for negative shifts).
    UNSCALED = "unscaled_normalized"
    # A real playback-speed difference with a global linear relation — trim
    # and time-scale the whole track once (the constant-offset patcher).
    GLOBAL_TIME_SCALE = "constant_offset"
    # No clean global relation — patch per-scene subsegments with atempo.
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
    """Translate PairMatcher video timestamps to one selected audio stream.

    PairMatcher timestamps are produced by ``showinfo`` and may be rebased by a
    container.  Decoded FLAC, in contrast, is deliberately rebased to its first
    audio-content sample.  This transform makes the two origins explicit so no
    patcher can use a raw RHS video timestamp as an audio cut position.
    """

    mapping_to_video_ms: int
    audio_content_start_ms: int
    audio_content_end_ms: int | None = None

    def video_to_stream_ms(self, timestamp_ms: int) -> int:
        """Return the timestamp on the selected stream's source timeline."""
        return timestamp_ms + self.mapping_to_video_ms

    def video_to_rebased_audio_ms(self, timestamp_ms: int) -> int:
        """Return the timestamp on the selected decoded-FLAC timeline."""
        return self.video_to_stream_ms(timestamp_ms) - self.audio_content_start_ms

    def rebased_audio_to_video_ms(self, timestamp_ms: int) -> int:
        """Return the PairMatcher timestamp corresponding to decoded-FLAC time."""
        return timestamp_ms - self.mapping_to_video_ms + self.audio_content_start_ms


@dataclass(frozen=True)
class AudioPatchRequest:
    """All immutable inputs shared by Melt audio patch strategies."""

    working_dir: str
    output_path: str
    base_video: VideoStreamRef
    source_audio: AudioStreamRef
    base_audio: AudioStreamRef | None
    mapping: tuple[tuple[int, int], ...]
    target_interval: TimelineInterval
    output_interval: TimelineInterval
    base_duration_ms: int
    source_timeline: VideoToAudioTimeline
    base_timeline: VideoToAudioTimeline | None
    fill_gaps_from_base: bool
    lhs_frames: FramesInfo
    rhs_frames: FramesInfo
    # PairMatcher timestamps may be rebased while the selected base video
    # retains a positive container PTS.  Patched audio is muxed beside that
    # untouched video, so translate every physical patch position back to the
    # base stream's output timeline before returning an mkvmerge --sync value.
    output_mapping_to_video_ms: int = 0


@dataclass(frozen=True)
class AudioPatchResult:
    """A validated patched audio track and its exact output-timeline placement."""

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
    """A requested source-stream window and the samples physically available in it.

    ``requested`` stays on the selected audio stream's timestamp timeline.  It
    deliberately retains portions before/after physical audio content, so the
    executor can carry those virtual gaps through a later time transform and
    place the emitted samples on the output timeline correctly.
    """

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


class _PreparedStreams(NamedTuple):
    entries: list[_StreamEntry]
    input_files: set[str]


class MeltPerformer(TrackTimelineMixin):
    def __init__(
        self,
        logger: logging.Logger,
        interruption: generic_utils.InterruptibleProcess,
        workspace: files_utils.Workspace,
        output_dir: str,
        cache: MeltCache | None = None,
        fill_audio_gaps: bool = False,
    ) -> None:
        self.logger = logger
        self.interruption = interruption
        self.output_dir = output_dir
        self.cache = cache
        self.fill_audio_gaps = fill_audio_gaps
        self._normalized_audio_cache: dict[tuple[str, int, int | None, int | None], str] = {}
        self._pair_match_cache: dict[tuple[str, str], _PairMatchResult] = {}
        self._media_info_cache: dict[str, dict[str, Any]] = {}
        self._stream_info_cache: dict[tuple[str, str, int], dict[str, Any] | None] = {}
        self.workspace = workspace

    def process_duplicates(self, plan: list[dict[str, Any]]) -> None:
        planned_items = [item for item in plan if item.get("groups")]
        for item in tqdm(planned_items, desc="Titles", unit="title", **generic_utils.get_tqdm_defaults(), position=0):
            title = item["title"]
            groups = item.get("groups", [])

            self.logger.info("Processing title: %s (%d group(s))", title, len(groups))

            for group in tqdm(groups, desc="Videos", unit="video", **generic_utils.get_tqdm_defaults(), position=1):
                self.interruption.check_for_stop()

                output_name = group["output_name"]
                files = group.get("files", [])
                file_ids = {f: i + 1 for i, f in enumerate(files)}
                for f, fid in file_ids.items():
                    self.logger.info("  #%d: %s", fid, f)

                # Use analysis results
                streams_info = group.get("streams", {})
                attachments = group.get("attachments", [])
                video_streams = [VideoStreamRef(*stream) for stream in streams_info.get("video", [])]
                audio_streams = [AudioStreamRef(*stream) for stream in streams_info.get("audio", [])]
                subtitle_streams = [SubtitleStreamRef(*stream) for stream in streams_info.get("subtitle", [])]
                attachment_refs = [AttachmentRef(*attachment) for attachment in attachments]
                required_input_files = self._collect_required_input_files(
                    video_streams,
                    audio_streams,
                    subtitle_streams,
                    attachment_refs,
                )

                output = self._build_output_path(title, output_name)
                self._reject_output_input_alias(output, set(files) | required_input_files)

                output_parent = os.path.dirname(output)
                os.makedirs(output_parent, exist_ok=True)

                self._normalized_audio_cache.clear()
                self._pair_match_cache.clear()
                files_details = group.get("files_details", {})

                with self.workspace.staging_for(output) as staged_output:
                    # Convert streams to unified list (and patch audios if needed)
                    prepared_streams = self._prepare_stream_entries(
                        video_streams,
                        audio_streams,
                        subtitle_streams,
                        required_input_files,
                        attachment_refs,
                        file_ids,
                        files_details,
                    )

                    # Sort streams by language alphabetically, unknown languages last
                    streams_list_sorted = sorted(
                        prepared_streams.entries,
                        key=lambda stream: (stream.language is None, stream.language or ""),
                    )

                    # Decide which track should be default
                    default_audio_stream = next((s for s in prepared_streams.entries if s.stream_type == "audio"), None)
                    default_audio_lang = default_audio_stream.language if default_audio_stream else None
                    preferred_audio = self._choose_preferred_audio(
                        group.get("audio_prod_lang"),
                        streams_list_sorted,
                        default_audio_lang,
                    )

                    generation_args = self.build_mkvmerge_args(
                        staged_output.path,
                        streams_list_sorted,
                        attachment_refs,
                        preferred_audio,
                        prepared_streams.input_files,
                    )

                    self.logger.info("Generating file: %s", self._display_path(output))

                    process_utils.raise_on_error(
                        process_utils.start_process("mkvmerge", generation_args, show_progress=True, logger=self.logger)
                    )

                    video_utils.validate_media_output(staged_output.path, logger=self.logger)
                    staged_output.commit()

                self.logger.info("%s saved.", output)

    @staticmethod
    def _resolved_path(path: str) -> str:
        return os.path.normcase(os.path.realpath(os.path.abspath(path)))

    def _reject_output_input_alias(self, output: str, input_files: Iterable[str]) -> None:
        resolved_output = self._resolved_path(output)
        for input_path in input_files:
            if resolved_output == self._resolved_path(input_path):
                raise RuntimeError(
                    f"Output path {output} aliases input path {input_path}; "
                    "refusing to overwrite an input file."
                )

    def build_mkvmerge_args(
        self,
        output_path: str,
        streams_list_sorted: Sequence[_StreamEntry],
        attachments: Sequence[AttachmentRef],
        preferred_audio: _StreamEntry | None,
        required_input_files: Iterable[str],
    ) -> list[str]:
        generation_args: list[str] = ["-o", output_path]
        files_opts: dict[str, dict[str, Any]] = {
            path: {
                "video": [],
                "audio": [],
                "subtitle": [],
                "attachments": [],
                "languages": {},
                "defaults": set(),
                "sync_offsets": {},
            }
            for path in required_input_files
        }

        # Collect per-file options and track order
        track_order: list[str] = []
        for stream in streams_list_sorted:
            fo: dict[str, Any] = files_opts[stream.path]
            fo[stream.stream_type].append(stream.tid)
            fo["languages"][stream.tid] = stream.language or "und"
            if stream.stream_type in ("audio", "subtitle") and preferred_audio and stream == preferred_audio:
                fo["defaults"].add(stream.tid)
            if stream.sync_offset_ms is not None:
                fo["sync_offsets"][stream.tid] = stream.sync_offset_ms
            file_index = generic_utils.get_key_position(files_opts, stream.path)
            track_order.append(f"{file_index}:{stream.tid}")

        for file_path, tid in attachments:
            fo = files_opts[file_path]
            fo["attachments"].append(tid)

        # Serialize options into mkvmerge args, file by file
        _STREAM_TYPE_OPTS = {
            "video":       ("--video-tracks",    "--no-video"),
            "audio":       ("--audio-tracks",    "--no-audio"),
            "subtitle":    ("--subtitle-tracks",  "--no-subtitles"),
            "attachments": ("--attachments",      "--no-attachments"),
        }

        for file_path, fo in files_opts.items():
            for stype, (include_flag, exclude_flag) in _STREAM_TYPE_OPTS.items():
                if fo[stype]:
                    generation_args.extend([include_flag, ",".join(str(i) for i in fo[stype])])
                else:
                    generation_args.append(exclude_flag)

            for tid, lang in fo["languages"].items():
                generation_args.extend(["--language", f"{tid}:{lang}"])

            if preferred_audio:
                for tid in fo["audio"] + fo["subtitle"]:
                    flag = "yes" if tid in fo["defaults"] else "no"
                    generation_args.extend(["--default-track", f"{tid}:{flag}"])

            for tid, offset_ms in fo["sync_offsets"].items():
                generation_args.extend(["--sync", f"{tid}:{offset_ms}"])

            generation_args.append(file_path)

        if track_order:
            generation_args.extend(["--track-order", ",".join(track_order)])

        return generation_args

    def patch_audio_constant_offset(
        self,
        wd: str,
        base_video: VideoStreamRef,
        source_audio: AudioStreamRef,
        base_audio: AudioStreamRef | None,
        output_path: str,
        segment_pairs: list[tuple[int, int]],
        *,
        fill_gaps_from_base: bool = True,
    ) -> int:
        """Compatibility entry point for one globally scaled audio patch.

        The production path constructs :class:`AudioPatchRequest` before
        selecting a strategy.  Keeping this small wrapper preserves the focused
        unit-level seam while ensuring it executes the exact same pipeline.
        """
        base_duration_ms = video_utils.get_video_duration(base_video.path, logger=self.logger)
        if base_duration_ms is None:
            raise RuntimeError(f"Could not determine base video duration for {base_video.path}.")
        request = self._create_audio_patch_request(
            wd,
            base_video,
            source_audio,
            base_audio,
            output_path,
            segment_pairs,
            base_duration_ms,
            fill_gaps_from_base=fill_gaps_from_base,
        )
        return self._execute_audio_patch(request, _AudioStrategy.GLOBAL_TIME_SCALE).timeline_start_ms

    def _create_audio_patch_request(
        self,
        working_dir: str,
        base_video: VideoStreamRef,
        source_audio: AudioStreamRef,
        base_audio: AudioStreamRef | None,
        output_path: str,
        mapping: Sequence[tuple[int, int]],
        base_duration_ms: int,
        *,
        lhs_frames: FramesInfo | None = None,
        rhs_frames: FramesInfo | None = None,
        fill_gaps_from_base: bool,
        output_end_ms: int | None = None,
    ) -> AudioPatchRequest:
        if not mapping:
            raise RuntimeError("Cannot patch audio without matched frame pairs.")

        mapping_tuple = tuple(mapping)
        segment = self._segment_range(mapping_tuple)
        source_video = VideoStreamRef(source_audio.path, 0, None)
        source_timeline = self._video_to_audio_timeline(source_video, source_audio, rhs_frames or {})
        base_timeline = (
            self._video_to_audio_timeline(base_video, base_audio, lhs_frames or {})
            if base_audio is not None
            else None
        )
        return AudioPatchRequest(
            working_dir=working_dir,
            output_path=output_path,
            base_video=base_video,
            source_audio=source_audio,
            base_audio=base_audio,
            mapping=mapping_tuple,
            target_interval=TimelineInterval(segment.lhs_start, segment.lhs_end),
            output_interval=TimelineInterval(0, output_end_ms if output_end_ms is not None else base_duration_ms),
            base_duration_ms=base_duration_ms,
            source_timeline=source_timeline,
            base_timeline=base_timeline,
            fill_gaps_from_base=fill_gaps_from_base,
            lhs_frames=lhs_frames or {},
            rhs_frames=rhs_frames or {},
            output_mapping_to_video_ms=self._mapping_to_video_ms(base_video, lhs_frames or {}),
        )

    def _mapping_to_video_ms(
        self,
        video_stream: VideoStreamRef,
        frames: FramesInfo,
    ) -> int:
        """Return the base-stream PTS corresponding to PairMatcher timestamp zero."""
        video_info = self._stream_info(video_stream.path, "video", video_stream.stream_index)
        video_start_ms = self._stream_start_offset_ms(video_info)
        mapping_start_ms = min(frames) if frames else video_start_ms
        return video_start_ms - mapping_start_ms

    def _video_to_audio_timeline(
        self,
        video_stream: VideoStreamRef,
        audio_stream: AudioStreamRef,
        frames: FramesInfo,
    ) -> VideoToAudioTimeline:
        """Build the explicit mapping-video-to-selected-audio transform."""
        audio_info = self._audio_stream_info(audio_stream)
        return VideoToAudioTimeline(
            mapping_to_video_ms=self._mapping_to_video_ms(video_stream, frames),
            audio_content_start_ms=self._audio_content_start_ms(audio_info),
            audio_content_end_ms=self._stream_end_offset_ms(audio_info),
        )

    def _execute_audio_patch(
        self,
        request: AudioPatchRequest,
        strategy: _AudioStrategy,
        *,
        segment_count: int = 1,
        min_subsegment_duration: float = 30.0,
        unscaled_shift_ms: int = 0,
    ) -> AudioPatchResult:
        """Decode, transform, concatenate, and validate one audio patch request.

        Every strategy shares this executor.  A strategy only decides how the
        mapping is divided and the time transform of each resulting segment.
        """
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

        if strategy == _AudioStrategy.SUBSEGMENT and request.lhs_frames and request.rhs_frames:
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
            "Audio patch (%s): base=[%d…%d] ms, source=[%d…%d] ms, segments=%d",
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
                if strategy == _AudioStrategy.GLOBAL_TIME_SCALE
                else "source_unscaled"
                if strategy == _AudioStrategy.UNSCALED
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
                if strategy == _AudioStrategy.GLOBAL_TIME_SCALE
                else "source_unscaled_normalized"
                if strategy == _AudioStrategy.UNSCALED
                else f"source_segment_{index}_scaled"
            )
            source_parts.append(self._scale_audio_part(
                raw_part,
                source_per_base,
                os.path.join(working_dir, f"{scaled_label}.flac"),
                source_params[1],
                strategy,
                label=f"{scaled_label} audio",
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

        self._concat_and_encode(
            [part.path for part in source_parts],
            head_part is not None,
            head_part.path if head_part is not None else "",
            tail_part is not None,
            tail_part.path if tail_part is not None else "",
            os.path.join(working_dir, "concat.txt"),
            request.output_path,
            channel_layout=self._get_audio_channel_layout(request.source_audio),
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
            if strategy == _AudioStrategy.UNSCALED
            else
            "global-time-scale"
            if strategy == _AudioStrategy.GLOBAL_TIME_SCALE
            else f"subsegment-atempo ({len(source_parts)} segments)"
        )
        return AudioPatchResult(
            stream=AudioStreamRef(request.output_path, 0, request.source_audio.language),
            timeline_start_ms=timeline_start_ms,
            duration_ms=actual_duration_ms,
            transformation=transformation,
        )

    def _build_strategy_audio_segments(
        self,
        request: AudioPatchRequest,
        strategy: _AudioStrategy,
        *,
        segment_count: int,
        min_subsegment_duration: float,
        unscaled_shift_ms: int,
    ) -> list[_AudioMappingSegment]:
        """Return strategy-specific time transforms, without touching audio data."""
        if strategy == _AudioStrategy.UNSCALED:
            # ``unscaled_shift_ms`` is measured in stream space, while this
            # segment must stay in PairMatcher's RHS video space.  Translate
            # it through both mapping origins; the common decoder then adds
            # the source origin exactly once when it cuts source-stream PTS.
            # This preserves small AAC/container phases (such as 22 ms) but
            # does not turn a shared positive container offset into a fake
            # half-second audio prefix.
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
        if strategy == _AudioStrategy.SUBSEGMENT:
            return self._build_audio_subsegments(
                request.mapping,
                segment_count,
                min_subsegment_duration,
            )

        source_range = self._segment_range(request.mapping)
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
        segment = MeltPerformer._segment_range(mapping)
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
        """Decode one video-time interval while retaining unavailable audio gaps."""
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

        self._decode_audio_stream_to_flac(
            stream,
            output_path,
            # The audio file can carry only physical samples.  Keep the wider
            # requested range in ``planned_source_window`` for later placement, but
            # cut the FLAC exactly to the available range so decoder tail
            # padding never turns a virtual suffix into extra audio.
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

        # Stream/container duration metadata can extend beyond the decodable
        # samples (for example after AAC priming removal).  Treat that early
        # EOF as a virtual suffix in the same source-window model used for a
        # declared stream end.  Every patch strategy subsequently scales and
        # places this window through the common executor.
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
        strategy: _AudioStrategy,
        *,
        label: str,
    ) -> _AudioPart:
        if source_per_base <= 0:
            raise RuntimeError("Audio mapping has no positive source duration.")
        expected_duration_ms = round(source_part.duration_ms / source_per_base)
        if not self._needs_fps_scaling(source_per_base):
            self._validate_audio_duration(
                source_part.duration_ms,
                expected_duration_ms,
                label,
                sample_rate=sample_rate,
            )
            return source_part

        if strategy == _AudioStrategy.GLOBAL_TIME_SCALE:
            filter_arg = f"asetrate={sample_rate * source_per_base:.12g},aresample={sample_rate}"
        else:
            # The factor must retain enough precision for a film-length segment;
            # three decimal places can accumulate hundreds of milliseconds of drift.
            filter_arg = f"atempo={source_per_base:.12g}"
        process_utils.raise_on_error(
            process_utils.start_process("ffmpeg", [
                "-y", "-i", source_part.path,
                "-filter:a", filter_arg,
                "-sample_fmt", "s32", "-c:a", "flac",
                output_path,
            ], logger=self.logger)
        )
        actual_duration_ms = video_utils.get_video_duration(output_path, logger=self.logger)
        if actual_duration_ms is None:
            raise RuntimeError(f"Could not determine duration for {label}.")
        self._validate_audio_duration(
            actual_duration_ms,
            expected_duration_ms,
            label,
            sample_rate=sample_rate,
            filter_tail_samples=(
                self._ATEMPO_TAIL_SAMPLES
                if strategy == _AudioStrategy.SUBSEGMENT
                else 0
            ),
        )
        return _AudioPart(output_path, actual_duration_ms, source_part.source_window)

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
        """Return the physical output range after virtual source gaps are removed."""
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

    @staticmethod
    def _collect_required_input_files(
        video_streams: Sequence[VideoStreamRef],
        audio_streams: Sequence[AudioStreamRef],
        subtitle_streams: Sequence[SubtitleStreamRef],
        attachments: Sequence[AttachmentRef],
    ) -> set[str]:
        required_input_files: set[str] = set()
        required_input_files |= {p for (p, _, _) in video_streams}
        required_input_files |= {p for (p, _, _) in audio_streams}
        required_input_files |= {p for (p, _, _) in subtitle_streams}
        required_input_files |= {info.path for info in attachments}
        return required_input_files

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format seconds as a compact human-readable time string."""
        if seconds < 0:
            return f"-{MeltPerformer._fmt_time(-seconds)}"
        h = int(seconds // 3600)
        remainder = seconds - h * 3600
        m = int(remainder // 60)
        s = remainder - m * 60
        if h > 0:
            return f"{h}:{m:02d}:{s:04.1f}"
        if m > 0:
            return f"{m}:{s:04.1f}"
        return f"{s:.1f}s"

    def _render_overlap_diagram(
        self,
        lhs_id: int,
        rhs_id: int,
        lhs_duration_ms: int,
        rhs_duration_ms: int,
        summary: CoverageSummary,
    ) -> list[str]:
        """Build ASCII overlap diagram lines for two partially-overlapping files.

        All positions are projected onto the lhs (base) timeline so that speed
        differences between files don't distort the visual overlap.  That
        projection can surprise: the rhs bar and its timestamps are stretched
        or shrunk by the speed factor, so they may exceed the file's real
        duration — when the factor is not 1, a trailing note spells this out.
        """
        lhs_dur = lhs_duration_ms / 1000
        rhs_dur = rhs_duration_ms / 1000
        lhs_sg = summary["lhs_start_gap_s"]
        rhs_sg = summary["rhs_start_gap_s"]
        lhs_eg = summary["lhs_end_gap_s"]
        rhs_eg = summary["rhs_end_gap_s"]

        # Compute speed factor to project rhs times onto lhs timeline.
        # shared_dur_lhs / shared_dur_rhs gives how much rhs time stretches on the lhs axis.
        shared_dur_lhs = lhs_dur - lhs_sg - lhs_eg
        shared_dur_rhs = rhs_dur - rhs_sg - rhs_eg
        speed = shared_dur_lhs / shared_dur_rhs if shared_dur_rhs > 0 else 1.0

        # Unified timeline positions (lhs starts at t=0, rhs projected onto lhs time)
        lhs_s, lhs_e = 0.0, lhs_dur
        rhs_s = lhs_sg - rhs_sg * speed
        rhs_e = (lhs_dur - lhs_eg) + rhs_eg * speed

        t_min = min(lhs_s, rhs_s)
        t_max = max(lhs_e, rhs_e)
        if t_max - t_min <= 0:
            return []

        _THRESH = 0.0  # ignore gaps smaller than _THRESH
        _MIN_GAP = 6   # minimum column width for a visible gap
        _W = 70        # total diagram width in columns

        left_gap = abs(lhs_s - rhs_s)
        right_gap = abs(lhs_e - rhs_e)
        has_left = left_gap > _THRESH
        has_right = right_gap > _THRESH

        left_w = _MIN_GAP if has_left else 0
        right_w = _MIN_GAP if has_right else 0
        mid_w = _W - left_w - right_w

        # Determine bar column ranges; when there's no gap both bars share that edge
        lhs_c0 = 0 if (lhs_s <= rhs_s or not has_left) else left_w
        rhs_c0 = 0 if (rhs_s <= lhs_s or not has_left) else left_w
        lhs_c1 = _W if (lhs_e >= rhs_e or not has_right) else left_w + mid_w
        rhs_c1 = _W if (rhs_e >= lhs_e or not has_right) else left_w + mid_w

        def _make_bar(c0: int, c1: int, label: str) -> str:
            width = max(c1 - c0, len(label) + 2)
            inner = width - 2
            pl = (inner - len(label)) // 2
            pr = inner - len(label) - pl
            return " " * c0 + "|" + "_" * pl + label + "_" * pr + "|"

        bar_lhs = _make_bar(lhs_c0, lhs_c1, f"#{lhs_id}")
        bar_rhs = _make_bar(rhs_c0, rhs_c1, f"#{rhs_id}")

        # Collect edge timestamps on the unified timeline (relative to t_min)
        edge_list: list[tuple[int, float]] = [
            (lhs_c0, lhs_s - t_min), (rhs_c0, rhs_s - t_min),
            (lhs_c1, lhs_e - t_min), (rhs_c1, rhs_e - t_min),
        ]
        seen: set[tuple[int, int]] = set()
        edges: list[tuple[int, str]] = []
        for col, t in sorted(edge_list):
            key = (col, round(t * 10))
            if key not in seen:
                seen.add(key)
                edges.append((col, self._fmt_time(t)))

        # Place labels on up to two rows, avoiding overlap
        rows: list[list[str]] = [
            [" "] * (_W + 20),
            [" "] * (_W + 20),
        ]
        row_end = [-1, -1]
        for c, label in edges:
            for r in range(2):
                if c >= row_end[r] and c + len(label) <= len(rows[r]):
                    for i, ch in enumerate(label):
                        rows[r][c + i] = ch
                    row_end[r] = c + len(label) + 1
                    break

        lines = [bar_lhs, bar_rhs]
        for row in rows:
            s = "".join(row).rstrip()
            if s:
                lines.append(s)
        if abs(speed - 1.0) >= 0.001:
            lines.append(
                f"(#{rhs_id} plays at a different speed; its bar and times are projected "
                f"onto #{lhs_id}'s timeline — {self._fmt_time(rhs_dur)} of #{rhs_id} "
                f"spans {self._fmt_time(rhs_e - rhs_s)} there)"
            )
        return lines

    def _log_coverage(
        self,
        lhs_path: str,
        rhs_path: str,
        mappings: list[tuple[int, int]],
        lhs_duration_ms: int,
        rhs_duration_ms: int,
        lhs_fps: float,
        rhs_fps: float,
        lhs_id: int,
        rhs_id: int,
    ) -> None:
        """Log an exact coverage report for a compared pair of files.

        Files of equal length that share all content are dismissed in one
        line.  Everything else gets the full report: exact edge gaps (to the
        millisecond, no thresholds hide small differences), both durations,
        the speed ratio, and the overlap diagram.  Files of different length
        are by definition not identical — with one exception spelled out
        explicitly: the same frames end to end played at a different frame
        rate.
        """
        summary = PairMatcher.coverage_summary(
            mappings,
            lhs_duration_ms,
            rhs_duration_ms,
            lhs_fps=lhs_fps,
            rhs_fps=rhs_fps,
        )
        lhs_name = os.path.basename(lhs_path)
        rhs_name = os.path.basename(rhs_path)
        ratio = summary["ratio"]
        same_speed = abs(ratio - 1.0) < 0.001
        same_length = lhs_duration_ms == rhs_duration_ms

        if same_length and same_speed and summary["full_coverage"]:
            self.logger.info(
                "Files are 100%% visually identical: %s ↔ %s",
                lhs_name, rhs_name,
            )
            return

        lhs_len = generic_utils.ms_to_time(lhs_duration_ms)
        rhs_len = generic_utils.ms_to_time(rhs_duration_ms)
        if summary["full_coverage"]:
            # The only way identical content yields different lengths: the
            # same frames played at a different frame rate.
            self.logger.info(
                "Files share all content but play at different speeds "
                "(same frames, different frame rate; speed ratio %.5f): "
                "#%d (%s, %s) ↔ #%d (%s, %s)",
                ratio, lhs_id, lhs_name, lhs_len, rhs_id, rhs_name, rhs_len,
            )
        else:
            self.logger.info(
                "Files are NOT fully identical: #%d (%s, %s) ↔ #%d (%s, %s)",
                lhs_id, lhs_name, lhs_len, rhs_id, rhs_name, rhs_len,
            )
            if not same_speed:
                self.logger.info("  speed ratio: %.5f", ratio)
        self.logger.info(
            "  shared content starts at %.3f s in #%d and at %.3f s in #%d",
            summary["lhs_start_gap_s"], lhs_id, summary["rhs_start_gap_s"], rhs_id,
        )
        self.logger.info(
            "  shared content ends %.3f s before end of #%d and %.3f s before end of #%d",
            summary["lhs_end_gap_s"], lhs_id, summary["rhs_end_gap_s"], rhs_id,
        )

        diagram = self._render_overlap_diagram(
            lhs_id, rhs_id, lhs_duration_ms, rhs_duration_ms, summary,
        )
        if diagram:
            self.logger.info("Overlap diagram:\n%s", "\n".join(diagram))

    @staticmethod
    def _segment_range(pairs: Sequence[tuple[int, int]]) -> _SegmentRange:
        """Return the bounding lhs/rhs range from a list of (lhs, rhs) pairs."""
        left, right = zip(*pairs)
        return _SegmentRange(min(left), max(left), min(right), max(right))

    def _get_audio_params(self, audio_stream: AudioStreamRef) -> tuple[int, int, str]:
        """Return (channels, sample_rate, sample_fmt) for a selected stream."""
        stream = self._audio_stream_info(audio_stream)
        if stream is None:
            raise RuntimeError(
                f"Audio stream #{audio_stream.stream_index} not found in {audio_stream.path}."
            )
        return int(stream["channels"]), int(stream["sample_rate"]), stream["sample_fmt"]

    def _get_audio_channel_layout(self, audio_stream: AudioStreamRef) -> str | None:
        """Return the selected stream's channel layout, or None."""
        stream = self._audio_stream_info(audio_stream)
        if stream is None:
            raise RuntimeError(
                f"Audio stream #{audio_stream.stream_index} not found in {audio_stream.path}."
            )
        return stream.get("channel_layout") or None

    _FPS_RATIO_TOLERANCE = 0.001
    # A start correction below this is normal probe noise; above it the audio
    # placement genuinely moves — logged, and rejected in fill-gaps mode where
    # it cannot be compensated.
    _SIGNIFICANT_START_CORRECTION_MS = 50
    # Effective-fps ratios closer to 1.0 than this are treated as the same
    # playback speed (probe noise), not a real tempo difference.
    _MIN_TEMPO_RATIO_DELTA = 0.005
    # A mapping counts as pure frame-rate drift only when the matched spans
    # agree in wall-clock duration within this fraction.
    _MAX_SAME_SPEED_SPAN_DELTA = 0.01
    # An effective-fps tempo ratio may replace the matched pairs' time mapping
    # only when both agree within this fraction; a larger disagreement means
    # the files are not frame-for-frame and the fps ratio says nothing about
    # the time mapping.
    _MAX_FPS_VS_OBSERVED_RATIO_ERROR = 0.05
    # Collapsing a linear mapping to one global time-scale requires the
    # matched span to cover at least this fraction of the base video;
    # projecting a global tempo from a small matched window is guesswork.
    _MIN_LINEAR_REBUILD_COVERAGE = 0.80
    # Number of subsegments the per-scene audio patcher splits the mapped
    # range into.
    _SUBSEGMENT_COUNT = 20
    # FFmpeg's atempo filter can retain up to one 1024-sample processing block
    # at EOF.  Its duration check must account for that bounded filter tail,
    # while final placement validation still catches accumulated error.
    _ATEMPO_TAIL_SAMPLES = 1024

    @staticmethod
    def _needs_fps_scaling(ratio: float) -> bool:
        """Return True when *ratio* deviates enough from 1.0 to need asetrate scaling."""
        return abs(ratio - 1.0) >= MeltPerformer._FPS_RATIO_TOLERANCE

    @staticmethod
    def _audio_duration_tolerance_ms(
        sample_rate: int,
        *,
        segment_count: int = 1,
        encoded: bool = False,
        filter_tail_samples: int = 0,
    ) -> int:
        """Return the measurable duration budget for a FLAC/AAC transformation.

        Each independently time-scaled segment can round by one sample.  The
        final AAC encode can additionally round to one 1024-sample AAC frame.
        Some filters, such as atempo, can retain a bounded tail block at EOF.
        ffprobe then reports the result in milliseconds.  This is a physical
        error budget, not a percentage that grows with film length.
        """
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
        """Raise when duration exceeds the sample/filter-derived precision budget."""
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
        """Return a FLAC-compatible sample format (FLAC does not support float formats)."""
        base = sample_fmt.removesuffix("p")
        if base in ("flt", "dbl"):
            return "s32"
        return sample_fmt

    # Channel layouts that have a standard AAC channel configuration index.
    # Non-standard layouts (e.g., "5.1(side)") force the AAC encoder to use a
    # Program Config Element (PCE) which many decoders/muxers handle poorly,
    # leading to missing channel_layout metadata and garbled channel ordering.
    _AAC_STANDARD_LAYOUTS = frozenset({
        "mono", "stereo", "3.0", "4.0", "5.0", "5.1", "6.1", "7.1",
    })

    @staticmethod
    def _concat_and_encode(
        parts: list[str],
        has_head: bool,
        head_path: str,
        has_tail: bool,
        tail_path: str,
        concat_list_path: str,
        output_path: str,
        channel_layout: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Concatenate audio parts (head + middle segments + tail) and encode to AAC.

        All input parts are expected to be FLAC. The concat demuxer merges them
        and encodes directly to AAC in a single ffmpeg pass.

        *channel_layout*, when given, is the source channel layout string
        (e.g., ``"5.1(side)"``).  Non-standard layouts are normalized to the
        closest standard AAC configuration to avoid PCE encoding issues.
        """
        def _esc(p: str) -> str:
            return p.replace("'", "'\\''" )

        with open(concat_list_path, "w", encoding="utf-8") as f:
            if has_head:
                f.write(f"file '{_esc(head_path)}'\n")
            for seg in parts:
                f.write(f"file '{_esc(seg)}'\n")
            if has_tail:
                f.write(f"file '{_esc(tail_path)}'\n")

        needs_layout_fix = (
            channel_layout is not None
            and channel_layout not in MeltPerformer._AAC_STANDARD_LAYOUTS
        )
        layout_args: list[str] = []
        if needs_layout_fix:
            # Force standard layout selection via aformat filter so the AAC
            # encoder uses a channel configuration index (no PCE).
            allowed = "|".join(sorted(MeltPerformer._AAC_STANDARD_LAYOUTS))
            layout_args = ["-af", f"aformat=channel_layouts={allowed}"]

        process_utils.raise_on_error(
            process_utils.start_process("ffmpeg", [
                "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                *layout_args,
                "-c:a", "aac",
                output_path,
            ], logger=logger)
        )

    def _patch_audio_segment(
        self,
        wd: str,
        base_video: VideoStreamRef,
        source_audio: AudioStreamRef,
        base_audio: AudioStreamRef | None,
        output_path: str,
        segment_pairs: list[tuple[int, int]],
        segment_count: int,
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo,
        min_subsegment_duration: float = 30.0,
        *,
        fill_gaps_from_base: bool = True,
    ) -> int:
        """Compatibility entry point for a subsegment patch request."""
        base_duration_ms = video_utils.get_video_duration(base_video.path, logger=self.logger)
        if base_duration_ms is None:
            raise RuntimeError(f"Could not determine base video duration for {base_video.path}.")
        request = self._create_audio_patch_request(
            wd,
            base_video,
            source_audio,
            base_audio,
            output_path,
            segment_pairs,
            base_duration_ms,
            lhs_frames=lhs_frames,
            rhs_frames=rhs_frames,
            fill_gaps_from_base=fill_gaps_from_base,
        )
        return self._execute_audio_patch(
            request,
            _AudioStrategy.SUBSEGMENT,
            segment_count=segment_count,
            min_subsegment_duration=min_subsegment_duration,
        ).timeline_start_ms

    def _build_output_path(self, title: str, output_name: str) -> str:
        return os.path.join(self.output_dir, title, output_name + ".mkv")

    def _display_path(self, path: str) -> str:
        return files_utils.format_path(path, self.output_dir)

    def _pair_match(
        self,
        mwd: str,
        video_path_base: str,
        source_path: str,
        file_ids: dict[str, int],
    ) -> _PairMatchResult:
        """Return the (cached) PairMatcher result for a (base, source) pair."""
        lhs_id = file_ids[video_path_base]
        rhs_id = file_ids[source_path]
        cache_key = (video_path_base, source_path)
        match_result = self._pair_match_cache.get(cache_key)

        if match_result is None:
            duration = video_utils.get_video_duration(source_path)
            matcher = PairMatcher(
                self.interruption, mwd, video_path_base, source_path,
                self.logger.getChild("PairMatcher"),
                lhs_label=f"#{lhs_id}", rhs_label=f"#{rhs_id}",
                cache=self.cache,
            )
            match_result = _PairMatchResult(
                matching=matcher.create_segments_mapping(),
                source_duration=duration,
            )
            self._pair_match_cache[cache_key] = match_result
        else:
            self.logger.info(
                "Reusing PairMatcher results for #%d ↔ #%d (same video pair).",
                lhs_id, rhs_id,
            )
        return match_result

    def _patch_mismatched_audio(
        self,
        base_video: VideoStreamRef,
        audio_stream: AudioStreamRef,
        base_audio: AudioStreamRef | None,
        base_duration: int,
        base_audio_end: int | None,
        file_ids: dict[str, int],
    ) -> AudioPatchResult:
        """Run PairMatcher and apply the appropriate audio patching strategy."""
        video_path_base = base_video.path
        audio_path = audio_stream.path
        stream_index = audio_stream.stream_index
        with self.workspace.scoped_dir("matching") as mwd, \
             generic_utils.TqdmBouncingBar(desc="Processing", **generic_utils.get_tqdm_defaults()):
            first_match = (video_path_base, audio_path) not in self._pair_match_cache
            match_result = self._pair_match(mwd, video_path_base, audio_path, file_ids)

            matching = match_result.matching
            lhs_all_frames = matching.lhs_all_frames
            rhs_all_frames = matching.rhs_all_frames
            duration = match_result.source_duration

            # Coverage is a property of the file pair, not of the chosen audio
            # strategy — report it right after the matcher, once per pair.
            if first_match and matching.mapping:
                self._log_coverage(
                    video_path_base,
                    audio_path,
                    matching.mapping,
                    base_duration,
                    duration,
                    matching.lhs_fps,
                    matching.rhs_fps,
                    file_ids[video_path_base],
                    file_ids[audio_path],
                )

            # Holes in the shared scene sequence (content present on one side
            # only, e.g. commercial-break cuts) cannot be patched over: every
            # audio strategy assumes the mapped content is continuous.  Fail
            # loudly instead of producing a subtly broken track.
            discontinuities = PairMatcher.find_content_discontinuities(matching.mapping)
            if discontinuities:
                for lhs_from, lhs_to, rhs_from, rhs_to, deficit in discontinuities:
                    self.logger.error(
                        "  Content discontinuity: %s-%s in #%d ↔ %s-%s in #%d (%+d ms)",
                        generic_utils.ms_to_time(lhs_from),
                        generic_utils.ms_to_time(lhs_to),
                        file_ids[video_path_base],
                        generic_utils.ms_to_time(rhs_from),
                        generic_utils.ms_to_time(rhs_to),
                        file_ids[audio_path],
                        deficit,
                    )
                raise RuntimeError(
                    "Inputs share content with holes in the scene sequence "
                    "(e.g. commercial-break cuts) — patching audio across such "
                    "discontinuities is not supported yet"
                )

            strategy = self._choose_audio_strategy(
                video_path_base, audio_path, matching,
            )
            self.logger.info("  Audio strategy: %s", strategy.value)
            mapping = self._prepare_audio_mapping(
                strategy, video_path_base, audio_path, matching,
            )

            unscaled_shift_ms = 0
            if strategy == _AudioStrategy.UNSCALED:
                stream_bias_ms = (
                    self._mapping_stream_bias_ms(video_path_base, matching.lhs_all_frames)
                    - self._mapping_stream_bias_ms(audio_path, matching.rhs_all_frames)
                )
                unscaled_shift_ms = self._unscaled_timeline_shift_ms(
                    mapping, matching.lhs_fps, stream_bias_ms,
                )
                if unscaled_shift_ms:
                    self.logger.info(
                        "  Unscaled timeline shift: %+d ms (matched content offset)",
                        unscaled_shift_ms,
                    )

            self.logger.debug(
                "Audio patching: base_duration=%d ms, source_duration=%d ms, "
                "mapping_relation=%s, lhs_fps=%.3f, rhs_fps=%.3f, mapping_pairs=%d",
                base_duration, duration, matching.relation.value,
                matching.lhs_fps, matching.rhs_fps, len(mapping),
            )
            if mapping:
                self.logger.debug(
                    "  Mapping range: lhs=[%d … %d] ms, rhs=[%d … %d] ms",
                    mapping[0][0], mapping[-1][0], mapping[0][1], mapping[-1][1],
                )

            patched_audio = self._temporary_audio_path("patched_audio", stream_index)
            request = self._create_audio_patch_request(
                mwd,
                base_video,
                audio_stream,
                base_audio,
                patched_audio,
                mapping,
                base_duration,
                lhs_frames=lhs_all_frames,
                rhs_frames=rhs_all_frames,
                fill_gaps_from_base=self.fill_audio_gaps,
                output_end_ms=(
                    base_audio_end
                    if strategy == _AudioStrategy.UNSCALED and base_audio_end is not None
                    else base_duration
                ),
            )
            result = self._execute_audio_patch(
                request,
                strategy,
                segment_count=self._SUBSEGMENT_COUNT if strategy == _AudioStrategy.SUBSEGMENT else 1,
                unscaled_shift_ms=unscaled_shift_ms,
            )
            self.logger.info("  Desired audio start: %d ms", result.timeline_start_ms)
            return result


    def _choose_audio_strategy(
        self,
        video_path_base: str,
        audio_path: str,
        matching: SegmentsMappingResult,
    ) -> _AudioStrategy:
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
        ``_unscaled_timeline_shift_ms``.

        The mapping the chosen strategy must be applied with is built
        separately by ``_prepare_audio_mapping``.
        """
        mapping = matching.mapping

        seg_raw = self._segment_range(mapping)
        raw_source_span = seg_raw.rhs_end - seg_raw.rhs_start
        raw_base_span = seg_raw.lhs_end - seg_raw.lhs_start
        raw_source_per_base = raw_source_span / raw_base_span if raw_base_span else 1.0
        frame_rate_only_drift = self._is_frame_rate_only_drift_mapping(
            video_path_base,
            audio_path,
            mapping,
            matching.lhs_all_frames,
            matching.rhs_all_frames,
            matching.lhs_fps,
            matching.rhs_fps,
        )

        if matching.relation == MappingRelation.GLOBAL_LINEAR:
            # Scale only when the span ratio is a real speed change and not
            # merely a same-speed frame-rate conversion.
            needs_scaling = self._needs_fps_scaling(raw_source_per_base) and not frame_rate_only_drift
            return _AudioStrategy.GLOBAL_TIME_SCALE if needs_scaling else _AudioStrategy.UNSCALED

        if frame_rate_only_drift:
            return _AudioStrategy.UNSCALED

        return _AudioStrategy.SUBSEGMENT

    def _prepare_audio_mapping(
        self,
        strategy: _AudioStrategy,
        video_path_base: str,
        audio_path: str,
        matching: SegmentsMappingResult,
    ) -> list[tuple[int, int]]:
        """Build the audio mapping the chosen strategy must be applied with.

        UNSCALED keeps the raw pairs (they only derive a placement shift);
        GLOBAL_TIME_SCALE needs the strict cleanup for its trim ranges;
        SUBSEGMENT may additionally collapse an effectively linear match to
        one clean line (``_try_collapse_linear_mapping``) so the subsegment
        patcher scales correctly — container metadata may sharpen the line's
        slope there but never override what the pairs observe.  Mapping edges
        are not touched here — boundary decisions belong to ``PairMatcher``.
        """
        if strategy == _AudioStrategy.UNSCALED:
            return matching.mapping

        mapping = self._strict_audio_mapping(matching.mapping)
        if strategy == _AudioStrategy.GLOBAL_TIME_SCALE:
            return mapping

        tempo_ratio = self._effective_tempo_ratio(
            video_path_base,
            audio_path,
            matching.lhs_all_frames,
            matching.rhs_all_frames,
        )
        recovered_mapping = self._try_collapse_linear_mapping(
            mapping,
            matching.lhs_all_frames,
            tempo_ratio,
        )
        if recovered_mapping is not None:
            mapping = recovered_mapping
            self.logger.info(
                "  Collapsed linear audio mapping: using %s-%s ↔ %s-%s",
                generic_utils.ms_to_time(mapping[0][0]),
                generic_utils.ms_to_time(mapping[-1][0]),
                generic_utils.ms_to_time(mapping[0][1]),
                generic_utils.ms_to_time(mapping[-1][1]),
            )
        return mapping

    @staticmethod
    def _mapping_stream_bias_ms(path: str, frames: FramesInfo) -> int:
        """Offset from *path*'s mapping space to its stream space.

        The frame timestamps feeding the pair mapping are container-dependent:
        some demuxers keep the video stream's start offset in frame PTS (mp4),
        others hand out frames rebased to zero (mkv).  The gap between the
        probed stream start and the first probed frame measures which
        convention this side uses, so mapping deltas from two files can be
        compared in one (stream) space.
        """
        if not frames:
            return 0
        info = video_utils.get_video_full_info(path)
        stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if stream is None:
            return 0
        return MeltPerformer._stream_start_offset_ms(stream) - min(frames)

    @staticmethod
    def _unscaled_timeline_shift_ms(
        mapping: list[tuple[int, int]],
        lhs_fps: float | None,
        stream_bias_ms: int = 0,
    ) -> int:
        """Base-minus-source timeline shift carried by an unscaled mapping.

        UNSCALED plays the source audio without changing speed, so the only degree of
        freedom left is where the track starts on the base timeline.  The
        matched pairs yield that shift as the median of ``lhs - rhs``,
        corrected by ``stream_bias_ms`` — the difference of the two sides'
        mapping-to-stream biases (see ``_mapping_stream_bias_ms``) — so the
        shift lands in stream space, the space tracks are placed in.  There a
        container start offset cancels out: content restored to its canonical
        position by such an offset yields a zero shift and keeps being placed
        by its content start alone.  Content can only be offset by dropped or
        added frames, so a genuine shift puts the median on the whole-frame
        grid of the base (residual under a millisecond); the shift is accepted
        only there.  Frame-rate-drift pairs carry deltas quantized to the
        coarser side's frame grid instead, and their median can sit anywhere
        within ±half of that frame — off-grid, rejected here as matcher noise
        rather than rounded up to a spurious whole frame.
        """
        if not mapping or not lhs_fps or lhs_fps <= 0:
            return 0
        frame_ms = 1000 / lhs_fps
        median_delta = statistics.median(lhs - rhs for lhs, rhs in mapping) + stream_bias_ms
        frames = round(median_delta / frame_ms)
        if abs(median_delta - frames * frame_ms) > frame_ms / 4:
            return 0
        return round(frames * frame_ms)

    @staticmethod
    def _strict_audio_mapping(mapping: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Collapse duplicate boundary matches before deriving audio trim ranges."""
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

    @staticmethod
    def _video_playable_duration_ms(path: str) -> int | None:
        info = video_utils.get_video_full_info(path)
        stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if stream is None:
            return None

        end_ms = MeltPerformer._stream_end_offset_ms(stream)
        if end_ms is None:
            return None
        return max(0, end_ms - MeltPerformer._stream_start_offset_ms(stream))

    @staticmethod
    def _effective_video_fps(path: str, frames: FramesInfo) -> float | None:
        if len(frames) < 2:
            return None

        duration_ms = MeltPerformer._video_playable_duration_ms(path)
        if duration_ms is None or duration_ms <= 0:
            return None

        frame_span = MeltPerformer._video_frame_span(path, frames)
        if frame_span <= 0:
            return None

        return frame_span * 1000 / duration_ms

    @staticmethod
    def _effective_tempo_ratio(
        base_video: str,
        source_video: str,
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
    ) -> float | None:
        """Tempo ratio implied by the two files' effective fps, or None when unknown."""
        lhs_effective_fps = MeltPerformer._effective_video_fps(base_video, lhs_all_frames)
        rhs_effective_fps = MeltPerformer._effective_video_fps(source_video, rhs_all_frames)
        if lhs_effective_fps is None or rhs_effective_fps is None or rhs_effective_fps <= 0:
            return None
        return lhs_effective_fps / rhs_effective_fps

    @staticmethod
    def _video_frame_span(path: str, frames: FramesInfo) -> int:
        info = video_utils.get_video_full_info(path)
        stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if stream is not None:
            try:
                frame_count = int(stream.get("nb_frames") or 0)
            except (TypeError, ValueError):
                frame_count = 0
            if frame_count > 1:
                return frame_count - 1
        return len(frames) - 1

    @staticmethod
    def _try_collapse_linear_mapping(
        mapping: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        tempo_ratio: float | None,
    ) -> list[tuple[int, int]] | None:
        """Collapse a linear mapping to the two endpoints of a tempo-built line.

        The pairs are frame-quantized and can carry a corrupted anchor from a
        low-entropy zone; *tempo_ratio* (the effective-fps tempo from
        container metadata) is free of both defects, so when it agrees with
        the pairs' slope within ``_MAX_FPS_VS_OBSERVED_RATIO_ERROR``,
        describes a real speed change, and the matched span covers
        ≥ ``_MIN_LINEAR_REBUILD_COVERAGE`` of the base, the mapping is
        replaced with a two-point line at that tempo — one global time-scale.
        """
        if not lhs_all_frames or tempo_ratio is None or tempo_ratio <= 0:
            return None

        pairs = sorted(mapping)
        slope = PairMatcher._linear_mapping_slope(pairs)
        if slope is None:
            return None
        if abs(tempo_ratio / slope - 1.0) >= MeltPerformer._MAX_FPS_VS_OBSERVED_RATIO_ERROR:
            return None
        if abs(tempo_ratio - 1.0) < MeltPerformer._MIN_TEMPO_RATIO_DELTA:
            return None

        first_lhs, first_rhs = pairs[0]
        last_lhs, _ = pairs[-1]
        lhs_span = last_lhs - first_lhs
        lhs_frame_span = max(lhs_all_frames) - min(lhs_all_frames)
        if lhs_frame_span <= 0 or lhs_span < lhs_frame_span * MeltPerformer._MIN_LINEAR_REBUILD_COVERAGE:
            return None

        return [
            (first_lhs, first_rhs),
            (last_lhs, first_rhs + round(lhs_span * tempo_ratio)),
        ]

    @staticmethod
    def _is_frame_rate_only_drift_mapping(
        base_video: str,
        source_video: str,
        mapping: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
        lhs_nominal_fps: float | None,
        rhs_nominal_fps: float | None,
    ) -> bool:
        if len(mapping) < 2 or lhs_nominal_fps is None or rhs_nominal_fps is None or rhs_nominal_fps <= 0:
            return False

        tempo_ratio = MeltPerformer._effective_tempo_ratio(
            base_video, source_video, lhs_all_frames, rhs_all_frames,
        )
        if tempo_ratio is None:
            return False
        nominal_ratio = lhs_nominal_fps / rhs_nominal_fps
        if abs(tempo_ratio - 1.0) < MeltPerformer._MIN_TEMPO_RATIO_DELTA:
            return False
        if abs(tempo_ratio / nominal_ratio - 1.0) >= MeltPerformer._MIN_TEMPO_RATIO_DELTA:
            return False

        observed_pairs = sorted(mapping)
        if len(observed_pairs) >= 4:
            observed_pairs = observed_pairs[1:-1]

        seg = MeltPerformer._segment_range(observed_pairs)
        target_duration = seg.lhs_end - seg.lhs_start
        source_duration = seg.rhs_end - seg.rhs_start
        if target_duration <= 0 or source_duration <= 0:
            return False

        observed_ratio = source_duration / target_duration
        return abs(observed_ratio - 1.0) < MeltPerformer._MAX_SAME_SPEED_SPAN_DELTA

    @staticmethod
    def _video_track_duration(
        path: str,
        files_details: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> int | None:
        """Return video track duration in ms from pre-probed plan data.

        Falls back to container-level ffprobe duration when plan data is
        unavailable.
        """
        details = files_details.get(path)
        if details:
            for track in details.get("tracks", {}).get("video", []):
                if not track.get("attached_pic", False):
                    length = track.get("length")
                    if length is not None:
                        return length
        return video_utils.get_video_duration(path, logger=logger)

    def _temporary_audio_path(self, label: str, stream_index: int) -> str:
        return self.workspace.unique_file(f"{label}_{stream_index}", "mka")

    def _prepare_normalized_unscaled_audio(
        self,
        source_stream: AudioStreamRef,
        desired_end_ms: int | None = None,
        desired_start_ms: int | None = None,
    ) -> AudioStreamRef:
        """Prepare an audio stream for mkvmerge without changing playback speed.

        The flow stays in a single FLAC-domain pass: decode to a priming-free
        FLAC, optionally trim it to the output timeline, then encode to AAC
        exactly once.

        ``desired_start_ms`` is where the decoded stream's first sample lands on
        the output timeline; the caller muxes the track at that position.  A
        negative value means that much of the head precedes the timeline and is
        cut here instead (the caller then muxes at zero).  ``desired_end_ms``
        bounds the track to the output-timeline end.

        Every extra AAC encode re-applies encoder-delay priming on builds that
        expose it (see ``track_timeline``), shifting the track ~21 ms per pass —
        hence the single-encode discipline, shared with the constant-offset
        patch path.
        """
        source_path = source_stream.path
        stream_index = source_stream.stream_index
        cache_key = (source_path, stream_index, desired_end_ms, desired_start_ms)
        cached_path = self._normalized_audio_cache.get(cache_key)
        if cached_path is not None:
            return AudioStreamRef(cached_path, 0, source_stream.language)

        flac_path = self._temporary_audio_path("normalized_unscaled_flac", stream_index)
        self._decode_audio_stream_to_flac(
            source_stream,
            flac_path,
            sample_fmt="s32",
            logger=self.logger,
        )

        prepared_flac = flac_path
        if desired_end_ms is not None:
            if desired_start_ms is None:
                desired_start_ms = self._source_stream_start_offset_ms(flac_path, "audio", 0)
            window_start_ms = max(0, -desired_start_ms)
            window_end_ms = max(window_start_ms, desired_end_ms - desired_start_ms)
            prepared_flac = self._temporary_audio_path("normalized_unscaled_trim", stream_index)
            trim_filter = (
                f"atrim=start={window_start_ms / 1000:.6f}:end={window_end_ms / 1000:.6f},"
                "asetpts=PTS-STARTPTS"
            )
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-i", flac_path, "-map", "0:a:0",
                    "-filter:a", trim_filter,
                    "-sample_fmt", "s32", "-c:a", "flac", prepared_flac,
                ], logger=self.logger)
            )

        output_path = self._temporary_audio_path("normalized_unscaled_audio", stream_index)
        channel_layout = self._get_audio_channel_layout(source_stream)
        self._concat_and_encode(
            [prepared_flac], False, "", False, "",
            self.workspace.unique_file(f"normalized_unscaled_concat_{stream_index}", "txt"),
            output_path,
            channel_layout=channel_layout,
            logger=self.logger,
        )

        self._normalized_audio_cache[cache_key] = output_path
        return AudioStreamRef(output_path, 0, source_stream.language)

    def _base_output_end_ms(
        self,
        video_path_base: str,
        video_streams: Sequence[VideoStreamRef],
        audio_streams: Sequence[AudioStreamRef],
    ) -> int | None:
        ends: list[int] = []
        for path, stream_index, _language in video_streams:
            if path == video_path_base:
                end = self._source_stream_end_offset_ms(path, "video", stream_index)
                if end is not None:
                    ends.append(end)
        for path, stream_index, _language in audio_streams:
            if path == video_path_base:
                end = self._source_stream_end_offset_ms(path, "audio", stream_index)
                if end is not None:
                    ends.append(end)
        return max(ends) if ends else None

    def _base_audio_end_ms(
        self,
        video_path_base: str,
        audio_streams: Sequence[AudioStreamRef],
    ) -> int | None:
        ends: list[int] = []
        for path, stream_index, _language in audio_streams:
            if path == video_path_base:
                end = self._source_stream_end_offset_ms(path, "audio", stream_index)
                if end is not None:
                    ends.append(end)
        return max(ends) if ends else None

    def _prepare_stream_entries(
        self,
        video_streams: Sequence[VideoStreamRef],
        audio_streams: Sequence[AudioStreamRef],
        subtitle_streams: Sequence[SubtitleStreamRef],
        required_input_files: set[str],
        attachments: Sequence[AttachmentRef],
        file_ids: dict[str, int],
        files_details: dict[str, Any] | None = None,
    ) -> _PreparedStreams:
        video_streams = [VideoStreamRef(*stream) for stream in video_streams]
        audio_streams = [AudioStreamRef(*stream) for stream in audio_streams]
        subtitle_streams = [SubtitleStreamRef(*stream) for stream in subtitle_streams]
        attachments = [AttachmentRef(*attachment) for attachment in attachments]
        streams_list: list[_StreamEntry] = []
        input_files = set(required_input_files)
        base_video = video_streams[0]
        video_path_base = base_video.path
        base_audio = next((stream for stream in audio_streams if stream.path == video_path_base), None)
        details = files_details or {}
        base_duration = self._video_track_duration(video_path_base, details, logger=self.logger)
        base_output_end_ms = self._base_output_end_ms(video_path_base, video_streams, audio_streams)
        base_audio_end_ms = self._base_audio_end_ms(video_path_base, audio_streams)
        protected_paths = (
            {p for (p, _, _) in video_streams}
            | {p for (p, _, _) in subtitle_streams}
            | {p for (p, _) in attachments}
        )

        for (path, stream_index, language) in video_streams:
            video_desired_start_ms = self._source_stream_start_offset_ms(path, "video", stream_index)
            streams_list.append(_StreamEntry(
                "video",
                stream_index,
                path,
                language,
                self._track_sync_offset_ms(path, "video", stream_index, video_desired_start_ms),
            ))

        for audio_stream in audio_streams:
            path, stream_index, language = audio_stream
            audio_desired_start_ms: int | None = self._audio_content_start_ms(
                self._stream_info(path, "audio", stream_index)
            )
            duration = self._video_track_duration(path, details, logger=self.logger)
            if _is_length_mismatch(base_duration, duration):
                assert base_duration is not None  # guaranteed by _is_length_mismatch
                original_path = path
                patched = self._patch_mismatched_audio(
                    base_video, audio_stream, base_audio, base_duration, base_audio_end_ms, file_ids,
                )
                path = patched.stream.path
                stream_index = patched.stream.stream_index
                audio_desired_start_ms = patched.timeline_start_ms
                input_files.add(path)
                if original_path not in protected_paths:
                    input_files.discard(original_path)
            else:
                # Length-matching audio is direct-passthrough by default: keep
                # the original file/stream and let final mkvmerge mux it.  Use
                # the FLAC-domain normalization flow only when direct mkvmerge
                # would change timing semantics or the track must be trimmed to
                # the output timeline.
                needs_mkvmerge_normalization = self._audio_needs_mkvmerge_normalization(audio_stream)
                trim_end_ms: int | None = None
                if path != video_path_base and audio_desired_start_ms is not None and base_output_end_ms is not None:
                    audio_end_ms = self._source_stream_end_offset_ms(path, "audio", stream_index)
                    if audio_end_ms is not None and audio_end_ms > base_output_end_ms:
                        trim_end_ms = base_output_end_ms
                needs_unscaled_preparation = needs_mkvmerge_normalization or trim_end_ms is not None
                if needs_unscaled_preparation:
                    original_path = path
                    prepared_stream = self._prepare_normalized_unscaled_audio(
                        audio_stream,
                        desired_end_ms=trim_end_ms,
                        desired_start_ms=audio_desired_start_ms,
                    )
                    path = prepared_stream.path
                    stream_index = prepared_stream.stream_index
                    input_files.add(path)
                    if original_path not in protected_paths:
                        input_files.discard(original_path)
            streams_list.append(_StreamEntry(
                "audio",
                stream_index,
                path,
                language,
                self._track_sync_offset_ms(path, "audio", stream_index, audio_desired_start_ms),
            ))

        for (path, stream_index, language) in subtitle_streams:
            desired_start_ms = self._source_stream_start_offset_ms(path, "subtitle", stream_index)
            streams_list.append(_StreamEntry(
                "subtitle",
                stream_index,
                path,
                language,
                self._track_sync_offset_ms(path, "subtitle", stream_index, desired_start_ms),
            ))

        return _PreparedStreams(streams_list, input_files)

    def _choose_preferred_audio(
        self,
        audio_prod_lang: str | None,
        streams_list_sorted: Sequence[_StreamEntry],
        default_audio_lang: str | None,
    ) -> _StreamEntry | None:
        preferred_lang = language_utils.unify_lang(audio_prod_lang) if audio_prod_lang else default_audio_lang

        preferred_audio = next(
            (info for info in streams_list_sorted if info.stream_type == "audio" and info.language == preferred_lang),
            None,
        )
        if audio_prod_lang:
            language_name = language_utils.language_name(audio_prod_lang)
            if preferred_audio:
                self.logger.info("Setting production audio language '%s' as default.", language_name)
            else:
                self.logger.warning("Production audio language '%s' not found among audio streams.", language_name)

        return preferred_audio
