"""Alignment of an output group's sources onto its base video timeline.

Each source whose video timeline differs from the base video is matched
once with ``PairMatcher``, its coverage is reported, and its mapping is kept
only when ``mapping_validation`` accepts it.  All of that happens before any
track work, so a refused source fails the group before minutes of audio
processing.  Tracks of accepted sources are then fitted onto the base
timeline with that mapping: audio through ``audio_strategy`` and
``AudioPatcher``, subtitles through ``SubtitleAlignment``.
"""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..tool import ToolRuntimeContext
from ..utils import generic_utils
from . import audio_strategy, debug_routines, mapping_validation
from .audio_patcher import (
    AudioPatcher,
    AudioPatchRequest,
    AudioPatchResult,
    AudioStrategy,
    TimelineInterval,
    VideoToAudioTimeline,
    segment_range,
)
from .mapping_validation import DEFAULT_MAX_TEMPORARY_FRAME_OFFSET
from .melt_cache import MeltCache
from .melt_common import (
    AudioStreamRef,
    FrameSpan,
    SubtitleStreamRef,
    VideoStreamRef,
    content_video_tracks,
    file_tracks,
    video_track_duration_ms,
)
from .pair_matcher import PairMatcher, SegmentsMappingResult
from .subtitle_alignment import SubtitleAlignment
from .track_timeline import TrackTimeline


@dataclass
class AlignmentGroup:
    """The base timeline of one output group and the sources aligned onto it."""

    file_ids: dict[str, int]
    base_video: VideoStreamRef
    base_audio: AudioStreamRef | None = None
    base_duration_ms: int | None = None
    base_audio_end_ms: int | None = None
    files_details: Mapping[str, Any] = field(default_factory=dict)
    alignment_paths: frozenset[str] = frozenset()
    # Validated PairMatcher mappings by source path, filled before any track
    # is aligned.
    aligned_sources: dict[str, SegmentsMappingResult] = field(default_factory=dict)


class SourceAligner:
    """Matches a group's mismatched sources and fits their tracks to the base."""

    def __init__(
        self,
        logger: logging.Logger,
        context: ToolRuntimeContext,
        timeline: TrackTimeline,
        audio_patcher: AudioPatcher,
        *,
        cache: MeltCache | None = None,
        fill_audio_gaps: bool = False,
        max_temporary_frame_offset: float = DEFAULT_MAX_TEMPORARY_FRAME_OFFSET,
    ) -> None:
        if (
            not math.isfinite(max_temporary_frame_offset)
            or max_temporary_frame_offset < DEFAULT_MAX_TEMPORARY_FRAME_OFFSET
        ):
            raise ValueError(
                "Maximum temporary frame offset must be finite and at least "
                f"{DEFAULT_MAX_TEMPORARY_FRAME_OFFSET:g}."
            )
        self.logger = logger
        self.interruption = context.interruption
        self.workspace = context.workspace
        self.media_analysis = context.media_analysis
        self.timeline = timeline
        self.audio_patcher = audio_patcher
        self.subtitle_alignment = SubtitleAlignment(self.workspace, logger)
        self.cache = cache
        self.fill_audio_gaps = fill_audio_gaps
        self.max_temporary_frame_offset = max_temporary_frame_offset

    def align_sources(
        self,
        group: AlignmentGroup,
        audio_streams: Sequence[AudioStreamRef],
        subtitle_streams: Sequence[SubtitleStreamRef],
    ) -> None:
        """Match and validate every aligned source once, before any track work.

        A refused source fails the group before any of its tracks, or another
        source's tracks, is processed.  The refusal names the first track type
        that would have used the mapping.
        """
        aligned_subtitles = [
            stream for stream in subtitle_streams if stream.path in group.alignment_paths
        ]
        for subtitle_stream in aligned_subtitles:
            self.subtitle_alignment.validate_text_track(
                subtitle_stream,
                group.file_ids,
                group.files_details,
            )

        consumers: dict[str, str] = {}
        for stream in audio_streams:
            if stream.path in group.alignment_paths:
                consumers.setdefault(stream.path, "audio")
        for subtitle_stream in aligned_subtitles:
            consumers.setdefault(subtitle_stream.path, "subtitle")
        for source_path in sorted(consumers, key=group.file_ids.__getitem__):
            self._align_source(group, source_path, consumer=consumers[source_path])

    def patch_audio(
        self,
        group: AlignmentGroup,
        audio_stream: AudioStreamRef,
    ) -> AudioPatchResult:
        """Patch *audio_stream* onto the base timeline with its source's mapping."""
        base_video = group.base_video
        source_video = self._primary_video_ref(audio_stream.path, group.files_details)
        base_duration = group.base_duration_ms
        assert base_duration is not None
        base_audio_end = group.base_audio_end_ms
        matching = group.aligned_sources[audio_stream.path]
        with self.workspace.scoped_dir("audio_patch") as mwd, \
             generic_utils.TqdmBouncingBar(desc="Processing", **generic_utils.get_tqdm_defaults()):
            plan = self._plan_audio(base_video, source_video, matching)
            strategy = plan.strategy
            mapping = plan.mapping

            self.logger.debug(
                "Audio patching: base_duration=%d ms, mapping_relation=%s, "
                "lhs_fps=%.3f, rhs_fps=%.3f, mapping_pairs=%d",
                base_duration, matching.relation.value,
                matching.lhs_fps, matching.rhs_fps, len(mapping),
            )
            if mapping:
                self.logger.debug(
                    "  Mapping range: lhs=[%d … %d] ms, rhs=[%d … %d] ms",
                    mapping[0][0], mapping[-1][0], mapping[0][1], mapping[-1][1],
                )

            patched_audio = self.audio_patcher.temporary_audio_path(
                "patched_audio",
                audio_stream.ffprobe_stream_index,
            )
            request = self._create_audio_patch_request(
                mwd,
                base_video,
                source_video,
                audio_stream,
                group.base_audio,
                patched_audio,
                plan,
                base_duration,
                matching=matching,
                fill_gaps_from_base=self.fill_audio_gaps,
                output_end_ms=(
                    base_audio_end
                    if strategy == AudioStrategy.UNSCALED and base_audio_end is not None
                    else base_duration
                ),
            )
            result = self.audio_patcher.execute(request)
            self.logger.info("  Desired audio start: %d ms", result.timeline_start_ms)
            return result

    def align_subtitle(
        self,
        group: AlignmentGroup,
        subtitle_stream: SubtitleStreamRef,
    ) -> tuple[SubtitleStreamRef, dict[str, Any]]:
        """Retime *subtitle_stream* onto the base timeline with its source's mapping."""
        details, subtitle_format, description = self.subtitle_alignment.validate_text_track(
            subtitle_stream,
            group.file_ids,
            group.files_details,
        )
        source_video = self._primary_video_ref(subtitle_stream.path, group.files_details)
        matching = group.aligned_sources[subtitle_stream.path]
        return self.subtitle_alignment.align(
            subtitle_stream,
            matching,
            details=details,
            subtitle_format=subtitle_format,
            description=description,
            source_mapping_to_video_ms=self._mapping_to_video_ms(
                source_video,
                matching.rhs_frame_span,
            ),
            output_mapping_to_video_ms=self._mapping_to_video_ms(
                group.base_video,
                matching.lhs_frame_span,
            ),
        )

    def _align_source(
        self,
        group: AlignmentGroup,
        source_path: str,
        *,
        consumer: str,
    ) -> None:
        """Match *source_path*, report its coverage and keep it if it is safe."""
        assert group.base_duration_ms is not None
        base_path = group.base_video.path
        file_ids = group.file_ids
        # The matcher's diagnostics live in its working directory, so the
        # verdict is reached while that directory exists.
        with self.workspace.scoped_dir("matching") as mwd:
            with generic_utils.TqdmBouncingBar(desc="Matching", **generic_utils.get_tqdm_defaults()):
                matching = self._pair_match(group, mwd, source_path)

            if matching.mapping:
                source_duration_ms = video_track_duration_ms(
                    source_path,
                    group.files_details,
                    logger=self.logger,
                )
                assert source_duration_ms is not None
                debug_routines.log_coverage(
                    self.logger,
                    matching.mapping,
                    group.base_duration_ms,
                    source_duration_ms,
                    matching.lhs_fps,
                    matching.rhs_fps,
                    file_ids[base_path],
                    file_ids[source_path],
                )

            mapping_validation.validate_mapping(
                matching,
                consumer=consumer,
                lhs_id=file_ids[base_path],
                rhs_id=file_ids[source_path],
                max_temporary_frame_offset=self.max_temporary_frame_offset,
                logger=self.logger,
            )
        group.aligned_sources[source_path] = matching

    def _pair_match(
        self,
        group: AlignmentGroup,
        mwd: str,
        source_path: str,
    ) -> SegmentsMappingResult:
        """Run PairMatcher for the group's base video and *source_path*."""
        video_path_base = group.base_video.path
        lhs_id = group.file_ids[video_path_base]
        rhs_id = group.file_ids[source_path]
        self.logger.info("Matching video content: #%d - #%d", lhs_id, rhs_id)
        matcher = PairMatcher(
            self.interruption, mwd, video_path_base, source_path,
            self.logger.getChild("PairMatcher"),
            lhs_label=f"#{lhs_id}", rhs_label=f"#{rhs_id}",
            cache=self.cache,
            media_analysis_session=self.media_analysis,
        )
        return matcher.create_segments_mapping()

    def _plan_audio(
        self,
        base_video: VideoStreamRef,
        source_video: VideoStreamRef,
        matching: SegmentsMappingResult,
    ) -> audio_strategy.AudioPlan:
        """Probe the facts the audio strategy needs and plan the patch."""
        lhs_span, rhs_span = matching.lhs_frame_span, matching.rhs_frame_span
        plan = audio_strategy.plan_audio(
            matching,
            tempo_ratio=self._effective_tempo_ratio(
                base_video.path,
                source_video.path,
                lhs_span,
                rhs_span,
            ),
            stream_bias_ms=(
                self._mapping_to_video_ms(base_video, lhs_span)
                - self._mapping_to_video_ms(source_video, rhs_span)
            ),
        )
        self.logger.info("  Audio strategy: %s", plan.strategy.value)
        mapping = plan.mapping
        if plan.collapsed_linear_mapping:
            self.logger.info(
                "  Collapsed linear audio mapping: using %s-%s ↔ %s-%s",
                generic_utils.ms_to_time(mapping[0][0]),
                generic_utils.ms_to_time(mapping[-1][0]),
                generic_utils.ms_to_time(mapping[0][1]),
                generic_utils.ms_to_time(mapping[-1][1]),
            )
        if plan.unscaled_shift_ms:
            self.logger.info(
                "  Unscaled timeline shift: %+d ms (matched content offset)",
                plan.unscaled_shift_ms,
            )
        return plan

    def _create_audio_patch_request(
        self,
        working_dir: str,
        base_video: VideoStreamRef,
        source_video: VideoStreamRef,
        source_audio: AudioStreamRef,
        base_audio: AudioStreamRef | None,
        output_path: str,
        plan: audio_strategy.AudioPlan,
        base_duration_ms: int,
        *,
        matching: SegmentsMappingResult | None = None,
        fill_gaps_from_base: bool,
        output_end_ms: int | None = None,
    ) -> AudioPatchRequest:
        if not plan.mapping:
            raise RuntimeError("Cannot patch audio without matched frame pairs.")

        mapping_tuple = tuple(plan.mapping)
        segment = segment_range(mapping_tuple)
        lhs_span = matching.lhs_frame_span if matching is not None else None
        rhs_span = matching.rhs_frame_span if matching is not None else None
        source_timeline = self._video_to_audio_timeline(source_video, source_audio, rhs_span)
        base_timeline = (
            self._video_to_audio_timeline(base_video, base_audio, lhs_span)
            if base_audio is not None
            else None
        )
        return AudioPatchRequest(
            working_dir=working_dir,
            output_path=output_path,
            source_audio=source_audio,
            base_audio=base_audio,
            mapping=mapping_tuple,
            target_interval=TimelineInterval(segment.lhs_start, segment.lhs_end),
            output_interval=TimelineInterval(0, output_end_ms if output_end_ms is not None else base_duration_ms),
            source_timeline=source_timeline,
            base_timeline=base_timeline,
            fill_gaps_from_base=fill_gaps_from_base,
            lhs_frames=matching.lhs_all_frames if matching is not None else {},
            rhs_frames=matching.rhs_all_frames if matching is not None else {},
            strategy=plan.strategy,
            output_mapping_to_video_ms=self._mapping_to_video_ms(base_video, lhs_span),
            segment_count=plan.segment_count,
            unscaled_shift_ms=plan.unscaled_shift_ms,
        )

    def _mapping_to_video_ms(
        self,
        video_stream: VideoStreamRef,
        frame_span: FrameSpan | None,
    ) -> int:
        """Offset from *video_stream*'s mapping space to its stream space.

        The frame timestamps feeding the pair mapping are container-dependent:
        some demuxers keep the video stream's start offset in frame PTS (mp4),
        others hand out frames rebased to zero (mkv).  The gap between the
        probed stream start and the first decoded frame measures which
        convention this side uses, so mapping deltas from two files can be
        compared in one (stream) space.
        """
        if frame_span is None:
            return 0
        video_info = self.timeline.stream_info(
            video_stream.path,
            "video",
            video_stream.ffprobe_stream_index,
        )
        return self.timeline.stream_start_offset_ms(video_info) - frame_span.first_ms

    def _video_to_audio_timeline(
        self,
        video_stream: VideoStreamRef,
        audio_stream: AudioStreamRef,
        frame_span: FrameSpan | None,
    ) -> VideoToAudioTimeline:
        """Build the explicit mapping-video-to-selected-audio transform."""
        audio_info = self.timeline.audio_stream_info(audio_stream)
        return VideoToAudioTimeline(
            mapping_to_video_ms=self._mapping_to_video_ms(video_stream, frame_span),
            audio_content_start_ms=self.timeline.audio_content_start_ms(audio_info),
            audio_content_end_ms=self.timeline.stream_end_offset_ms(audio_info),
        )

    def _effective_video_fps(self, path: str, frame_span: FrameSpan | None) -> float | None:
        if frame_span is None or frame_span.frame_count < 2:
            return None

        info = self.timeline.media_info(path)
        stream = next(
            (item for item in info.get("streams", []) if item.get("codec_type") == "video"),
            None,
        )
        if stream is None:
            return None
        end_ms = TrackTimeline.stream_end_offset_ms(stream)
        if end_ms is None:
            return None
        duration_ms = max(
            0,
            end_ms - TrackTimeline.stream_start_offset_ms(stream),
        )
        if duration_ms <= 0:
            return None

        try:
            frame_count = int(stream.get("nb_frames") or 0)
        except (TypeError, ValueError):
            frame_count = 0
        if frame_count < 2:
            frame_count = frame_span.frame_count
        return (frame_count - 1) * 1000 / duration_ms

    def _effective_tempo_ratio(
        self,
        base_video: str,
        source_video: str,
        lhs_frame_span: FrameSpan | None,
        rhs_frame_span: FrameSpan | None,
    ) -> float | None:
        """Tempo ratio implied by the two files' effective fps, or None when unknown."""
        lhs_effective_fps = self._effective_video_fps(base_video, lhs_frame_span)
        rhs_effective_fps = self._effective_video_fps(source_video, rhs_frame_span)
        if lhs_effective_fps is None or rhs_effective_fps is None or rhs_effective_fps <= 0:
            return None
        return lhs_effective_fps / rhs_effective_fps

    @staticmethod
    def _primary_video_ref(path: str, files_details: Mapping[str, Any]) -> VideoStreamRef:
        """The first fully identified content video track of *path*."""
        for track in content_video_tracks(file_tracks(files_details, path, "video")):
            mkvmerge_track_id = track.get("tid")
            ffprobe_stream_index = track.get("ffprobe_stream_index")
            if isinstance(mkvmerge_track_id, int) and isinstance(ffprobe_stream_index, int):
                return VideoStreamRef(
                    path,
                    mkvmerge_track_id,
                    ffprobe_stream_index,
                    track.get("language"),
                )
        raise RuntimeError(f"No fully identified video track found in {path}.")
