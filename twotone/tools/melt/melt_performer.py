import logging
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from tqdm import tqdm

from ..tool import ToolRuntimeContext
from ..utils import (
    files_utils,
    generic_utils,
    language_utils,
    process_utils,
    video_utils,
)
from .audio_patcher import AudioPatcher
from .mapping_validation import DEFAULT_MAX_TEMPORARY_FRAME_OFFSET
from .melt_cache import MeltCache
from .melt_common import (
    AttachmentRef,
    AudioStreamRef,
    StreamType,
    SubtitleStreamRef,
    VideoStreamRef,
    planned_video_duration_ms,
    video_track_duration_ms,
)
from .melt_plan import MeltTitlePlan
from .source_aligner import AlignmentGroup, SourceAligner
from .track_timeline import TrackTimeline, mkvmerge_preserves_start


# Position of each track type in the output file's track order.
_OUTPUT_TRACK_TYPE_ORDER: dict[StreamType, int] = {"video": 0, "audio": 1, "subtitle": 2}


class _StreamEntry(NamedTuple):
    stream_type: StreamType
    mkvmerge_track_id: int
    path: str
    language: str | None
    sync_offset_ms: int | None = None
    name: str | None = None
    forced: bool | None = None
    default: bool | None = None


class _PreparedStreams(NamedTuple):
    entries: list[_StreamEntry]
    input_files: set[str]


_MKVMERGE_TRACK_SELECTION_FLAGS = {
    "video":       ("--video-tracks",    "--no-video"),
    "audio":       ("--audio-tracks",    "--no-audio"),
    "subtitle":    ("--subtitle-tracks",  "--no-subtitles"),
    "attachments": ("--attachments",      "--no-attachments"),
}


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


@dataclass
class _MkvmergeFileOptions:
    """Per-input mkvmerge options collected before serialization."""

    track_ids: dict[str, list[int]] = field(
        default_factory=lambda: {stype: [] for stype in _MKVMERGE_TRACK_SELECTION_FLAGS},
    )
    languages: dict[int, str] = field(default_factory=dict)
    defaults: set[int] = field(default_factory=set)
    default_overrides: dict[int, bool] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    forced: dict[int, bool] = field(default_factory=dict)
    sync_offsets: dict[int, int] = field(default_factory=dict)


class MeltPerformer:
    def __init__(
        self,
        logger: logging.Logger,
        context: ToolRuntimeContext,
        output_dir: str,
        cache: MeltCache | None = None,
        fill_audio_gaps: bool = False,
        max_temporary_frame_offset: float = DEFAULT_MAX_TEMPORARY_FRAME_OFFSET,
    ) -> None:
        self.logger = logger
        self.interruption = context.interruption
        self.output_dir = output_dir
        self.workspace = context.workspace
        self.timeline = TrackTimeline(self.workspace, self.logger, context.media_analysis)
        self.audio_patcher = AudioPatcher(self.logger, self.timeline)
        self.aligner = SourceAligner(
            self.logger,
            context,
            self.timeline,
            self.audio_patcher,
            cache=cache,
            fill_audio_gaps=fill_audio_gaps,
            max_temporary_frame_offset=max_temporary_frame_offset,
        )

    def process_duplicates(self, plan: Sequence[MeltTitlePlan]) -> None:
        planned_items = [item for item in plan if item.groups]
        for item in tqdm(planned_items, desc="Titles", unit="title", **generic_utils.get_tqdm_defaults(), position=0):
            title = item.title
            groups = item.groups

            self.logger.info("Processing title: %s (%d group(s))", title, len(groups))

            for group in tqdm(groups, desc="Videos", unit="video", **generic_utils.get_tqdm_defaults(), position=1):
                self.interruption.check_for_stop()

                input_files = group.input_files
                file_ids = input_files.ids

                # Use analysis results
                video_streams = group.video_streams or ()
                audio_streams = group.audio_streams or ()
                subtitle_streams = group.subtitle_streams or ()
                attachment_refs = group.attachments
                required_input_files = self._collect_required_input_files(
                    video_streams,
                    audio_streams,
                    subtitle_streams,
                    attachment_refs,
                )

                output = self._build_output_path(title, group.output_name)
                self._reject_output_input_alias(
                    output,
                    set(input_files.paths) | required_input_files,
                )

                output_parent = os.path.dirname(output)
                os.makedirs(output_parent, exist_ok=True)

                self.audio_patcher.reset_cache()
                files_details = group.files_details

                with self.workspace.staging_for(output) as staged_output:
                    # Convert streams to unified list (and patch audios if needed)
                    prepared_streams = self._prepare_stream_entries(
                        video_streams,
                        audio_streams,
                        subtitle_streams,
                        attachment_refs,
                        file_ids,
                        files_details,
                        alignment_paths=group.alignment_paths,
                    )

                    # Video, audio, then subtitles; within each type
                    # alphabetically by language, unknown languages last.
                    streams_list_sorted = sorted(
                        prepared_streams.entries,
                        key=lambda stream: (
                            _OUTPUT_TRACK_TYPE_ORDER[stream.stream_type],
                            stream.language is None,
                            stream.language or "",
                        ),
                    )

                    # Decide which track should be default
                    default_audio_stream = next((s for s in prepared_streams.entries if s.stream_type == "audio"), None)
                    default_audio_lang = default_audio_stream.language if default_audio_stream else None
                    preferred_audio = self._choose_preferred_audio(
                        group.audio_prod_lang,
                        streams_list_sorted,
                        default_audio_lang,
                    )

                    generation_args = self.build_mkvmerge_args(
                        staged_output.path,
                        streams_list_sorted,
                        attachment_refs,
                        preferred_audio,
                        prepared_streams.input_files,
                        self._extract_chapters(group.chapter_source)
                        if group.chapter_source
                        else None,
                    )

                    self.logger.info("Generating file: %s", self._display_path(output))

                    process_utils.raise_on_error(
                        process_utils.start_process("mkvmerge", generation_args, show_progress=True, logger=self.logger)
                    )

                    expected_duration_ms = (
                        planned_video_duration_ms(
                            video_streams[0].path,
                            files_details,
                            video_streams[0].mkvmerge_track_id,
                        )
                        if video_streams
                        else None
                    )
                    video_utils.validate_media_output(
                        staged_output.path,
                        expected_stream_counts=Counter(
                            stream.stream_type for stream in streams_list_sorted
                        ),
                        expected_duration_ms=expected_duration_ms,
                        logger=self.logger,
                    )
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
        chapters_file: str | None = None,
    ) -> list[str]:
        generation_args: list[str] = ["-o", output_path]
        if chapters_file:
            generation_args.extend(["--chapters", chapters_file])
        files_opts = {path: _MkvmergeFileOptions() for path in required_input_files}
        file_indices = {path: index for index, path in enumerate(files_opts)}

        # Collect per-file options and track order
        track_order: list[str] = []
        for stream in streams_list_sorted:
            fo = files_opts[stream.path]
            tid = stream.mkvmerge_track_id
            fo.track_ids[stream.stream_type].append(tid)
            fo.languages[tid] = stream.language or "und"
            if stream.stream_type in ("audio", "subtitle") and preferred_audio and stream == preferred_audio:
                fo.defaults.add(tid)
            if stream.default is not None:
                fo.default_overrides[tid] = stream.default
            if stream.name is not None:
                fo.names[tid] = stream.name
            if stream.forced is not None:
                fo.forced[tid] = stream.forced
            if stream.sync_offset_ms is not None:
                fo.sync_offsets[tid] = stream.sync_offset_ms
            track_order.append(f"{file_indices[stream.path]}:{tid}")

        for file_path, tid in attachments:
            files_opts[file_path].track_ids["attachments"].append(tid)

        # Serialize options into mkvmerge args, file by file
        for file_path, fo in files_opts.items():
            for stype, (include_flag, exclude_flag) in _MKVMERGE_TRACK_SELECTION_FLAGS.items():
                ids = fo.track_ids[stype]
                if ids:
                    generation_args.extend([include_flag, ",".join(str(i) for i in ids)])
                else:
                    generation_args.append(exclude_flag)

            for tid, lang in fo.languages.items():
                generation_args.extend(["--language", f"{tid}:{lang}"])

            for tid, name in fo.names.items():
                generation_args.extend(["--track-name", f"{tid}:{name}"])

            for tid, forced in fo.forced.items():
                generation_args.extend(["--forced-display-flag", f"{tid}:{_yes_no(forced)}"])

            if preferred_audio:
                for tid in fo.track_ids["audio"] + fo.track_ids["subtitle"]:
                    generation_args.extend(["--default-track", f"{tid}:{_yes_no(tid in fo.defaults)}"])
            else:
                for tid, default in fo.default_overrides.items():
                    generation_args.extend(["--default-track", f"{tid}:{_yes_no(default)}"])

            for tid, offset_ms in fo.sync_offsets.items():
                generation_args.extend(["--sync", f"{tid}:{offset_ms}"])

            generation_args.append("--no-chapters")
            generation_args.append(file_path)

        if track_order:
            generation_args.extend(["--track-order", ",".join(track_order)])

        return generation_args

    def _extract_chapters(self, source_path: str) -> str:
        chapters_file = self.workspace.unique_file("chapters", "xml")
        self.logger.info("Copying chapters from: %s", self._display_path(source_path))
        process_utils.raise_on_error(
            process_utils.start_process(
                "mkvextract",
                [source_path, "chapters", chapters_file],
                logger=self.logger,
            )
        )
        return chapters_file

    @staticmethod
    def _collect_required_input_files(
        video_streams: Sequence[VideoStreamRef],
        audio_streams: Sequence[AudioStreamRef],
        subtitle_streams: Sequence[SubtitleStreamRef],
        attachments: Sequence[AttachmentRef],
    ) -> set[str]:
        return (
            {stream.path for stream in video_streams}
            | {stream.path for stream in audio_streams}
            | {stream.path for stream in subtitle_streams}
            | {attachment.path for attachment in attachments}
        )

    def _build_output_path(self, title: str, output_name: str) -> str:
        return os.path.join(self.output_dir, title, output_name + ".mkv")

    def _display_path(self, path: str) -> str:
        return files_utils.format_path(path, self.output_dir)

    def _max_stream_end_ms(
        self,
        path: str,
        stream_type: StreamType,
        streams: Sequence[VideoStreamRef | AudioStreamRef],
    ) -> int | None:
        ends = [
            end
            for stream in streams
            if stream.path == path
            and (end := self.timeline.source_stream_end_offset_ms(
                stream.path,
                stream_type,
                stream.ffprobe_stream_index,
            )) is not None
        ]
        return max(ends, default=None)

    def _base_output_end_ms(
        self,
        group: AlignmentGroup,
        video_streams: Sequence[VideoStreamRef],
    ) -> int | None:
        """End of the output timeline: the later of the base video and audio ends."""
        ends = [
            end
            for end in (
                self._max_stream_end_ms(group.base_video.path, "video", video_streams),
                group.base_audio_end_ms,
            )
            if end is not None
        ]
        return max(ends, default=None)

    def _base_audio_end_ms(
        self,
        video_path_base: str,
        audio_streams: Sequence[AudioStreamRef],
    ) -> int | None:
        return self._max_stream_end_ms(video_path_base, "audio", audio_streams)

    def _stream_entry(
        self,
        stream_type: StreamType,
        stream: VideoStreamRef | AudioStreamRef | SubtitleStreamRef,
        desired_start_ms: int | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> _StreamEntry:
        metadata = metadata or {}
        return _StreamEntry(
            stream_type,
            stream.mkvmerge_track_id,
            stream.path,
            stream.language,
            self.timeline.track_sync_offset_ms(
                stream.path,
                stream_type,
                stream.ffprobe_stream_index,
                desired_start_ms,
            ),
            metadata.get("name") or metadata.get("title"),
            bool(metadata["forced"]) if "forced" in metadata else None,
            bool(metadata["default"]) if "default" in metadata else None,
        )

    def _prepare_video_entries(
        self,
        video_streams: Sequence[VideoStreamRef],
    ) -> list[_StreamEntry]:
        entries: list[_StreamEntry] = []
        for stream in video_streams:
            desired_start_ms = self.timeline.source_stream_start_offset_ms(
                stream.path,
                "video",
                stream.ffprobe_stream_index,
            )
            entries.append(self._stream_entry("video", stream, desired_start_ms))
        return entries

    def _prepare_audio_entries(
        self,
        audio_streams: Sequence[AudioStreamRef],
        group: AlignmentGroup,
        base_output_end_ms: int | None,
    ) -> list[_StreamEntry]:
        base_video = group.base_video
        entries: list[_StreamEntry] = []
        for audio_stream in audio_streams:
            path = audio_stream.path
            output_stream = audio_stream
            preserves_base_container_timing = (
                path == base_video.path
                and mkvmerge_preserves_start(path)
            )
            if preserves_base_container_timing:
                desired_start_ms = self.timeline.source_stream_start_offset_ms(
                    path,
                    "audio",
                    audio_stream.ffprobe_stream_index,
                )
            else:
                desired_start_ms = self.timeline.audio_content_start_ms(
                    self.timeline.audio_stream_info(audio_stream)
                )

            if path in group.alignment_paths:
                patched = self.aligner.patch_audio(group, audio_stream)
                output_stream = patched.stream
                desired_start_ms = patched.timeline_start_ms
            else:
                normalization_reason = (
                    self.timeline.audio_mkvmerge_normalization_reason(audio_stream)
                    if not preserves_base_container_timing
                    else None
                )
                trim_end_ms: int | None = None
                if path != base_video.path and base_output_end_ms is not None:
                    audio_end_ms = self.timeline.source_stream_end_offset_ms(
                        path,
                        "audio",
                        audio_stream.ffprobe_stream_index,
                    )
                    if audio_end_ms is not None and audio_end_ms > base_output_end_ms:
                        trim_end_ms = base_output_end_ms

                if normalization_reason is not None or trim_end_ms is not None:
                    reasons = []
                    if normalization_reason is not None:
                        reasons.append(normalization_reason)
                    if trim_end_ms is not None:
                        reasons.append(
                            "trimming to the output timeline at "
                            f"{generic_utils.ms_to_time(trim_end_ms)}"
                        )
                    self.logger.info(
                        "Preparing audio track #%d from file #%d for muxing: %s.",
                        audio_stream.mkvmerge_track_id,
                        group.file_ids[path],
                        " and ".join(reasons),
                    )
                    output_stream = self.audio_patcher.prepare_normalized_unscaled_audio(
                        audio_stream,
                        desired_end_ms=trim_end_ms,
                        desired_start_ms=desired_start_ms,
                    )

            entries.append(self._stream_entry("audio", output_stream, desired_start_ms))
        return entries

    def _prepare_subtitle_entries(
        self,
        subtitle_streams: Sequence[SubtitleStreamRef],
        group: AlignmentGroup,
    ) -> list[_StreamEntry]:
        entries: list[_StreamEntry] = []
        for subtitle_stream in subtitle_streams:
            output_stream = subtitle_stream
            output_metadata: dict[str, Any] = {}
            if subtitle_stream.path in group.alignment_paths:
                output_stream, output_metadata = self.aligner.align_subtitle(
                    group, subtitle_stream,
                )
                desired_start_ms = None
            else:
                desired_start_ms = self.timeline.source_stream_start_offset_ms(
                    subtitle_stream.path,
                    "subtitle",
                    subtitle_stream.ffprobe_stream_index,
                )
            entries.append(self._stream_entry(
                "subtitle",
                output_stream,
                desired_start_ms,
                output_metadata,
            ))
        return entries

    def _group_context(
        self,
        video_streams: Sequence[VideoStreamRef],
        audio_streams: Sequence[AudioStreamRef],
        file_ids: dict[str, int],
        files_details: Mapping[str, Any],
        alignment_paths: Iterable[str],
    ) -> AlignmentGroup:
        base_video = video_streams[0]
        return AlignmentGroup(
            file_ids=file_ids,
            base_video=base_video,
            base_audio=next(
                (stream for stream in audio_streams if stream.path == base_video.path),
                None,
            ),
            base_duration_ms=video_track_duration_ms(
                base_video.path,
                files_details,
                logger=self.logger,
                mkvmerge_track_id=base_video.mkvmerge_track_id,
            ),
            base_audio_end_ms=self._base_audio_end_ms(
                base_video.path,
                audio_streams,
            ),
            files_details=files_details,
            alignment_paths=frozenset(alignment_paths),
        )

    def _prepare_stream_entries(
        self,
        video_streams: Sequence[VideoStreamRef],
        audio_streams: Sequence[AudioStreamRef],
        subtitle_streams: Sequence[SubtitleStreamRef],
        attachments: Sequence[AttachmentRef],
        file_ids: dict[str, int],
        files_details: Mapping[str, Any] | None = None,
        alignment_paths: Iterable[str] = (),
    ) -> _PreparedStreams:
        self.timeline.set_input_file_ids(file_ids)
        group = self._group_context(
            video_streams,
            audio_streams,
            file_ids,
            files_details or {},
            alignment_paths,
        )
        self.aligner.align_sources(group, audio_streams, subtitle_streams)
        entries = self._prepare_video_entries(video_streams)
        entries.extend(self._prepare_audio_entries(
            audio_streams,
            group,
            self._base_output_end_ms(group, video_streams),
        ))
        entries.extend(self._prepare_subtitle_entries(subtitle_streams, group))

        input_files = (
            {entry.path for entry in entries}
            | {attachment.path for attachment in attachments}
        )
        return _PreparedStreams(entries, input_files)

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
