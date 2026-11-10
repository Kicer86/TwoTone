import logging
import math
import os
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from tqdm import tqdm

from ..tool import ToolRuntimeContext
from ..utils import (
    files_utils,
    generic_utils,
    language_utils,
    process_utils,
    requirements_utils,
    video_utils,
)
from . import debug_routines, matching_analysis
from .audio_patcher import (
    AudioPatcher,
    AudioPatchRequest,
    AudioPatchResult,
    TimelineInterval,
    VideoToAudioTimeline,
    segment_range,
)
from .audio_patcher import (
    AudioStrategy as _AudioStrategy,
)
from .melt_cache import MeltCache
from .melt_common import (
    AttachmentRef,
    AudioStreamRef,
    FramesInfo,
    StreamType,
    SubtitleStreamRef,
    VideoStreamRef,
)
from .melt_plan import MeltTitlePlan
from .pair_matcher import (
    MappingRelation,
    PairMatcher,
    SegmentsMappingResult,
)
from .subtitle_alignment import SubtitleAlignment
from .track_timeline import TrackTimeline


class _PairMatchResult(NamedTuple):
    """A cached PairMatcher outcome for one (base, source) video pair."""
    matching: SegmentsMappingResult
    source_duration: int


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


class MeltPerformer:
    def __init__(
        self,
        logger: logging.Logger,
        context: ToolRuntimeContext,
        output_dir: str,
        cache: MeltCache | None = None,
        fill_audio_gaps: bool = False,
        max_temporary_frame_offset: float = PairMatcher.DEFAULT_MAX_TEMPORARY_FRAME_OFFSET,
    ) -> None:
        self.logger = logger
        self.interruption = context.interruption
        self.output_dir = output_dir
        self.cache = cache
        self.fill_audio_gaps = fill_audio_gaps
        self.max_temporary_frame_offset = max_temporary_frame_offset
        self._pair_match_cache: dict[tuple[str, str], _PairMatchResult] = {}
        self.media_analysis = context.media_analysis
        self.workspace = context.workspace
        self.timeline = TrackTimeline(self.workspace, self.logger)
        self.audio_patcher = AudioPatcher(self.logger, self.timeline)
        self.subtitle_alignment = SubtitleAlignment(self.workspace, self.logger)

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
                self._pair_match_cache.clear()
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
                        alignment_paths=set(group.alignment_paths),
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
                        self._planned_video_duration_ms(video_streams[0], files_details)
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

    @staticmethod
    def _planned_video_duration_ms(
        base_video: VideoStreamRef,
        files_details: Mapping[str, Any],
    ) -> int | None:
        tracks = files_details.get(base_video.path, {}).get("tracks", {}).get("video", [])
        for track in tracks:
            if track.get("tid") == base_video.mkvmerge_track_id and not track.get("attached_pic", False):
                return track.get("length")
        return None

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
        files_opts: dict[str, dict[str, Any]] = {
            path: {
                "video": [],
                "audio": [],
                "subtitle": [],
                "attachments": [],
                "languages": {},
                "defaults": set(),
                "default_overrides": {},
                "names": {},
                "forced": {},
                "sync_offsets": {},
            }
            for path in required_input_files
        }

        # Collect per-file options and track order
        track_order: list[str] = []
        for stream in streams_list_sorted:
            fo: dict[str, Any] = files_opts[stream.path]
            fo[stream.stream_type].append(stream.mkvmerge_track_id)
            fo["languages"][stream.mkvmerge_track_id] = stream.language or "und"
            if stream.stream_type in ("audio", "subtitle") and preferred_audio and stream == preferred_audio:
                fo["defaults"].add(stream.mkvmerge_track_id)
            if stream.default is not None:
                fo["default_overrides"][stream.mkvmerge_track_id] = stream.default
            if stream.name is not None:
                fo["names"][stream.mkvmerge_track_id] = stream.name
            if stream.forced is not None:
                fo["forced"][stream.mkvmerge_track_id] = stream.forced
            if stream.sync_offset_ms is not None:
                fo["sync_offsets"][stream.mkvmerge_track_id] = stream.sync_offset_ms
            file_index = generic_utils.get_key_position(files_opts, stream.path)
            track_order.append(f"{file_index}:{stream.mkvmerge_track_id}")

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

            for tid, name in fo["names"].items():
                generation_args.extend(["--track-name", f"{tid}:{name}"])

            for tid, forced in fo["forced"].items():
                generation_args.extend(["--forced-display-flag", f"{tid}:{'yes' if forced else 'no'}"])

            if preferred_audio:
                for tid in fo["audio"] + fo["subtitle"]:
                    flag = "yes" if tid in fo["defaults"] else "no"
                    generation_args.extend(["--default-track", f"{tid}:{flag}"])
            else:
                for tid, default in fo["default_overrides"].items():
                    generation_args.extend(["--default-track", f"{tid}:{'yes' if default else 'no'}"])

            for tid, offset_ms in fo["sync_offsets"].items():
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

    def _create_audio_patch_request(
        self,
        working_dir: str,
        base_video: VideoStreamRef,
        source_video: VideoStreamRef,
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
        segment = segment_range(mapping_tuple)
        source_timeline = self._video_to_audio_timeline(source_video, source_audio, rhs_frames or {})
        base_timeline = (
            self._video_to_audio_timeline(base_video, base_audio, lhs_frames or {})
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
        video_info = self.timeline._stream_info(
            video_stream.path,
            "video",
            video_stream.ffprobe_stream_index,
        )
        video_start_ms = self.timeline._stream_start_offset_ms(video_info)
        mapping_start_ms = min(frames) if frames else video_start_ms
        return video_start_ms - mapping_start_ms

    def _video_to_audio_timeline(
        self,
        video_stream: VideoStreamRef,
        audio_stream: AudioStreamRef,
        frames: FramesInfo,
    ) -> VideoToAudioTimeline:
        """Build the explicit mapping-video-to-selected-audio transform."""
        audio_info = self.timeline._audio_stream_info(audio_stream)
        return VideoToAudioTimeline(
            mapping_to_video_ms=self._mapping_to_video_ms(video_stream, frames),
            audio_content_start_ms=self.timeline._audio_content_start_ms(audio_info),
            audio_content_end_ms=self.timeline._stream_end_offset_ms(audio_info),
        )

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

    _FPS_RATIO_TOLERANCE = 0.001
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

    @staticmethod
    def _needs_fps_scaling(ratio: float) -> bool:
        """Classify a global span as a real playback-speed change.

        This is a strategy-selection policy, not a decision to skip a
        physical transform for an individual audio part.  Per-part scaling is
        instead determined by the sample-derived duration budget in
        :meth:`_scale_audio_part`.
        """
        return (
            ratio >= 1.0 + MeltPerformer._FPS_RATIO_TOLERANCE
            or ratio <= 1.0 - MeltPerformer._FPS_RATIO_TOLERANCE
        )

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
            self.logger.info("Matching video content: #%d - #%d", lhs_id, rhs_id)
            duration = video_utils.get_video_duration(source_path)
            matcher = PairMatcher(
                self.interruption, mwd, video_path_base, source_path,
                self.logger.getChild("PairMatcher"),
                lhs_label=f"#{lhs_id}", rhs_label=f"#{rhs_id}",
                cache=self.cache,
                media_analysis_session=self.media_analysis,
                max_temporary_frame_offset=self.max_temporary_frame_offset,
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

    def _validated_pair_match(
        self,
        working_dir: str,
        base_path: str,
        source_path: str,
        base_duration_ms: int,
        file_ids: dict[str, int],
        *,
        consumer: str,
    ) -> _PairMatchResult:
        """Return one safe, cached mapping for an aligned source track."""
        first_match = (base_path, source_path) not in self._pair_match_cache
        match_result = self._pair_match(
            working_dir,
            base_path,
            source_path,
            file_ids,
        )
        matching = match_result.matching

        if first_match and matching.mapping:
            debug_routines.log_coverage(
                self.logger,
                matching.mapping,
                base_duration_ms,
                match_result.source_duration,
                matching.lhs_fps,
                matching.rhs_fps,
                file_ids[base_path],
                file_ids[source_path],
            )

        discontinuities = matching.content_discontinuities
        if matching.relation is MappingRelation.GENERIC:
            discontinuities = discontinuities or tuple(
                matching_analysis.find_content_discontinuities(matching.mapping)
            )

        if matching.timing_discontinuities:
            debug_routines.log_timing_discontinuities(
                self.logger,
                matching.timing_discontinuities,
                consumer=consumer,
                lhs_id=file_ids[base_path],
                rhs_id=file_ids[source_path],
                debug_path=matching.timing_discontinuity_debug_path,
            )
            raise RuntimeError(
                f"Unsupported {consumer} timeline discontinuity: the matched PTS mapping "
                "has a PTS step despite continuous decoded video"
            )

        if discontinuities:
            debug_routines.log_content_discontinuities(
                self.logger,
                discontinuities,
                lhs_id=file_ids[base_path],
                rhs_id=file_ids[source_path],
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
        return match_result

    def _patch_mismatched_audio(
        self,
        base_video: VideoStreamRef,
        source_video: VideoStreamRef,
        audio_stream: AudioStreamRef,
        base_audio: AudioStreamRef | None,
        base_duration: int,
        base_audio_end: int | None,
        file_ids: dict[str, int],
    ) -> AudioPatchResult:
        """Run PairMatcher and apply the appropriate audio patching strategy."""
        self.timeline.set_input_file_ids(file_ids)
        video_path_base = base_video.path
        audio_path = audio_stream.path
        ffprobe_stream_index = audio_stream.ffprobe_stream_index
        with self.workspace.scoped_dir("matching") as mwd, \
             generic_utils.TqdmBouncingBar(desc="Processing", **generic_utils.get_tqdm_defaults()):
            match_result = self._validated_pair_match(
                mwd,
                video_path_base,
                audio_path,
                base_duration,
                file_ids,
                consumer="audio",
            )

            matching = match_result.matching
            lhs_all_frames = matching.lhs_all_frames
            rhs_all_frames = matching.rhs_all_frames
            duration = match_result.source_duration

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
                    mapping,
                    matching.lhs_fps,
                    stream_bias_ms=stream_bias_ms,
                    rhs_fps=matching.rhs_fps,
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

            patched_audio = self.audio_patcher.temporary_audio_path(
                "patched_audio",
                ffprobe_stream_index,
            )
            request = self._create_audio_patch_request(
                mwd,
                base_video,
                source_video,
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
            result = self.audio_patcher.execute(
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

        seg_raw = segment_range(mapping)
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

        mapping = matching_analysis.strict_mapping(matching.mapping)
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

    def _mapping_stream_bias_ms(self, path: str, frames: FramesInfo) -> int:
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
        info = self.timeline._media_info(path)
        stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if stream is None:
            return 0
        return TrackTimeline._stream_start_offset_ms(stream) - min(frames)

    @staticmethod
    def _unscaled_timeline_shift_ms(
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
        mapping-to-stream biases (see ``_mapping_stream_bias_ms``) — so the
        shift lands in stream space, the space tracks are placed in.  There a
        container start offset cancels out: content restored to its canonical
        position by such an offset yields a zero shift and keeps being placed
        by its content start alone.

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

    def _effective_video_fps(self, path: str, frames: FramesInfo) -> float | None:
        if len(frames) < 2:
            return None

        info = self.timeline._media_info(path)
        stream = next(
            (item for item in info.get("streams", []) if item.get("codec_type") == "video"),
            None,
        )
        if stream is None:
            return None
        end_ms = TrackTimeline._stream_end_offset_ms(stream)
        if end_ms is None:
            return None
        duration_ms = max(
            0,
            end_ms - TrackTimeline._stream_start_offset_ms(stream),
        )
        if duration_ms <= 0:
            return None

        try:
            frame_count = int(stream.get("nb_frames") or 0)
        except (TypeError, ValueError):
            frame_count = 0
        frame_span = frame_count - 1 if frame_count > 1 else len(frames) - 1

        return frame_span * 1000 / duration_ms

    def _effective_tempo_ratio(
        self,
        base_video: str,
        source_video: str,
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
    ) -> float | None:
        """Tempo ratio implied by the two files' effective fps, or None when unknown."""
        lhs_effective_fps = self._effective_video_fps(base_video, lhs_all_frames)
        rhs_effective_fps = self._effective_video_fps(source_video, rhs_all_frames)
        if lhs_effective_fps is None or rhs_effective_fps is None or rhs_effective_fps <= 0:
            return None
        return lhs_effective_fps / rhs_effective_fps

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
        slope = matching_analysis.linear_mapping_slope(pairs)
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

    def _is_frame_rate_only_drift_mapping(
        self,
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

        tempo_ratio = self._effective_tempo_ratio(
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

        seg = segment_range(observed_pairs)
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

    def _base_output_end_ms(
        self,
        video_path_base: str,
        video_streams: Sequence[VideoStreamRef],
        audio_streams: Sequence[AudioStreamRef],
    ) -> int | None:
        ends: list[int] = []
        for video_stream in video_streams:
            if video_stream.path == video_path_base:
                end = self.timeline._source_stream_end_offset_ms(
                    video_stream.path,
                    "video",
                    video_stream.ffprobe_stream_index,
                )
                if end is not None:
                    ends.append(end)
        for audio_stream in audio_streams:
            if audio_stream.path == video_path_base:
                end = self.timeline._source_stream_end_offset_ms(
                    audio_stream.path,
                    "audio",
                    audio_stream.ffprobe_stream_index,
                )
                if end is not None:
                    ends.append(end)
        return max(ends) if ends else None

    def _base_audio_end_ms(
        self,
        video_path_base: str,
        audio_streams: Sequence[AudioStreamRef],
    ) -> int | None:
        ends: list[int] = []
        for stream in audio_streams:
            if stream.path == video_path_base:
                end = self.timeline._source_stream_end_offset_ms(
                    stream.path,
                    "audio",
                    stream.ffprobe_stream_index,
                )
                if end is not None:
                    ends.append(end)
        return max(ends) if ends else None

    @staticmethod
    def _primary_video_ref(path: str, files_details: dict[str, Any]) -> VideoStreamRef:
        tracks = files_details.get(path, {}).get("tracks", {}).get("video", [])
        for track in tracks:
            if track.get("attached_pic", False):
                continue
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

    def _patch_mismatched_subtitle(
        self,
        base_video: VideoStreamRef,
        source_video: VideoStreamRef,
        subtitle_stream: SubtitleStreamRef,
        base_duration_ms: int,
        file_ids: dict[str, int],
        files_details: Mapping[str, Any],
    ) -> tuple[SubtitleStreamRef, dict[str, Any]]:
        details, subtitle_format, description = self.subtitle_alignment.validate_text_track(
            subtitle_stream,
            file_ids,
            files_details,
        )
        with self.workspace.scoped_dir("matching") as matching_dir:
            match_result = self._validated_pair_match(
                matching_dir,
                base_video.path,
                source_video.path,
                base_duration_ms,
                file_ids,
                consumer="subtitle",
            )

        matching = match_result.matching
        return self.subtitle_alignment.align(
            subtitle_stream,
            matching,
            details=details,
            subtitle_format=subtitle_format,
            description=description,
            source_mapping_to_video_ms=self._mapping_to_video_ms(
                source_video,
                matching.rhs_all_frames,
            ),
            output_mapping_to_video_ms=self._mapping_to_video_ms(
                base_video,
                matching.lhs_all_frames,
            ),
        )

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
            self.timeline._track_sync_offset_ms(
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
            desired_start_ms = self.timeline._source_stream_start_offset_ms(
                stream.path,
                "video",
                stream.ffprobe_stream_index,
            )
            entries.append(self._stream_entry("video", stream, desired_start_ms))
        return entries

    def _prepare_audio_entries(
        self,
        audio_streams: Sequence[AudioStreamRef],
        *,
        base_video: VideoStreamRef,
        base_audio: AudioStreamRef | None,
        base_duration_ms: int | None,
        base_output_end_ms: int | None,
        base_audio_end_ms: int | None,
        file_ids: dict[str, int],
        files_details: dict[str, Any],
        alignment_paths: set[str],
    ) -> list[_StreamEntry]:
        entries: list[_StreamEntry] = []
        for audio_stream in audio_streams:
            path = audio_stream.path
            output_stream = audio_stream
            extension = os.path.splitext(path)[1].lower()
            preserves_base_container_timing = (
                path == base_video.path
                and extension in TrackTimeline._MKVMERGE_PRESERVES_START_EXTENSIONS
            )
            if preserves_base_container_timing:
                desired_start_ms = self.timeline._source_stream_start_offset_ms(
                    path,
                    "audio",
                    audio_stream.ffprobe_stream_index,
                )
            else:
                desired_start_ms = self.timeline._audio_content_start_ms(
                    self.timeline._audio_stream_info(audio_stream)
                )

            if path in alignment_paths:
                assert base_duration_ms is not None
                source_video = self._primary_video_ref(path, files_details)
                patched = self._patch_mismatched_audio(
                    base_video,
                    source_video,
                    audio_stream,
                    base_audio,
                    base_duration_ms,
                    base_audio_end_ms,
                    file_ids,
                )
                output_stream = patched.stream
                desired_start_ms = patched.timeline_start_ms
            else:
                normalization_reason = (
                    self.timeline._audio_mkvmerge_normalization_reason(audio_stream)
                    if not preserves_base_container_timing
                    else None
                )
                trim_end_ms: int | None = None
                if path != base_video.path and base_output_end_ms is not None:
                    audio_end_ms = self.timeline._source_stream_end_offset_ms(
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
                        file_ids[path],
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
        *,
        base_video: VideoStreamRef,
        base_duration_ms: int | None,
        file_ids: dict[str, int],
        files_details: dict[str, Any],
        alignment_paths: set[str],
    ) -> list[_StreamEntry]:
        entries: list[_StreamEntry] = []
        for subtitle_stream in subtitle_streams:
            output_stream = subtitle_stream
            output_metadata: dict[str, Any] = {}
            if subtitle_stream.path in alignment_paths:
                assert base_duration_ms is not None
                source_video = self._primary_video_ref(subtitle_stream.path, files_details)
                output_stream, output_metadata = self._patch_mismatched_subtitle(
                    base_video,
                    source_video,
                    subtitle_stream,
                    base_duration_ms,
                    file_ids,
                    files_details,
                )
                desired_start_ms = None
            else:
                desired_start_ms = self.timeline._source_stream_start_offset_ms(
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

    def _prepare_stream_entries(
        self,
        video_streams: Sequence[VideoStreamRef],
        audio_streams: Sequence[AudioStreamRef],
        subtitle_streams: Sequence[SubtitleStreamRef],
        attachments: Sequence[AttachmentRef],
        file_ids: dict[str, int],
        files_details: dict[str, Any] | None = None,
        alignment_paths: set[str] | None = None,
    ) -> _PreparedStreams:
        self.timeline.set_input_file_ids(file_ids)

        base_video = video_streams[0]
        base_audio = next(
            (stream for stream in audio_streams if stream.path == base_video.path),
            None,
        )
        details = files_details or {}
        paths_to_align = alignment_paths or set()
        base_duration_ms = self._video_track_duration(
            base_video.path,
            details,
            logger=self.logger,
        )
        base_output_end_ms = self._base_output_end_ms(
            base_video.path,
            video_streams,
            audio_streams,
        )
        base_audio_end_ms = self._base_audio_end_ms(
            base_video.path,
            audio_streams,
        )

        entries = self._prepare_video_entries(video_streams)
        entries.extend(self._prepare_audio_entries(
            audio_streams,
            base_video=base_video,
            base_audio=base_audio,
            base_duration_ms=base_duration_ms,
            base_output_end_ms=base_output_end_ms,
            base_audio_end_ms=base_audio_end_ms,
            file_ids=file_ids,
            files_details=details,
            alignment_paths=paths_to_align,
        ))
        entries.extend(self._prepare_subtitle_entries(
            subtitle_streams,
            base_video=base_video,
            base_duration_ms=base_duration_ms,
            file_ids=file_ids,
            files_details=details,
            alignment_paths=paths_to_align,
        ))

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


setattr(
    MeltPerformer._prepare_audio_entries,
    requirements_utils._REQUIRED_DEPS_ATTR,
    {
        AudioPatcher.execute,
        AudioPatcher.prepare_normalized_unscaled_audio,
        SubtitleAlignment.align,
    },
)
