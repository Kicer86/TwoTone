import logging
import os
import shutil

from typing import Any, Iterable, NamedTuple, Sequence
from tqdm import tqdm

from ..utils import files_utils, generic_utils, language_utils, process_utils, video_utils
from .debug_routines import DebugRoutines
from .melt_cache import MeltCache
from .pair_matcher import MappingRelation, PairMatcher
from .melt_common import FramesInfo, StreamType, _ensure_working_dir, _is_length_mismatch


class _SegmentRange(NamedTuple):
    lhs_start: int
    lhs_end: int
    rhs_start: int
    rhs_end: int


class _PairMatchResult(NamedTuple):
    mapping: list[tuple[int, int]]
    lhs_all_frames: FramesInfo
    rhs_all_frames: FramesInfo
    mapping_relation: MappingRelation
    base_duration: int
    source_duration: int
    lhs_fps: float
    rhs_fps: float


class _StreamEntry(NamedTuple):
    stream_type: StreamType
    tid: int
    path: str
    language: str | None
    sync_offset_ms: int | None = None


class _PreparedStreams(NamedTuple):
    entries: list[_StreamEntry]
    input_files: set[str]


class MeltPerformer:
    def __init__(
        self,
        logger: logging.Logger,
        interruption: generic_utils.InterruptibleProcess,
        working_dir: str,
        output_dir: str,
        tolerance_ms: int,
        cache: MeltCache | None = None,
        fill_audio_gaps: bool = False,
    ) -> None:
        self.logger = logger
        self.interruption = interruption
        self.output_dir = output_dir
        self.tolerance_ms = tolerance_ms
        self.cache = cache
        self.fill_audio_gaps = fill_audio_gaps
        self._normalized_audio_cache: dict[tuple[str, int, int | None, int | None], str] = {}
        self._temporary_audio_counter = 0
        self._pair_match_cache: dict[tuple[str, str], _PairMatchResult] = {}
        self._aac_priming_exposed_cache: bool | None = None
        self.wd = _ensure_working_dir(working_dir)

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
                video_streams: list[tuple[str, int, str | None]] = streams_info.get("video", [])
                audio_streams: list[tuple[str, int, str | None]] = streams_info.get("audio", [])
                subtitle_streams: list[tuple[str, int, str | None]] = streams_info.get("subtitle", [])
                required_input_files = self._collect_required_input_files(
                    video_streams,
                    audio_streams,
                    subtitle_streams,
                    attachments,
                )

                output = self._build_output_path(title, output_name)
                if os.path.exists(output):
                    self.logger.info("Output file %s exists, removing it.", output)
                    os.remove(output)

                output_parent = os.path.dirname(output)
                os.makedirs(output_parent, exist_ok=True)

                if len(required_input_files) == 1:
                    # only one file is being used, just copy it to the output dir
                    first_file_path = list(required_input_files)[0]
                    self._copy_single_input(first_file_path, output)
                else:
                    # Convert streams to unified list (and patch audios if needed)
                    self._normalized_audio_cache.clear()
                    self._temporary_audio_counter = 0
                    self._pair_match_cache.clear()
                    files_details = group.get("files_details", {})
                    prepared_streams = self._prepare_stream_entries(
                        video_streams,
                        audio_streams,
                        subtitle_streams,
                        required_input_files,
                        attachments,
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
                        output,
                        streams_list_sorted,
                        attachments,
                        preferred_audio,
                        prepared_streams.input_files,
                    )

                    self.logger.info("Generating file: %s", self._display_path(output))

                    process_utils.raise_on_error(
                        process_utils.start_process("mkvmerge", generation_args, show_progress=True, logger=self.logger)
                    )

                    self.logger.info("%s saved.", output)

    def build_mkvmerge_args(
        self,
        output_path: str,
        streams_list_sorted: Sequence[_StreamEntry],
        attachments: Sequence[tuple[str, int]],
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
        base_video: str,
        source_video: str,
        output_path: str,
        segment_pairs: list[tuple[int, int]],
        *,
        use_silence: bool = False,
    ) -> int:
        """Replace audio using a single global time-scale for constant-offset cases.

        Instead of splitting into many subsegments and applying per-segment atempo,
        this trims and time-scales the source audio directly from the video file,
        avoiding full audio extraction.

        When *use_silence* is True, head/tail gaps are skipped entirely — the
        caller is expected to position the track via mkvmerge ``--sync``.

        Returns the effective sync offset (ms) for positioning the produced
        audio track on the base-video timeline.
        """

        wd = os.path.join(wd, "audio_extraction")
        os.makedirs(wd, exist_ok=True)

        # Compute global segment range (milliseconds)
        seg = self._segment_range(segment_pairs)
        seg1_start, seg1_end = seg.lhs_start, seg.lhs_end
        seg2_start, seg2_end = seg.rhs_start, seg.rhs_end

        # 1. Extract head/tail directly from base video, normalized to source params
        source_params = self._get_audio_params(source_video, logger=self.logger)
        base_duration_ms = video_utils.get_video_duration(base_video, logger=self.logger)
        has_head = seg1_start > 0 and not use_silence
        has_tail = seg1_end < base_duration_ms and not use_silence

        # Video-frame ratio (true fps relationship, independent of audio track).
        # Both spans come from matched video frames, so the ratio captures only the
        # real playback-rate relationship.  It must not be skewed by where the audio
        # stream happens to start relative to its video, otherwise a constant-offset
        # source whose audio merely leads its video gets a spurious global time-scale.
        source_dur = seg2_end - seg2_start
        target_dur = seg1_end - seg1_start
        video_ratio = target_dur / source_dur if source_dur else 1.0
        fps_ratio = source_dur / target_dur if target_dur else 1.0
        needs_scaling = self._needs_fps_scaling(fps_ratio)

        self.logger.info(
            "Audio patch (constant-offset): base=[%d…%d] ms, source=[%d…%d] ms, fps_ratio=%.4f",
            seg1_start, seg1_end, seg2_start, seg2_end, fps_ratio,
        )

        head_path = os.path.join(wd, "head.flac")
        tail_path = os.path.join(wd, "tail.flac")

        if has_head:
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-to", str(seg1_start / 1000),
                    "-i", base_video, "-map", "0:a:0",
                    *self._normalize_args(source_params),
                    "-c:a", "flac", head_path,
                ], logger=self.logger)
            )
        if has_tail:
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-ss", str(seg1_end / 1000),
                    "-i", base_video, "-map", "0:a:0",
                    *self._normalize_args(source_params),
                    "-c:a", "flac", tail_path,
                ], logger=self.logger)
            )

        # 2. Trim + time-scale source audio.
        #    The scaling ratio is derived from video-frame timestamps (fps
        #    relationship), NOT from measured audio duration.

        trimmed_audio = os.path.join(wd, "source_trimmed.flac")
        # Decode the source audio window, with lossy-codec encoder-delay priming
        # removed deterministically so the patched track is not shifted by one AAC
        # frame (~21 ms) on ffmpeg builds that decode the priming as real samples.
        self._decode_source_audio_to_flac(
            source_video,
            trimmed_audio,
            sample_fmt=self._flac_safe_fmt(source_params[2]),
            trim_start_ms=seg2_start,
            trim_end_ms=seg2_end,
            logger=self.logger,
        )

        actual_source_dur = video_utils.get_video_duration(trimmed_audio, logger=self.logger)
        expected_scaled_dur = round(actual_source_dur * video_ratio)

        # _patched_audio_start_ms already places the source by its priming-aware
        # content start: on builds that expose AAC priming _audio_content_start_ms
        # folds the encoder-delay frame into the offset to match the priming-exposed
        # decode, and absorbing builds omit it.  That makes the offset correct on both
        # conventions, so no extra per-container priming correction is applied here.
        sync_offset = self._patched_audio_start_ms(
            source_video,
            seg1_start,
            seg2_start,
            video_ratio,
        )
        start_correction = sync_offset - seg1_start
        if not use_silence and start_correction > 50:
            raise RuntimeError(
                f"Audio start correction of {start_correction} ms detected "
                f"in fill-audio-gaps mode. "
                f"The source container's audio starts later than its video, "
                f"which cannot be compensated when head/tail are filled from "
                f"the base file. Use default (silence) mode instead."
            )

        # Scale with VIDEO-frame ratio (true fps relationship)
        scaled_audio = os.path.join(wd, "source_scaled.flac")
        if not needs_scaling:
            scaled_audio = trimmed_audio
        else:
            sample_rate = source_params[1]
            adjusted_rate = sample_rate * fps_ratio
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-i", trimmed_audio,
                    "-filter:a", f"asetrate={adjusted_rate:.6f},aresample={sample_rate}",
                    "-sample_fmt", "s32", "-c:a", "flac",
                    scaled_audio,
                ], logger=self.logger)
            )

        scaled_dur = video_utils.get_video_duration(scaled_audio, logger=self.logger)
        self._validate_audio_duration(scaled_dur, expected_scaled_dur, "scaled audio")

        # 3. Concatenate and encode to AAC
        channel_layout = self._get_audio_channel_layout(source_video, logger=self.logger)
        self._concat_and_encode(
            [scaled_audio], has_head, head_path, has_tail, tail_path,
            os.path.join(wd, "concat.txt"), output_path,
            channel_layout=channel_layout,
            logger=self.logger,
        )

        return sync_offset

    @staticmethod
    def _collect_required_input_files(
        video_streams: Sequence[tuple[str, int, str | None]],
        audio_streams: Sequence[tuple[str, int, str | None]],
        subtitle_streams: Sequence[tuple[str, int, str | None]],
        attachments: Sequence[tuple[str, int]],
    ) -> set[str]:
        required_input_files: set[str] = set()
        required_input_files |= {p for (p, _, _) in video_streams}
        required_input_files |= {p for (p, _, _) in audio_streams}
        required_input_files |= {p for (p, _, _) in subtitle_streams}
        required_input_files |= {info[0] for info in attachments}
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
        summary: dict[str, Any],
    ) -> list[str]:
        """Build ASCII overlap diagram lines for two partially-overlapping files.

        All positions are projected onto the lhs (base) timeline so that speed
        differences between files don't distort the visual overlap.
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
        """Log a human-readable coverage report after PairMatcher finishes."""
        summary = PairMatcher.coverage_summary(
            mappings,
            lhs_duration_ms,
            rhs_duration_ms,
            lhs_fps=lhs_fps,
            rhs_fps=rhs_fps,
        )
        lhs_name = os.path.basename(lhs_path)
        rhs_name = os.path.basename(rhs_path)

        if summary["full_coverage"]:
            ratio = summary["ratio"]
            if abs(ratio - 1.0) < 0.001:
                self.logger.info(
                    "Files are 100%% visually identical: %s ↔ %s",
                    lhs_name, rhs_name,
                )
            else:
                self.logger.info(
                    "Files are 100%% visually identical (speed ratio %.4f): %s ↔ %s",
                    ratio, lhs_name, rhs_name,
                )
        else:
            parts: list[str] = []
            lhs_sg = summary["lhs_start_gap_s"]
            rhs_sg = summary["rhs_start_gap_s"]
            lhs_eg = summary["lhs_end_gap_s"]
            rhs_eg = summary["rhs_end_gap_s"]

            if lhs_sg > 0.04 or rhs_sg > 0.04:
                parts.append(
                    f"shared content starts at {lhs_sg:.1f}s in #{lhs_id} "
                    f"and {rhs_sg:.1f}s in #{rhs_id}"
                )
            if lhs_eg > 0.04 or rhs_eg > 0.04:
                parts.append(
                    f"shared content ends {lhs_eg:.1f}s before end of #{lhs_id} "
                    f"and {rhs_eg:.1f}s before end of #{rhs_id}"
                )

            detail = "; ".join(parts) if parts else "partial overlap"
            self.logger.info(
                "Files are NOT fully identical — %s",
                detail,
            )

            diagram = self._render_overlap_diagram(
                lhs_id, rhs_id, lhs_duration_ms, rhs_duration_ms, summary,
            )
            if diagram:
                self.logger.info("Overlap diagram:\n%s", "\n".join(diagram))

    @classmethod
    def _source_audio_priming(
        cls,
        video_path: str,
        logger: logging.Logger | None = None,
        *,
        audio_stream_index: int | None = None,
    ) -> tuple[str | None, int, int, dict[str, Any]]:
        """Return (codec_name, initial_padding_samples, sample_rate, stream).

        ``initial_padding`` is the lossy-codec encoder-delay (priming) sample count the
        container signals via Matroska *CodecDelay*; it is 0 for MP4/MOV (where the same
        delay is carried by an edit list that every ffmpeg build honours on decode).
        ``audio_stream_index`` selects a specific stream by its absolute container
        index; when omitted the first audio stream is used.
        """
        info = video_utils.get_video_full_info(video_path, logger=logger)
        streams = info.get("streams", [])
        stream: dict[str, Any]
        if audio_stream_index is not None:
            stream = next(
                (s for s in streams
                 if s.get("codec_type") == "audio" and s.get("index") == audio_stream_index),
                {},
            )
        else:
            stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
        codec = stream.get("codec_name")
        try:
            init_pad = int(stream.get("initial_padding") or 0)
        except (TypeError, ValueError):
            init_pad = 0
        try:
            sample_rate = int(stream.get("sample_rate") or 0)
        except (TypeError, ValueError):
            sample_rate = 0
        return codec, init_pad, sample_rate, stream

    def _audio_starts_just_before_video_adjustment_ms(self, path: str) -> int:
        info = video_utils.get_video_full_info(path)
        video_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
        video_start_ms = self._stream_start_offset_ms(video_stream)
        audio_start_ms = self._audio_content_start_ms(audio_stream)
        delta_ms = video_start_ms - audio_start_ms
        if 0 < delta_ms <= 50:
            return delta_ms
        return 0

    def _decode_source_audio_to_flac(
        self,
        source_video: str,
        output_path: str,
        *,
        sample_fmt: str = "s32",
        trim_start_ms: int | None = None,
        trim_end_ms: int | None = None,
        audio_stream_index: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Decode the first audio stream of *source_video* to FLAC with the lossy-codec
        encoder-delay priming removed deterministically across all ffmpeg builds.

        AAC and similar lapped-transform codecs prepend ``initial_padding`` priming
        samples that must be skipped on decode.  Some ffmpeg builds skip them
        automatically (honouring Matroska *CodecDelay*), others decode them as real
        leading samples — shifting the track by ~21 ms (one AAC frame).  To get
        identical output everywhere, when the source carries CodecDelay priming we
        strip that signalling by remuxing the AAC bitstream to ADTS (which has no
        priming metadata), decode the now-always-present priming, and trim exactly
        ``initial_padding`` samples ourselves.

        ``trim_start_ms`` / ``trim_end_ms`` select a window of the *priming-free*
        stream, matching what a priming-honouring decoder would have produced.  These
        bounds are expressed on the source's content timeline.  When the priming strip
        drops the container timestamps, re-anchor the window to the priming-independent
        content start so exposing and absorbing ffmpeg builds select identical samples.
        """
        codec, init_pad, sample_rate, stream = self._source_audio_priming(
            source_video, logger=logger, audio_stream_index=audio_stream_index,
        )
        source_map = f"0:{audio_stream_index}" if audio_stream_index is not None else "0:a:0"

        prime_offset_s = 0.0
        window_anchor_ms = 0
        input_path = source_video
        input_map = source_map
        adts_path: str | None = None
        if codec == "aac" and init_pad > 0 and sample_rate > 0:
            adts_path = f"{output_path}.priming.aac"
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-i", source_video, "-map", source_map,
                    "-c:a", "copy", "-f", "adts", adts_path,
                ], logger=logger)
            )
            input_path = adts_path
            input_map = "0:a:0"  # the remuxed ADTS file carries only the selected stream
            prime_offset_s = init_pad / sample_rate
            # ADTS carries no container timestamps, so the decoded frames start at 0
            # instead of the source's content start.  The raw start_time on builds
            # exposing AAC priming is one encoder-delay frame too early, so use the
            # same priming-independent content start as final track placement.
            window_anchor_ms = self._audio_content_start_ms(stream)

        filters: list[str] = []
        if prime_offset_s or trim_start_ms is not None or trim_end_ms is not None:
            start_s = prime_offset_s + max(0, (trim_start_ms or 0) - window_anchor_ms) / 1000
            atrim = f"atrim=start={start_s:.6f}"
            if trim_end_ms is not None:
                end_s = prime_offset_s + max(0, trim_end_ms - window_anchor_ms) / 1000
                atrim += f":end={end_s:.6f}"
            filters.append(atrim)
            filters.append("asetpts=PTS-STARTPTS")

        args = ["-y", "-i", input_path, "-map", input_map]
        if filters:
            args += ["-filter:a", ",".join(filters)]
        args += ["-sample_fmt", sample_fmt, "-c:a", "flac", output_path]
        try:
            process_utils.raise_on_error(process_utils.start_process("ffmpeg", args, logger=logger))
        finally:
            if adts_path and os.path.exists(adts_path):
                os.remove(adts_path)

    def _extract_audio_to_flac(self, video_path: str, output_path: str, logger: logging.Logger | None = None) -> None:
        self._decode_source_audio_to_flac(video_path, output_path, sample_fmt="s32", logger=logger)

    @staticmethod
    def _segment_range(pairs: Sequence[tuple[int, int]]) -> _SegmentRange:
        """Return the bounding lhs/rhs range from a list of (lhs, rhs) pairs."""
        left, right = zip(*pairs)
        return _SegmentRange(min(left), max(left), min(right), max(right))

    @staticmethod
    def _get_audio_params(audio_path: str, logger: logging.Logger | None = None) -> tuple[int, int, str]:
        """Return (channels, sample_rate, sample_fmt) of the first audio stream."""
        info = video_utils.get_video_full_info(audio_path, logger=logger)
        stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
        return int(stream["channels"]), int(stream["sample_rate"]), stream["sample_fmt"]

    @staticmethod
    def _get_audio_channel_layout(audio_path: str, logger: logging.Logger | None = None) -> str | None:
        """Return the channel layout string of the first audio stream, or None."""
        info = video_utils.get_video_full_info(audio_path, logger=logger)
        stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
        return stream.get("channel_layout") or None

    def _aac_priming_exposed(self) -> bool:
        """Whether this ffmpeg build exposes AAC encoder-delay priming as a negative
        container start time instead of absorbing it.

        Builds disagree: some report a freshly encoded AAC stream's ``start_time`` as
        ``-encoder_delay`` (priming exposed, kept in the decoded samples), others as 0
        (priming absorbed).  The same convention applies when *reading* a source AAC
        stream, so on exposing builds a positive-offset (asO) audio stream's reported
        ``start_time`` is short by one frame (~21 ms) — which otherwise mis-places the
        patched track by exactly that frame.  Detected once and cached per run.
        """
        if self._aac_priming_exposed_cache is None:
            exposed = False
            probe = os.path.join(self.wd, f"_priming_probe_{os.getpid()}.mka")
            try:
                process_utils.raise_on_error(
                    process_utils.start_process("ffmpeg", [
                        "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                        "-t", "0.5", "-c:a", "aac", probe,
                    ], logger=self.logger)
                )
                info = video_utils.get_video_full_info(probe, logger=self.logger)
                stream: dict[str, Any] = next(
                    (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
                    {},
                )
                exposed = float(stream.get("start_time") or 0.0) < -0.001
            except Exception:  # noqa: BLE001 - detection failure falls back to "absorbed"
                exposed = False
            finally:
                if os.path.exists(probe):
                    os.remove(probe)
            self._aac_priming_exposed_cache = exposed
        return self._aac_priming_exposed_cache

    def _audio_content_start_ms(self, stream: dict[str, Any] | None) -> int:
        """Audio stream's true content start (ms), priming-convention independent.

        On builds that expose AAC priming, the reported ``start_time`` is short by the
        encoder delay; add it back so positive-offset placement matches absorbing
        builds.  ``_stream_start_offset_ms`` clamps negatives to 0, so asR streams are
        unaffected; only positive (asO) offsets carry the leaked frame.
        """
        base = self._stream_start_offset_ms(stream)
        if not stream or stream.get("codec_name") != "aac" or not self._aac_priming_exposed():
            return base
        try:
            pad = int(stream.get("initial_padding") or 0)
            sample_rate = int(stream.get("sample_rate") or 0)
            raw_start = float(stream.get("start_time") or 0.0)
        except (TypeError, ValueError):
            return base
        if pad <= 0 or sample_rate <= 0:
            return base
        return max(0, round((raw_start + pad / sample_rate) * 1000))

    def _patched_audio_start_ms(
        self,
        source_video: str,
        seg1_start: int,
        seg2_start: int,
        video_ratio: float,
    ) -> int:
        """Return timeline start for a decoded source-audio trim."""
        info = video_utils.get_video_full_info(source_video)
        video_stream = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
        video_start_ms = self._stream_start_offset_ms(video_stream)
        audio_start_ms = self._audio_content_start_ms(audio_stream)

        audio_start_correction = 0
        if audio_start_ms > video_start_ms and audio_start_ms > seg2_start:
            audio_start_correction = round((audio_start_ms - seg2_start) * video_ratio)

        try:
            container_start = float(info.get("format", {}).get("start_time") or 0.0)
        except (TypeError, ValueError):
            container_start = 0.0

        positive_timeline_correction = 0
        timeline_start_ms = max(video_start_ms, audio_start_ms)
        if container_start > 0 and timeline_start_ms > seg2_start:
            positive_timeline_correction = max(0, round(timeline_start_ms * video_ratio) - seg1_start)

        correction = max(audio_start_correction, positive_timeline_correction)
        if correction > 50:
            self.logger.info(
                "Audio trim start correction: %d ms → sync offset: %d ms",
                correction,
                seg1_start + correction,
            )
        return seg1_start + correction

    _FPS_RATIO_TOLERANCE = 0.001

    @staticmethod
    def _needs_fps_scaling(ratio: float) -> bool:
        """Return True when *ratio* deviates enough from 1.0 to need asetrate scaling."""
        return abs(ratio - 1.0) >= MeltPerformer._FPS_RATIO_TOLERANCE

    _MAX_DURATION_DEVIATION = 0.05  # 5%

    def _validate_audio_duration(self, actual_ms: int, expected_ms: int, label: str) -> None:
        """Raise if *actual_ms* deviates from *expected_ms* by more than 5%."""
        if expected_ms == 0:
            return
        deviation = abs(actual_ms - expected_ms) / expected_ms
        if deviation > self._MAX_DURATION_DEVIATION:
            raise RuntimeError(
                f"Audio duration mismatch in {label}: "
                f"got {actual_ms} ms, expected {expected_ms} ms "
                f"(deviation: {deviation:.1%}, max allowed: {self._MAX_DURATION_DEVIATION:.0%})"
            )

    @staticmethod
    def _flac_safe_fmt(sample_fmt: str) -> str:
        """Return a FLAC-compatible sample format (FLAC does not support float formats)."""
        base = sample_fmt.removesuffix("p")
        if base in ("flt", "dbl"):
            return "s32"
        return sample_fmt

    @staticmethod
    def _normalize_args(params: tuple[int, int, str]) -> list[str]:
        """Return ffmpeg args that re-encode audio to match *params* (channels, sample_rate, sample_fmt)."""
        channels, sample_rate, sample_fmt = params
        sample_fmt = MeltPerformer._flac_safe_fmt(sample_fmt)
        return ["-ac", str(channels), "-ar", str(sample_rate), "-sample_fmt", sample_fmt]

    @staticmethod
    def _extract_head_tail(
        base_audio: str,
        base_video: str,
        seg_start_ms: int,
        seg_end_ms: int,
        head_path: str,
        tail_path: str,
        normalize_to: tuple[int, int, str] | None = None,
        logger: logging.Logger | None = None,
    ) -> tuple[bool, bool]:
        """Extract head/tail segments from the base audio track.

        When *normalize_to* is given as (channels, sample_rate, sample_fmt),
        head and tail are re-encoded to match those parameters so that FLAC
        concatenation with the main segment works without parameter mismatches.

        Returns (has_head, has_tail) indicating which parts were extracted.
        """
        norm_args: list[str] = []
        if normalize_to:
            norm_args = MeltPerformer._normalize_args(normalize_to)
        else:
            norm_args = ["-sample_fmt", "s32"]

        has_head = seg_start_ms > 0
        if has_head:
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-ss", "0", "-to", str(seg_start_ms / 1000),
                    "-i", base_audio, *norm_args, "-c:a", "flac", head_path,
                ], logger=logger)
            )

        base_duration_ms = video_utils.get_video_duration(base_video, logger=logger)
        has_tail = seg_end_ms < base_duration_ms
        if has_tail:
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-ss", str(seg_end_ms / 1000),
                    "-i", base_audio, *norm_args, "-c:a", "flac", tail_path,
                ], logger=logger)
            )

        return has_head, has_tail

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
        base_video: str,
        source_video: str,
        output_path: str,
        segment_pairs: list[tuple[int, int]],
        segment_count: int,
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo,
        min_subsegment_duration: float = 30.0,
        *,
        use_silence: bool = False,
    ) -> None:
        """Replace an audio segment in the base video with time-adjusted audio from another video.

        The replacement is split into smaller, corresponding subsegments to better handle drift.
        When *use_silence* is True, head/tail gaps are skipped entirely — the
        caller is expected to position the track via mkvmerge ``--sync``.
        """

        wd = os.path.join(wd, "audio_extraction")
        debug_wd = os.path.join(wd, "debug")
        os.makedirs(wd, exist_ok=True)
        os.makedirs(debug_wd, exist_ok=True)

        v2_audio = os.path.join(wd, "v2_audio.flac")
        head_path = os.path.join(wd, "head.flac")
        tail_path = os.path.join(wd, "tail.flac")

        debug = DebugRoutines(debug_wd, lhs_frames, rhs_frames)

        # Compute global segment range (milliseconds)
        seg = self._segment_range(segment_pairs)
        seg1_start, seg1_end = seg.lhs_start, seg.lhs_end

        # 1. Extract audio tracks
        if not use_silence:
            v1_audio = os.path.join(wd, "v1_audio.flac")
            self._extract_audio_to_flac(base_video, v1_audio, logger=self.logger)
        self._extract_audio_to_flac(source_video, v2_audio, logger=self.logger)

        source_params = self._get_audio_params(v2_audio, logger=self.logger)

        # 2. Extract head/tail from base audio (skipped when use_silence — caller uses --sync)
        if use_silence:
            has_head = False
            has_tail = False
        else:
            has_head, has_tail = self._extract_head_tail(
                v1_audio, base_video, seg1_start, seg1_end, head_path, tail_path,
                normalize_to=source_params,
                logger=self.logger,
            )

        # 3. Generate subsegment split points from provided mapping pairs
        total_left_duration = seg1_end - seg1_start
        left_targets = [seg1_start + i * total_left_duration // segment_count for i in range(segment_count + 1)]

        def closest_pair(value: int, pairs: Sequence[tuple[int, int]]) -> tuple[int, int]:
            return min(pairs, key=lambda p: abs(p[0] - value))

        selected_pairs = [closest_pair(t, segment_pairs) for t in left_targets]

        # Merge short segments with a neighbor
        cleaned_pairs: list[tuple[int, int, int, int]] = []
        i = 0
        while i < len(selected_pairs) - 1:
            l_start = selected_pairs[i][0]
            l_end = selected_pairs[i + 1][0]
            r_start = selected_pairs[i][1]
            r_end = selected_pairs[i + 1][1]

            l_dur = l_end - l_start
            r_dur = r_end - r_start

            if l_dur < min_subsegment_duration * 1000 or r_dur < min_subsegment_duration * 1000:
                if i + 2 < len(selected_pairs):
                    selected_pairs[i + 1] = selected_pairs[i + 2]
                    del selected_pairs[i + 2]
                    continue
                if i > 0:
                    prev = cleaned_pairs[-1]
                    cleaned_pairs[-1] = (prev[0], l_end, prev[2], r_end)
                    i += 1
                    continue

            cleaned_pairs.append((l_start, l_end, r_start, r_end))
            i += 1

        debug.dump_pairs(cleaned_pairs)

        # 4. Extract, time-scale and collect replacement parts
        temp_segments: list[str] = []
        for idx, (l_start, l_end, r_start, r_end) in enumerate(cleaned_pairs):
            left_duration = l_end - l_start
            right_duration = r_end - r_start
            ratio = right_duration / left_duration if left_duration else 1.0

            if abs(ratio - 1.0) > 0.10:
                self.logger.error("Segment %d duration mismatch exceeds 10%%", idx)

            raw_cut = os.path.join(wd, f"cut_{idx}.flac")
            scaled_cut = os.path.join(wd, f"scaled_{idx}.flac")

            process_utils.raise_on_error(
                process_utils.start_process(
                    "ffmpeg", [
                        "-y",
                        "-ss", str(r_start / 1000),
                        "-to", str(r_end / 1000),
                        "-i", v2_audio,
                        "-c:a", "flac",
                        raw_cut,
                    ],
                    logger=self.logger,
                )
            )

            process_utils.raise_on_error(
                process_utils.start_process(
                    "ffmpeg", [
                        "-y",
                        "-i", raw_cut,
                        "-filter:a", f"atempo={ratio:.3f}",
                        "-sample_fmt", "s32", "-c:a", "flac",
                        scaled_cut,
                    ],
                    logger=self.logger,
                )
            )

            temp_segments.append(scaled_cut)

        # 5. Concatenate and encode
        channel_layout = self._get_audio_channel_layout(source_video, logger=self.logger)
        self._concat_and_encode(
            temp_segments, has_head, head_path, has_tail, tail_path,
            os.path.join(wd, "concat.txt"), output_path,
            channel_layout=channel_layout,
            logger=self.logger,
        )

    def _build_output_path(self, title: str, output_name: str) -> str:
        return os.path.join(self.output_dir, title, output_name + ".mkv")

    def _display_path(self, path: str) -> str:
        return files_utils.format_path(path, self.output_dir)

    def _copy_single_input(self, input_path: str, output_path: str) -> None:
        self.logger.info(
            "File %s is superior. Using it whole as output %s.",
            self._display_path(input_path), self._display_path(output_path),
        )
        shutil.copy2(input_path, output_path)

    def _patch_mismatched_audio(
        self,
        video_path_base: str,
        audio_stream: tuple[str, int],
        base_duration: int,
        base_audio_end: int | None,
        file_ids: dict[str, int],
    ) -> tuple[str, int, int | None]:
        """Run PairMatcher and apply the appropriate audio patching strategy.

        Returns (patched_path, new_stream_index, desired_start_ms).
        """
        audio_path, stream_index = audio_stream
        with files_utils.ScopedDirectory(os.path.join(self.wd, "matching")) as mwd, \
             generic_utils.TqdmBouncingBar(desc="Processing", **generic_utils.get_tqdm_defaults()):
            lhs_id = file_ids[video_path_base]
            rhs_id = file_ids[audio_path]
            cache_key = (video_path_base, audio_path)
            match_result = self._pair_match_cache.get(cache_key)

            if match_result is None:
                duration = video_utils.get_video_duration(audio_path)
                matcher = PairMatcher(
                    self.interruption, mwd, video_path_base, audio_path,
                    self.logger.getChild("PairMatcher"),
                    lhs_label=f"#{lhs_id}", rhs_label=f"#{rhs_id}",
                    cache=self.cache,
                )
                mapping_result = matcher.create_segments_mapping()
                match_result = _PairMatchResult(
                    mapping=mapping_result.mapping,
                    lhs_all_frames=mapping_result.lhs_all_frames,
                    rhs_all_frames=mapping_result.rhs_all_frames,
                    mapping_relation=mapping_result.relation,
                    base_duration=base_duration,
                    source_duration=duration,
                    lhs_fps=matcher.lhs_fps,
                    rhs_fps=matcher.rhs_fps,
                )
                self._pair_match_cache[cache_key] = match_result
            else:
                self.logger.info(
                    "Reusing PairMatcher results for #%d ↔ #%d (same video pair).",
                    lhs_id, rhs_id,
                )

            mapping = match_result.mapping
            lhs_all_frames = match_result.lhs_all_frames
            rhs_all_frames = match_result.rhs_all_frames
            mapping_relation = match_result.mapping_relation
            duration = match_result.source_duration

            if (
                mapping_relation == MappingRelation.LINEAR_FRAME_DRIFT
                or (
                    mapping_relation == MappingRelation.GENERIC
                    and self._is_frame_rate_only_drift_mapping(
                        video_path_base,
                        audio_path,
                        mapping,
                        lhs_all_frames,
                        rhs_all_frames,
                        match_result.lhs_fps,
                        match_result.rhs_fps,
                    )
                )
            ):
                self.logger.info("  Audio strategy: drift_passthrough")
                desired_start_ms = self._audio_content_start_ms(
                    self._stream_info(audio_path, "audio", stream_index)
                )
                patched_audio, patched_index = self._prepare_passthrough_audio(
                    audio_path,
                    stream_index,
                    desired_end_ms=base_audio_end if base_audio_end is not None else base_duration,
                    desired_start_ms=desired_start_ms,
                )
                self.logger.info("  Desired audio start: %d ms", desired_start_ms)
                return patched_audio, patched_index, desired_start_ms

            mapping = self._strict_audio_mapping(mapping)

            extrapolated_mapping = None
            if mapping_relation == MappingRelation.GENERIC:
                extrapolated_mapping = self._try_effective_fps_audio_mapping(
                    video_path_base,
                    audio_path,
                    mapping,
                    lhs_all_frames,
                    rhs_all_frames,
                    match_result.lhs_fps,
                    match_result.rhs_fps,
                )
            if extrapolated_mapping is None:
                extrapolated_mapping = self._try_sparse_linear_audio_extrapolation(
                    mapping,
                    lhs_all_frames,
                    rhs_all_frames,
                    match_result.lhs_fps,
                    match_result.rhs_fps,
                )
            if extrapolated_mapping is not None:
                mapping = extrapolated_mapping
                mapping_relation = MappingRelation.LINEAR_FRAME_DRIFT
                self.logger.info(
                    "  Sparse linear audio extrapolation: using %s-%s ↔ %s-%s",
                    generic_utils.ms_to_time(mapping[0][0]),
                    generic_utils.ms_to_time(mapping[-1][0]),
                    generic_utils.ms_to_time(mapping[0][1]),
                    generic_utils.ms_to_time(mapping[-1][1]),
                )

            self._log_coverage(
                video_path_base,
                audio_path,
                mapping,
                base_duration,
                duration,
                match_result.lhs_fps,
                match_result.rhs_fps,
                lhs_id,
                rhs_id,
            )

            self.logger.debug(
                "Audio patching: base_duration=%d ms, source_duration=%d ms, "
                "mapping_relation=%s, lhs_fps=%.3f, rhs_fps=%.3f, mapping_pairs=%d",
                base_duration, duration, mapping_relation.value,
                match_result.lhs_fps, match_result.rhs_fps, len(mapping),
            )
            if mapping:
                self.logger.debug(
                    "  Mapping range: lhs=[%d … %d] ms, rhs=[%d … %d] ms",
                    mapping[0][0], mapping[-1][0], mapping[0][1], mapping[-1][1],
                )

            use_silence = not self.fill_audio_gaps
            use_constant_offset = mapping_relation == MappingRelation.CONSTANT_FRAME_OFFSET
            self.logger.info("  Audio strategy: %s", "constant_offset" if use_constant_offset else "subsegment")

            patched_audio = os.path.join(self.wd, f"tmp_{os.getpid()}_{stream_index}.mka")
            if use_constant_offset:
                effective_sync = self.patch_audio_constant_offset(mwd, video_path_base, audio_path, patched_audio, mapping, use_silence=use_silence)
            else:
                self._patch_audio_segment(mwd, video_path_base, audio_path, patched_audio, mapping, 20, lhs_all_frames, rhs_all_frames, use_silence=use_silence)
                effective_sync = min(p[0] for p in mapping)

            if use_silence:
                desired_start_ms = effective_sync
                self.logger.info("  Desired audio start: %d ms", desired_start_ms)
                return patched_audio, 0, desired_start_ms

        return patched_audio, 0, None

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
    def _try_effective_fps_audio_mapping(
        base_video: str,
        source_video: str,
        mapping: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
        lhs_nominal_fps: float | None = None,
        rhs_nominal_fps: float | None = None,
    ) -> list[tuple[int, int]] | None:
        """Build a linear audio mapping from effective video FPS when matches are generic."""
        if len(mapping) < 2:
            return None

        lhs_effective_fps = MeltPerformer._effective_video_fps(base_video, lhs_all_frames)
        rhs_effective_fps = MeltPerformer._effective_video_fps(source_video, rhs_all_frames)
        if lhs_effective_fps is None or rhs_effective_fps is None or rhs_effective_fps <= 0:
            return None

        tempo_ratio = lhs_effective_fps / rhs_effective_fps
        if abs(tempo_ratio - 1.0) < 0.005:
            return None

        seg = MeltPerformer._segment_range(mapping)
        target_duration = seg.lhs_end - seg.lhs_start
        if target_duration <= 0:
            return None

        if MeltPerformer._is_frame_rate_only_drift_mapping(
            base_video,
            source_video,
            mapping,
            lhs_all_frames,
            rhs_all_frames,
            lhs_nominal_fps,
            rhs_nominal_fps,
        ):
            return None

        lhs_frame_duration = max(lhs_all_frames) - min(lhs_all_frames)
        if lhs_frame_duration > 0 and target_duration < lhs_frame_duration * 0.80:
            return None

        source_duration = round(target_duration * tempo_ratio)
        if source_duration <= 0:
            return None

        return [
            (seg.lhs_start, seg.rhs_start),
            (seg.lhs_end, seg.rhs_start + source_duration),
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

        lhs_effective_fps = MeltPerformer._effective_video_fps(base_video, lhs_all_frames)
        rhs_effective_fps = MeltPerformer._effective_video_fps(source_video, rhs_all_frames)
        if lhs_effective_fps is None or rhs_effective_fps is None or rhs_effective_fps <= 0:
            return False

        tempo_ratio = lhs_effective_fps / rhs_effective_fps
        nominal_ratio = lhs_nominal_fps / rhs_nominal_fps
        if abs(tempo_ratio - 1.0) < 0.005:
            return False
        if abs(tempo_ratio / nominal_ratio - 1.0) >= 0.005:
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
        return abs(observed_ratio - 1.0) < 0.01

    @staticmethod
    def _try_sparse_linear_audio_extrapolation(
        mapping: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
        lhs_fps: float,
        rhs_fps: float,
    ) -> list[tuple[int, int]] | None:
        """Extrapolate sparse but linear mappings for global audio time-scaling."""
        if len(mapping) < 3 or not lhs_all_frames or not rhs_all_frames:
            return None

        pairs = sorted(mapping)
        first_lhs, first_rhs = pairs[0]
        last_lhs, last_rhs = pairs[-1]
        lhs_span = last_lhs - first_lhs
        if lhs_span <= 0:
            return None

        rhs_per_lhs = (last_rhs - first_rhs) / lhs_span
        if rhs_per_lhs <= 0:
            return None

        max_residual_ms = 1000
        for lhs_time, rhs_time in pairs[1:-1]:
            predicted_rhs = first_rhs + (lhs_time - first_lhs) * rhs_per_lhs
            if abs(predicted_rhs - rhs_time) > max_residual_ms:
                return None

        lhs_start = min(lhs_all_frames)
        lhs_end = max(lhs_all_frames)
        rhs_start = min(rhs_all_frames)
        rhs_end = max(rhs_all_frames)
        actual_rhs_start = rhs_start
        actual_rhs_end = rhs_end
        lhs_effective_fps = (
            (len(lhs_all_frames) - 1) * 1000 / (lhs_end - lhs_start)
            if len(lhs_all_frames) > 1 and lhs_end > lhs_start
            else lhs_fps
        )
        rhs_effective_fps = (
            (len(rhs_all_frames) - 1) * 1000 / (rhs_end - rhs_start)
            if len(rhs_all_frames) > 1 and rhs_end > rhs_start
            else rhs_fps
        )
        if lhs_effective_fps > 0 and rhs_effective_fps > 0:
            fps_projected_rhs_end = rhs_start + round((lhs_end - lhs_start) * lhs_effective_fps / rhs_effective_fps)
            rhs_end = max(rhs_end, fps_projected_rhs_end)

        predicted_rhs_start = round(first_rhs + (lhs_start - first_lhs) * rhs_per_lhs)
        extrapolated_rhs_start = max(rhs_start, min(rhs_end, predicted_rhs_start))
        if lhs_effective_fps > 0 and rhs_effective_fps > 0:
            extrapolated_rhs_end = extrapolated_rhs_start + round(
                (lhs_end - lhs_start) * lhs_effective_fps / rhs_effective_fps
            )
        else:
            predicted_rhs_end = round(first_rhs + (lhs_end - first_lhs) * rhs_per_lhs)
            extrapolated_rhs_end = max(rhs_start, min(rhs_end, predicted_rhs_end))

        if extrapolated_rhs_end <= extrapolated_rhs_start:
            return None

        result = list(pairs)
        if lhs_start < first_lhs and actual_rhs_start < first_rhs and extrapolated_rhs_start < first_rhs:
            result.insert(0, (lhs_start, extrapolated_rhs_start))
        if lhs_end > last_lhs and actual_rhs_end > last_rhs and extrapolated_rhs_end > last_rhs:
            result.append((lhs_end, extrapolated_rhs_end))

        if len(result) == len(pairs):
            return None

        result = MeltPerformer._strict_audio_mapping(result)
        start_extended = result[0] != pairs[0]
        end_extended = result[-1] != pairs[-1]
        if not start_extended and not end_extended:
            return None

        return result

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

    _MKVMERGE_PRESERVES_START_EXTENSIONS = frozenset({".mkv", ".mk3d", ".mka", ".webm"})

    @staticmethod
    def _stream_info(path: str, stream_type: str, stream_index: int) -> dict[str, Any] | None:
        info = video_utils.get_video_full_info(path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") != stream_type:
                continue
            try:
                probed_index = int(stream.get("index", -1))
            except (TypeError, ValueError):
                continue
            if probed_index == stream_index:
                return stream
        return None

    @staticmethod
    def _audio_stream_info(path: str, stream_index: int) -> dict[str, Any] | None:
        return MeltPerformer._stream_info(path, "audio", stream_index)

    @staticmethod
    def _stream_start_offset_ms(stream: dict[str, Any] | None) -> int:
        if stream is None:
            return 0
        try:
            start_time = float(stream.get("start_time") or 0.0)
        except (TypeError, ValueError):
            return 0
        return max(0, round(start_time * 1000))

    @staticmethod
    def _stream_end_offset_ms(stream: dict[str, Any] | None) -> int | None:
        if stream is None:
            return None

        duration = stream.get("duration")
        if duration is not None:
            try:
                return MeltPerformer._stream_start_offset_ms(stream) + round(float(duration) * 1000)
            except (TypeError, ValueError):
                pass

        tag_duration = stream.get("tags", {}).get("DURATION")
        if tag_duration is not None:
            return generic_utils.time_to_ms(tag_duration)

        return None

    def _source_stream_start_offset_ms(self, path: str, stream_type: str, stream_index: int) -> int:
        return self._stream_start_offset_ms(self._stream_info(path, stream_type, stream_index))

    def _source_stream_end_offset_ms(self, path: str, stream_type: str, stream_index: int) -> int | None:
        return self._stream_end_offset_ms(self._stream_info(path, stream_type, stream_index))

    def _mkvmerge_input_start_offset_ms(self, path: str, stream_type: str, stream_index: int) -> int:
        extension = os.path.splitext(path)[1].lower()
        if extension in self._MKVMERGE_PRESERVES_START_EXTENSIONS:
            return self._source_stream_start_offset_ms(path, stream_type, stream_index)
        return 0

    def _track_sync_offset_ms(
        self,
        path: str,
        stream_type: str,
        stream_index: int,
        desired_start_ms: int | None,
    ) -> int | None:
        if desired_start_ms is None:
            return None
        input_start_ms = self._mkvmerge_input_start_offset_ms(path, stream_type, stream_index)
        sync_offset_ms = desired_start_ms - input_start_ms
        return sync_offset_ms if sync_offset_ms else None

    def _audio_needs_mkvmerge_normalization(self, path: str, stream_index: int) -> bool:
        """Return True when direct mkvmerge remux can shift decoded AAC timing.

        Non-Matroska AAC always needs a priming-free remux: mkvmerge would otherwise
        drop the container start offset.  Matroska normally preserves start, so raw
        passthrough is fine -- *except* when the AAC stream carries encoder-delay
        priming (CodecDelay / ``initial_padding``).  Remuxed raw, such a stream is
        placed build-dependently: builds that expose priming shift its decoded
        content by one frame (~21 ms) relative to a priming-stripped base track.
        Route those through the FLAC-domain flow too so the priming is removed
        deterministically, exactly as for non-Matroska inputs.
        """
        stream = self._audio_stream_info(path, stream_index)
        if stream is None or stream.get("codec_name") != "aac":
            return False
        extension = os.path.splitext(path)[1].lower()
        if extension not in self._MKVMERGE_PRESERVES_START_EXTENSIONS:
            return True
        try:
            init_pad = int(stream.get("initial_padding") or 0)
        except (TypeError, ValueError):
            init_pad = 0
        return init_pad > 0

    def _temporary_audio_path(self, label: str, stream_index: int) -> str:
        path = os.path.join(
            self.wd,
            f"tmp_{os.getpid()}_{label}_{self._temporary_audio_counter}_{stream_index}.mka",
        )
        self._temporary_audio_counter += 1
        return path

    def _prepare_passthrough_audio(
        self,
        source_path: str,
        stream_index: int,
        desired_end_ms: int | None = None,
        desired_start_ms: int | None = None,
    ) -> tuple[str, int]:
        """Prepare an unscaled (length-matching) audio stream for mkvmerge through a
        single FLAC-domain flow: decode to a priming-free FLAC, optionally trim its
        tail to the output-timeline end, then encode to AAC exactly once.

        Feeding AAC straight to mkvmerge, or re-encoding AAC→AAC, re-applies the
        encoder-delay priming on ffmpeg builds that expose it, accumulating ~21 ms
        per pass and shifting the track.  Routing every processing step through FLAC
        and encoding AAC only once keeps decoded timing identical across builds —
        the same single-encode discipline the constant-offset patch path relies on.
        """
        cache_key = (source_path, stream_index, desired_end_ms, desired_start_ms)
        cached_path = self._normalized_audio_cache.get(cache_key)
        if cached_path is not None:
            return cached_path, 0

        flac_path = self._temporary_audio_path("passthrough_flac", stream_index)
        self._decode_source_audio_to_flac(
            source_path,
            flac_path,
            sample_fmt="s32",
            audio_stream_index=stream_index,
            logger=self.logger,
        )

        prepared_flac = flac_path
        if desired_end_ms is not None:
            if desired_start_ms is None:
                desired_start_ms = self._source_stream_start_offset_ms(flac_path, "audio", 0)
            relative_end_ms = max(0, desired_end_ms - desired_start_ms)
            prepared_flac = self._temporary_audio_path("passthrough_trim", stream_index)
            trim_filter = (
                f"atrim=start=0.000000:end={relative_end_ms / 1000:.6f},"
                "asetpts=PTS-STARTPTS"
            )
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-i", flac_path, "-map", "0:a:0",
                    "-filter:a", trim_filter,
                    "-sample_fmt", "s32", "-c:a", "flac", prepared_flac,
                ], logger=self.logger)
            )

        output_path = self._temporary_audio_path("passthrough_audio", stream_index)
        channel_layout = self._get_audio_channel_layout(source_path, logger=self.logger)
        self._concat_and_encode(
            [prepared_flac], False, "", False, "",
            os.path.join(self.wd, f"passthrough_concat_{os.getpid()}_{stream_index}.txt"),
            output_path,
            channel_layout=channel_layout,
            logger=self.logger,
        )

        self._normalized_audio_cache[cache_key] = output_path
        return output_path, 0

    def _base_output_end_ms(
        self,
        video_path_base: str,
        video_streams: Sequence[tuple[str, int, str | None]],
        audio_streams: Sequence[tuple[str, int, str | None]],
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
        audio_streams: Sequence[tuple[str, int, str | None]],
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
        video_streams: Sequence[tuple[str, int, str | None]],
        audio_streams: Sequence[tuple[str, int, str | None]],
        subtitle_streams: Sequence[tuple[str, int, str | None]],
        required_input_files: set[str],
        attachments: Sequence[tuple[str, int]],
        file_ids: dict[str, int],
        files_details: dict[str, Any] | None = None,
    ) -> _PreparedStreams:
        streams_list: list[_StreamEntry] = []
        input_files = set(required_input_files)
        video_path_base, _, _ = video_streams[0]
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

        for (path, stream_index, language) in audio_streams:
            # Priming-convention independent content start: on builds that expose AAC
            # encoder-delay priming, a positive-offset (asO) stream's raw start_time is
            # short by one frame; _audio_content_start_ms adds it back so placement
            # matches absorbing builds (a no-op for non-AAC / asR streams).
            audio_desired_start_ms: int | None = self._audio_content_start_ms(
                self._stream_info(path, "audio", stream_index)
            )
            duration = self._video_track_duration(path, details, logger=self.logger)
            if _is_length_mismatch(base_duration, duration, self.tolerance_ms):
                assert base_duration is not None  # guaranteed by _is_length_mismatch
                original_path = path
                path, stream_index, audio_desired_start_ms = self._patch_mismatched_audio(
                    video_path_base, (path, stream_index), base_duration, base_audio_end_ms, file_ids,
                )
                input_files.add(path)
                if original_path not in protected_paths:
                    input_files.discard(original_path)
            else:
                # Length-matching audio is passed through unchanged, except for two
                # mkvmerge concerns handled by a single FLAC-domain flow (decode →
                # optional tail trim → one AAC encode): AAC needs priming-free remux
                # for non-Matroska inputs, and audio overrunning the output timeline
                # must be trimmed.  Both used to be separate AAC re-encodes that
                # accumulated encoder-delay priming on builds that expose it.
                needs_normalization = self._audio_needs_mkvmerge_normalization(path, stream_index)
                trim_end_ms: int | None = None
                if path != video_path_base and audio_desired_start_ms is not None and base_output_end_ms is not None:
                    audio_end_ms = self._source_stream_end_offset_ms(path, "audio", stream_index)
                    if audio_end_ms is not None and audio_end_ms > base_output_end_ms + self.tolerance_ms:
                        trim_end_ms = base_output_end_ms
                if needs_normalization or trim_end_ms is not None:
                    original_path = path
                    path, stream_index = self._prepare_passthrough_audio(
                        path, stream_index,
                        desired_end_ms=trim_end_ms,
                        desired_start_ms=audio_desired_start_ms,
                    )
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
