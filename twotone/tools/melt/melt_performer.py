import enum
import logging
import os
import shutil

from typing import Any, Iterable, NamedTuple, Sequence
from tqdm import tqdm

from ..utils import files_utils, generic_utils, language_utils, process_utils, video_utils
from .debug_routines import DebugRoutines
from .melt_cache import MeltCache
from .pair_matcher import PairMatcher
from .melt_common import FramesInfo, _ensure_working_dir, _is_length_mismatch


class _SegmentRange(NamedTuple):
    lhs_start: int
    lhs_end: int
    rhs_start: int
    rhs_end: int


class _AudioStrategy(enum.Enum):
    STREAM_COPY = "stream_copy"        # no re-encoding, shift via --sync
    CONSTANT_OFFSET = "constant_offset"  # single global time-scale
    SUBSEGMENT = "subsegment"          # per-subsegment atempo


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
        self._sync_offsets: dict[str, int] = {}
        self.wd = _ensure_working_dir(working_dir)

    def process_duplicates(self, plan: list[dict[str, Any]]) -> None:
        visible_items = [item for item in plan if item.get("groups") or item.get("skipped_groups")]
        planned_items = [item for item in visible_items if item.get("groups")]
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
                    self.logger.info("  #%d: %s", fid, self._display_path(f))

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
                    self._sync_offsets.clear()
                    files_details = group.get("files_details", {})
                    streams_list = self._prepare_stream_entries(
                        video_streams,
                        audio_streams,
                        subtitle_streams,
                        required_input_files,
                        attachments,
                        file_ids,
                        files_details,
                    )

                    # Sort streams by language alphabetically, unknown languages last
                    streams_list_sorted = sorted(streams_list, key=lambda stream: (stream[3] is None, stream[3] or ""))

                    # Decide which track should be default
                    default_audio_stream = next((s for s in streams_list if s[0] == "audio"), None)
                    default_audio_lang = default_audio_stream[3] if default_audio_stream else None
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
                        required_input_files,
                    )

                    self.logger.info("Generating file: %s", self._display_path(output))

                    process_utils.raise_on_error(
                        process_utils.start_process("mkvmerge", generation_args, show_progress=True)
                    )

                    self.logger.info("%s saved.", output)

    def build_mkvmerge_args(
        self,
        output_path: str,
        streams_list_sorted: Sequence[tuple[str, int, str, str | None]],
        attachments: Sequence[tuple[str, int]],
        preferred_audio: tuple[str, int, str, str | None] | None,
        required_input_files: Iterable[str],
    ) -> list[str]:
        generation_args: list[str] = ["-o", output_path]
        files_opts: dict[str, dict[str, Any]] = {
            path: {"video": [], "audio": [], "subtitle": [], "attachments": [], "languages": {}, "defaults": set()}
            for path in required_input_files
        }

        # Collect per-file options and track order
        track_order: list[str] = []
        for stream_type, tid, file_path, language in streams_list_sorted:
            fo: dict[str, Any] = files_opts[file_path]
            fo[stream_type].append(tid)
            fo["languages"][tid] = language or "und"
            if stream_type in ("audio", "subtitle") and preferred_audio and (stream_type, tid, file_path, language) == preferred_audio:
                fo["defaults"].add(tid)
            file_index = generic_utils.get_key_position(files_opts, file_path)
            track_order.append(f"{file_index}:{tid}")

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

            sync_offset = self._sync_offsets.get(file_path)
            if sync_offset is not None:
                for tid in fo["audio"]:
                    generation_args.extend(["--sync", f"{tid}:{sync_offset}"])

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
        avoiding full audio extraction. When the durations already match
        (ratio ≈ 1.0) and no head/tail is needed, the audio is stream-copied
        without any re-encoding.

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
        source_params = self._get_audio_params(source_video)
        base_duration_ms = video_utils.get_video_duration(base_video)
        has_head = seg1_start > 0 and not use_silence
        has_tail = seg1_end < base_duration_ms and not use_silence

        # Video-frame ratio (true fps relationship, independent of audio track)
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
                ])
            )
        if has_tail:
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-ss", str(seg1_end / 1000),
                    "-i", base_video, "-map", "0:a:0",
                    *self._normalize_args(source_params),
                    "-c:a", "flac", tail_path,
                ])
            )

        # 2. Trim + time-scale source audio.
        #    The scaling ratio is derived from video-frame timestamps (fps
        #    relationship), NOT from measured audio duration.  If the audio
        #    stream in the container is shorter than the video, the deficit
        #    is handled via sync_offset (start) and natural end (no padding).

        # Fast path: no head/tail + fps ratio ≈ 1.0 → stream-copy, no re-encoding at all
        if not has_head and not has_tail and not needs_scaling:
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y",
                    "-ss", str(seg2_start / 1000),
                    "-to", str(seg2_end / 1000),
                    "-i", source_video,
                    "-map", "0:a:0", "-c:a", "copy",
                    output_path,
                ])
            )
            actual_dur = video_utils.get_video_duration(output_path)
            deficit = source_dur - actual_dur

            if not use_silence and deficit > 50:
                raise RuntimeError(
                    f"Audio deficit of {deficit} ms detected in fill-audio-gaps mode. "
                    f"The source container's audio starts later than its video, "
                    f"which cannot be compensated when head/tail are filled from "
                    f"the base file. Use default (silence) mode instead."
                )

            self._validate_audio_duration(actual_dur, source_dur, "stream-copied audio")
            return self._sync_offset_from_deficit(seg1_start, deficit, video_ratio)

        trimmed_audio = os.path.join(wd, "source_trimmed.flac")
        process_utils.raise_on_error(
            process_utils.start_process("ffmpeg", [
                "-y",
                "-ss", str(seg2_start / 1000),
                "-to", str(seg2_end / 1000),
                "-i", source_video,
                "-map", "0:a:0",
                "-sample_fmt", self._flac_safe_fmt(source_params[2]),
                "-c:a", "flac",
                trimmed_audio,
            ])
        )

        actual_source_dur = video_utils.get_video_duration(trimmed_audio)
        expected_scaled_dur = round(actual_source_dur * video_ratio)

        # Compute sync offset from the measured trim deficit.
        # If the audio stream starts later than the -ss timestamp (common in
        # AVI), the trimmed output is shorter than requested.  The deficit
        # tells us how much audio is missing at the start, so the sync offset
        # must shift forward to compensate.
        deficit = source_dur - actual_source_dur

        if not use_silence and deficit > 50:
            raise RuntimeError(
                f"Audio deficit of {deficit} ms detected in fill-audio-gaps mode. "
                f"The source container's audio starts later than its video, "
                f"which cannot be compensated when head/tail are filled from "
                f"the base file. Use default (silence) mode instead."
            )

        sync_offset = self._sync_offset_from_deficit(seg1_start, deficit, video_ratio)

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
                ])
            )

        scaled_dur = video_utils.get_video_duration(scaled_audio)
        self._validate_audio_duration(scaled_dur, expected_scaled_dur, "scaled audio")

        # 3. Concatenate and encode to AAC
        channel_layout = self._get_audio_channel_layout(source_video)
        self._concat_and_encode(
            [scaled_audio], has_head, head_path, has_tail, tail_path,
            os.path.join(wd, "concat.txt"), output_path,
            channel_layout=channel_layout,
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

        _THRESH = 0.1  # ignore gaps smaller than 100ms
        _MIN_GAP = 6   # minimum column width for a visible gap
        _W = 70         # total diagram width in columns

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
        lhs_id: int,
        rhs_id: int,
    ) -> None:
        """Log a human-readable coverage report after PairMatcher finishes."""
        summary = PairMatcher.coverage_summary(mappings, lhs_duration_ms, rhs_duration_ms)
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
                    f"shared content starts at {lhs_sg:.1f}s in {lhs_name} "
                    f"and {rhs_sg:.1f}s in {rhs_name}"
                )
            if lhs_eg > 0.04 or rhs_eg > 0.04:
                parts.append(
                    f"shared content ends {lhs_eg:.1f}s before end of {lhs_name} "
                    f"and {rhs_eg:.1f}s before end of {rhs_name}"
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

    @staticmethod
    def _extract_audio_to_flac(video_path: str, output_path: str) -> None:
        process_utils.raise_on_error(
            process_utils.start_process("ffmpeg", ["-y", "-i", video_path, "-map", "0:a:0", "-sample_fmt", "s32", "-c:a", "flac", output_path])
        )

    @staticmethod
    def _segment_range(pairs: Sequence[tuple[int, int]]) -> _SegmentRange:
        """Return the bounding lhs/rhs range from a list of (lhs, rhs) pairs."""
        left, right = zip(*pairs)
        return _SegmentRange(min(left), max(left), min(right), max(right))

    @staticmethod
    def _get_audio_params(audio_path: str) -> tuple[int, int, str]:
        """Return (channels, sample_rate, sample_fmt) of the first audio stream."""
        info = video_utils.get_video_full_info(audio_path)
        stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
        return int(stream["channels"]), int(stream["sample_rate"]), stream["sample_fmt"]

    @staticmethod
    def _get_audio_channel_layout(audio_path: str) -> str | None:
        """Return the channel layout string of the first audio stream, or None."""
        info = video_utils.get_video_full_info(audio_path)
        stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
        return stream.get("channel_layout") or None

    def _sync_offset_from_deficit(
        self,
        seg1_start: int,
        deficit: int,
        video_ratio: float,
    ) -> int:
        """Compute mkvmerge --sync offset from the measured trim deficit.

        When the audio stream in a container starts later than the video
        (common in AVI), trimming from a video-frame timestamp yields output
        shorter than requested.  The *deficit* (requested − actual) tells us
        how much audio is missing at the start.  We shift the sync offset
        forward by ``deficit * video_ratio`` to compensate.
        """
        if deficit <= 50:
            return seg1_start
        correction = round(deficit * video_ratio)
        sync_offset = seg1_start + correction
        self.logger.info(
            "Audio deficit: %d ms → sync offset: %d ms (base: %d + correction: %d)",
            deficit, sync_offset, seg1_start, correction,
        )
        return sync_offset

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
                ])
            )

        base_duration_ms = video_utils.get_video_duration(base_video)
        has_tail = seg_end_ms < base_duration_ms
        if has_tail:
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-ss", str(seg_end_ms / 1000),
                    "-i", base_audio, *norm_args, "-c:a", "flac", tail_path,
                ])
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
            ])
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
            self._extract_audio_to_flac(base_video, v1_audio)
        self._extract_audio_to_flac(source_video, v2_audio)

        source_params = self._get_audio_params(v2_audio)

        # 2. Extract head/tail from base audio (skipped when use_silence — caller uses --sync)
        if use_silence:
            has_head = False
            has_tail = False
        else:
            has_head, has_tail = self._extract_head_tail(
                v1_audio, base_video, seg1_start, seg1_end, head_path, tail_path,
                normalize_to=source_params,
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
                    ]
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
                    ]
                )
            )

            temp_segments.append(scaled_cut)

        # 5. Concatenate and encode
        channel_layout = self._get_audio_channel_layout(source_video)
        self._concat_and_encode(
            temp_segments, has_head, head_path, has_tail, tail_path,
            os.path.join(wd, "concat.txt"), output_path,
            channel_layout=channel_layout,
        )

    def _build_output_path(self, title: str, output_name: str) -> str:
        return os.path.join(self.output_dir, title, output_name + ".mkv")

    def _display_path(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.output_dir)
        except ValueError:
            return path

    def _copy_single_input(self, input_path: str, output_path: str) -> None:
        self.logger.info(
            "File %s is superior. Using it whole as output %s.",
            self._display_path(input_path), self._display_path(output_path),
        )
        shutil.copy2(input_path, output_path)

    def _shift_audio_no_reencode(
        self,
        source_video: str,
        output_path: str,
        segment_pairs: list[tuple[int, int]],
    ) -> int:
        """Trim source audio with stream-copy and return a sync offset for mkvmerge.

        For the simplest case (constant offset, ratio ≈ 1.0): no decoding at all.
        The returned offset (ms) should be applied as ``--sync TID:<offset>`` in mkvmerge.
        """
        seg = self._segment_range(segment_pairs)
        seg2_start, seg2_end = seg.rhs_start, seg.rhs_end
        sync_offset = seg.lhs_start
        expected_dur = seg2_end - seg2_start

        process_utils.raise_on_error(
            process_utils.start_process("ffmpeg", [
                "-y",
                "-ss", str(seg2_start / 1000),
                "-to", str(seg2_end / 1000),
                "-i", source_video,
                "-map", "0:a:0", "-c:a", "copy",
                output_path,
            ])
        )

        actual_dur = video_utils.get_video_duration(output_path)
        self._validate_audio_duration(actual_dur, expected_dur, "stream-copied audio (no reencode)")
        deficit = expected_dur - actual_dur

        return self._sync_offset_from_deficit(sync_offset, deficit, 1.0)

    def _patch_mismatched_audio(
        self,
        video_path_base: str,
        audio_path: str,
        video_tid: int,
        stream_index: int,
        base_duration: int,
        file_ids: dict[str, int] | None,
    ) -> tuple[str, int]:
        """Run PairMatcher and apply the appropriate audio patching strategy.

        Returns (patched_path, new_stream_index).
        """
        duration = video_utils.get_video_duration(audio_path)
        with files_utils.ScopedDirectory(os.path.join(self.wd, "matching")) as mwd, \
             generic_utils.TqdmBouncingBar(desc="Processing", **generic_utils.get_tqdm_defaults()):
            lhs_id = file_ids.get(video_path_base, 1) if file_ids else 1
            rhs_id = file_ids.get(audio_path, 2) if file_ids else 2
            matcher = PairMatcher(
                self.interruption, mwd, video_path_base, audio_path,
                self.logger.getChild("PairMatcher"),
                lhs_label=f"#{lhs_id}", rhs_label=f"#{rhs_id}",
                cache=self.cache,
            )
            mapping, lhs_all_frames, rhs_all_frames, constant_offset = matcher.create_segments_mapping()

            self._log_coverage(video_path_base, audio_path, mapping, base_duration, duration, lhs_id, rhs_id)

            self.logger.info(
                "Audio patching: base_duration=%d ms, source_duration=%d ms, "
                "constant_offset=%s, lhs_fps=%.3f, rhs_fps=%.3f, mapping_pairs=%d",
                base_duration, duration, constant_offset,
                matcher.lhs_fps, matcher.rhs_fps, len(mapping),
            )
            if mapping:
                self.logger.info(
                    "  Mapping range: lhs=[%d … %d] ms, rhs=[%d … %d] ms",
                    mapping[0][0], mapping[-1][0], mapping[0][1], mapping[-1][1],
                )

            use_silence = not self.fill_audio_gaps
            strategy = self._choose_audio_strategy(constant_offset, use_silence, mapping)
            self.logger.info("  Audio strategy: %s", strategy.value)

            if strategy == _AudioStrategy.STREAM_COPY:
                patched_audio = os.path.join(self.wd, f"tmp_{os.getpid()}_{video_tid}_{stream_index}.mka")
                sync_offset = self._shift_audio_no_reencode(audio_path, patched_audio, mapping)
                self._sync_offsets[patched_audio] = sync_offset
                return patched_audio, 0

            patched_audio = os.path.join(self.wd, f"tmp_{os.getpid()}_{video_tid}_{stream_index}.mka")
            if strategy == _AudioStrategy.CONSTANT_OFFSET:
                effective_sync = self.patch_audio_constant_offset(mwd, video_path_base, audio_path, patched_audio, mapping, use_silence=use_silence)
            else:
                self._patch_audio_segment(mwd, video_path_base, audio_path, patched_audio, mapping, 20, lhs_all_frames, rhs_all_frames, use_silence=use_silence)
                effective_sync = min(p[0] for p in mapping)

            if use_silence:
                self._sync_offsets[patched_audio] = effective_sync
                self.logger.info("  Sync offset (--sync): %d ms", effective_sync)

        return patched_audio, 0

    def _choose_audio_strategy(
        self,
        constant_offset: bool,
        use_silence: bool,
        mapping: list[tuple[int, int]],
    ) -> _AudioStrategy:
        """Pick the lightest audio patching strategy that fits the constraints."""
        if not constant_offset:
            return _AudioStrategy.SUBSEGMENT
        if use_silence:
            seg = self._segment_range(mapping)
            source_dur = seg.rhs_end - seg.rhs_start
            target_dur = seg.lhs_end - seg.lhs_start
            ratio = source_dur / target_dur if target_dur else 1.0
            if not self._needs_fps_scaling(ratio):
                return _AudioStrategy.STREAM_COPY
        return _AudioStrategy.CONSTANT_OFFSET

    @staticmethod
    def _video_track_duration(path: str, files_details: dict[str, Any]) -> int | None:
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
        return video_utils.get_video_duration(path)

    def _prepare_stream_entries(
        self,
        video_streams: Sequence[tuple[str, int, str | None]],
        audio_streams: Sequence[tuple[str, int, str | None]],
        subtitle_streams: Sequence[tuple[str, int, str | None]],
        required_input_files: set[str],
        attachments: Sequence[tuple[str, int]],
        file_ids: dict[str, int] | None = None,
        files_details: dict[str, Any] | None = None,
    ) -> list[tuple[str, int, str, str | None]]:
        streams_list: list[tuple[str, int, str, str | None]] = []
        video_path_base, video_tid, _ = video_streams[0]
        details = files_details or {}
        base_duration = self._video_track_duration(video_path_base, details)
        protected_paths = (
            {p for (p, _, _) in video_streams}
            | {p for (p, _, _) in subtitle_streams}
            | {p for (p, _) in attachments}
        )

        for (path, stream_index, language) in video_streams:
            streams_list.append(("video", stream_index, path, language))

        for (path, stream_index, language) in audio_streams:
            duration = self._video_track_duration(path, details)
            if _is_length_mismatch(base_duration, duration, self.tolerance_ms):
                original_path = path
                path, stream_index = self._patch_mismatched_audio(
                    video_path_base, path, video_tid, stream_index, base_duration, file_ids,
                )
                required_input_files.add(path)
                if original_path not in protected_paths:
                    required_input_files.discard(original_path)
            streams_list.append(("audio", stream_index, path, language))

        for (path, stream_index, language) in subtitle_streams:
            streams_list.append(("subtitle", stream_index, path, language))

        return streams_list

    def _choose_preferred_audio(
        self,
        audio_prod_lang: str | None,
        streams_list_sorted: Sequence[tuple[str, int, str, str | None]],
        default_audio_lang: str | None,
    ) -> tuple[str, int, str, str | None] | None:
        preferred_lang = language_utils.unify_lang(audio_prod_lang) if audio_prod_lang else default_audio_lang

        preferred_audio = next(
            (info for info in streams_list_sorted if info[0] == "audio" and info[3] == preferred_lang),
            None,
        )
        if audio_prod_lang:
            language_name = language_utils.language_name(audio_prod_lang)
            if preferred_audio:
                self.logger.info("Setting production audio language '%s' as default.", language_name)
            else:
                self.logger.warning("Production audio language '%s' not found among audio streams.", language_name)

        return preferred_audio
