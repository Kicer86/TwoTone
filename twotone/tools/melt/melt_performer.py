import logging
import os
import shutil

from typing import Any, Iterable, Sequence
from tqdm import tqdm

from ..utils import files_utils, generic_utils, language_utils, process_utils, video_utils
from .debug_routines import DebugRoutines
from .pair_matcher import PairMatcher
from .melt_common import FramesInfo, _ensure_working_dir, _is_length_mismatch


class MeltPerformer:
    def __init__(
        self,
        logger: logging.Logger,
        interruption: generic_utils.InterruptibleProcess,
        working_dir: str,
        output_dir: str,
        tolerance_ms: int,
    ) -> None:
        self.logger = logger
        self.interruption = interruption
        self.output_dir = output_dir
        self.tolerance_ms = tolerance_ms
        self.wd = _ensure_working_dir(working_dir)

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

    def _log_coverage(
        self,
        lhs_path: str,
        rhs_path: str,
        mappings: list[tuple[int, int]],
        lhs_duration_ms: int,
        rhs_duration_ms: int,
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
                    f"start mismatch: {lhs_sg:.1f}s in {lhs_name}, "
                    f"{rhs_sg:.1f}s in {rhs_name}"
                )
            if lhs_eg > 0.04 or rhs_eg > 0.04:
                parts.append(
                    f"end mismatch: {lhs_eg:.1f}s in {lhs_name}, "
                    f"{rhs_eg:.1f}s in {rhs_name}"
                )

            detail = "; ".join(parts) if parts else "partial overlap"
            self.logger.info(
                "Files are NOT fully identical — %s: %s ↔ %s",
                detail, lhs_name, rhs_name,
            )

    def _build_mkvmerge_args(
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
        for file_path, fo in files_opts.items():
            if fo["video"]:
                generation_args.extend(["--video-tracks", ",".join(str(i) for i in fo["video"])])
            else:
                generation_args.append("--no-video")

            if fo["audio"]:
                generation_args.extend(["--audio-tracks", ",".join(str(i) for i in fo["audio"])])
            else:
                generation_args.append("--no-audio")

            if fo["subtitle"]:
                generation_args.extend(["--subtitle-tracks", ",".join(str(i) for i in fo["subtitle"])])
            else:
                generation_args.append("--no-subtitles")

            if fo["attachments"]:
                generation_args.extend(["--attachments", ",".join(str(i) for i in fo["attachments"])])
            else:
                generation_args.append("--no-attachments")

            for tid, lang in fo["languages"].items():
                generation_args.extend(["--language", f"{tid}:{lang}"])

            if preferred_audio:
                for tid in fo["audio"] + fo["subtitle"]:
                    flag = "yes" if tid in fo["defaults"] else "no"
                    generation_args.extend(["--default-track", f"{tid}:{flag}"])

            generation_args.append(file_path)

        if track_order:
            generation_args.extend(["--track-order", ",".join(track_order)])

        return generation_args

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
    ) -> None:
        """Replace an audio segment in the base video with time-adjusted audio from another video.

        The replacement is split into smaller, corresponding subsegments to better handle drift.
        """

        wd = os.path.join(wd, "audio_extraction")
        debug_wd = os.path.join(wd, "debug")
        os.makedirs(wd, exist_ok=True)
        os.makedirs(debug_wd, exist_ok=True)

        v1_audio = os.path.join(wd, "v1_audio.flac")
        v2_audio = os.path.join(wd, "v2_audio.flac")
        head_path = os.path.join(wd, "head.flac")
        tail_path = os.path.join(wd, "tail.flac")

        debug = DebugRoutines(debug_wd, lhs_frames, rhs_frames)

        # Compute global segment range (milliseconds)
        left_points = [p[0] for p in segment_pairs]
        right_points = [p[1] for p in segment_pairs]
        seg1_start, seg1_end = min(left_points), max(left_points)

        # 1. Extract main audio tracks
        process_utils.raise_on_error(
            process_utils.start_process("ffmpeg", ["-y", "-i", base_video, "-map", "0:a:0", "-c:a", "flac", v1_audio])
        )
        process_utils.raise_on_error(
            process_utils.start_process("ffmpeg", ["-y", "-i", source_video, "-map", "0:a:0", "-c:a", "flac", v2_audio])
        )

        # 2. Extract head and tail from base audio
        process_utils.raise_on_error(
            process_utils.start_process("ffmpeg", ["-y", "-ss", "0", "-to", str(seg1_start / 1000), "-i", v1_audio, "-c:a", "flac", head_path])
        )
        process_utils.raise_on_error(
            process_utils.start_process("ffmpeg", ["-y", "-ss", str(seg1_end / 1000), "-i", v1_audio, "-c:a", "flac", tail_path])
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
                self.logger.error(f"Segment {idx} duration mismatch exceeds 10%")

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
                        "-c:a", "flac",
                        scaled_cut,
                    ]
                )
            )

            temp_segments.append(scaled_cut)

        # 5. Concatenate head + replacement parts + tail
        concat_list = os.path.join(wd, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            f.write(f"file '{head_path}'\n")
            for seg in temp_segments:
                f.write(f"file '{seg}'\n")
            f.write(f"file '{tail_path}'\n")

        merged_flac = os.path.join(wd, "merged.flac")
        process_utils.raise_on_error(
            process_utils.start_process(
                "ffmpeg", [
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list,
                    "-c:a", "flac", merged_flac
                ],
            )
        )

        # 6. Encode to final audio format
        process_utils.raise_on_error(
            process_utils.start_process(
                "ffmpeg",
                ["-y", "-i", merged_flac, "-c:a", "aac", "-movflags", "+faststart", output_path],
            )
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
            f"File {self._display_path(input_path)} is superior. Using it whole as output {self._display_path(output_path)}."
        )
        shutil.copy2(input_path, output_path)

    def _prepare_stream_entries(
        self,
        video_streams: Sequence[tuple[str, int, str | None]],
        audio_streams: Sequence[tuple[str, int, str | None]],
        subtitle_streams: Sequence[tuple[str, int, str | None]],
        required_input_files: set[str],
        attachments: Sequence[tuple[str, int]],
    ) -> list[tuple[str, int, str, str | None]]:
        streams_list: list[tuple[str, int, str, str | None]] = []
        video_path_base, video_tid, _ = video_streams[0]
        base_duration = video_utils.get_video_duration(video_path_base)
        protected_paths = (
            {p for (p, _, _) in video_streams}
            | {p for (p, _, _) in subtitle_streams}
            | {p for (p, _) in attachments}
        )

        for (path, stream_index, language) in video_streams:
            streams_list.append(("video", stream_index, path, language))

        for (path, stream_index, language) in audio_streams:
            duration = video_utils.get_video_duration(path)
            if _is_length_mismatch(base_duration, duration, self.tolerance_ms):
                original_path = path
                with files_utils.ScopedDirectory(os.path.join(self.wd, "matching")) as mwd, \
                     generic_utils.TqdmBouncingBar(desc="Processing", **generic_utils.get_tqdm_defaults()):
                    matcher = PairMatcher(self.interruption, mwd, video_path_base, path, self.logger.getChild("PairMatcher"))
                    mapping, lhs_all_frames, rhs_all_frames = matcher.create_segments_mapping()

                    self._log_coverage(video_path_base, path, mapping, base_duration, duration)

                    patched_audio = os.path.join(self.wd, f"tmp_{os.getpid()}_{video_tid}_{stream_index}.m4a")
                    self._patch_audio_segment(mwd, video_path_base, path, patched_audio, mapping, 20, lhs_all_frames, rhs_all_frames)
                    path = patched_audio
                    stream_index = 0
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
                self.logger.info(f"Setting production audio language '{language_name}' as default.")
            else:
                self.logger.warning(f"Production audio language '{language_name}' not found among audio streams.")

        return preferred_audio

    def process_duplicates(self, plan: list[dict[str, Any]]) -> None:
        visible_items = [item for item in plan if item.get("groups") or item.get("skipped_groups")]
        planned_items = [item for item in visible_items if item.get("groups")]
        for item in tqdm(planned_items, desc="Titles", unit="title", **generic_utils.get_tqdm_defaults(), position=0):
            title = item["title"]
            groups = item.get("groups", [])

            for group in tqdm(groups, desc="Videos", unit="video", **generic_utils.get_tqdm_defaults(), position=1):
                self.interruption.check_for_stop()

                output_name = group["output_name"]

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
                    self.logger.info(f"Output file {output} exists, removing it.")
                    os.remove(output)

                output_parent = os.path.dirname(output)
                os.makedirs(output_parent, exist_ok=True)

                if len(required_input_files) == 1:
                    # only one file is being used, just copy it to the output dir
                    first_file_path = list(required_input_files)[0]
                    self._copy_single_input(first_file_path, output)
                else:
                    # Convert streams to unified list (and patch audios if needed)
                    streams_list = self._prepare_stream_entries(
                        video_streams,
                        audio_streams,
                        subtitle_streams,
                        required_input_files,
                        attachments,
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

                    generation_args = self._build_mkvmerge_args(
                        output,
                        streams_list_sorted,
                        attachments,
                        preferred_audio,
                        required_input_files,
                    )

                    self.logger.info(f"Generating file: {self._display_path(output)}")

                    process_utils.raise_on_error(
                        process_utils.start_process("mkvmerge", generation_args, show_progress=True)
                    )

                    self.logger.info(f"{output} saved.")
