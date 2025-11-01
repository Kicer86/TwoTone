
import argparse
import logging
import os
import re
import shutil

from overrides import override
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from tqdm import tqdm

from ..tool import Tool
from ..utils import files_utils, generic_utils, language_utils, process_utils, video_utils
from .attachments_picker import AttachmentsPicker
from .debug_routines import DebugRoutines
from .duplicates_source import DuplicatesSource
from .jellyfin import JellyfinSource
from .pair_matcher import PairMatcher
from .static_source import StaticSource
from .streams_picker import StreamsPicker

FramesInfo = Dict[int, Dict[str, str]]

def _split_path_fix(value: str) -> List[str]:
    pattern = r'"((?:[^"\\]|\\.)*?)"'

    matches = re.findall(pattern, value)
    return [match.replace(r'\"', '"') for match in matches]


class Melter:
    def __init__(
            self,
            logger: logging.Logger,
            interruption: generic_utils.InterruptibleProcess,
            duplicates_source: DuplicatesSource,
            wd: str,
            output: str,
            allow_length_mismatch: bool = False,
            allow_language_guessing: bool = False,
        ):
        self.logger = logger
        self.interruption = interruption
        self.duplicates_source = duplicates_source
        self.debug_it: int = 0
        self.wd = os.path.join(wd, str(os.getpid()))
        self.output = output
        self.allow_length_mismatch = allow_length_mismatch
        self.allow_language_guessing = allow_language_guessing
        self.tolerance_ms = 100

        os.makedirs(self.wd, exist_ok=True)

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
        process_utils.start_process("ffmpeg", ["-y", "-i", base_video, "-map", "0:a:0", "-c:a", "flac", v1_audio])
        process_utils.start_process("ffmpeg", ["-y", "-i", source_video, "-map", "0:a:0", "-c:a", "flac", v2_audio])

        # 2. Extract head and tail from base audio
        process_utils.start_process("ffmpeg", ["-y", "-ss", "0", "-to", str(seg1_start / 1000), "-i", v1_audio, "-c:a", "flac", head_path])
        process_utils.start_process("ffmpeg", ["-y", "-ss", str(seg1_end / 1000), "-i", v1_audio, "-c:a", "flac", tail_path])

        # 3. Generate subsegment split points from provided mapping pairs
        total_left_duration = seg1_end - seg1_start
        left_targets = [seg1_start + i * total_left_duration / segment_count for i in range(segment_count + 1)]

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

            process_utils.start_process(
                "ffmpeg", [
                    "-y",
                    "-ss", str(r_start / 1000),
                    "-to", str(r_end / 1000),
                    "-i", v2_audio,
                    "-c:a", "flac",
                    raw_cut
                ],
            )

            process_utils.start_process(
                "ffmpeg", [
                    "-y",
                    "-i", raw_cut,
                    "-filter:a", f"atempo={ratio:.3f}",
                    "-c:a", "flac",
                    scaled_cut
                ],
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
        process_utils.start_process(
            "ffmpeg", [
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-c:a", "flac", merged_flac
            ],
        )

        # 6. Encode to final audio format
        process_utils.start_process(
            "ffmpeg",
            ["-y", "-i", merged_flac, "-c:a", "aac", "-movflags", "+faststart", output_path],
        )

    def _stream_short_details(self, stype: str, stream: Dict[str, Any]) -> str:
        def fmt_fps(value: str) -> str | None:
            try:
                fps = generic_utils.fps_str_to_float(str(value))
            except Exception:
                return None

            if abs(fps - round(fps)) < 0.01:
                return str(int(round(fps)))
            return f"{fps:.2f}"

        if stype == "video":
            width = stream.get("width")
            height = stream.get("height")
            fps = stream.get("fps")
            codec = stream.get("codec")
            length = stream.get("length")
            length_formatted = generic_utils.ms_to_time(length) if length else None
            details = []
            if width and height:
                fps_val = fmt_fps(fps) if fps else None
                if fps_val:
                    details.append(f"{width}x{height}@{fps_val}")
                else:
                    details.append(f"{width}x{height}")
            elif fps:
                fps_val = fmt_fps(fps)
                if fps_val:
                    details.append(f"{fps_val}fps")
            if codec:
                details.append(codec)

            if length_formatted:
                details.append(f"duration: {length_formatted}")

            return ", ".join(details)
        if stype == "audio":
            channels = stream.get("channels")
            sample_rate = stream.get("sample_rate")
            details = []
            if channels:
                details.append(f"{channels}ch")
            if sample_rate:
                details.append(f"{sample_rate}Hz")
            return ", ".join(details)
        if stype == "subtitle":
            fmt = stream.get("format")
            return fmt or ""
        return ""

    def _print_file_details(self, file: str, details: Dict[str, Any], ids: Dict[str, int]) -> None:
        def formatter(key: str, value: Any) -> str:
            if key == "fps":
                try:
                    fps = generic_utils.fps_str_to_float(str(value))
                    return f"{fps:.3f}"
                except Exception:
                    return str(value)
            if key == "length":
                return generic_utils.ms_to_time(value) if value else "-"
            return str(value) if value else "-"

        def show(key: str) -> bool:
            if key == "tid":
                return False
            else:
                return True

        file_id = ids[file]
        self.logger.info(f"File #{file_id} details:")
        tracks = details["tracks"]
        attachments = details["attachments"]

        for stream_type, streams in tracks.items():
            self.logger.info(f"  {stream_type}: {len(streams)} track(s)")
            for stream in streams:
                lang_name = language_utils.language_name(stream.get("language"))
                short = self._stream_short_details(stream_type, stream)

                info = lang_name
                if short:
                    info += f" ({short})"

                sid = stream.get("tid")
                self.logger.info(f"    #{sid}: {info}")

        for attachment in attachments:
            file_name = attachment["file_name"]
            self.logger.info(f"  attachment: {file_name}")

        # more details for debug
        for stream_type, streams in tracks.items():
            self.logger.debug(f"\t{stream_type}:")

            for stream in streams:
                sid = stream.get("tid")
                self.logger.debug(f"\t#{sid}:")
                for key, value in stream.items():
                    if show(key):
                        key_title = key + ":"
                        self.logger.debug(
                            f"\t\t{key_title:<16}{formatter(key, value)}")

    def _print_streams_details(self, ids: Dict[str, int], all_streams: Iterable[Tuple[str, Iterable[Tuple[str, int, str | None]]]], tracks: Dict[str, Dict]) -> None:
        for stype, type_stream in all_streams:
            for stream in type_stream:
                path = stream[0]
                tid = stream[1]
                language = language_utils.language_name(stream[2])

                stream_details = None
                track_infos = tracks.get(path, {}).get(stype, [])
                for info in track_infos:
                    if info.get("tid") == tid:
                        stream_details = self._stream_short_details(stype, info)
                        break

                extra = f" ({stream_details})" if stream_details else ""

                file_id = ids[path]
                self.logger.info(f"{stype} track #{tid}: {language} from file #{file_id}{extra}")

    def _print_attachments_details(self, ids: Dict[str, int], all_attachments: Iterable[Tuple[str, int]]) -> None:
        for stream in all_attachments:
            path = stream[0]
            tid = stream[1]

            file_id = ids[path]
            self.logger.info(f"Attachment ID #{tid} from file #{file_id}")

    def _is_length_mismatch(self, base_ms: int | None, other_ms: int | None) -> bool:
        if base_ms is None or other_ms is None:
            return False
        return abs(base_ms - other_ms) > self.tolerance_ms

    def _pick_streams(self, tracks: Dict[str, Any], ids: Dict[str, int]) -> Tuple[List[Tuple[str, int, str | None]], List[Tuple[str, int, str | None]], List[Tuple[str, int, str | None]]]:
        picker_wd = os.path.join(self.wd, "stream_picker")
        streams_picker = StreamsPicker(self.logger, self.duplicates_source, picker_wd, allow_language_guessing=self.allow_language_guessing)
        video_streams, audio_streams, subtitle_streams = streams_picker.pick_streams(tracks, ids)
        return video_streams, audio_streams, subtitle_streams

    def _probe_inputs(self, files: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        details_full = {file: video_utils.get_video_data_mkvmerge(file, enrich=True) for file in files}
        attachments = {file: info["attachments"] for file, info in details_full.items()}
        tracks = {file: info["tracks"] for file, info in details_full.items()}
        return details_full, attachments, tracks

    def _validate_input_files(
        self,
        tracks: Dict[str, Any],
        ids: Dict[str, int],
        video_streams: List[Tuple[str, int, str | None]],
        audio_streams: List[Tuple[str, int, str | None]],
        subtitle_streams: List[Tuple[str, int, str | None]],
    ) -> Tuple[bool, List[Tuple[str, int]]]:
        # Validate lengths across used files

        # Base length for detailed checks
        v_path, v_tid, _ = video_streams[0]
        base_length = tracks[v_path]["video"][v_tid]["length"]

        # Subtitle mismatch (unsupported)
        for path, _, _ in subtitle_streams:
            length = tracks[path]["video"][0]["length"]
            if self._is_length_mismatch(base_length, length):
                file_id = ids[path]
                self.logger.error(
                    f"Subtitles stream from file #{file_id} has length different than length of video stream from file {v_path}. This is not supported yet"
                )
                return False, []

        # Audio lengths valdiation
        for path, tid, _ in audio_streams:
            length = tracks[path]["video"][0]["length"]
            if self._is_length_mismatch(base_length, length):
                file_id = ids[path]
                base_file_id = ids[v_path]
                self.logger.warning(f"Audio stream from file #{file_id} has length different than length of video stream from file #{base_file_id}. Check for --allow-length-mismatch option to allow this.")

                if self.allow_length_mismatch:
                    self.logger.info("Audio length mismatch detected; audio will be time-adjusted during processing.")

                else:
                    return False

        return True

    def _analyze_group(self, files: List[str], ids: Dict[str, int]) -> Dict[str, Any] | None:
        # Probe inputs and print details
        details_full, attachments, tracks = self._probe_inputs(files)
        for file, file_details in details_full.items():
            self._print_file_details(file, file_details, ids)

        # Pick streams
        try:
            video_streams, audio_streams, subtitle_streams = self._pick_streams(tracks, ids)
        except RuntimeError as err:
            self.logger.error(err)
            return None

        # Validate and compute audio patch requirements
        ok = self._validate_input_files(tracks, ids, video_streams, audio_streams, subtitle_streams)
        if not ok:
            return None

        # Attachments picking
        picked_attachments = AttachmentsPicker(self.logger).pick_attachments(attachments)

        # Present proposed output
        self.logger.info("Streams used to create output video file:")
        self._print_streams_details(
            ids,
            (
                ("video", video_streams),
                ("audio", audio_streams),
                ("subtitle", subtitle_streams),
            ),
            tracks,
        )
        self._print_attachments_details(ids, picked_attachments)

        # Prepare plan entity
        return {
            "streams": {
                "video": video_streams,
                "audio": audio_streams,
                "subtitle": subtitle_streams,
            },
            "attachments": picked_attachments,
        }

    def _prepare_duplicates_set(self, duplicates: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Prepare groups of duplicate files and output names per title.

        Returns a plan in the form:
        [
          {"title": str, "groups": [{"files": [str,...], "output_name": str}, ...]},
          ...
        ]
        """
        def process_entries(entries: List[str]) -> List[Tuple[List[str], str]]:
            # Returns list of: (group of duplicates, output base name)

            def file_without_ext(path: str) -> str:
                dir, name, _ = files_utils.split_path(path)
                return os.path.join(dir, name)

            if all(os.path.isdir(p) for p in entries):
                dirs = entries

                if len(dirs) == 1:
                    # Special case: single dir → treat all files as one group of duplicates
                    dir_path = dirs[0]
                    media_files = [
                        os.path.join(root, file)
                        for root, _, filenames in os.walk(dir_path)
                        for file in filenames
                        if video_utils.is_video(file)
                    ]
                    media_files.sort()
                    output_name = file_without_ext(os.path.relpath(media_files[0], dir_path)) if media_files else "output"
                    return [(media_files, output_name)]

                # Multiple dirs → group matching files by position
                files_per_dir = []
                for dir_path in dirs:
                    media_files = [
                        os.path.join(root, file)
                        for root, _, filenames in os.walk(dir_path)
                        for file in filenames
                        if video_utils.is_video(file)
                    ]
                    media_files.sort()
                    files_per_dir.append(media_files)

                sorted_file_lists = [list(entry) for entry in zip(*files_per_dir)]
                first_file_fullnames = [os.path.relpath(path[0], dirs[0]) for path in sorted_file_lists]
                first_file_names = [file_without_ext(path) for path in first_file_fullnames]

                return [(files_group, output_name) for files_group, output_name in zip(sorted_file_lists, first_file_names)]

            else:
                # List of individual files
                first_file_fullname = os.path.basename(entries[0])
                first_file_name = Path(first_file_fullname).stem
                return [(entries, first_file_name)]

        plan: List[Dict[str, Any]] = []
        for title, entries in duplicates.items():
            files_groups = process_entries(entries)
            item = {
                "title": title,
                "groups": [{"files": files, "output_name": output_name} for files, output_name in files_groups]
            }
            plan.append(item)

        return plan

    def analyze_duplicates(self, duplicates: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        base_plan = self._prepare_duplicates_set(duplicates)

        analysis_plan: List[Dict[str, Any]] = []
        for item in base_plan:
            title = item["title"]
            groups = item["groups"]

            # Title header
            self.logger.info("-------------------------" + "-" * len(title))
            self.logger.info(f"Analyzing duplicates for {title}")
            self.logger.info("-------------------------" + "-" * len(title))

            analyzed_groups: List[Dict[str, Any]] = []
            for group in groups:
                files = group["files"]
                output_name = group["output_name"]

                if len(groups) > 1:
                    self.logger.info("------------------------------------")
                    self.logger.info("Processing group of duplicated files")
                    self.logger.info("------------------------------------")

                ids = {file: i + 1 for i, file in enumerate(files)}
                for file, id in ids.items():
                    self.logger.info(f"#{id}: {file}")

                # analysis for group
                plan_details = self._analyze_group(files, ids)
                if plan_details is None:
                    self.logger.info("Skipping output generation")
                else:
                    analyzed_groups.append({
                        "files": files,
                        "output_name": output_name,
                        **plan_details,
                    })

            analysis_plan.append({
                "title": title,
                "groups": analyzed_groups,
            })

        return analysis_plan

    def _collect_required_input_files(
        self,
        video_streams: Sequence[Tuple[str, int, str | None]],
        audio_streams: Sequence[Tuple[str, int, str | None]],
        subtitle_streams: Sequence[Tuple[str, int, str | None]],
        attachments: Sequence[Tuple[str, int]],
    ) -> set[str]:
        required_input_files: set[str] = set()
        required_input_files |= {p for (p, _, _) in video_streams}
        required_input_files |= {p for (p, _, _) in audio_streams}
        required_input_files |= {p for (p, _, _) in subtitle_streams}
        required_input_files |= {info[0] for info in attachments}
        return required_input_files

    def _build_output_path(self, title: str, output_name: str) -> str:
        return os.path.join(self.output, title, output_name + ".mkv")

    def _copy_single_input(self, input_path: str, output_path: str) -> None:
        self.logger.info(f"File {input_path} is superior. Using it whole as an output.")
        shutil.copy2(input_path, output_path)

    def _prepare_stream_entries(
        self,
        video_streams: Sequence[Tuple[str, int, str | None]],
        audio_streams: Sequence[Tuple[str, int, str | None]],
        subtitle_streams: Sequence[Tuple[str, int, str | None]],
        video_path_base: str,
        video_tid: int,
        required_input_files: set[str],
    ) -> List[Tuple[str, int, str, str | None]]:
        streams_list: List[Tuple[str, int, str, str | None]] = []

        base_duration = video_utils.get_video_duration(video_path_base)

        for (path, stream_index, language) in video_streams:
            streams_list.append(("video", stream_index, path, language))

        for (path, stream_index, language) in audio_streams:
            duration = video_utils.get_video_duration(path)
            if self._is_length_mismatch(base_duration, duration):
                with files_utils.ScopedDirectory(os.path.join(self.wd, "matching")) as mwd, \
                     generic_utils.TqdmBouncingBar(desc="Processing", **generic_utils.get_tqdm_defaults()):
                    matcher = PairMatcher(self.interruption, mwd, video_path_base, path, self.logger.getChild("PairMatcher"))
                    mapping, lhs_all_frames, rhs_all_frames = matcher.create_segments_mapping()
                    patched_audio = os.path.join(self.wd, f"tmp_{os.getpid()}_{video_tid}_{stream_index}.m4a")
                    self._patch_audio_segment(mwd, video_path_base, path, patched_audio, mapping, 20, lhs_all_frames, rhs_all_frames)
                    path = patched_audio
                    stream_index = 0
                    required_input_files.add(path)
            streams_list.append(("audio", stream_index, path, language))

        for (path, stream_index, language) in subtitle_streams:
            streams_list.append(("subtitle", stream_index, path, language))

        return streams_list

    def _choose_preferred_audio(
        self,
        streams_list_sorted: Sequence[Tuple[str, int, str, str | None]],
        default_video_path: str,
        default_audio_lang: str | None,
    ) -> Tuple[str | None, Tuple[str, int, str, str | None] | None]:
        metadata = self.duplicates_source.get_metadata_for(default_video_path)
        prod_lang = metadata.get("audio_prod_lang")
        preferred_lang = language_utils.unify_lang(prod_lang) if prod_lang else default_audio_lang

        preferred_audio = next(
            (info for info in streams_list_sorted if info[0] == "audio" and info[3] == preferred_lang),
            None,
        )
        if prod_lang:
            language_name = language_utils.language_name(prod_lang)
            if preferred_audio:
                self.logger.info(f"Setting production audio language '{language_name}' as default.")
            else:
                self.logger.warning(f"Production audio language '{language_name}' not found among audio streams.")

        return preferred_lang, preferred_audio

    def _build_mkvmerge_args(
        self,
        output_path: str,
        streams_list_sorted: Sequence[Tuple[str, int, str, str | None]],
        attachments: Sequence[Tuple[str, int]],
        preferred_audio: Tuple[str, int, str, str | None] | None,
        required_input_files: Iterable[str],
    ) -> List[str]:
        generation_args: List[str] = ["-o", output_path]
        files_opts: Dict[str, Dict[str, Any]] = {
            path: {"video": [], "audio": [], "subtitle": [], "attachments": [], "languages": {}, "defaults": set()}
            for path in required_input_files
        }

        # Collect per-file options and track order
        track_order: List[str] = []
        for stream_type, tid, file_path, language in streams_list_sorted:
            fo: Dict[str, Any] = files_opts[file_path]
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

            for tid in fo["audio"] + fo["subtitle"]:
                flag = "yes" if tid in fo["defaults"] else "no"
                generation_args.extend(["--default-track", f"{tid}:{flag}"])

            generation_args.append(file_path)

        if track_order:
            generation_args.extend(["--track-order", ",".join(track_order)])

        return generation_args

    def process_duplicates_set(self, plan: List[Dict[str, Any]]):
        for item in tqdm(plan, desc="Titles", unit="title", **generic_utils.get_tqdm_defaults(), position=0):
            title = item["title"]
            groups = item["groups"]

            for group in tqdm(groups, desc="Videos", unit="video", **generic_utils.get_tqdm_defaults(), position=1):
                self.interruption._check_for_stop()

                output_name = group["output_name"]

                # Use analysis results
                streams_info = group.get("streams", {})
                attachments = group.get("attachments", [])
                video_streams: List[Tuple[str, int, str | None]] = streams_info.get("video", [])
                audio_streams: List[Tuple[str, int, str | None]] = streams_info.get("audio", [])
                subtitle_streams: List[Tuple[str, int, str | None]] = streams_info.get("subtitle", [])
                required_input_files = self._collect_required_input_files(video_streams, audio_streams, subtitle_streams, attachments)

                output = self._build_output_path(title, output_name)
                if os.path.exists(output):
                    self.logger.info(f"Output file {output} exists, removing it.")
                    os.remove(output)

                output_dir = os.path.dirname(output)
                os.makedirs(output_dir, exist_ok=True)

                if len(required_input_files) == 1:
                    # only one file is being used, just copy it to the output dir
                    first_file_path = list(required_input_files)[0]
                    self._copy_single_input(first_file_path, output)
                else:
                    # Convert streams to unified list (and patch audios if needed)
                    video_path_base, video_tid, _ = video_streams[0]
                    streams_list = self._prepare_stream_entries(
                        video_streams,
                        audio_streams,
                        subtitle_streams,
                        video_path_base,
                        video_tid,
                        required_input_files
                    )

                    # Sort streams by language alphabetically
                    streams_list_sorted = sorted(streams_list, key=lambda stream: stream[3] if stream[3] else "")

                    # Decide which track should be default
                    default_video_stream = next(filter(lambda s: s[0] == "video", streams_list))
                    default_audio_stream = next((s for s in streams_list if s[0] == "audio"), None)
                    default_audio_lang = default_audio_stream[3] if default_audio_stream else None
                    _, preferred_audio = self._choose_preferred_audio(
                        streams_list_sorted, default_video_stream[2], default_audio_lang
                    )

                    generation_args = self._build_mkvmerge_args(
                        output, streams_list_sorted, attachments, preferred_audio, required_input_files
                    )

                    self.logger.info("Starting output file generation from chosen streams.")
                    process_utils.raise_on_error(
                        process_utils.start_process("mkvmerge", generation_args, show_progress=True)
                    )
                    self.logger.info(f"{output} saved.")


class RequireJellyfinServer(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, "jellyfin_server", None) is None:
            parser.error(f"{option_string} requires --jellyfin-server to be specified")
        setattr(namespace, self.dest, values)


class MeltTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self._analysis_results: List[Dict[str, Any]] | None = None
        self._data_source: DuplicatesSource | None = None
        self._interruption: generic_utils.InterruptibleProcess | None = None

    @override
    def setup_parser(self, parser: argparse.ArgumentParser):
        self.parser = parser

        jellyfin_group = parser.add_argument_group("Jellyfin source")
        jellyfin_group.add_argument('--jellyfin-server',
                                    help='URL to the Jellyfin server which will be used as a source of video files duplicates')
        jellyfin_group.add_argument('--jellyfin-token',
                                    action=RequireJellyfinServer,
                                    help='Access token (http://server:8096/web/#/dashboard/keys)')
        jellyfin_group.add_argument('--jellyfin-path-fix',
                                    action=RequireJellyfinServer,
                                    help='Specify a replacement pattern for file paths to ensure "melt" can access Jellyfin video files.\n\n'
                                         '"Melt" requires direct access to video files. If Jellyfin is not running on the same machine as "melt",\n'
                                         'you need to set up network access to Jellyfin’s video storage and specify how paths should be resolved.\n\n'
                                         'For example, suppose Jellyfin runs on a Linux machine where the video library is stored at "/srv/videos" (a shared directory).\n'
                                         'If "melt" is running on another Linux machine that accesses this directory remotely at "/mnt/shared_videos,"\n'
                                         'you need to map "/srv/videos" (Jellyfin’s path) to "/mnt/shared_videos" (the path accessible on the machine running "melt").\n\n'
                                         'In this case, use: --jellyfin-path-fix \\"/srv/videos\\",\\"/mnt/shared_videos\\" to define the replacement pattern.' \
                                         'Please mind that \\ to preserve \" are crucial')

        manual_group = parser.add_argument_group("Manual input source")
        manual_group.add_argument('-t', '--title',
                                  help='Video (movie or series when directory is provided as an input) title.')
        manual_group.add_argument('-i', '--input', dest='input_files', action='append',
                                  help='Add an input video file or directory with video files (can be specified multiple times).\n'
                                       'path can be followed with a comma and some additional parameters:\n'
                                       'audio_lang:XXX       - information about audio language (like eng, de or pl).\n'
                                       'audio_prod_lang:XXX - original/production audio language.\n\n'
                                       'Example of usage:\n'
                                       '--input some/path/file.mp4,audio_lang:jp --input some/path/file.mp4,audio_lang:eng\n\n'
                                       'If files are provided with this option, all of them are treated as duplicates of given title.\n'
                                       'If directoriess are provided, a \'series\' mode is being used and melt will list and sort files from each dir, and corresponding '
                                       'files from provided directories will be grouped as duplicates.\n'
                                       'If only one directory is provided as input, all files found inside will be treated as duplicates of the title.\n'
                                       'No other scenarios and combinations of inputs are supported.')

        # global options
        parser.add_argument('-o', '--output-dir',
                            help="Directory for output files",
                            required = True)

        parser.add_argument('--allow-length-mismatch', action='store_true',
                            help='[EXPERIMENTAL] Continue processing even if input video lengths differ.\n'
                                 'This may require additional processing that can consume significant time and disk space.')

        parser.add_argument('--allow-language-guessing', action='store_true',
                            help='If audio language is not provided in file metadata, try find language codes (like EN or DE) in file names')


    @override
    def analyze(self, args, logger: logging.Logger, working_dir: str):
        # Reset cached state
        self._analysis_results = None
        self._data_source = None
        self._interruption = generic_utils.InterruptibleProcess()

        # Build data source based on arguments
        if args.jellyfin_server:
            path_fix = _split_path_fix(args.jellyfin_path_fix) if args.jellyfin_path_fix else None

            if path_fix and len(path_fix) != 2:
                self.parser.error(f"Invalid content for --jellyfin-path-fix argument. Got: {path_fix}")

            self._data_source = JellyfinSource(interruption=self._interruption,
                                               url=args.jellyfin_server,
                                               token=args.jellyfin_token,
                                               path_fix=path_fix,
                                               logger=logger.getChild("JellyfinSource"))
        elif args.input_files:
            title = args.title
            input_entries = args.input_files

            if not title:
                self.parser.error(f"Missing required option: --title")

            src = StaticSource(interruption=self._interruption)

            for input in input_entries:
                # split by ',' but respect ""
                input_split = re.findall(r'(?:[^,"]|"(?:\\"|[^"])*")+', input)
                path = input_split[0]

                if not os.path.exists(path):
                    raise ValueError(f"Path {path} does not exist")

                audio_lang = ""
                audio_prod_lang = ""

                if len(input_split) > 1:
                    for extra_arg in input_split[1:]:
                        if extra_arg[:11] == "audio_lang:":
                            audio_lang = extra_arg[11:]
                        if extra_arg[:15] == "audio_prod_lang:":
                            audio_prod_lang = extra_arg[15:]

                src.add_entry(title, path)

                if audio_lang:
                    src.add_metadata(path, "audio_lang", audio_lang)
                if audio_prod_lang:
                    src.add_metadata(path, "audio_prod_lang", audio_prod_lang)

            self._data_source = src

        # If no source, nothing to analyze
        if not self._data_source:
            logger.info("No input source specified. Nothing to analyze.")
            return

        logger.info("Collecting duplicates for analysis")
        duplicates = self._data_source.collect_duplicates()

        melter = Melter(logger,
                        self._interruption,
                        self._data_source,
                        wd = working_dir,
                        output = args.output_dir,
                        allow_length_mismatch = args.allow_length_mismatch,
                        allow_language_guessing = args.allow_language_guessing,
        )

        self._analysis_results = melter.analyze_duplicates(duplicates)

    @override
    def perform(self, args, logger: logging.Logger, working_dir: str):
        plan = self._analysis_results
        data_source = self._data_source
        interruption = self._interruption or generic_utils.InterruptibleProcess()
        # clear cached results early to free memory
        self._analysis_results = None
        # Quick exit if no analysis was done
        if not plan or not data_source:
            logger.info("No analysis results, nothing to melt.")
            return

        melter = Melter(logger,
                        interruption,
                        data_source,
                        wd = working_dir,
                        output = args.output_dir,
                        allow_length_mismatch = args.allow_length_mismatch,
                        allow_language_guessing = args.allow_language_guessing,
        )

        # Use precomputed plan to skip re-grouping
        melter.process_duplicates_set(plan)
