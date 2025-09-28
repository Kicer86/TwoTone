
import argparse
import logging
import os
import re
import shutil

from collections import defaultdict
from overrides import override
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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


class Melter():
    def __init__(
            self,
            logger: logging.Logger,
            interruption: generic_utils.InterruptibleProcess,
            duplicates_source: DuplicatesSource,
            live_run: bool,
            wd: str,
            output: str,
            allow_length_mismatch: bool = False,
            allow_language_guessing: bool = False,
        ):
        self.logger = logger
        self.interruption = interruption
        self.duplicates_source = duplicates_source
        self.live_run = live_run
        self.debug_it: int = 0
        self.wd = os.path.join(wd, str(os.getpid()))
        self.output = output
        self.allow_length_mismatch = allow_length_mismatch
        self.allow_language_guessing = allow_language_guessing

        os.makedirs(self.wd, exist_ok=True)

    def _patch_audio_segment(
        self,
        wd: str,
        video1_path: str,
        video2_path: str,
        output_path: str,
        segment_pairs: list[tuple[int, int]],
        segment_count: int,
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo,
        min_subsegment_duration: float = 30.0,
    ):
        """
        Replaces a segment of audio in video1 with a segment from video2 (after adjusting its duration),
        split into smaller corresponding subsegments.

        :param wd: Working directory for intermediate files.
        :param video1_path: Path to the first video (base).
        :param video2_path: Path to the second video (source of audio segment).
        :param segment_pairs: list of (timestamp_v1_ms, timestamp_v2_ms) pairs
        :param segment_count: how many subsegments to split the entire segment into
        :param output_path: Path to final audio output
        :param min_subsegment_duration: minimum duration in seconds below which a subsegment is merged with neighbor
        """

        wd = os.path.join(wd, "audio_extraction")
        debug_wd = os.path.join(wd, "debug")
        os.makedirs(wd)
        os.makedirs(debug_wd)

        v1_audio = os.path.join(wd, "v1_audio.flac")
        v2_audio = os.path.join(wd, "v2_audio.flac")
        head_path = os.path.join(wd, "head.flac")
        tail_path = os.path.join(wd, "tail.flac")

        debug = DebugRoutines(debug_wd, lhs_frames, rhs_frames)

        # Compute global segment range
        s1_all = [p[0] for p in segment_pairs]
        s2_all = [p[1] for p in segment_pairs]
        seg1_start, seg1_end = min(s1_all), max(s1_all)
        seg2_start, seg2_end = min(s2_all), max(s2_all)

        # 1. Extract main audio
        process_utils.start_process("ffmpeg", ["-y", "-i", video1_path, "-map", "0:a:0", "-c:a", "flac", v1_audio])
        process_utils.start_process("ffmpeg", ["-y", "-i", video2_path, "-map", "0:a:0", "-c:a", "flac", v2_audio])

        # 2. Extract head and tail
        process_utils.start_process("ffmpeg", ["-y", "-ss", "0", "-to", str(seg1_start / 1000), "-i", v1_audio, "-c:a", "flac", head_path])
        process_utils.start_process("ffmpeg", ["-y", "-ss", str(seg1_end / 1000), "-i", v1_audio, "-c:a", "flac", tail_path])

        # 3. Generate subsegment split points using pair list boundaries
        total_left_duration = seg1_end - seg1_start
        left_targets = [seg1_start + i * total_left_duration / segment_count for i in range(segment_count + 1)]

        def closest_pair(value, pairs):
            return min(pairs, key=lambda p: abs(p[0] - value))

        selected_pairs = [closest_pair(t, segment_pairs) for t in left_targets]

        # Merge short segments with the shorter neighbor
        cleaned_pairs = []
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
                elif i > 0:
                    prev = cleaned_pairs[-1]
                    cleaned_pairs[-1] = (prev[0], l_end, prev[2], r_end)
                    i += 1
                    continue

            cleaned_pairs.append((l_start, l_end, r_start, r_end))
            i += 1

        debug.dump_pairs(cleaned_pairs)

        temp_segments = []
        for idx, (l_start, l_end, r_start, r_end) in enumerate(cleaned_pairs):
            left_duration = l_end - l_start
            right_duration = r_end - r_start
            ratio = right_duration / left_duration

            if abs(ratio - 1.0) > 0.10:
                self.logger.error(f"Segment {idx} duration mismatch exceeds 10%")

            raw_cut = os.path.join(wd, f"cut_{idx}.flac")
            scaled_cut = os.path.join(wd, f"scaled_{idx}.flac")

            process_utils.start_process("ffmpeg", [
                "-y", "-ss", str(r_start / 1000), "-to", str(r_end / 1000),
                "-i", v2_audio, "-c:a", "flac", raw_cut
            ])

            process_utils.start_process("ffmpeg", [
                "-y", "-i", raw_cut,
                "-filter:a", f"atempo={ratio:.3f}",
                "-c:a", "flac", scaled_cut
            ])

            temp_segments.append(scaled_cut)

        # 4. Combine all audio
        concat_list = os.path.join(wd, "concat.txt")
        with open(concat_list, "w") as f:
            f.write(f"file '{head_path}'\n")
            for seg in temp_segments:
                f.write(f"file '{seg}'\n")
            f.write(f"file '{tail_path}'\n")

        merged_flac = os.path.join(wd, "merged.flac")
        process_utils.start_process("ffmpeg", [
            "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c:a", "flac", merged_flac
        ])

        # 5. Re-encode to output file
        process_utils.start_process("ffmpeg", [
            "-y", "-i", merged_flac, "-c:a", "aac", "-movflags", "+faststart", output_path
        ])

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

    def _print_file_details(self, file: str, details: Dict[str, Any], ids: Dict[str, int]):
        def formatter(key: str, value: Any) -> str:
            if key == "fps":
                return eval(value)
            elif key == "length":
                return generic_utils.ms_to_time(value) if value else "-"
            else:
                return value if value else "-"

        def show(key: str) -> bool:
            if key == "tid":
                return False
            else:
                return True

        id = ids[file]
        self.logger.info(f"File #{id} details:")
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

                id = stream.get("tid")
                self.logger.info(f"    #{id}: {info}")

        for attachment in attachments:
            file_name = attachment["file_name"]
            self.logger.info(f"  attachment: {file_name}")

        # more details for debug
        for stream_type, streams in tracks.items():
            self.logger.debug(f"\t{stream_type}:")

            for stream in streams:
                id = stream.get("tid")
                self.logger.debug(f"\t#{id}:")
                for key, value in stream.items():
                    if show(key):
                        key_title = key + ":"
                        self.logger.debug(
                            f"\t\t{key_title:<16}{formatter(key, value)}")

    def _print_streams_details(self, ids: Dict[str, int], all_streams: List, tracks: Dict[str, Dict]):
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

                id = ids[path]
                self.logger.info(f"{stype} track #{tid}: {language} from file #{id}{extra}")

    def _print_attachements_details(self, ids: Dict[str, int], all_attachments: List):
         for stream in all_attachments:
            path = stream[0]
            tid = stream[1]

            id = ids[path]
            self.logger.info(f"Attachment ID #{tid} from file #{id}")

    def _process_duplicates(self, duplicates: List[str], ids: Dict[str, int]) -> Tuple[Dict, List] | None:
        # analyze files in terms of quality and available content
        # use mkvmerge-based probing enriched with ffprobe data
        details_full = {file: video_utils.get_video_data_mkvmerge(file, enrich=True) for file in duplicates}
        attachments = {file: info["attachments"] for file, info in details_full.items()}
        tracks = {file: info["tracks"] for file, info in details_full.items()}

        # print input file details
        for file, file_details in details_full.items():
            self._print_file_details(file, file_details, ids)

        streams_picker = StreamsPicker(self.logger, self.duplicates_source, allow_language_guessing = self.allow_language_guessing)
        try:
            video_streams, audio_streams, subtitle_streams = streams_picker.pick_streams(tracks, ids)
        except RuntimeError as re:
            self.logger.error(re)
            return None

        # verify if all chosen streams come from inputs of similar length
        used_paths = {path for path, _, _ in (video_streams + audio_streams + subtitle_streams)}
        lengths = [tracks[path]["video"][0]["length"] for path in used_paths]
        if len(lengths) > 1:
            base = lengths[0]
            if any(abs(base - l) > 100 for l in lengths[1:]):
                self.logger.warning("Input video lengths differ. Check for --allow-length-mismatch option.")
                if not self.allow_length_mismatch:
                    return None

        picked_attachments = AttachmentsPicker(self.logger).pick_attachments(attachments)

        # print proposed output file
        self.logger.info("Streams used to create output video file:")
        self._print_streams_details(
            ids,
            [(stype, streams) for stype, streams in zip(["video", "audio", "subtitle"], [video_streams, audio_streams, subtitle_streams])],
            tracks,
        )
        self._print_attachements_details(ids, picked_attachments)

        # build streams mapping
        streams = defaultdict(list)

        #   process video streams
        for path, tid, language in video_streams:
            streams[path].append({
                "tid": tid,
                "language": language,
                "type": "video",
            })

        #   process audio streams

        #       check if input files are of the same length
        video_stream = video_streams[0]
        video_stream_path = video_stream[0]
        video_stream_index = video_stream[1]

        base_length = tracks[video_stream_path]["video"][video_stream_index]["length"]
        file_name = 0
        self.logger.debug(f"Using video file {video_stream_path}:{video_stream_index} as a base")

        for path, tid, language in audio_streams:
            length = tracks[path]["video"][0]["length"]

            if abs(base_length - length) > 100:
                id = ids[path]
                self.logger.warning(f"Audio stream from file #{id} has length different than length of video stream from file {video_stream_path}.")

                if self.live_run:
                    self.logger.info("Starting videos comparison to solve mismatching lengths.")
                    # more than 100ms difference in length, perform content matching

                    with files_utils.ScopedDirectory(os.path.join(self.wd, "matching")) as mwd, \
                         generic_utils.TqdmBouncingBar(desc="Processing", **generic_utils.get_tqdm_defaults()):

                        pairMatcher = PairMatcher(self.interruption, mwd, video_stream_path, path, self.logger.getChild("PairMatcher"))

                        mapping, lhs_all_frames, rhs_all_frames = pairMatcher.create_segments_mapping()
                        output_file = os.path.join(self.wd, f"tmp_{file_name}.m4a")
                        self._patch_audio_segment(mwd, video_stream_path, path, output_file, mapping, 20, lhs_all_frames, rhs_all_frames)

                        file_name += 1
                        path = output_file
                        tid = 0
                else:
                    self.logger.info("Skipping videos comparison to solve mismatching lengths due to dry run.")

            streams[path].append({
                "tid": tid,
                "language": language,
                "type": "audio",
            })

        # process subtitle streams
        for path, tid, language in subtitle_streams:
            length = tracks[path]["video"][0]["length"]

            if abs(base_length - length) > 100:
                id = ids[path]
                error = f"Subtitles stream from file #{id} has length different than length of video stream from file {video_stream_path}. This is not supported yet"
                self.logger.error(error)
                raise RuntimeError(f"Unsupported case: {error}")

            streams[path].append({
                "tid": tid,
                "language": language,
                "type": "subtitle",
            })

        return streams, picked_attachments

    def _process_duplicates_set(self, duplicates: Dict[str, List[str]]):
        def process_entries(entries: List[str]) -> List[Tuple[List[str], str]]:
            # function returns list of: group of duplicates and name for final output file.
            # when dirs are provided as entries, files from each dir are collected and sorted
            # and files on the same position are grouped

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

        for title, entries in tqdm(duplicates.items(), desc="Titles", unit="title", **generic_utils.get_tqdm_defaults(), position=0):

            self.logger.info( "-------------------------" + "-" * len(title))
            self.logger.info(f"Analyzing duplicates for {title}")
            self.logger.info( "-------------------------" + "-" * len(title))

            files_groups = process_entries(entries)

            for files, output_name in tqdm(files_groups, desc="Videos", unit="video", **generic_utils.get_tqdm_defaults(), position=1):
                self.interruption._check_for_stop()

                if len(files_groups) > 1:
                    self.logger.info("------------------------------------")
                    self.logger.info("Processing group of duplicated files")
                    self.logger.info("------------------------------------")

                ids = {}
                for i, file in enumerate(files):
                    id = i + 1
                    ids[file] = id
                    self.logger.info(f"#{id}: {file}")

                result = self._process_duplicates(files, ids)
                if result is None:
                    self.logger.info("Skipping output generation")
                    continue

                streams, attachments = result
                required_input_files = { file_path for file_path in streams }
                required_input_files |= { info[0] for info in attachments }

                output = os.path.join(self.output, title, output_name + ".mkv")
                if os.path.exists(output):
                    if self.live_run:
                        self.logger.info(f"Output file {output} exists, removing it.")
                        os.remove(output)
                    else:
                        self.logger.info(f"Dry run, skipping step: output file {output} exists.")

                output_dir = os.path.dirname(output)
                os.makedirs(output_dir, exist_ok=True)

                if len(required_input_files) == 1:
                    # only one file is being used, just copy it to the output dir
                    first_file_path = list(required_input_files)[0]

                    if self.live_run:
                        self.logger.info(f"File {first_file_path} is superior. Using it whole as an output.")
                        shutil.copy2(first_file_path, output)
                    else:
                        self.logger.info(f"Dry run, skipping step: using whole {first_file_path} file as an output.")
                else:
                    generation_args = ["-o", output]
                    files_opts = {
                        path: {"video": [], "audio": [], "subtitle": [], "attachments": [], "languages": {}, "defaults": set()}
                        for path in required_input_files
                    }

                    # convert streams to list for later sorting
                    streams_list = []
                    for path, infos in streams.items():
                        for info in infos:
                            stream_type = info["type"]
                            stream_index = info["tid"]
                            language = info.get("language", None)

                            streams_list.append((stream_type, stream_index, path, language))

                    # sort streams by language alphabetically
                    streams_list_sorted = sorted(streams_list, key=lambda stream: stream[3] if stream[3] else "")

                    # decide which track should be default
                    default_video_stream = next(filter(lambda stream: stream[0] == "video", streams_list))
                    default_audio_stream = next(filter(lambda stream: stream[0] == "audio", streams_list), None)
                    metadata = self.duplicates_source.get_metadata_for(default_video_stream[2])
                    prod_lang = metadata.get("audio_prod_lang")
                    if prod_lang:
                        preferred_lang = language_utils.unify_lang(prod_lang)
                    else:
                        preferred_lang = default_audio_stream[3] if default_audio_stream else None

                    def find_preferred_audio():
                        for info in streams_list_sorted:
                            if info[0] == "audio" and info[3] == preferred_lang:
                                return info

                        return None

                    preferred_audio = find_preferred_audio()

                    if prod_lang:
                        language_name = language_utils.language_name(prod_lang)
                        if preferred_audio:
                            self.logger.info(f"Setting production audio language '{language_name}' as default.")
                        else:
                            self.logger.warning(f"Production audio language '{language_name}' not found among audio streams.")

                    # collect per-file options and track order
                    track_order = []
                    for stream in streams_list_sorted:
                        stream_type, tid, file_path, language = stream
                        fo: Dict = files_opts[file_path]
                        fo[stream_type].append(tid)
                        fo["languages"][tid] = language or "und"
                        if stream_type in ("audio", "subtitle") and stream == preferred_audio:
                            fo["defaults"].add(tid)
                        file_index = generic_utils.get_key_position(files_opts, file_path)
                        track_order.append(f"{file_index}:{tid}")

                    for file_path, tid in attachments:
                        fo: Dict = files_opts[file_path]
                        fo["attachments"].append(tid)

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

                    if self.live_run:
                        self.logger.info("Starting output file generation from chosen streams.")
                        process_utils.raise_on_error(process_utils.start_process("mkvmerge", generation_args, show_progress = True))
                        self.logger.info(f"{output} saved.")
                    else:
                        self.logger.info("Dry run, skipping output file generation")

    def melt(self):
        with files_utils.ScopedDirectory(self.wd) as wd, logging_redirect_tqdm():
            self.logger.debug(f"Starting `melt` with live run: {self.live_run} and working dir: {self.wd}")
            self.logger.info("Finding duplicates")
            duplicates = self.duplicates_source.collect_duplicates()
            self._process_duplicates_set(duplicates)


class RequireJellyfinServer(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, "jellyfin_server", None) is None:
            parser.error(f"{option_string} requires --jellyfin-server to be specified")
        setattr(namespace, self.dest, values)


class MeltTool(Tool):
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
    def run(self, args, no_dry_run: bool, logger: logging.Logger, working_dir: str):
        interruption = generic_utils.InterruptibleProcess()

        data_source = None
        if args.jellyfin_server:
            path_fix = _split_path_fix(args.jellyfin_path_fix) if args.jellyfin_path_fix else None

            if path_fix and len(path_fix) != 2:
                self.parser.error(f"Invalid content for --jellyfin-path-fix argument. Got: {path_fix}")

            data_source = JellyfinSource(interruption=interruption,
                                         url=args.jellyfin_server,
                                         token=args.jellyfin_token,
                                         path_fix=path_fix,
                                         logger=logger.getChild("JellyfinSource"))
        elif args.input_files:
            title = args.title
            input_entries = args.input_files

            if not title:
                self.parser.error(f"Missing required option: --title")

            data_source = StaticSource(interruption=interruption)

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

                data_source.add_entry(title, path)

                if audio_lang:
                    data_source.add_metadata(path, "audio_lang", audio_lang)
                if audio_prod_lang:
                    data_source.add_metadata(path, "audio_prod_lang", audio_prod_lang)

        melter = Melter(logger,
                        interruption,
                        data_source,
                        live_run = no_dry_run,
                        wd = working_dir,
                        output = args.output_dir,
                        allow_length_mismatch = args.allow_length_mismatch,
                        allow_language_guessing = args.allow_language_guessing,
        )

        melter.melt()
