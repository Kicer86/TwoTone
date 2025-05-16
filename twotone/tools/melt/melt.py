
import argparse
import logging
import os
import re

from overrides import override
from typing import Any, Dict, List, Optional, Tuple

from .. import utils
from ..tool import Tool
from ..utils2 import files, process, video
from .debug_routines import DebugRoutines
from .duplicates_source import DuplicatesSource
from .jellyfin import JellyfinSource
from .pair_matcher import PairMatcher
from .static_source import StaticSource

FramesInfo = Dict[int, Dict[str, str]]

def _split_path_fix(value: str) -> List[str]:
    pattern = r'"((?:[^"\\]|\\.)*?)"'

    matches = re.findall(pattern, value)
    return [match.replace(r'\"', '"') for match in matches]


def iter_starting_with(d: dict, start_key) -> Tuple:
    """Yield (key, value) pairs starting from start_key, then the rest in order."""
    if start_key not in d:
        raise KeyError(f"{start_key} not found in dictionary")

    yielded = set()

    # Yield the starting key first
    yield start_key, d[start_key]
    yielded.add(start_key)

    # Yield the rest in order
    for k, v in d.items():
        if k not in yielded:
            yield k, v


class Melter():
    def __init__(self, logger, interruption: utils.InterruptibleProcess, duplicates_source: DuplicatesSource, live_run: bool):
        self.logger = logger
        self.interruption = interruption
        self.duplicates_source = duplicates_source
        self.live_run = live_run
        self.debug_it: int = 0

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
        :param output_path: Path to final video output
        :param min_subsegment_duration: minimum duration in seconds below which a subsegment is merged with neighbor
        """

        wd = os.path.join(wd, "audio extraction")
        debug_wd = os.path.join(wd, "debug")
        os.makedirs(wd)
        os.makedirs(debug_wd)

        v1_audio = os.path.join(wd, "v1_audio.flac")
        v2_audio = os.path.join(wd, "v2_audio.flac")
        head_path = os.path.join(wd, "head.flac")
        tail_path = os.path.join(wd, "tail.flac")
        final_audio = os.path.join(wd, "final_audio.m4a")
        final_audio = os.path.join(wd, "final_audio.m4a")

        debug = DebugRoutines(debug_wd, lhs_frames, rhs_frames)

        # Compute global segment range
        s1_all = [p[0] for p in segment_pairs]
        s2_all = [p[1] for p in segment_pairs]
        seg1_start, seg1_end = min(s1_all), max(s1_all)
        seg2_start, seg2_end = min(s2_all), max(s2_all)

        # 1. Extract main audio
        process.start_process("ffmpeg", ["-y", "-i", video1_path, "-map", "0:a:0", "-c:a", "flac", v1_audio])
        process.start_process("ffmpeg", ["-y", "-i", video2_path, "-map", "0:a:0", "-c:a", "flac", v2_audio])

        # 2. Extract head and tail
        process.start_process("ffmpeg", ["-y", "-ss", "0", "-to", str(seg1_start / 1000), "-i", v1_audio, "-c:a", "flac", head_path])
        process.start_process("ffmpeg", ["-y", "-ss", str(seg1_end / 1000), "-i", v1_audio, "-c:a", "flac", tail_path])

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

            process.start_process("ffmpeg", [
                "-y", "-ss", str(r_start / 1000), "-to", str(r_end / 1000),
                "-i", v2_audio, "-c:a", "flac", raw_cut
            ])

            process.start_process("ffmpeg", [
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
        process.start_process("ffmpeg", [
            "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c:a", "flac", merged_flac
        ])

        # 5. Re-encode to AAC
        process.start_process("ffmpeg", [
            "-y", "-i", merged_flac, "-c:a", "aac", "-movflags", "+faststart", final_audio
        ])

        # 6. Generate final MKV
        utils.generate_mkv(
            output_path=output_path,
            input_video=video1_path,
            subtitles=[],
            audios=[{"path": final_audio, "language": "eng", "default": True}]
        )

    def _print_file_details(self, file: str, details: Dict[str, Any]):
        def formatter(key: str, value: any) -> str:
            if key == "fps":
                return eval(value)
            elif key == "length":
                return utils.ms_to_time(value)
            else:
                return value if value else "-"

        def show(key: str) -> bool:
            if key == "tid":
                return False
            else:
                return True

        self.logger.debug(f"File {file} details:")
        for stream_type, streams in details.items():
            self.logger.debug(f"\t{stream_type}:")

            for i, stream in enumerate(streams):
                self.logger.debug(f"\t#{i + 1}:")
                for key, value in stream.items():
                    if show(key):
                        key_title = key + ":"
                        self.logger.debug(
                            f"\t\t{key_title:<16}{formatter(key, value)}")

    def _pick_best_video(self, files_details: Dict[str, Dict]) -> str:
        best_file = None
        best_stream = None

        # todo: handle many video streams
        default_video_stream = 0

        for file, details in files_details.items():
            if best_file is None:
                best_file = file
                best_stream = details
                continue

            lhs_video_stream = best_stream["video"][default_video_stream]
            rhs_video_stream = details["video"][default_video_stream]

            if lhs_video_stream["width"] < rhs_video_stream["width"] and lhs_video_stream["height"] < rhs_video_stream["height"]:
                best_file = file
                best_stream = details
            elif lhs_video_stream["width"] > rhs_video_stream["width"] and lhs_video_stream["height"] > rhs_video_stream["height"]:
                continue
            else:
                # equal or mixed width/height
                if eval(lhs_video_stream["fps"]) < eval(rhs_video_stream["fps"]):
                    best_file = file
                    best_stream = details

        return best_file, default_video_stream

    def _pick_unique_streams(
        self,
        files_details: Dict[str, Dict],
        best_file: str,
        stream_type: str,
        unique_keys: List[str],
        preference_keys: List[str],
        fallback_languages: Optional[Dict[str, str]] = None,
        override_languages: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, int, str]]:
        """
        Select unique streams of a given type across multiple files.

        Args:
            files_details: Mapping from file path to metadata.
            best_file: File to prioritize when collecting streams.
            stream_type: Stream type to extract (e.g., "audio", "subtitles").
            unique_keys: Keys that define stream uniqueness (first must be 'language').
            preference_keys: Keys used to resolve ties (e.g., ["bitrate"]).
            fallback_languages: Optional per-file fallback if stream["language"] is None.
            override_languages: Optional per-file override to force stream["language"].

        Returns:
            List of (file_path, stream_id, language) tuples.
        """

        assert unique_keys and unique_keys[0] == "language", "First unique_key must be 'language'"

        stream_index = {}

        for path, details in iter_starting_with(files_details, best_file):
            for index, stream in enumerate(details.get(stream_type, [])):
                # Determine language
                lang = stream.get("language")
                if override_languages and path in override_languages:
                    original_lang = lang
                    lang = override_languages[path]
                    tid = stream["tid"]
                    if original_lang:
                        self.logger.info(f"Overriding {stream_type} stream #{tid} language {original_lang} with {lang} for file {path}")
                    else:
                        self.logger.info(f"Setting {stream_type} stream #{tid} missing language to {lang} for file {path}")
                elif (not lang) and fallback_languages and path in fallback_languages:
                    original_lang = lang
                    lang = fallback_languages[path]
                    tid = stream["tid"]
                    self.logger.info(f"Setting {stream_type} stream #{tid} missing language to {lang} for file {path}")

                # Build a modified copy of the stream for comparison
                stream_view = stream.copy()
                stream_view["language"] = lang

                # Build unique key based on stream view
                unique_key = tuple(stream_view.get(k) for k in unique_keys)
                language = unique_key[0] if unique_key else "default"

                current = {"file": path, "index": index, "details": stream}

                if unique_key not in stream_index:
                    stream_index[unique_key] = current
                    continue

                existing = stream_index[unique_key]
                existing_view = existing["details"].copy()
                existing_lang = existing_view.get("language")
                if override_languages and existing["file"] in override_languages:
                    existing_lang = override_languages[existing["file"]]
                elif (not existing_lang) and fallback_languages and existing["file"] in fallback_languages:
                    existing_lang = fallback_languages[existing["file"]]
                existing_view["language"] = existing_lang

                if preference_keys:
                    better = False
                    for key in preference_keys:
                        old = existing_view.get(key)
                        new = stream_view.get(key)
                        if old is None or new is None:
                            continue
                        if new > old:
                            better = True
                            break
                        elif new < old:
                            break
                    if better:
                        stream_index[unique_key] = current

        # Flatten result
        result = []
        for unique_key, entry in stream_index.items():
            index = entry["index"]
            path = entry["file"]
            language = unique_key[0]
            result.append((path, index, language))

        return result

    def _process_duplicates(self, duplicates: List[str]):
        with files.ScopedDirectory("/tmp/twotone/melter") as wd:
            # analyze files in terms of quality and available content
            def video_details(path: str):
                details = video.get_video_data2(path)

                return details

            details = {file: video_details(file) for file in duplicates}

            for file, file_details in details.items():
                self._print_file_details(file, file_details)

            # pick video stream
            best_video, video_stream = self._pick_best_video(details)

            # pick audio streams
            forced_audio_language = {path: self.duplicates_source.get_metadata_for(path).get("audio_lang") for path in details}
            forced_audio_language = {path: lang for path, lang in forced_audio_language.items() if lang}
            audio_streams = self._pick_unique_streams(details, best_video, "audio", ["language", "channels"], ["sample_rate"], override_languages=forced_audio_language)

            # pick subtitle streams
            forced_subtitle_language = {path: self.duplicates_source.get_metadata_for(path).get("subtitle_lang") for path in details}
            forced_subtitle_language = {path: lang for path, lang in forced_subtitle_language.items() if lang}
            subtitle_streams = self._pick_unique_streams(details, best_video, "subtitle", ["language"], [], override_languages=forced_subtitle_language)

            # validate video files
            used_video_files = set()
            used_video_files.add(best_video)
            used_video_files.update({stream[0] for stream in audio_streams})
            used_video_files.update({stream[0] for stream in subtitle_streams})

            if len(used_video_files) == 1:
                self.logger.info(f"File {used_video_files[0]} contains best content. Other files are not needed.")

                # todo: just copy to output dir
            else:
                # check if input files are of the same lenght
                base_lenght = details[best_video]["video"][video_stream]["length"]

                for path, index, _ in audio_streams:
                    lenght = details[path]["video"][index]["length"]

                    if abs(base_lenght - lenght) > 100:     # more than 100ms difference in lenght, perform content matching
                        pairMatcher = PairMatcher(wd, best_video, path, self.logger.getChild("PairMatcher"))

                        mapping, lhs_all_frames, rhs_all_frames = pairMatcher.create_segments_mapping()
                        self._patch_audio_segment(wd, duplicates[0], duplicates[1], os.path.join(wd, "final.mkv"), mapping, 20, lhs_all_frames, rhs_all_frames)

        return


        video_details = [video.get_video_data2(video_file) for video_file in duplicates]
        video_lengths = {video.video_tracks[0].length for video in video_details}

        if len(video_lengths) == 1:
            # all files are of the same lenght
            # remove all but first one
            logging.info("Removing exact duplicates. Leaving one copy")
            if self.live_run:
                for file in duplicates[1:]:
                    os.remove(file)
        else:
            logging.warning("Videos have different lengths, skipping")


    def _process_duplicates_set(self, duplicates: Dict[str, List[str]]):
        for title, files in duplicates.items():
            logging.info(f"Analyzing duplicates for {title}")

            self._process_duplicates(files)


    def melt(self):
        self.logger.info("Finding duplicates")
        duplicates = self.duplicates_source.collect_duplicates()
        self._process_duplicates_set(duplicates)
        #print(json.dumps(duplicates, indent=4))


class RequireJellyfinServer(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, "jellyfin_server", None) is None:
            parser.error(
                f"{option_string} requires --jellyfin-server to be specified")
        setattr(namespace, self.dest, values)


class MeltTool(Tool):
    @override
    def setup_parser(self, parser: argparse.ArgumentParser):

        jellyfin_group = parser.add_argument_group("Jellyfin source")
        jellyfin_group.add_argument('--jellyfin-server',
                                    help='URL to the Jellyfin server which will be used as a source of video files duplicates')
        jellyfin_group.add_argument('--jellyfin-token',
                                    action=RequireJellyfinServer,
                                    help='Access token (http://server:8096/web/#/dashboard/keys)')
        jellyfin_group.add_argument('--jellyfin-path-fix',
                                    action=RequireJellyfinServer,
                                    help='Specify a replacement pattern for file paths to ensure "melt" can access Jellyfin video files.\n\n'
                                         '"Melt" requires direct access to video files. If Jellyfin is not running on the same machine as "melt,"\n'
                                         'you must set up network access to Jellyfin’s video storage and specify how paths should be resolved.\n\n'
                                         'For example, suppose Jellyfin runs on a Linux machine where the video library is stored at "/srv/videos" (a shared directory).\n'
                                         'If "melt" is running on another Linux machine that accesses this directory remotely at "/mnt/shared_videos,"\n'
                                         'you need to map "/srv/videos" (Jellyfin’s path) to "/mnt/shared_videos" (the path accessible on the machine running "melt").\n\n'
                                         'In this case, use: --jellyfin-path-fix "/srv/videos","/mnt/shared_videos" to define the replacement pattern.')

        manual_group = parser.add_argument_group("Manual input source")
        manual_group.add_argument('-t', '--title',
                                  help='Video (movie or series when directory is provided as an input) title.')
        manual_group.add_argument('-i', '--input', dest='input_files', action='append',
                                  help='Add an input video file or directory with video files (can be specified multiple times).\n'
                                       'path can be followed with a comma and some additional parameters:\n'
                                       'audio_lang:XXX  - information about audio language (like eng, de or pl).\n\n'
                                       'Example of usage:\n'
                                       '--input some/path/file.mp4,audio_lang:jp')


    @override
    def run(self, args, no_dry_run: bool, logger: logging.Logger):
        interruption = utils.InterruptibleProcess()

        data_source = None
        if args.jellyfin_server:
            path_fix = _split_path_fix(args.jellyfin_path_fix) if args.jellyfin_path_fix else None

            if path_fix and len(path_fix) != 2:
                raise ValueError(f"Invalid content for --jellyfin-path-fix argument. Got: {path_fix}")

            data_source = JellyfinSource(interruption=interruption,
                                         url=args.jellyfin_server,
                                         token=args.jellyfin_token,
                                         path_fix=path_fix)
        elif args.input_files:
            title = args.title
            input_entries = args.input_files

            data_source = StaticSource(interruption=interruption)

            for input in input_entries:
                # split by ',' but respect ""
                input_split = re.findall(r'(?:[^,"]|"(?:\\"|[^"])*")+', input)
                path = input_split[0]

                if not os.path.exists(path):
                    raise ValueError(f"Path {path} does not exist")

                audio_lang = ""

                if len(input_split) > 1:
                    for extra_arg in input_split[1:]:
                        if extra_arg[:11] == "audio_lang:":
                            audio_lang = extra_arg[11:]

                data_source.add_entry(title, path)

                if audio_lang:
                    data_source.add_metadata(path, "audio_lang", audio_lang)

        melter = Melter(logger, interruption, data_source, live_run = no_dry_run)
        melter.melt()
