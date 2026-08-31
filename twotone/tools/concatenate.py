
import argparse
import logging
import os
import re
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any
from overrides import override
from tqdm import tqdm

from .tool import EmptyPlan, Plan, Tool
from twotone.tools.utils import generic_utils, process_utils, video_utils, files_utils


class Concatenate(generic_utils.InterruptibleProcess):
    def __init__(self, logger: logging.Logger, workspace: files_utils.Workspace):
        super().__init__(logger)
        self.logger = logger
        self.workspace = workspace

    def analyze(self, path: str, ignore_warnings: bool = False) -> dict[str, list[tuple[str, int]]] | None:
        self.logger.info(f"Collecting video files from path {path}")
        video_files = video_utils.collect_video_files(path, self)

        self.logger.info("Finding splitted videos")
        parts_regex = re.compile("(.*[^0-9a-z]+)(cd\\d+)([^0-9a-z]+.*)", re.IGNORECASE)

        splitted = []
        for video_file in video_files:
            if parts_regex.match(video_file):
                splitted.append(video_file)
            else:
                self.logger.debug(f"File {video_file} does not match pattern")

        self.logger.info("Matching videos")
        matched_videos: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for video in splitted:
            match = parts_regex.search(video)
            if match is None:
                continue

            path = match.group(1)
            if path[-1] == os.sep:
                # movie path is like: /dir/movie/cd1.mp4
                # repeat last dir name as a base for output video file
                last_dir_name = os.path.basename(os.path.normpath(path))
                name_without_part_number = os.path.join(path, last_dir_name) + match.group(3)
            else:
                # movie path is like: /dir/movie/movie cd1.mp4.
                # remove last char before CDXXX as it is most likely space or hyphen and use it as a base for output video file
                name_without_part_number = path[:-1] + match.group(3)

            # ffmpeg does not support rmvb container, use mp4
            dir, name, extn = files_utils.split_path(name_without_part_number)
            if extn.lower() == "rmvb":
                name_without_part_number = os.path.join(dir, name + ".mkv")

            part = match.group(2)
            partNo = int(part[2:])                                                                              # drop 'CD'
            matched_videos[name_without_part_number].append((video, partNo))

        self.logger.info("Processing groups")
        warnings = False
        sorted_videos: dict[str, list[tuple[str, int]]] = {}
        valid_videos: dict[str, list[tuple[str, int]]] = {}
        for common_name, details in matched_videos.items():

            # sort parts by part number [1]
            details = sorted(details, key = lambda detail: detail[1])
            sorted_videos[common_name] = details
            group_has_warning = False

            if os.path.lexists(common_name):
                self.logger.error("Output file already exists, skipping group: %s", common_name)
                warnings = True
                group_has_warning = True

            # collect all part numbers
            parts = []
            for _, partNo in details:
                parts.append(partNo)

            if len(parts) < 2:
                self.logger.warning(f"There are less than two parts for video represented under a common name: {common_name}")
                warnings = True
                group_has_warning = True

            # expect parts to be numbered from 1 to N
            for i, value in enumerate(parts):
                if i + 1 != value:
                    self.logger.warning(f"There is a mismatch in CD numbers for a group of files represented under a common name: {common_name}")
                    warnings = True
                    group_has_warning = True

            if group_has_warning and ignore_warnings:
                continue
            valid_videos[common_name] = details

        if warnings and not ignore_warnings:
            self.logger.error("Fix above warnings and try again")
            return None

        if ignore_warnings:
            return valid_videos
        return sorted_videos

    def analyze_files(self, input_files: list[str], output: str) -> dict[str, list[tuple[str, int]]] | None:
        if len(input_files) < 2:
            self.logger.error("At least two video files are required for concatenation")
            return None

        invalid_files = [path for path in input_files if not os.path.isfile(path) or not video_utils.is_video(path)]
        if invalid_files:
            for path in invalid_files:
                self.logger.error("Not a video file: %s", path)
            return None

        resolved_output = os.path.normcase(os.path.realpath(os.path.abspath(output)))
        if os.path.lexists(output):
            self.logger.error("Output file already exists: %s", output)
            return None
        if resolved_output in {os.path.normcase(os.path.realpath(os.path.abspath(path))) for path in input_files}:
            self.logger.error("Output file must not be one of the input files: %s", output)
            return None

        return {output: [(path, part_number) for part_number, path in enumerate(input_files, start=1)]}

    @staticmethod
    def _is_mkv_concatenation(input_files: list[str], output: str) -> bool:
        return Path(output).suffix.lower() == ".mkv" and all(Path(path).suffix.lower() == ".mkv" for path in input_files)

    def _validate_mkv_tracks(self, input_files: list[str]) -> bool:
        reference_path = input_files[0]
        reference_tracks = video_utils.get_video_full_info_mkvmerge(reference_path, logger=self.logger).get("tracks", [])
        reference_by_id = {track.get("id"): track for track in reference_tracks}

        for path in input_files[1:]:
            tracks = video_utils.get_video_full_info_mkvmerge(path, logger=self.logger).get("tracks", [])
            tracks_by_id = {track.get("id"): track for track in tracks}
            if reference_by_id.keys() != tracks_by_id.keys():
                self.logger.error("MKV files have different track IDs: %s", path)
                return False

            for track_id, reference_track in reference_by_id.items():
                track = tracks_by_id[track_id]
                if self._mkv_track_signature(reference_track) != self._mkv_track_signature(track):
                    self.logger.error("MKV files have incompatible track %s: %s", track_id, path)
                    return False

        return True

    @staticmethod
    def _mkv_track_signature(track: dict[str, Any]) -> tuple[Any, Any]:
        return track.get("type"), track.get("codec")

    def _perform_mkv_concatenation(self, input_files: list[str], output: str) -> bool:
        if not self._validate_mkv_tracks(input_files):
            self.logger.error("Skipping MKV concatenation because input track layouts differ: %s", output)
            return False

        mkvmerge_args = ["-o", output, input_files[0]]
        for input_file in input_files[1:]:
            mkvmerge_args.extend(["+", input_file])

        self.logger.info("Concatenating MKV files into %s", output)
        status = process_utils.start_process("mkvmerge", mkvmerge_args, logger=self.logger)
        if status.returncode in (0, 1) and os.path.exists(output):
            return True

        self.logger.error("Problems with MKV concatenation, skipping file %s", output)
        self.logger.debug(status.stdout)
        self.logger.debug(status.stderr)
        return False

    def _perform_non_mkv_concatenation(self, input_files: list[str], output: str) -> bool:
        audio_codec = "copy"
        for input_file in input_files:
            file_details = video_utils.get_video_data(input_file, logger=self.logger)
            audio_streams = file_details.get("audio", [])
            for audio_stream in audio_streams:
                codec = audio_stream.get("codec")
                if codec and codec.lower() == "cook":
                    audio_codec = "aac"
                    break

        def escape_path(path: str) -> str:
            return path.replace("'", "'\\''")

        # The concat demuxer resolves relative paths against this temporary
        # list file, which lives in the working directory rather than the
        # caller's current directory.
        input_file_content = [f"file '{escape_path(os.path.abspath(input_file))}'" for input_file in input_files]
        with self.workspace.text_file("\n".join(input_file_content), "txt") as input_file:
            ffmpeg_args = ["-f", "concat", "-safe", "0", "-i", input_file, "-c:v", "copy", "-c:a", audio_codec, output]

            self.logger.info(f"Concatenating files into {output} file")
            status = process_utils.start_process("ffmpeg", ffmpeg_args, logger=self.logger)
            if status.returncode == 0:
                return True

        self.logger.error(f"Problems with concatenation, skipping file {output}")
        self.logger.debug(status.stdout)
        self.logger.debug(status.stderr)
        return False

    def perform(self, sorted_videos: dict[str, list[tuple[str, int]]]) -> None:
        self.logger.info("Starting concatenation")
        for output, details in tqdm(sorted_videos.items(), desc="Concatenating", unit="movie", **generic_utils.get_tqdm_defaults()):
            self.check_for_stop()

            input_files = [video for video, _ in details]

            with self.workspace.staging_for(output) as staged_output:
                if self._is_mkv_concatenation(input_files, output):
                    succeeded = self._perform_mkv_concatenation(input_files, staged_output.path)
                else:
                    succeeded = self._perform_non_mkv_concatenation(input_files, staged_output.path)

                if succeeded:
                    video_utils.validate_media_output(staged_output.path, logger=self.logger)
                    staged_output.commit()

            if succeeded:
                for input_file in input_files:
                    try:
                        os.remove(input_file)
                    except OSError as error:
                        self.logger.warning("Concatenated output was saved, but could not remove input %s: %s", input_file, error)


class ConcatenateTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    @override
    def setup_parser(self, parser: argparse.ArgumentParser):
        parser.description = (
            "Concatenate is a tool for concatenating video files splitted into many files into one.\n"
            "For example if you have movie consisting of two files: movie-cd1.avi and movie-cd2.avi\n"
            "then 'concatenate' tool will glue them into one file 'movie.avi'.\n"
            "If your files come with subtitle files, you may want to use 'merge' tool first\n"
            "to merge video files with corresponding subtitle files.\n"
            "Otherwise you will end up with one video file and two subtitle files for cd1 and cd2 which will be useless now"
        )
        parser.add_argument('inputs',
                            nargs='+',
                            help='One directory to scan recursively, or video files to concatenate in this order.')
        parser.add_argument('--output', '-o',
                            help='Output path when concatenating explicitly provided video files.')
        parser.add_argument('--ignore-warnings',
                            action='store_true',
                            help='Skip videos with warnings and continue with valid groups.')

    @override
    def analyze(self, args, logger: logging.Logger, workspace: files_utils.Workspace) -> Plan:
        concatenator = Concatenate(logger, workspace=workspace)
        inputs = args.inputs
        directories = [path for path in inputs if os.path.isdir(path)]
        files = [path for path in inputs if os.path.isfile(path)]
        invalid_inputs = [path for path in inputs if path not in directories and path not in files]

        if invalid_inputs:
            for path in invalid_inputs:
                logger.error("Input path does not exist: %s", path)
            return EmptyPlan()

        if directories and files:
            logger.error("Provide either one directory or a list of video files, not both")
            return EmptyPlan()

        if directories:
            if len(directories) != 1:
                logger.error("Provide exactly one directory to scan")
                return EmptyPlan()
            if args.output:
                logger.error("--output is only supported when concatenating explicitly provided files")
                return EmptyPlan()
            analysis = concatenator.analyze(directories[0], ignore_warnings=args.ignore_warnings)
        else:
            if not args.output:
                logger.error("--output is required when concatenating explicitly provided files")
                return EmptyPlan()
            analysis = concatenator.analyze_files(files, args.output)
        if analysis is None:
            return EmptyPlan()

        for output, details in analysis.items():
            input_files = [path for path, _ in details]
            if concatenator._is_mkv_concatenation(input_files, output) and not concatenator._validate_mkv_tracks(input_files):
                logger.error("MKV concatenation requires matching track IDs, types, and codecs")
                return EmptyPlan()
        return ConcatenatePlan(items=analysis)

    @override
    def perform(self, args, logger: logging.Logger, workspace: files_utils.Workspace, plan: Plan) -> None:
        if not isinstance(plan, ConcatenatePlan):
            raise TypeError(f"Expected ConcatenatePlan, got {type(plan).__name__}")

        concatenator = Concatenate(logger, workspace)
        concatenator.perform(plan.items)


@dataclass
class ConcatenatePlan:
    items: dict[str, list[tuple[str, int]]]

    def is_empty(self) -> bool:
        return not self.items

    def input_files(self) -> set[str]:
        return {
            path
            for details in self.items.values()
            for path, _part in details
        }

    def render(self, logger: logging.Logger) -> None:
        if not self.items:
            logger.info("No videos to concatenate.")
            return

        logger.info("Planned concatenations: %d", len(self.items))
        for output, details in self.items.items():
            paths = [path for path, _ in details]
            common_path = os.path.commonpath(paths) if paths else ""
            if common_path:
                logger.info("Files from %s:", common_path)
                for path, _ in details:
                    logger.info("  %s", os.path.relpath(path, common_path))
            else:
                for path, part in details:
                    logger.info("  part %d: %s", part, path)
            logger.info("  -> %s", output)
