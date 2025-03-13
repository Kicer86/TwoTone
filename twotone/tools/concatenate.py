
import argparse
import logging
import os
import re
from collections import defaultdict
from overrides import override
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from . import utils
from .tool import Tool


class Concatenate(utils.InterruptibleProcess):
    def __init__(self, live_run: bool):
        super().__init__()

        self.live_run = live_run

    def run(self, path: str):
        logging.info(f"Collecting video files from path {path}")
        video_files = utils.collect_video_files(path, self)

        logging.info("Finding splitted videos")
        parts_regex = re.compile("(.*[^0-9a-z]+)(cd\\d+)([^0-9a-z]+.*)", re.IGNORECASE)

        splitted = []
        for video_file in video_files:
            if parts_regex.match(video_file):
                splitted.append(video_file)
            else:
                logging.debug(f"File {video_file} does not match pattern")

        logging.info("Matching videos")
        matched_videos = defaultdict(list)
        for video in splitted:
            match = parts_regex.search(video)

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

            part = match.group(2)
            partNo = int(part[2:])                                                                              # drop 'CD'
            matched_videos[name_without_part_number].append((video, partNo))

        logging.info("Processing groups")
        warnings = False
        sorted_videos = {}
        for common_name, details in matched_videos.items():

            # sort parts by part number [1]
            details = sorted(details, key = lambda detail: detail[1])
            sorted_videos[common_name] = details

            # collect all part numbers
            parts = []
            for _, partNo in details:
                parts.append(partNo)

            if len(parts) < 2:
                logging.warning(f"There are less than two parts for video represented under a common name: {common_name}")
                warnings = True

            # expect parts to be numbered from 1 to N
            for i, value in enumerate(parts):
                if i + 1 != value:
                    logging.warning(f"There is a mismatch in CD numbers for a group of files represented under a common name: {common_name}")
                    warnings = True

        if warnings:
            logging.error("Fix above warnings and try again")
            return

        logging.info("Files to be concatenated (in given order):")
        for common_name, details in sorted_videos.items():
            paths = [path for path, _ in details]
            common_path = os.path.commonpath(paths)
            logging.info(f"Files from {common_path}:")

            cl = len(common_path) + 1
            for path in paths:
                logging.info(f"\t{path[cl:]}")

            logging.info(f"\t->{common_name}")

        logging.info("Starting concatenation")
        with logging_redirect_tqdm():
            for output, details in tqdm(sorted_videos.items(), desc="Concatenating", unit="movie", **utils.get_tqdm_defaults()):
                self._check_for_stop()

                input_files = [video for video, _ in details]

                def escape_path(path: str) -> str:
                    return path.replace("'", "'\\''")

                input_file_content = [f"file '{escape_path(input_file)}'" for input_file in input_files]
                with utils.TempFileManager("\n".join(input_file_content), "txt") as input_file:
                    ffmpeg_args = ["-f", "concat", "-safe", "0", "-i", input_file, "-c", "copy", output]

                    logging.info(f"Concatenating files into {output} file")
                    if self.live_run:
                        status = utils.start_process("ffmpeg", ffmpeg_args)
                        if status.returncode == 0:
                            for input_file in input_files:
                                os.remove(input_file)
                        else:
                            logging.error(f"Problems with concatenation, skipping file {output}")
                            logging.debug(status.stdout)
                            logging.debug(status.stderr)
                            if os.path.exists(output):
                                os.remove(output)
                    else:
                        logging.info("Dry run, skipping concatenation")


class ConcatenateTool(Tool):
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
        parser.add_argument('videos_path',
                            nargs=1,
                            help='Path with videos to concatenate.')

    @override
    def run(self, args):
        concatenate = Concatenate(args.no_dry_run)
        concatenate.run(args.videos_path[0])
