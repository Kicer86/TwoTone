
import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
from overrides import override
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from . import utils
from .tool import Tool
from .utils2 import process


class Fixer(utils.InterruptibleProcess):
    def __init__(self, logger: logging.Logger, really_fix: bool):
        super().__init__()
        self.logger = logger
        self._do_fix = really_fix

    def _print_broken_videos(self, broken_videos_info: [(utils.VideoInfo, [int])]):
        self.logger.info(f"Found {len(broken_videos_info)} broken videos:")
        for broken_video in broken_videos_info:
            self.logger.info(f"{len(broken_video[1])} broken subtitle(s) in {broken_video[0].path} found")

    def _no_resolver(self, video_track: utils.VideoTrack, content: str):
        self.logger.error("Cannot fix the file, no idea how to do it.")
        return None

    def _long_tail_resolver(self, video_track: utils.VideoTrack, content: str):
        timestamps = list(utils.subrip_time_pattern.finditer(content))
        last_timestamp = timestamps[-1]
        time_from, time_to = map(utils.time_to_ms, last_timestamp.groups())
        length = video_track.length
        new_time_to = min(time_from + 5000, length)

        begin_pos = last_timestamp.start(1)
        end_pos = last_timestamp.end(2)

        content = content[:begin_pos] + f"{utils.ms_to_time(time_from)} --> {utils.ms_to_time(new_time_to)}" + content[end_pos:]
        return content

    def _fps_scale_resolver(self, video_track: utils.VideoTrack, content: str):
        target_fps = utils.fps_str_to_float(video_track.fps)
        multiplier = utils.ffmpeg_default_fps / target_fps

        return utils.alter_subrip_subtitles_times(content, multiplier)

    def _get_resolver(self, content: str, video_length: int):
        timestamps = list(utils.subrip_time_pattern.finditer(content))
        if len(timestamps) == 0:
            return self._no_resolver

        # check if last subtitle is beyond limit
        last_timestamp = timestamps[-1]
        time_from, time_to = map(utils.time_to_ms, last_timestamp.groups())

        if time_from < video_length and time_to > video_length:
            return self._long_tail_resolver

        if time_from > video_length and time_to > video_length:
            return self._fps_scale_resolver

        return self._no_resolver

    def _fix_subtitle(self, broken_subtitle, video_info: utils.VideoInfo) -> bool:
        video_track = video_info.video_tracks[0]

        with open(broken_subtitle, 'r', encoding='utf-8') as file:
            content = file.read()

        # figure out what is broken
        resolver = self._get_resolver(content, video_track.length)
        new_content = resolver(video_track, content)

        if new_content is None:
            self.logger.warning("Subtitles not fixed")
            return False
        else:
            with open(broken_subtitle, 'w', encoding='utf-8') as file:
                file.write(new_content)
            return True

    def _extract_all_subtitles(self,video_file: str, subtitles: [utils.Subtitle], wd: str) -> [utils.SubtitleFile]:
        result = []
        options = ["tracks", video_file]

        for subtitle in subtitles:
            outputfile = f"{wd}/{subtitle.tid}.srt"
            subtitleFile = utils.SubtitleFile(path=outputfile, language=subtitle.language, encoding="utf8")

            result.append(subtitleFile)
            options.append(f"{subtitle.tid}:{outputfile}")

        process.start_process("mkvextract", options)

        return result

    def _repair_videos(self, broken_videos_info: [(utils.VideoInfo, [int])]):
        self._print_broken_videos(broken_videos_info)
        self.logger.info("Fixing videos")

        with logging_redirect_tqdm():
            for broken_video in tqdm(broken_videos_info, desc="Fixing", unit="video", leave=False, smoothing=0.1, mininterval=.2, disable=utils.hide_progressbar()):
                self._check_for_stop()

                video_info = broken_video[0]
                broken_subtitiles = broken_video[1]

                with tempfile.TemporaryDirectory() as wd_dir:
                    video_file = video_info.path
                    self.logger.info(f"Fixing subtitles in file {video_file}")
                    self.logger.debug("Extracting subtitles from file")
                    subtitles = self._extract_all_subtitles(video_file, video_info.subtitles, wd_dir)
                    broken_subtitles_paths = [subtitles[i] for i in broken_subtitiles]

                    status = all(self._fix_subtitle(broken_subtitile.path, video_info) for broken_subtitile in broken_subtitles_paths)

                    if status:
                        if self._do_fix:
                            # remove all subtitles from video
                            self.logger.debug("Removing existing subtitles from file")
                            video_without_subtitles = video_file + ".nosubtitles.mkv"
                            process.start_process("mkvmerge", ["-o", video_without_subtitles, "-S", video_file])

                            # add fixed subtitles to video
                            self.logger.debug("Adding fixed subtitles to file")
                            temporaryVideoPath = video_file + ".fixed.mkv"
                            utils.generate_mkv(input_video=video_without_subtitles, output_path=temporaryVideoPath, subtitles=subtitles)

                            # overwrite broken video with fixed one
                            os.replace(temporaryVideoPath, video_file)

                            # remove temporary file
                            os.remove(video_without_subtitles)
                        else:
                            self.logger.info("Not applying fixes - dry run mode.")
                    else:
                        self.logger.debug("Skipping video due to errors")

    def _check_if_broken(self, video_file: str): # -> (utils.VideoInfo, [int]) | None:    // FIXME
        self.logger.debug(f"Processing file {video_file}")

        def diff(a, b):
            return abs(a - b) / max(a, b)

        video_info = utils.get_video_data(video_file)
        video_length = video_info.video_tracks[0].length

        if video_length is None:
            self.logger.warning(f"File {video_file} has unknown length. Cannot proceed.")
            return None

        broken_subtitiles = []

        for i in range(len(video_info.subtitles)):
            subtitle = video_info.subtitles[i]

            if not subtitle.format == "subrip":
                self.logger.warning(f"Cannot analyse subtitle #{i} of {video_file}: unsupported format '{subtitle.format}'")
                continue

            length = subtitle.length
            if length is not None and length > video_length * 1.001:                 # use 0.1% error margin as for some reason valid subtitles may appear longer than video
                broken_subtitiles.append(i)

        if len(broken_subtitiles) == 0:
            self.logger.debug("No issues found")
            return None

        self.logger.debug(f"Issues found in {video_file}")
        return (video_info, broken_subtitiles)

    def _process_dir(self, path: str) -> []:
        broken_videos = []
        video_files = []

        self.logger.debug(f"Finding videos in {path}")
        for cd, _, files in os.walk(path, followlinks = True):
            for file in files:
                self._check_for_stop()
                file_path = os.path.join(cd, file)

                if utils.is_video(file_path):
                    video_files.append(file_path)

        self.logger.debug("Analysing videos")
        with logging_redirect_tqdm():
            for video in tqdm(video_files, desc="Analysing videos", unit="video", leave=False, smoothing=0.1, mininterval=.2, disable=utils.hide_progressbar()):
                self._check_for_stop()
                broken_video = self._check_if_broken(video)
                if broken_video is not None:
                    broken_videos.append(broken_video)

        return broken_videos

    def process_dir(self, path: str):
        broken_videos = self._process_dir(path)

        self._repair_videos(broken_videos)


class FixerTool(Tool):
    @override
    def setup_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument('videos_path',
                            nargs=1,
                            help='Path with videos to analyze.')

    @override
    def run(self, args, no_dry_run: bool, logger: logging.Logger):
        for tool in ["mkvmerge", "mkvextract", "ffprobe"]:
            path = shutil.which(tool)
            if path is None:
                raise RuntimeError(f"{tool} not found in PATH")
            else:
                logging.debug(f"{tool} path: {path}")

        logger.info("Searching for broken files")
        fixer = Fixer(logger, no_dry_run)
        fixer.process_dir(args.videos_path[0])
        logger.info("Done")
