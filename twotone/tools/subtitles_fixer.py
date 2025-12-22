
import argparse
import logging
import os
import pysubs2

from functools import partial
from overrides import override
from tqdm import tqdm
from typing import Callable

from .tool import Tool
from twotone.tools.utils import generic_utils, process_utils, subtitles_utils, video_utils


class Fixer(generic_utils.InterruptibleProcess):
    def __init__(self, logger: logging.Logger, working_dir: str) -> None:
        super().__init__()
        self.logger = logger
        self.working_dir = working_dir

    def _print_broken_videos(self, broken_videos_info: list[tuple[dict, list[int]]]) -> None:
        self.logger.info(f"Found {len(broken_videos_info)} broken videos:")
        for broken_video in broken_videos_info:
            self.logger.info(f"{len(broken_video[1])} broken subtitle(s) in {broken_video[0]['path']} found")

    def _no_resolver(self, video_track: dict, content: pysubs2.SSAFile) -> None:
        self.logger.error("Cannot fix the file, no idea how to do it.")
        return None

    def _drop_subtitle_resolver(self, video_track: dict, content: pysubs2.SSAFile) -> pysubs2.SSAFile | None:
        self.logger.warning("No idea how to fix subtitle, removing it")
        return None

    def _long_tail_resolver(self, video_track: dict, content: pysubs2.SSAFile) -> pysubs2.SSAFile:
        last_timestamp = content[-1]
        time_from = last_timestamp.start
        length = video_track["length"]
        new_time_to = min(time_from + 5000, length)

        content[-1].end = new_time_to

        return content

    def _fps_scale_resolver(self, video_track: dict, content: pysubs2.SSAFile, target_fps: float) -> pysubs2.SSAFile:
        content.transform_framerate(subtitles_utils.ffmpeg_default_fps, target_fps)

        return content

    def _get_resolver(self, content: pysubs2.SSAFile, video_track: dict, drop_broken: bool) -> Callable[[dict, pysubs2.SSAFile], pysubs2.SSAFile | None]:
        if len(content) == 0:
            return self._drop_subtitle_resolver if drop_broken else self._no_resolver

        video_length = video_track["length"]
        video_fps = generic_utils.fps_str_to_float(video_track["fps"])

        # check if last subtitle is beyond limit
        last_timestamp = content[-1]
        time_from = last_timestamp.start
        time_to = last_timestamp.end

        if time_from < video_length and time_to > video_length:
            return self._long_tail_resolver

        if time_from > video_length and time_to > video_length and time_to * subtitles_utils.ffmpeg_default_fps / video_fps < video_length:
            return partial(self._fps_scale_resolver, target_fps = video_fps)

        return self._drop_subtitle_resolver if drop_broken else self._no_resolver

    def _fix_subtitle(self, broken_subtitle: str, video_info: dict, drop_broken: bool) -> bool:
        video_track = video_info["video"][0]

        subs = subtitles_utils.open_subtitle_file(broken_subtitle, fps = video_track["fps"])
        if not subs:
            msg = f"Failed to open subtitles file: {broken_subtitle}"
            raise OSError(msg)

        # figure out what is broken
        resolver = self._get_resolver(subs, video_track, drop_broken)
        new_content = resolver(video_track, subs)

        if new_content is None:
            self.logger.warning("Subtitle not fixed")
            return False
        else:
            new_content.save(broken_subtitle)
            return True

    def _extract_all_subtitles(self, video_file: str, subtitles: list[dict], wd: str) -> dict[str, dict]:
        """Extract all subtitle tracks using subtitles_utils.extract_subtitle_to_temp.

        Builds a stable list of SubtitleFile objects matching the order of
        the provided `subtitles` metadata. The underlying extractor appends
        the tid to the filename and auto-detects proper extension.
        """
        if not subtitles:
            return {}

        tids = [s["tid"] for s in subtitles]
        base_tmp = os.path.join(wd, "subtitle")

        tid_to_path = video_utils.extract_subtitle_to_temp(video_file, tids, base_tmp, logger=self.logger)

        result: dict[str, dict] = {}
        for subtitle in subtitles:
            tid = subtitle["tid"]
            path = tid_to_path[tid]
            result[path] = subtitle

        return result

    def repair_videos(self, broken_videos_info: list[tuple[dict, list[int]]], drop_broken: bool = False) -> None:
        self._print_broken_videos(broken_videos_info)
        self.logger.info("Fixing videos")

        for broken_video in tqdm(broken_videos_info, desc="Fixing", unit="video", leave=False, smoothing=0.1, mininterval=.2, disable=generic_utils.hide_progressbar()):
            self._check_for_stop()

            video_info = broken_video[0]
            broken_subtitiles = broken_video[1]

            wd_dir = self.working_dir
            video_file = video_info["path"]
            self.logger.info(f"Fixing subtitles in file {video_file}")
            self.logger.debug("Extracting subtitles from file")
            subs_info = video_info.get("subtitle", [])
            subtitles = self._extract_all_subtitles(video_file, subs_info, wd_dir)
            broken_subtitles_paths = [path for path, subtitle in subtitles.items() if subtitle["tid"] in broken_subtitiles]

            try:
                new_subtitles = {}

                for path, subtitle in subtitles.items():
                    tid = subtitle["tid"]
                    include = self._fix_subtitle(path, video_info, drop_broken) if tid in broken_subtitiles else True

                    if include:
                        new_subtitles[path] = subtitle
                    else:
                        if drop_broken:
                            self.logger.warning(f"Skipping subtitle #{tid} as it could not be fixed")
                        else:
                            raise Exception("Unfixable subtitle found")

            except Exception as e:
                self.logger.error(e)
                self.logger.debug("Skipping video due to errors")
                return

            # remove all subtitles from video
            self.logger.debug("Removing existing subtitles from file")
            video_without_subtitles = video_file + ".nosubtitles.mkv"
            process_utils.start_process("mkvmerge", ["-o", video_without_subtitles, "-S", video_file])

            # add fixed subtitles to video
            self.logger.debug("Adding fixed subtitles to file")
            temporaryVideoPath = video_file + ".fixed.mkv"

            video_utils.generate_mkv(input_video = video_without_subtitles, output_path = temporaryVideoPath, subtitles = new_subtitles)

            # overwrite broken video with fixed one
            os.replace(temporaryVideoPath, video_file)

            # remove temporary file
            os.remove(video_without_subtitles)


    def _check_if_broken(self, video_file: str) -> tuple[dict, list[int]] | None:
        self.logger.debug(f"Processing file {video_file}")

        video_info = video_utils.get_video_data(video_file)
        video_info["path"] = video_file
        video_length = video_info["video"][0]["length"]

        if video_length is None:
            self.logger.warning(f"File {video_file} has unknown length. Cannot proceed.")
            return None

        broken_subtitiles = []

        for subtitle in video_info.get("subtitle", []):
            tid = subtitle["tid"]

            length = subtitle["length"]
            if length is not None and length > video_length * 1.001:                 # use 0.1% error margin as for some reason valid subtitles may appear longer than video
                broken_subtitiles.append(tid)

        if len(broken_subtitiles) == 0:
            self.logger.debug("No issues found")
            return None

        self.logger.info(f"Issues found in {video_file}")
        return (video_info, broken_subtitiles)

    def scan_directory(self, path: str) -> list[tuple[dict, list[int]]]:
        broken_videos = []
        video_files = []

        self.logger.debug(f"Finding videos in {path}")
        for cd, _, files in os.walk(path, followlinks = True):
            for file in files:
                self._check_for_stop()
                file_path = os.path.join(cd, file)

                if video_utils.is_video(file_path):
                    video_files.append(file_path)

        self.logger.debug("Analysing videos")
        for video in tqdm(video_files, desc="Analysing videos", unit="video", leave=False, smoothing=0.1, mininterval=.2, disable=generic_utils.hide_progressbar()):
            self._check_for_stop()
            broken_video = self._check_if_broken(video)
            if broken_video is not None:
                broken_videos.append(broken_video)

        return broken_videos

class FixerTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self._analysis_results: list[tuple[dict, list[int]]] | None = None

    @override
    def setup_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--drop-unfixable", "-d",
                            action="store_true",
                            help='Drop (remove) subtitles that cannot be fixed.')
        parser.add_argument('videos_path',
                            nargs=1,
                            help='Path with videos to analyze.')

    @override
    def analyze(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str) -> None:
        self._analysis_results = None
        process_utils.ensure_tools_exist(["mkvmerge", "mkvextract", "ffprobe"], logger)

        logger.info("Searching for broken files")

        fixer = Fixer(logger, working_dir=working_dir)
        self._analysis_results = fixer.scan_directory(args.videos_path[0])

    @override
    def perform(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str) -> None:
        broken_videos = self._analysis_results
        self._analysis_results = None
        if broken_videos is None:
            logger.info("No analysis results, nothing to fix.")
            return

        fixer = Fixer(logger, working_dir)
        fixer.repair_videos(broken_videos, drop_broken = args.drop_unfixable)
        logger.info("Done")
