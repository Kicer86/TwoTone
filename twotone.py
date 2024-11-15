
import argparse
import langid
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from collections import namedtuple
from pathlib import Path

import utils


work = True


class TwoTone(utils.InterruptibleProcess):

    def __init__(self, dry_run: bool, language: str, lang_priority: str):
        super().__init__()
        self.dry_run = dry_run
        self.language = language
        self.to_be_removed = []
        self.lang_priority = [] if not lang_priority or lang_priority == "" else lang_priority.split(",")

    def _get_temporary_file(self, ext: str) -> str:
        tmp_file = tempfile.mkstemp(suffix="."+ext)
        tmp_path = tmp_file[1]
        self._remove_later(tmp_path)
        return tmp_path

    def _register_input(self, path: str):
        if not self.dry_run:
            self._remove_later(path)

    def _remove_later(self, path: str):
        self.to_be_removed.append(path)

    def _remove(self):
        for file_to_remove in self.to_be_removed:
            os.remove(file_to_remove)

        self.to_be_removed.clear()

    def _build_subtitle_from_path(self, path: str) -> utils.SubtitleFile:
        encoding = utils.file_encoding(path)
        language = self.language if self.language != "auto" else self._guess_language(path, encoding)

        return utils.SubtitleFile(path, language, encoding)

    def _simple_subtitle_search(self, path: str) -> [utils.SubtitleFile]:
        video_name = Path(path).stem
        directory = Path(path).parent

        subtitles = []

        for subtitle_ext in ["txt", "srt"]:
            subtitle_file = video_name + "." + subtitle_ext
            subtitle_path = os.path.join(directory, subtitle_file)
            if os.path.exists(subtitle_path) and utils.is_subtitle(subtitle_path):
                subtitle = self._build_subtitle_from_path(subtitle_path)
                subtitles.append(subtitle)

        return subtitles

    def _recursive_subtitle_search(self, path: str) -> [utils.SubtitleFile]:
        found_subtitles = []
        found_subdirs = []

        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir():
                    found_subdirs.append(entry.path)
                elif entry.is_file():
                    if utils.is_video(entry.path):
                        # if there is a video file then all possible subtitles at this level (and below) belong to
                        # it, quit recursion for current directory
                        return []
                    elif utils.is_subtitle(entry.path):
                        found_subtitles.append(entry.path)

        # if we got here, then no video was found at this level
        subtitles = [self._build_subtitle_from_path(subtitle) for subtitle in found_subtitles]

        for subdir in found_subdirs:
            sub_subtitles = self._recursive_subtitle_search(subdir)
            subtitles.extend(subtitles)

        return subtitles

    def _aggressive_subtitle_search(self, path: str) -> [utils.SubtitleFile]:
        subtitles = self._simple_subtitle_search(path)
        directory = Path(path).parent

        for entry in os.scandir(directory):
            if entry.is_dir():
                sub_subtitles = self._recursive_subtitle_search(entry.path)
                subtitles.extend(sub_subtitles)
            elif entry.is_file() and utils.is_subtitle(entry.path):
                subtitle = self._build_subtitle_from_path(entry.path)
                subtitles.append(subtitle)

        return list(set(subtitles))

    @staticmethod
    def _get_index_for(l: [], value):
        try:
            return l.index(value)
        except ValueError:
            return len(l)

    def _sort_subtitles(self, subtitles: [utils.SubtitleFile]) -> [utils.SubtitleFile]:
        priorities = self.lang_priority.copy()
        priorities.append(None)
        subtitles_sorted = sorted(subtitles, key=lambda s: self._get_index_for(priorities, s.language))

        return subtitles_sorted

    def _convert_subtitle(self, video_fps: str, subtitle: utils.SubtitleFile) -> [utils.SubtitleFile]:
        converted_subtitle = subtitle

        if not self.dry_run:
            input_file = subtitle.path
            output_file = self._get_temporary_file("srt")
            encoding = subtitle.encoding if subtitle.encoding != "UTF-8-SIG" else "utf-8"

            status = utils.start_process("ffmpeg",
                                         ["-hide_banner", "-y", "-sub_charenc", encoding, "-i", input_file, output_file])

            if status.returncode == 0:
                # there is no way (as of now) to tell ffmpeg to convert subtitles with proper frame rate in mind.
                # so here some naive conversion is being done
                # see: https://trac.ffmpeg.org/ticket/10929
                #      https://trac.ffmpeg.org/ticket/3287
                if utils.is_subtitle_microdvd(subtitle):
                    fps = eval(video_fps)

                    # prepare new output file, and use previous one as new input
                    input_file = output_file
                    output_file = self._get_temporary_file("srt")

                    utils.fix_subtitles_fps(input_file, output_file, fps)

            else:
                raise RuntimeError(f"ffmpeg exited with unexpected error:\n{status.stderr.decode('utf-8')}")

            converted_subtitle = utils.SubtitleFile(output_file, subtitle.language, "utf-8")

        return converted_subtitle

    @staticmethod
    def _guess_language(path: str, encoding: str) -> str:
        result = ""

        with open(path, "r", encoding=encoding) as sf:
            content = sf.readlines()
            content_joined = "".join(content)
            result = langid.classify(content_joined)[0]

        return result

    def _merge(self, input_video: str, subtitles: [utils.SubtitleFile]):
        logging.info(f"Video file: {input_video}")

        video_dir, video_name, video_extension = utils.split_path(input_video)
        output_video = video_dir + "/" + video_name + "." + "mkv"

        # collect details about input file
        input_file_details = utils.get_video_data(input_video)

        # make sure output file does not exist
        i = 1
        while os.path.exists(output_video):
            output_video = video_dir + "/" + video_name + "." + str(i) + "." + "mkv"
            i += 1

        # register input for removal
        self._register_input(input_video)

        # set subtitles and languages
        sorted_subtitles = self._sort_subtitles(subtitles)
        sorted_subtitles_str = ", ".join([subtitle.language if subtitle.language is not None else "unknown" for subtitle in sorted_subtitles])
        logging.info(f"Merging with subtitles in languages: [{sorted_subtitles_str}]")

        prepared_subtitles = []
        for subtitle in sorted_subtitles:
            logging.debug(f"\tregister subtitle [{subtitle.language}]: {subtitle.path}")
            self._register_input(subtitle.path)

            # Subtitles are buggy sometimes, use ffmpeg to fix them.
            # Also makemkv does not handle MicroDVD subtitles, so convert all to SubRip.
            fps = input_file_details.video_tracks[0].fps
            converted_subtitle = self._convert_subtitle(fps, subtitle)

            prepared_subtitles.append(converted_subtitle)

        # perform
        logging.debug("\tMerge in progress...")
        if not self.dry_run:
            utils.generate_mkv(input_video=input_video, output_path=output_video, subtitles=prepared_subtitles)

        # Remove all inputs and temporary files. Only output file should left
        self._remove()

        logging.debug("\tDone")

    def _process_video(self, video_file: str, subtitles_fetcher):
        logging.debug(f"Analyzing subtitles for video: {video_file}")
        subtitles = subtitles_fetcher(video_file)

        if len(subtitles) == 0:
            return None
        else:
            return (video_file, subtitles)

    def _process_dir(self, path: str):
        logging.debug(f"Finding videos in {path}")
        videos_and_subtitles = []

        for cd, _, files in os.walk(path, followlinks = True):
            video_files = []
            for file in files:
                self._check_for_stop()
                file_path = os.path.join(cd, file)

                if utils.is_video(file_path):
                    video_files.append(file_path)

            # check if number of unique file names (excluding extensions) is equal to number of files (including extensions).
            # if no, then it means there are at least two video files with the same name but different extension.
            # this is a cumbersome situation so just don't allow it
            unique_names = set( Path(video).stem for video in video_files)
            if len(unique_names) != len(video_files):
                logging.warning(f"Two video files with the same name found in {cd}. This is not supported, skipping whole directory.")
                continue

            subtitles_finder = self._aggressive_subtitle_search if len(video_files) == 1 else self._simple_subtitle_search
            for video_file in video_files:
                vs = self._process_video(video_file, subtitles_finder)
                if vs is not None:
                    videos_and_subtitles.append(vs)

        return videos_and_subtitles

    def process_dir(self, path: str):
        logging.info(f"Looking for video and subtitle files in {path}")
        vas = self._process_dir(path)

        logging.info(f"Found {len(vas)} videos with subtitles to merge")
        for video, _ in vas:
            logging.debug(video)

        logging.info("Starting merge")
        with logging_redirect_tqdm():
            for video, subtitles in tqdm(vas, desc="Merging", unit="video", leave=False, smoothing=0.1, mininterval=.2, disable=utils.hide_progressbar()):
                self._check_for_stop()
                self._merge(video, subtitles)


def run(sys_args: [str]):
    parser = argparse.ArgumentParser(description='Combine many video/subtitle files into one mkv file. '
                                                 'By default program does nothing but showing what will be done. '
                                                 'Use --no-dry-run option to perform actual operation. '
                                                 'Please mind that ALL source files, so consider making a backup. '
                                                 'It is safe to stop this script with ctrl+c - it will quit '
                                                 'gracefully in a while.')
    parser.add_argument('videos_path',
                        nargs=1,
                        help='Path with videos to combine.')
    parser.add_argument("--no-dry-run", "-r",
                        action='store_true',
                        default=False,
                        help='Perform actual operation.')
    parser.add_argument("--language", "-l",
                        help='Language code for found subtitles. By default none is used. See mkvmerge '
                             '--list-languages for available languages. For automatic detection use: auto')
    parser.add_argument("--languages-priority", "-p",
                        help='Comma separated list of two letter language codes. Order on the list defines order of '
                             'subtitles appending.\nFor example, for --languages-priority pl,de,en,fr all '
                             'found subtitles will be ordered so polish goes as first, then german, english and '
                             'french. If there are subtitles in any other language, they will be append at '
                             'the end in undefined order')
    parser.add_argument("--verbose", action='store_true', default=False, help='Verbose output')

    args = parser.parse_args(sys_args)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    for tool in ["mkvmerge", "ffmpeg", "ffprobe"]:
        path = shutil.which(tool)
        if path is None:
            raise RuntimeError(f"{tool} not found in PATH")
        else:
            logging.debug(f"{tool} path: {path}")

    logging.info("Searching for movie and subtitle files to be merged")
    two_tone = TwoTone(dry_run=not args.no_dry_run,
                       language=args.language,
                       lang_priority=args.languages_priority)
    two_tone.process_dir(args.videos_path[0])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    try:
        run(sys.argv[1:])
    except RuntimeError as e:
        logging.error(f"Unexpected error occurred: {e}. Terminating")
        exit(1)

    if work:
        logging.info("Done")
    else:
        logging.warning("Quited due to SIGINT")
