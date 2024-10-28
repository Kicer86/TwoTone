
import argparse
import langid
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

import utils


work = True


class TwoTone:

    def __init__(self, dry_run: bool, language: str, lang_priority: str):
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

        prepared_subtitles = []
        for subtitle in sorted_subtitles:
            logging.info(f"\tadd subtitles [{subtitle.language}]: {subtitle.path}")
            self._register_input(subtitle.path)

            # Subtitles are buggy sometimes, use ffmpeg to fix them.
            # Also makemkv does not handle MicroDVD subtitles, so convert all to SubRip.

            fps = input_file_details.video_tracks[0].fps
            converted_subtitle = self._convert_subtitle(fps, subtitle)

            prepared_subtitles.append(converted_subtitle)

        # perform
        logging.info("\tMerge in progress...")
        if not self.dry_run:
            utils.generate_mkv(input_video=input_video, output_path=output_video, subtitles=prepared_subtitles)

        # Remove all inputs and temporary files. Only output file should left
        self._remove()

        logging.info("\tDone")

    def _process_video(self, video_file: str, subtitles_fetcher):
        logging.debug(f"Analyzing subtitles for video: {video_file}")
        subtitles = subtitles_fetcher(video_file)
        if subtitles:
            self._merge(video_file, subtitles)

    def process_dir(self, path: str):
        global work
        if not work:
            return

        video_files = []
        for entry in os.scandir(path):
            if entry.is_file() and utils.is_video(entry.path):
                video_files.append(entry.path)
            elif entry.is_dir():
                self.process_dir(entry.path)

        if len(video_files) == 1:
            self._process_video(video_files[0], self._aggressive_subtitle_search)
        elif len(video_files) > 1:
            for video_file in video_files:
                self._process_video(video_file, self._simple_subtitle_search)


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


def sig_handler(signum, frame):
    global work
    logging.warning("SIGINT received, stopping soon")
    work = False


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sig_handler)
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
