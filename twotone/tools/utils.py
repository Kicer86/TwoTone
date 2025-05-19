
import cchardet
import json
import logging
import math
import os.path
import py3langid as langid
import re
import signal
import subprocess
import sys
import tempfile
import uuid
from collections import namedtuple
from itertools import islice
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .utils2 import process, video
from .utils2.generic import fps_str_to_float, get_tqdm_defaults, ms_to_time, time_to_ms


SubtitleFile = namedtuple("Subtitle", "path language encoding")
Subtitle = namedtuple("Subtitle", "language default length tid format")
VideoTrack = namedtuple("VideoTrack", "fps length")
VideoInfo = namedtuple("VideoInfo", "video_tracks subtitles path")
ProcessResult = namedtuple("ProcessResult", "returncode stdout stderr")

subtitle_format1 = re.compile("[0-9]{1,2}:[0-9]{2}:[0-9]{2}:.*")
subtitle_format2 = re.compile("(?:0|1)\n[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} --> [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}\n", flags = re.MULTILINE)
microdvd_time_pattern = re.compile("\\{[0-9]+\\}\\{[0-9]+\\}.*")
subrip_time_pattern = re.compile(r'(\d+:\d{2}:\d{2},\d{3}) --> (\d+:\d{2}:\d{2},\d{3})')

ffmpeg_default_fps = 23.976                      # constant taken from https://trac.ffmpeg.org/ticket/3287


def file_encoding(file: str) -> str:
    detector = cchardet.UniversalDetector()

    with open(file, 'rb') as file:
        for line in file.readlines():
            detector.feed(line)
            if detector.done:
                break
        detector.close()

    encoding = detector.result["encoding"]

    return encoding


def is_video(file: str) -> bool:
    return Path(file).suffix[1:].lower() in ["mkv", "mp4", "avi", "mpg", "mpeg", "mov", "rmvb"]


def is_subtitle(file: str) -> bool:
    logging.debug(f"Checking file {file} for being subtitle")
    ext = file[-4:]

    if ext == ".srt" or ext == ".sub" or ext == ".txt":
        file = os.path.realpath(file)
        encoding = file_encoding(file)

        if encoding:
            logging.debug(f"\tOpening file with encoding {encoding}")

            with open(file, 'r', encoding = encoding) as text_file:
                head = "".join(islice(text_file, 5)).strip()

                for subtitle_format in [subtitle_format1, microdvd_time_pattern, subtitle_format2]:
                    if subtitle_format.match(head):
                        logging.debug("\tSubtitle format detected")
                        return True

    logging.debug("\tNot a subtitle file")
    return False


def is_subtitle_microdvd(subtitle: Subtitle) -> bool:
    with open(subtitle.path, 'r', encoding = subtitle.encoding) as text_file:
        head = "".join(islice(text_file, 5)).strip()

        if microdvd_time_pattern.match(head):
            return True

    return False


def guess_language(path: str, encoding: str) -> str:
    result = ""

    with open(path, "r", encoding=encoding) as sf:
        content = sf.readlines()
        content_joined = "".join(content)
        result = langid.classify(content_joined)[0]

    return result


def build_subtitle_from_path(path: str, language: str | None = "") -> SubtitleFile:
    """
        if language == None - use autodetection.
                       Empty string - no language
                       2/3 letter language code - use that language
    """
    encoding = file_encoding(path)
    language = guess_language(path, encoding) if language is None else language

    return SubtitleFile(path, language, encoding)


def build_audio_from_path(path: str, language: str | None = "") -> Dict:
    return {"path": path,
            "language": language,
    }


def guess_language(path: str, encoding: str) -> str:
    result = ""

    with open(path, "r", encoding=encoding) as sf:
        content = sf.readlines()
        content_joined = "".join(content)
        result = langid.classify(content_joined)[0]

    return result


def build_subtitle_from_path(path: str, language: str | None = "") -> SubtitleFile:
    """
        if language == None - use autodetection.
                       Empty string - no language
                       2/3 letter language code - use that language
    """
    encoding = file_encoding(path)
    language = guess_language(path, encoding) if language is None else language

    return SubtitleFile(path, language, encoding)


def alter_subrip_subtitles_times(content: str, multiplier: float) -> str:
    def multiply_time(match):
        time_from, time_to = map(time_to_ms, match.groups())
        time_from *= multiplier
        time_to *= multiplier

        time_from_srt = ms_to_time(time_from)
        time_to_srt = ms_to_time(time_to)

        return f"{time_from_srt} --> {time_to_srt}"

    content = subrip_time_pattern.sub(multiply_time, content)

    return content


def fix_subtitles_fps(input_path: str, output_path: str, subtitles_fps: float):
    """ fix subtitle's fps """
    multiplier = ffmpeg_default_fps / subtitles_fps

    # if no scaling is needed, make sure scale is set exactly to 1
    # and rewrite file as we need a copy in output_path anyway.
    # A simple file copying would do the job, but I just want to use the same
    # mechanism in all scenarios
    if math.isclose(multiplier, 1, rel_tol = 0.001):
        multiplier = 1

    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        content = infile.read()
        content = alter_subrip_subtitles_times(content, multiplier)
        outfile.write(content)


def get_video_frames_count(video_file: str):
    result = process.start_process("ffprobe", ["-v", "error", "-select_streams", "v:0", "-count_packets",
                                               "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", video_file])

    try:
        return int(result.stdout.strip())
    except ValueError:
        logging.error(f"Failed to get frame count for {video_file}")
        return None


def get_video_data(path: str) -> [VideoInfo]:

    def get_length(stream) -> int:
        """
            get lenght in milliseconds
        """
        length = None

        if "tags" in stream:
            tags = stream["tags"]
            duration = tags.get("DURATION", None)
            if duration is not None:
                length = time_to_ms(duration)

        if length is None:
            length = stream.get("duration", None)
            if length is not None:
                length = int(float(length) * 1000)

        return length

    output_json = video.get_video_full_info(path)

    subtitles = []
    video_tracks = []
    for stream in output_json["streams"]:
        stream_type = stream["codec_type"]
        if stream_type == "subtitle":
            if "tags" in stream:
                tags = stream["tags"]
                language = tags.get("language", None)
            else:
                language = None
            is_default = stream["disposition"]["default"]
            length = get_length(stream)
            tid = stream["index"]
            format = stream["codec_name"]

            subtitles.append(Subtitle(language, default=is_default, length=length, tid=tid, format=format))
        elif stream_type == "video":
            fps = stream["r_frame_rate"]
            length = get_length(stream)
            if length is None:
                length = video.get_video_duration(path)

            video_tracks.append(VideoTrack(fps=fps, length=length))

    return VideoInfo(video_tracks, subtitles, path)


def generate_mkv(output_path: str, input_video: str, subtitles: List[SubtitleFile] = [], audios: List[Dict] = []):
    # output
    options = ["-o", output_path]

    # set input
    options.append(input_video)

    # set audio tracks
    for i, audio in enumerate(audios):
        if "language" in audio and audio["language"]:
            options.extend(["--language", f"0:{audio['language']}"])

        if audio.get("default", False):
            options.extend(["--default-track", "0:yes"])
        else:
            options.extend(["--default-track", "0:no"])

        options.append(audio["path"])

    # set subtitles and languages
    for i, subtitle in enumerate(subtitles):
        lang = subtitle.language

        if lang:
            options.extend(["--language", f"0:{lang}"])

        if i == 0:
            options.extend(["--default-track", "0:yes"])
        else:
            options.extend(["--default-track", "0:no"])

        options.append(subtitle.path)

    # perform
    cmd = "mkvmerge"
    result = process.start_process(cmd, options)

    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"{cmd} exited with unexpected error:\n{result.stderr}")

    if not os.path.exists(output_path):
        logging.error("Output file was not created")
        raise RuntimeError(f"{cmd} did not create output file")

    # validate output file correctness
    output_file_details = get_video_data(output_path)
    input_file_details = get_video_data(input_video)

    if not compare_videos(input_file_details.video_tracks, output_file_details.video_tracks) or \
            len(input_file_details.subtitles) + len(subtitles) != len(output_file_details.subtitles):
        logging.error("Output file seems to be corrupted")
        raise RuntimeError("mkvmerge created a corrupted file")


def compare_videos(lhs: [VideoTrack], rhs: [VideoTrack]) -> bool:
    if len(lhs) != len(rhs):
        return False

    for lhs_item, rhs_item in zip(lhs, rhs):
        lhs_fps = fps_str_to_float(lhs_item.fps)
        rhs_fps = fps_str_to_float(rhs_item.fps)

        if lhs_fps == rhs_fps:
            return True

        diff = abs(lhs_fps - rhs_fps)

        # For videos with fps 1000000/33333 (≈30fps) mkvmerge generates video with 30/1 fps.
        # And videos with fps 29999/500 (≈60fps) it uses 60/1 fps.
        # I'm not sure if this is acceptable but at this moment let it be
        if diff > 0.0021:
            return False

    return True


def hide_progressbar() -> bool:
    return not sys.stdout.isatty() or 'unittest' in sys.modules


class InterruptibleProcess:
    def __init__(self):
        self._work = True
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logging.info(f"Got signal #{signum}. Exiting soon.")
        self._work = False

    def _check_for_stop(self):
        if not self._work:
            logging.warning("Exiting now due to received signal.")
            sys.exit(1)


def collect_video_files(path: str, interruptible: InterruptibleProcess) -> List[str]:
    video_files = []
    for cd, _, files in os.walk(path, followlinks = True):
        for file in files:
            interruptible._check_for_stop()
            file_path = os.path.join(cd, file)

            if is_video(file_path):
                video_files.append(file_path)

    return video_files


def get_unique_file_name(directory: str, extension: str) -> str:
    while True:
        file_name = f"{uuid.uuid4().hex}.{extension}"
        full_path = os.path.join(directory, file_name)

        if not os.path.exists(full_path):
            return full_path


class TempFileManager:
    def __init__(self, content: str, extension: str = None):
        self.content = content
        self.extension = extension
        self.filepath = None

    def __enter__(self):
        with tempfile.NamedTemporaryFile(delete = False, suffix = "." + self.extension, mode = 'w') as temp_file:
            self.filepath = temp_file.name
            temp_file.write(self.content)

        return self.filepath

    def __exit__(self, exc_type, exc_value, traceback):
        if self.filepath and os.path.exists(self.filepath):
            os.remove(self.filepath)
