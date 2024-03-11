
import cchardet
import json
import logging
import os.path
import re
import subprocess
from collections import namedtuple
from itertools import islice
from pathlib import Path

Subtitle = namedtuple("Subtitle", "language default")
VideoTrack = namedtuple("VideoTrack", "")
VideoInfo = namedtuple("VideoInfo", "video_tracks subtitles")
ProcessResult = namedtuple("ProcessResult", "returncode stdout stderr")

subtitle_format1 = re.compile("[0-9]{2}:[0-9]{2}:[0-9]{2}:.*")
subtitle_format2 = re.compile("\\{[0-9]+\\}\\{[0-9]+\\}.*")
subtitle_format3 = re.compile("(?:0|1)\n[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} --> [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}\n", flags = re.MULTILINE)


def start_process(process: str, args: [str]) -> ProcessResult:
    command = [process]
    command.extend(args)

    logging.debug(f"Starting {process} with options: {' '.join(args)}")
    sub_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    stdout, stderr = sub_process.communicate()

    logging.debug(f"Process finished with {sub_process.returncode}")

    return ProcessResult(sub_process.returncode, stdout, stderr)


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
    return Path(file).suffix[1:].lower() in ["mkv", "mp4", "avi", "mpg", "mpeg", "mov"]


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

                for subtitle_format in [subtitle_format1, subtitle_format2, subtitle_format3]:
                    if subtitle_format.match(head):
                        logging.debug("\tSubtitle format detected")
                        return True

    logging.debug("\tNot a subtitle file")
    return False


def get_video_data(path: str) -> [VideoInfo]:
    args = []
    args.extend(["-v", "quiet"])
    args.extend(["-print_format", "json"])
    args.append("-show_format")
    args.append("-show_streams")
    args.append(path)

    process = start_process("ffprobe", args)

    if process.returncode != 0:
        raise RuntimeError(f"ffprobe exited with unexpected error:\n{process.stderr.decode('utf-8')}")

    output_lines = process.stdout
    output_str = output_lines.decode('utf8')
    output_json = json.loads(output_lines)

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
            subtitles.append(Subtitle(language, is_default))
        elif stream_type == "video":
            video_tracks.append(VideoTrack())

    return VideoInfo(video_tracks, subtitles)


def split_path(path: str) -> (str, str, str):
    info = Path(path)

    return str(info.parent), info.stem, info.suffix[1:]
