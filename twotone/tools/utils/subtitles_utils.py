import cchardet
import logging
import math
import os
import re
import tempfile

from dataclasses import dataclass
from itertools import islice
from typing import Dict, Optional

import py3langid as langid

from .generic_utils import ms_to_time, time_to_ms


@dataclass
class SubtitleFile:
    path: str
    language: str
    encoding: str
    comment: str | None = None


@dataclass
class Subtitle:
    language: str
    default: int | bool
    length: int | None
    tid: int
    format: str


subtitle_format1 = re.compile("[0-9]{1,2}:[0-9]{2}:[0-9]{2}:.*")
subtitle_format2 = re.compile("(?:0|1)\n[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} --> "
                              "[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}\n", flags=re.MULTILINE)
microdvd_time_pattern = re.compile("\\{[0-9]+\\}\\{[0-9]+\\}.*")
weird_microdvd_time_pattern = re.compile("\\[[0-9]+\\]\\[[0-9]+\\].*")                          # [] instead of {}
subrip_time_pattern = re.compile(r'(\d+:\d{2}:\d{2},\d{3}) --> (\d+:\d{2}:\d{2},\d{3})')

ffmpeg_default_fps = 23.976  # constant taken from https://trac.ffmpeg.org/ticket/3287


def file_encoding(file: str) -> str:
    detector = cchardet.UniversalDetector()

    with open(file, 'rb') as file_obj:
        for line in file_obj.readlines():
            detector.feed(line)
            if detector.done:
                break
        detector.close()

    encoding = detector.result["encoding"]

    return encoding


def is_subtitle(file: str) -> bool:
    logging.debug(f"Checking file {file} for being subtitle")
    ext = file[-4:]

    if ext == ".srt" or ext == ".sub" or ext == ".txt":
        file = os.path.realpath(file)
        encoding = file_encoding(file)

        if encoding:
            logging.debug(f"\tOpening file with encoding {encoding}")

            with open(file, 'r', encoding=encoding) as text_file:
                head = "".join(islice(text_file, 5)).strip()

                for subtitle_format in [subtitle_format1, microdvd_time_pattern, weird_microdvd_time_pattern, subtitle_format2]:
                    if subtitle_format.match(head):
                        logging.debug("\tSubtitle format detected")
                        return True

    logging.debug("\tNot a subtitle file")
    return False


def is_subtitle_microdvd(subtitle: SubtitleFile) -> bool:
    with open(subtitle.path, 'r', encoding=subtitle.encoding) as text_file:
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


def extract_subtitle_to_temp(video_path: str, tid: int, stream_format: str, output_base_path: str, logger: Optional[logging.Logger] = None) -> Optional[str]:
    """Extract a subtitle track from a container to a file.

    - ``output_base_path`` should be a path without extension; the proper
      extension will be appended depending on ``stream_format``.
    - Returns full output path on success, or ``None`` if unsupported or failed.
    """
    fmt = (stream_format or "").lower()
    ext_map = {
        "subrip": ".srt",
        "srt": ".srt",
        "ass": ".ass",
        "ssa": ".ssa",
        "webvtt": ".vtt",
        "mov_text": ".srt",
        "text": ".srt",
    }

    suffix = ext_map.get(fmt)
    if not suffix:
        return None

    try:
        output_path = f"{output_base_path}{suffix}"
        # Local import to avoid circular dependency: process_utils -> video_utils -> subtitles_utils
        from . import process_utils
        status = process_utils.start_process("mkvextract", ["tracks", video_path, f"{tid}:{output_path}"])
        if status.returncode != 0:
            if logger:
                logger.debug(f"mkvextract failed (tid={tid}): {status.stderr}")
            # Ensure we don't leave partially created files
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            return None

        return output_path
    except Exception as e:
        if logger:
            logger.debug(f"Subtitle extraction failed (tid={tid}): {e}")
        return None


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
    return {"path": path, "language": language}


def alter_subrip_subtitles_times(content: str, multiplier: float) -> str:
    def multiply_time(match):
        time_from, time_to = map(time_to_ms, match.groups())
        time_from = int(time_from * multiplier)
        time_to = int(time_to * multiplier)

        time_from_srt = ms_to_time(time_from)
        time_to_srt = ms_to_time(time_to)

        return f"{time_from_srt} --> {time_to_srt}"

    content = subrip_time_pattern.sub(multiply_time, content)

    return content


def fix_subtitles_fps(input_path: str, output_path: str, subtitles_fps: float):
    """fix subtitle's fps"""
    multiplier = ffmpeg_default_fps / subtitles_fps

    if math.isclose(multiplier, 1, rel_tol=0.001):
        multiplier = 1

    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        content = infile.read()
        content = alter_subrip_subtitles_times(content, multiplier)
        outfile.write(content)
