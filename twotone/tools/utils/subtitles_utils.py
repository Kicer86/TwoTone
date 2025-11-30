
import cchardet
import logging
import mimetypes
import py3langid as langid
import pysubs2

from typing import Dict, Optional, List

from . import process_utils
from .data_structs import SubtitleFile


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


def open_subtitle_file(file: str, fps: float = ffmpeg_default_fps) -> Optional[pysubs2.SSAFile]:
    try:
        mime = mimetypes.guess_type(file)
        if mime and mime[0].startswith("text/"):
            encoding = file_encoding(file)
            subs = pysubs2.load(file, encoding = encoding, fps = fps)
            return subs
        else:
            return None

    except Exception as e:
        logging.debug(f"Error opening subtitle file {file}: {e}")
        return None


def is_subtitle(file: str) -> bool:
    logging.debug(f"Checking file {file} for being subtitle")

    subs = open_subtitle_file(file)
    if subs:
        logging.debug("\tSubtitle format detected")
        return True
    else:
        logging.debug("\tNot a subtitle file")
        return False


def is_subtitle_microdvd(subtitle: SubtitleFile) -> bool:
    subs = open_subtitle_file(subtitle.path)
    if subs and subs.format and subs.format.lower() == "microdvd":
        return True
    else:
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

    return SubtitleFile(path = path, language = language, encoding = encoding)


def build_subtitle_from_dict(path: str, data: dict) -> SubtitleFile:
    return SubtitleFile(
        path = path,
        language = data["language"],
        encoding = data.get("encoding", None),
        name = data["title"],
        default = data["default"],
        format = data["format"],
    )


def build_audio_from_path(path: str, language: str | None = "") -> Dict:
    return {"path": path, "language": language}


def fix_subtitles_fps(input_path: str, output_path: str, subtitles_fps: float):
    """fix subtitle's fps"""
    subs = open_subtitle_file(input_path)

    if subs:
        subs.transform_framerate(ffmpeg_default_fps, subtitles_fps)
        subs.save(output_path)
