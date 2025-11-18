import cchardet
import logging
import py3langid as langid
import pysubs2

from dataclasses import dataclass
from typing import Dict, Optional, List

from . import process_utils, video_utils


@dataclass
class SubtitleFile:
    path: str
    language: str | None
    encoding: str
    comment: str | None = None


@dataclass
class Subtitle:
    language: str
    default: int | bool
    length: int | None
    tid: int
    format: str

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
        encoding = file_encoding(file)
        subs = pysubs2.load(file, encoding = encoding, fps = fps)
        return subs

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


def extract_subtitle_to_temp(video_path: str, tids: List[int], output_base_path: str, logger: Optional[logging.Logger] = None) -> Dict[int, str]:
    """Extract subtitle tracks to temporary files.

    - Determines stream formats internally using video metadata.
    - Appends the track id to the output path: f"{output_base_path}.{tid}.{ext}".
    - Returns a mapping {tid: output_path} for all requested tids.
    """

    tids_list: List[int] = list(tids)

    # Map formats to file extensions
    ext_map = {
        "subrip": ".srt",
        "srt": ".srt",
        "ass": ".ass",
        "ssa": ".ssa",
        "webvtt": ".vtt",
        "mov_text": ".srt",
        "text": ".srt",
    }

    # Discover formats using video_utils
    try:
        info = video_utils.get_video_data(video_path)
        stream_fmt = {s.get("tid"): (s.get("format") or "").lower() for s in info.get("subtitle", [])}
    except Exception as e:
        stream_fmt = {}
        if logger:
            logger.debug(f"Failed to get stream info for '{video_path}': {e}")

    # Build mkvextract options
    tid_to_path: Dict[int, str] = {}
    options = ["tracks", video_path]
    for tid in tids_list:
        fmt = stream_fmt.get(tid, "")
        suffix = ext_map.get(fmt, ".srt")
        out_path = f"{output_base_path}.{tid}{suffix}"
        tid_to_path[tid] = out_path
        options.append(f"{tid}:{out_path}")

    try:
        status = process_utils.start_process("mkvextract", options)
        if status.returncode != 0 and logger:
            logger.debug(f"mkvextract failed for {video_path}: {status.stderr}")
    except Exception as e:
        if logger:
            logger.debug(f"Subtitle extraction failed for {video_path}: {e}")

    return tid_to_path


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


def fix_subtitles_fps(input_path: str, output_path: str, subtitles_fps: float):
    """fix subtitle's fps"""
    subs = open_subtitle_file(input_path)

    if subs:
        subs.transform_framerate(ffmpeg_default_fps, subtitles_fps)
        subs.save(output_path)
