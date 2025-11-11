import cchardet
import logging
import math
import os
import re

from dataclasses import dataclass
from itertools import islice
from typing import Dict, Optional, List

import py3langid as langid
import pysubs2

from .generic_utils import ms_to_time, time_to_ms
from . import process_utils, video_utils


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


# === pysubs2-based utilities (multi-format safe) ===

def load_subtitles(path: str, encoding: Optional[str] = None) -> pysubs2.SSAFile:
    """Load subtitles using pysubs2, respecting file extension and encoding.

    If encoding is not provided, detect using file_encoding().
    """
    enc = encoding or file_encoding(path)
    return pysubs2.load(path, encoding=enc)


def save_subtitles(subs: pysubs2.SSAFile, path: str, encoding: str = "utf-8") -> None:
    """Save subtitles to the given path using format inferred from extension."""
    subs.save(path, encoding=encoding)


def get_first_event_start_ms(subs: pysubs2.SSAFile) -> Optional[int]:
    """Return start time (ms) of earliest event, or None if empty."""
    if not subs.events:
        return None
    return min(e.start for e in subs.events)


def get_last_event_end_ms(subs: pysubs2.SSAFile) -> Optional[int]:
    """Return end time (ms) of last event, or None if empty."""
    if not subs.events:
        return None
    return max(e.end for e in subs.events)


def scale_times_inplace(subs: pysubs2.SSAFile, multiplier: float) -> None:
    """Scale all event times by multiplier (in place)."""
    if multiplier == 1.0:
        return
    for ev in subs.events:
        ev.start = int(round(ev.start * multiplier))
        ev.end = int(round(ev.end * multiplier))


def clamp_last_event_end_inplace(subs: pysubs2.SSAFile, video_length_ms: int, max_tail_ms: int = 5000) -> None:
    """Clamp the final event duration so it ends no later than min(start+max_tail_ms, video_length_ms)."""
    if not subs.events:
        return
    # Identify event with maximal end; if ties, pick the one with maximal start
    last = max(subs.events, key=lambda e: (e.end, e.start))
    target_end = min(last.start + max_tail_ms, int(video_length_ms))
    if last.end > target_end:
        last.end = target_end


def rebase_times_by_offset_inplace(subs: pysubs2.SSAFile, offset_ms: int) -> None:
    """Subtract offset_ms from all events; clamp to >= 0. Does not reorder events."""
    if offset_ms <= 0:
        return
    for ev in subs.events:
        new_start = ev.start - offset_ms
        new_end = ev.end - offset_ms
        ev.start = max(0, new_start)
        ev.end = max(0, new_end)


def detect_first_event_offset_ms(subs: pysubs2.SSAFile) -> Optional[int]:
    """Return earliest event start (ms), or None if empty. Alias for get_first_event_start_ms."""
    return get_first_event_start_ms(subs)


def copy_with_encoding(subs: pysubs2.SSAFile) -> pysubs2.SSAFile:
    """Return a shallow copy suitable for modifications while preserving metadata/scripts."""
    return subs.copy()
