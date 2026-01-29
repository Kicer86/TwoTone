import cchardet
import logging
from pathlib import Path
import py3langid as langid
import pysubs2

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(kw_only=True)
class SubtitleCommonData:
    name: str | None = None
    language: str | None = None
    default: int | bool = False
    forced: int | bool = False
    enabled: int | bool = True
    format: str | None = None

# for subtitle tracks in files
@dataclass(kw_only=True)
class Subtitle(SubtitleCommonData):
    length: int | None = None
    tid: int | None = None
    format: str | None = None

# for files
@dataclass(kw_only=True)
class SubtitleFile(SubtitleCommonData):
    path: str | None = None
    encoding: str | None = None

ffmpeg_default_fps = 23.976  # constant taken from https://trac.ffmpeg.org/ticket/3287
MAX_SUBTITLE_BYTES = 64 * 1024
SUBTITLE_EXTENSIONS = {
    ".ass",
    ".cap",
    ".dfxp",
    ".json",
    ".lrc",
    ".mpl",
    ".mpsub",
    ".pjs",
    ".rt",
    ".scc",
    ".srt",
    ".ssa",
    ".sub",
    ".ttml",
    ".txt",
    ".tmp",
    ".usf",
    ".vtt",
    ".sbv",
    ".stl",
    ".xml",
    ".idx",
}

FFPROBE_SUBTITLE_FORMATS = {
    "ass",
    "microdvd",
    "mpl2",
    "srt",
    "ssa",
    "subrip",
    "ttml",
    "vtt",
    "webvtt",
}

NON_AMBIGUOUS_SUBTITLE_EXTENSIONS = SUBTITLE_EXTENSIONS - {".txt", ".json", ".xml"}
FALLBACK_SUBTITLE_EXTENSIONS = {".sub"}


MKVMERGE_SUPPORTED_FORMATS = {
    "ass",
    "ssa",
    "srt",
    "subrip",
    "vtt",
    "webvtt",
}

MKVMERGE_UNSUPPORTED_FORMATS = {
    "json",
    "microdvd",
    "mpl2",
    "ttml",
    "tmp",
    "whisper_jax",
}


def subtitle_format_from_extension(path: str) -> str | None:
    ext = Path(path).suffix.lower()
    return pysubs2.formats.FILE_EXTENSION_TO_FORMAT_IDENTIFIER.get(ext)


def file_encoding(file: str) -> str:
    detector = cchardet.UniversalDetector()
    remaining = MAX_SUBTITLE_BYTES

    with open(file, 'rb') as file_obj:
        while remaining > 0:
            chunk = file_obj.read(min(4096, remaining))
            if not chunk:
                break
            detector.feed(chunk)
            remaining -= len(chunk)
            if detector.done:
                break
        detector.close()

    encoding = detector.result["encoding"]

    return encoding


def open_subtitle_file(file: str, fps: float = ffmpeg_default_fps) -> Optional[pysubs2.SSAFile]:
    try:
        encoding = file_encoding(file)
        subs = pysubs2.load(file, encoding = encoding, fps = fps)
        _strip_microdvd_header(subs, fps)
        return subs

    except Exception as e:
        logging.debug(f"Error opening subtitle file {file}: {e}")
        return None


def is_subtitle(file: str) -> bool:
    logging.debug(f"Checking file {file} for being subtitle")
    suffix = Path(file).suffix.lower()
    if suffix not in SUBTITLE_EXTENSIONS:
        logging.debug("\tNot a subtitle file")
        return False

    from . import process_utils

    status = process_utils.start_process(
        "ffprobe",
        ["-v", "error", "-show_entries", "format=format_name", "-of", "default=nw=1:nk=1", file],
    )
    if status.returncode == 0:
        formats = {fmt.strip().lower() for fmt in status.stdout.split(",") if fmt.strip()}
        if formats & FFPROBE_SUBTITLE_FORMATS:
            logging.debug("\tSubtitle format detected")
            return True

        if suffix in NON_AMBIGUOUS_SUBTITLE_EXTENSIONS:
            logging.debug("\tAssuming subtitle based on extension")
            return True
    else:
        if suffix in FALLBACK_SUBTITLE_EXTENSIONS:
            logging.debug("\tAssuming subtitle based on extension (ffprobe failed)")
            return True

    logging.debug("\tNot a subtitle file")
    return False


def _strip_microdvd_header(subs: pysubs2.SSAFile | None, fps: float | None = None) -> None:
    if not subs or not subs.format or subs.format.lower() != "microdvd":
        return
    if len(subs) == 0:
        return

    first = subs[0]
    if first.start != 0 or first.end != 0:
        return

    try:
        header_fps = float(first.text)
    except (TypeError, ValueError):
        return

    if fps is None or abs(header_fps - fps) < 0.05:
        del subs[0]


def is_subtitle_microdvd(subtitle: SubtitleFile) -> bool:
    assert subtitle.path

    subs = open_subtitle_file(subtitle.path)
    if subs and subs.format and subs.format.lower() == "microdvd":
        return True
    else:
        return False


def guess_subtitle_language(path: str, encoding: str) -> str:
    result = ""

    if not encoding:
        encoding = "utf-8"

    if encoding.lower() in {"ascii", "us-ascii"}:
        encoding = "utf-8"

    try:
        with open(path, "r", encoding=encoding, errors="replace") as sf:
            content = sf.read()
    except LookupError:
        with open(path, "r", encoding="utf-8", errors="replace") as sf:
            content = sf.read()

    result = langid.classify(content)[0]

    return result


def build_subtitle_from_path(path: str, language: str | None = "") -> SubtitleFile:
    """
    if language == None - use autodetection.
                   Empty string - no language
                   2/3 letter language code - use that language
    """
    encoding = file_encoding(path)
    language = guess_subtitle_language(path, encoding) if language is None else language

    return SubtitleFile(path = path, language = language, encoding = encoding)


def build_subtitle_from_dict(path: str, data: dict) -> SubtitleFile:
    return SubtitleFile(
        path = path,
        language = data["language"],
        encoding = data.get("encoding", None),
        name = data["title"],
        default = data["default"],
        enabled = data.get("enabled", True),
        forced = data["forced"],
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
