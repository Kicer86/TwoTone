import re

from typing import Any, Literal, NamedTuple, TypedDict

from ..utils import generic_utils


class FrameInfo(TypedDict):
    """A single probed video frame.

    ``frame_id`` is the ffmpeg frame ordinal; ``path`` is the extracted image
    on disk, or ``None`` while the frame is only probed.
    """
    frame_id: int
    path: str | None


FramesInfo = dict[int, FrameInfo]
StreamType = Literal["video", "audio", "subtitle"]


class VideoStreamRef(NamedTuple):
    path: str
    stream_index: int
    language: str | None


class AudioStreamRef(NamedTuple):
    path: str
    stream_index: int
    language: str | None


class SubtitleStreamRef(NamedTuple):
    path: str
    stream_index: int
    language: str | None


class AttachmentRef(NamedTuple):
    path: str
    stream_index: int


def _fmt_fps(value: str) -> str | None:
    try:
        fps = generic_utils.fps_str_to_float(str(value))
    except Exception:
        return None
    if abs(fps - round(fps)) < 0.01:
        return str(int(round(fps)))
    return f"{fps:.2f}"


def stream_short_details(stype: StreamType, stream: dict[str, Any]) -> str:
    """One-line human-readable summary of a stream's key properties."""
    match stype:
        case "video":
            width = stream.get("width")
            height = stream.get("height")
            fps = stream.get("fps")
            codec = stream.get("codec")
            length = stream.get("length")
            length_formatted = generic_utils.ms_to_time(length) if length else None
            details = []
            if width and height:
                fps_val = _fmt_fps(fps) if fps else None
                if fps_val:
                    details.append(f"{width}x{height}@{fps_val}")
                else:
                    details.append(f"{width}x{height}")
            elif fps:
                fps_val = _fmt_fps(fps)
                if fps_val:
                    details.append(f"{fps_val}fps")
            if codec:
                details.append(codec)
            if length_formatted:
                details.append(f"duration: {length_formatted}")
            return ", ".join(details)
        case "audio":
            channels = stream.get("channels")
            sample_rate = stream.get("sample_rate")
            details = []
            if channels:
                details.append(f"{channels}ch")
            if sample_rate:
                details.append(f"{sample_rate}Hz")
            return ", ".join(details)
        case "subtitle":
            fmt = stream.get("format") or stream.get("codec")
            return str(fmt) if fmt else ""
        case _:
            return ""


def _split_path_fix(value: str) -> list[str]:
    pattern = r'"((?:[^"\\]|\\.)*?)"'

    matches = re.findall(pattern, value)
    return [match.replace(r'\"', '"') for match in matches]


def _is_length_mismatch(base_ms: int | None, other_ms: int | None) -> bool:
    if base_ms is None or other_ms is None:
        return False
    return base_ms != other_ms
