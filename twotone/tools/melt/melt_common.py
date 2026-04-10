import os
import re

from typing import Any, Literal

from ..utils import generic_utils

FramesInfo = dict[int, dict[str, str]]
StreamType = Literal["video", "audio", "subtitle"]

DEFAULT_TOLERANCE_MS = 0


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


def _ensure_working_dir(working_dir: str) -> str:
    os.makedirs(working_dir, exist_ok=True)
    return working_dir


def _is_length_mismatch(base_ms: int | None, other_ms: int | None, tolerance_ms: int) -> bool:
    if base_ms is None or other_ms is None:
        return False
    return abs(base_ms - other_ms) > tolerance_ms
