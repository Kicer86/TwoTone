import logging
import os
import re

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any, Literal, NamedTuple, TypedDict

from ..utils import generic_utils, video_utils


class FrameInfo(TypedDict):
    """A single probed video frame.

    ``frame_id`` is the ffmpeg frame ordinal; ``path`` is the extracted image
    on disk, or ``None`` while the frame is only probed.
    """
    frame_id: int
    path: str | None


FramesInfo = dict[int, FrameInfo]


class FrameSpan(NamedTuple):
    """First and last decoded frame timestamps of a video, and its frame count."""
    first_ms: int
    last_ms: int
    frame_count: int

    @classmethod
    def of(cls, frames: FramesInfo) -> "FrameSpan | None":
        """Summarize *frames*, or return ``None`` when nothing was decoded."""
        if not frames:
            return None
        return cls(min(frames), max(frames), len(frames))

    @property
    def duration_ms(self) -> int:
        """Time from the first to the last decoded frame."""
        return self.last_ms - self.first_ms


StreamType = Literal["video", "audio", "subtitle"]


def format_timestamp(timestamp_ms: int) -> str:
    """Render one media position for people and exact diagnostics."""
    return f"{generic_utils.ms_to_time(timestamp_ms)} ({timestamp_ms} ms)"


def format_timestamp_range(start_ms: int, end_ms: int) -> str:
    """Render a media range in clock and raw-millisecond forms."""
    return (
        f"{generic_utils.ms_to_time(start_ms)}-"
        f"{generic_utils.ms_to_time(end_ms)} "
        f"({start_ms}-{end_ms} ms)"
    )


@dataclass(frozen=True)
class MeltInputFiles:
    """Stable, one-based references to the files in one Melt input group."""

    paths: tuple[str, ...]
    display_roots: tuple[str, ...] = ()

    def __init__(self, paths: list[str] | tuple[str, ...], display_roots: list[str] | tuple[str, ...] = ()) -> None:
        object.__setattr__(self, "paths", tuple(paths))
        object.__setattr__(self, "display_roots", tuple(display_roots))

    def id_for(self, path: str) -> int:
        return self.paths.index(path) + 1

    @property
    def ids(self) -> dict[str, int]:
        return {path: index for index, path in enumerate(self.paths, start=1)}

    def reference(self, path: str) -> str:
        return f"file #{self.id_for(path)}"

    def display_path(self, path: str) -> str:
        path_abs = os.path.abspath(path)
        containing_inputs = []
        for display_root in self.display_roots:
            root_abs = os.path.abspath(display_root)
            try:
                if os.path.commonpath([root_abs, path_abs]) == root_abs:
                    containing_inputs.append(display_root)
            except ValueError:
                continue

        if containing_inputs:
            closest_input = max(containing_inputs, key=lambda candidate: len(os.path.abspath(candidate)))
            relative_path = os.path.relpath(path_abs, os.path.abspath(closest_input))
            if relative_path != ".":
                return relative_path
        return path

    def render(self, logger: logging.Logger, prefix: str = "") -> None:
        for path in self.paths:
            logger.info("%s#%d: %s", prefix, self.id_for(path), self.display_path(path))


class VideoStreamRef(NamedTuple):
    path: str
    mkvmerge_track_id: int
    ffprobe_stream_index: int
    language: str | None


class AudioStreamRef(NamedTuple):
    path: str
    mkvmerge_track_id: int
    ffprobe_stream_index: int
    language: str | None


class SubtitleStreamRef(NamedTuple):
    path: str
    mkvmerge_track_id: int
    ffprobe_stream_index: int
    language: str | None


class AttachmentRef(NamedTuple):
    path: str
    mkvmerge_attachment_id: int


def file_tracks(
    files_details: Mapping[str, Any],
    path: str,
    stream_type: StreamType,
) -> list[dict[str, Any]]:
    """Return the probed tracks of one type for *path*, or an empty list."""
    return files_details.get(path, {}).get("tracks", {}).get(stream_type, [])


def find_track(tracks: Iterable[dict[str, Any]], tid: int) -> dict[str, Any] | None:
    """Return the track with mkvmerge id *tid*, if present."""
    return next((track for track in tracks if track.get("tid") == tid), None)


def content_video_tracks(tracks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return video tracks that are not attached pictures (cover art)."""
    return [track for track in tracks if not track.get("attached_pic", False)]


def planned_video_duration_ms(
    path: str,
    files_details: Mapping[str, Any],
    mkvmerge_track_id: int | None = None,
) -> int | None:
    """Video track length in ms from pre-probed plan data, or ``None``.

    Reads the track *mkvmerge_track_id*, or the first content video track
    with a known length when no track is named.
    """
    tracks = content_video_tracks(file_tracks(files_details, path, "video"))
    if mkvmerge_track_id is not None:
        track = find_track(tracks, mkvmerge_track_id)
        tracks = [track] if track is not None else []
    return next(
        (track["length"] for track in tracks if track.get("length") is not None),
        None,
    )


def video_track_duration_ms(
    path: str,
    files_details: Mapping[str, Any],
    logger: logging.Logger | None = None,
    mkvmerge_track_id: int | None = None,
) -> int | None:
    """Video track length in ms, falling back to the container duration.

    The ffprobe container duration is used only when plan data does not
    know the track length.
    """
    planned = planned_video_duration_ms(path, files_details, mkvmerge_track_id)
    if planned is not None:
        return planned
    return video_utils.get_video_duration(path, logger=logger)


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
