
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from . import process
from .generic import time_to_ms


def is_video(file: str) -> bool:
    return Path(file).suffix[1:].lower() in ["mkv", "mp4", "avi", "mpg", "mpeg", "mov", "rmvb"]


def get_video_frames_count(video_file: str):
    result = process.start_process("ffprobe", ["-v", "error", "-select_streams", "v:0", "-count_packets",
                                   "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", video_file])

    try:
        return int(result.stdout.strip())
    except ValueError:
        logging.error(f"Failed to get frame count for {video_file}")
        return None


def detect_scene_changes(file_path, threshold = 0.4) -> List[int]:
    """
        Run ffmpeg with a scene detection filter and extract scene change times.
        Function returns list of scene changes in milliseconds
    """

    args = [
        "-i", file_path,
        "-an",                                              # Ignore all audio streams
        "-sn",                                              # Ignore subtitle streams
        "-dn",                                              # Ignore data streams
        "-fps_mode", "auto",
        "-frame_pts", "true",                               # Ensure correct frame timestamps
        "-filter_complex", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]
    result = process.start_process("ffmpeg", args = args)

    # Look for lines with "pts_time:"; these indicate the timestamp of a scene change.
    scene_times = []
    pattern = re.compile(r"pts_time:(\d+\.\d+)")
    for line in result.stderr.splitlines():
        match = pattern.search(line)
        if match:
            time_s = float(match.group(1))
            time_ms = int(round(time_s * 1000))

            scene_times.append(time_ms)

    return sorted(set(scene_times))


def get_video_duration(video_file):
    """Get the duration of a video in milliseconds."""
    result = process.start_process("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_file])

    try:
        return int(float(result.stdout.strip())*1000)
    except ValueError:
        logging.error(f"Failed to get duration for {video_file}")
        return None


def get_video_full_info(path: str) -> str:
    args = []
    args.extend(["-v", "quiet"])
    args.extend(["-print_format", "json"])
    args.append("-show_format")
    args.append("-show_streams")
    args.append(path)

    result = process.start_process("ffprobe", args)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe exited with unexpected error:\n{result.stderr}")

    output_lines = result.stdout
    output_json = json.loads(output_lines)

    return output_json


def get_video_data2(path: str) -> Dict:

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

    output_json = get_video_full_info(path)

    streams = defaultdict(list)
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

            streams["subtitle"]. append({
                "language": language,
                "default": is_default,
                "length": length,
                "tid": tid,
                "format": format})
        elif stream_type == "video":
            fps = stream["r_frame_rate"]
            length = get_length(stream)
            if length is None:
                length = get_video_duration(path)

            streams["video"].append({"fps": fps, "length": length})

    return dict(streams)
