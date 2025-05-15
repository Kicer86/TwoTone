
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union, Tuple

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


def extract_timestamp_frame_mapping(video_path: str) -> Dict[int, int]:
    """
    Extracts a mapping of timestamp (seconds) to frame number from a video.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        dict: A dictionary mapping {timestamp in ms (int): frame number (int)}
    """

    args = [
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=coded_picture_number,pkt_dts_time",
        "-print_format", "flat",
        video_path
    ]

    # Run the command
    result = process.start_process("ffprobe", args)

    # Parse output
    timestamp_frame_map = {}
    for line in result.stdout.strip().split("\n"):
        match = re.match(r'frames\.frame\.(\d+)\.pkt_dts_time="([\d\.]+)"', line)
        if match:
            timestamp = float(match.group(2))  # Extract timestamp in seconds
            timestamp_ms = int(round(timestamp * 1000))

            frame_number = int(match.group(1))  # Extract frame number
            timestamp_frame_map[timestamp_ms] = frame_number

    return timestamp_frame_map


def extract_all_frames(video_path: str, target_dir: str, format: str = "jpeg", scale: Union[float, Tuple[int, int]] = 0.5) -> Dict[int, Dict]:
    """
        Function extracts all frames into the given directory (should be empty).
        Returns a dict mapping timestamp (ms) -> {'path': frame_path, 'frame': frame_number}
    """
    def run_ffmpeg(args):
        return process.start_process("ffmpeg", args=args)

    # Clear target directory
    def clean_target_dir():
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    scale_option = ""
    if isinstance(scale, float):
        if scale != 1.0:
            rscale = 1 / scale
            scale_option = f"scale=iw/{rscale}:ih/{rscale}"
    elif isinstance(scale, tuple):
        scale_option = f"scale={scale[0]}:{scale[1]}"
    else:
        raise RuntimeError("Invalid type for scale")

    output_pattern = os.path.join(target_dir, f"frame_%06d.{format}")

    def build_args(extra_args):
        filters = ["showinfo"]
        if scale_option:
            filters.append(scale_option)
        filters_arg = ",".join(filters)

        return [
            "-copyts",
            "-start_at_zero",
            "-i", video_path,
            "-an",                      # Ignore audio
            "-sn",                      # Ignore subtitles
            "-dn",                      # Ignore data streams
            "-frame_pts", "true",
            *extra_args,
            "-q:v", "2",
            "-vf", filters_arg,
            output_pattern
        ]

    fallback_options = [
        ["-fps_mode", "vfr"],
    ]

    for opts in fallback_options:
        clean_target_dir()
        args = build_args(opts)
        result = run_ffmpeg(args)

        if result.returncode == 0:
            break
    else:
        raise RuntimeError("ffmpeg failed with all fallback options.")

    frame_pattern = re.compile(r"n: *(\d+).*pts_time:([\d.]+)")
    frame_files = sorted(os.listdir(target_dir))

    mapping = {}
    f = 0
    for line in result.stderr.splitlines():
        match = frame_pattern.search(line)
        if match:
            frame_number = int(match.group(1))
            timestamp_ms = int(round(float(match.group(2)) * 1000))
            frame_file = frame_files[f]
            frame_id = int(frame_file[6:- (len(format) + 1)])
            mapping[timestamp_ms] = {"path": os.path.join(target_dir, frame_files[f]), "frame": frame_number, "frame_id": frame_id}
            f += 1

    print(f"Parsed frames: {f}, Frame files: {len(frame_files)}")

    return mapping


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

    def get_language(stream) -> Union[str | None]:
        if "tags" in stream:
            tags = stream["tags"]
            language = tags.get("language", None)
        else:
            language = None

        return language

    output_json = get_video_full_info(path)

    streams = defaultdict(list)
    for stream in output_json["streams"]:
        stream_type = stream["codec_type"]
        if stream_type == "subtitle":
            language = get_language(stream)
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

            width = stream["width"]
            height = stream["height"]
            bitrate = stream["bitrate"] if "bitrate" in stream else None
            codec = stream["codec_name"]

            streams["video"].append({
                "fps": fps,
                "length": length,
                "width": width,
                "height": height,
                "bitrate": bitrate,
                "codec": codec,
            })
        elif stream_type == "audio":
            language = get_language(stream)
            channels = stream["channels"]
            sample_rate = stream["sample_rate"]
            tid = stream["index"]

            streams["audio"].append({
                "language": language,
                "channels": channels,
                "sample_rate": sample_rate,
                "tid": tid,
            })

    return dict(streams)
