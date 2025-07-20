import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union, Tuple



from . import language_utils, process_utils
from .generic_utils import fps_str_to_float, time_to_ms
from .subtitles_utils import SubtitleFile




def is_video(file: str) -> bool:
    return Path(file).suffix[1:].lower() in ["mkv", "mp4", "avi", "mpg", "mpeg", "mov", "rmvb"]


def get_video_frames_count(video_file: str):
    result = process_utils.start_process("ffprobe", ["-v", "error", "-select_streams", "v:0", "-count_packets",
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
    result = process_utils.start_process("ffmpeg", args = args)

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
    result = process_utils.start_process("ffprobe", args)

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
        return process_utils.start_process("ffmpeg", args=args)

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

    #self.logger.debug(f"Parsed frames: {f}, Frame files: {len(frame_files)}")

    return mapping


def get_video_duration(video_file):
    """Get the duration of a video in milliseconds."""
    result = process_utils.start_process("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_file])

    try:
        return int(float(result.stdout.strip())*1000)
    except ValueError:
        logging.error(f"Failed to get duration for {video_file}")
        return None


def get_video_full_info(path: str) -> dict:
    args = []
    args.extend(["-v", "quiet"])
    args.extend(["-print_format", "json"])
    args.append("-show_format")
    args.append("-show_streams")
    args.append(path)

    result = process_utils.start_process("ffprobe", args)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe exited with unexpected error:\n{result.stderr}")

    output_lines = result.stdout
    output_json = json.loads(output_lines)

    return output_json


def get_video_data(path: str) -> Dict:

    def get_length(stream) -> int | None:
        """Return stream length in milliseconds if available."""
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

        return language_utils.unify_lang(language) if language else None

    output_json = get_video_full_info(path)

    streams = defaultdict(list)
    for stream in output_json["streams"]:
        stream_type = stream["codec_type"]
        tid = stream["index"]
        if stream_type == "subtitle":
            language = get_language(stream)
            is_default = stream["disposition"]["default"]
            length = get_length(stream)
            format = stream["codec_name"]

            streams["subtitle"].append({
                "language": language,
                "default": is_default,
                "length": length,
                "tid": tid,
                "format": format})
        elif stream_type == "video":
            fps = stream["r_frame_rate"]
            length = get_length(stream)
            disposition = stream.get("disposition", {})
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
                "tid": tid,
            })
        elif stream_type == "audio":
            language = get_language(stream)
            channels = stream["channels"]
            sample_rate = stream["sample_rate"]

            streams["audio"].append({
                "language": language,
                "channels": channels,
                "sample_rate": sample_rate,
                "tid": tid,
            })

    return dict(streams)


def get_video_full_info_mkvmerge(path: str) -> dict:
    """Return file information using ``mkvmerge -J``."""

    result = process_utils.start_process("mkvmerge", ["-J", path])

    if result.returncode != 0:
        raise RuntimeError(f"mkvmerge exited with unexpected error:\n{result.stderr}")

    return json.loads(result.stdout)


def get_video_data_mkvmerge(path: str, enrich: bool = False) -> Dict:
    """
        Return stream information parsed from ``mkvmerge -J`` output.
        For non mkv files, mkvmerge does not provide as much information as ffprobe.
        Set 'enrich' to True to enrich mkvmerge's outpput with data from ffprobe.
    """

    def find_ffprobe_track(track_id: int, ffprobe_info: {}):
        for streams in (ffprobe_info or {}).values():
            for stream in streams:
                if stream.get("tid", None) == track_id:
                    return stream

        return None

    def merge_properties(initial: Dict or None, update: Dict):
        if initial is None:
            return update

        output = initial
        for key, value in update.items():
            base_value = output.get(key, None)

            if base_value is None:
                output[key] = value
            elif value is None:
                pass
            elif value != base_value:
                if key != "codec":
                    logging.warning(f"Inconsistent data provided by mkvmerge ({key}: {value}) and ffprobe ({key}: {base_value})")
                output[key] = value

        return output

    info = get_video_full_info_mkvmerge(path)

    container_props = info.get("container", {}).get("properties", {})
    duration_ms = None
    if "duration" in container_props:
        try:
            duration_ms = int(float(container_props["duration"]) * 1000)
        except (TypeError, ValueError):
            duration_ms = None

    # process streams/tracks
    streams = defaultdict(list)
    ffprobe_info = get_video_data(path) if enrich else None

    for track in info.get("tracks", []):
        track_type = track.get("type")
        tid = track.get("id")
        props = track.get("properties", {})
        uid = props.get("uid", None)

        language = props.get("language")
        language = language_utils.unify_lang(language) if language else None

        track_initial_data = find_ffprobe_track(tid, ffprobe_info)

        if track_type == "video":
            dims = props.get("pixel_dimensions") or props.get("display_dimensions")
            width = height = None
            if dims and "x" in dims:
                try:
                    w, h = dims.split("x")
                    width = int(w)
                    height = int(h)
                except ValueError:
                    pass

            fps = props.get("frame_rate")
            if not fps:
                default_duration = props.get("default_duration")
                if default_duration:
                    try:
                        fps = 1_000_000_000 / float(default_duration)
                    except (TypeError, ValueError):
                        fps = None

            fps_str = str(fps) if fps else track_initial_data.get("fps", "0") if track_initial_data else "0"

            streams["video"].append(
                merge_properties(track_initial_data, {
                    "fps": fps_str,
                    "length": duration_ms,
                    "width": width,
                    "height": height,
                    "bitrate": None,
                    "codec": track.get("codec"),
                    "tid": tid,
                    "uid": uid,
                })
            )
        elif track_type == "audio":
            channels = props.get("audio_channels")
            sample_rate = props.get("audio_sampling_frequency")

            streams["audio"].append(
                merge_properties(track_initial_data, {
                    "language": language,
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "tid": tid,
                    "uid": uid,
                })
            )
        elif track_type in ("subtitles", "subtitle"):
            streams["subtitle"].append(
                merge_properties(track_initial_data, {
                    "language": language,
                    "default": props.get("default_track", False),
                    "length": duration_ms,
                    "tid": tid,
                    "uid": uid,
                    "format": track.get("codec"),
                })
            )

    # attachments
    attachments = []
    for attachment in info.get("attachments", []):
        content_type = attachment.get("content_type", "")
        if content_type[:5] == "image":
            props = track.get("properties", {})
            uid = props.get("uid", None)
            attachments.append(
            {
                "tid": attachment["id"],
                "uid": uid,
                "content_type": content_type,
                "file_name": attachment["file_name"],
            })

    return {
        "attachments": attachments,
        "tracks": dict(streams),
    }


def compare_videos(lhs: List[Dict], rhs: List[Dict]) -> bool:
    if len(lhs) != len(rhs):
        return False

    for lhs_item, rhs_item in zip(lhs, rhs):
        lhs_fps = fps_str_to_float(lhs_item["fps"])
        rhs_fps = fps_str_to_float(rhs_item["fps"])

        if lhs_fps == rhs_fps:
            return True

        diff = abs(lhs_fps - rhs_fps)

        # For videos with fps 1000000/33333 (≈30fps) mkvmerge generates video with 30/1 fps.
        # And videos with fps 29999/500 (≈60fps) it uses 60/1 fps.
        # I'm not sure if this is acceptable but at this moment let it be
        if diff > 0.0021:
            return False

    return True


def collect_video_files(path: str, interruptible) -> List[str]:
    video_files = []
    for cd, _, files in os.walk(path, followlinks=True):
        for file in files:
            interruptible._check_for_stop()
            file_path = os.path.join(cd, file)

            if is_video(file_path):
                video_files.append(file_path)

    return video_files


def generate_mkv(output_path: str, input_video: str, subtitles: List[SubtitleFile] | None = None, audios: List[Dict] | None = None, thumbnail: Union[str, None] = None):
    subtitles = subtitles or []
    audios = audios or []

    options = ["-o", output_path]
    options.append(input_video)

    for audio in audios:
        if "language" in audio and audio["language"]:
            options.extend(["--language", f"0:{audio['language']}"])

        if audio.get("default", False):
            options.extend(["--default-track", "0:yes"])
        else:
            options.extend(["--default-track", "0:no"])

        options.append(audio["path"])

    for i, subtitle in enumerate(subtitles):
        lang = subtitle.language

        if lang:
            options.extend(["--language", f"0:{lang}"])

        if i == 0:
            options.extend(["--default-track", "0:yes"])
        else:
            options.extend(["--default-track", "0:no"])

        options.append(subtitle.path)

    if thumbnail:
        options.extend(["--attach-file", thumbnail])

    cmd = "mkvmerge"
    result = process_utils.start_process(cmd, options)

    # validate result and output file
    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"{cmd} exited with unexpected error:\n{result.stderr}\n\nAnd output: {result.stdout}")

    if not os.path.exists(output_path):
        logging.error("Output file was not created")
        raise RuntimeError(f"{cmd} did not create output file")
