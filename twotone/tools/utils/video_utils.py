import json
import logging
import os
import platform
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from . import language_utils, process_utils, subtitles_utils
from .generic_utils import InterruptibleProcess, fps_str_to_float, get_tqdm_defaults, time_to_ms
from .subtitles_utils import SubtitleFile


def _start_ffmpeg_streaming(
    args: list[str],
    interruption: InterruptibleProcess | None = None,
    on_line: "Callable[[str], None] | None" = None,
) -> tuple[subprocess.Popen, list[str]]:
    """Start an ffmpeg subprocess and read its stderr line-by-line.

    Returns ``(process, stderr_lines)`` after ffmpeg finishes.
    Checks *interruption* between lines so ctrl+c can terminate the run.
    Terminates the subprocess on interruption.
    When *on_line* is given, it is called with each stderr line (e.g. for
    progress updates).
    """
    defaults = process_utils.DEFAULT_TOOL_OPTIONS.get("ffmpeg", [])
    full_args = list(args)
    for opt in reversed(defaults):
        if opt not in full_args:
            full_args.insert(0, opt)

    command = ["ffmpeg"] + full_args
    logging.debug(f"Starting ffmpeg {' '.join(full_args)}")

    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "universal_newlines": True,
        "bufsize": 1,
    }
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(command, **popen_kwargs)

    stderr_lines: list[str] = []
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)
            if on_line is not None:
                on_line(line)
            if interruption is not None and not interruption._work:
                proc.terminate()
                proc.wait()
                interruption.check_for_stop()  # raises SystemExit
    except Exception:
        proc.terminate()
        proc.wait()
        raise

    # Drain stdout (ffmpeg sends everything to stderr, but be safe)
    if proc.stdout:
        proc.stdout.read()

    proc.wait()
    return proc, stderr_lines


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


def detect_scene_changes(
    file_path: str,
    threshold: float = 0.4,
    logger: logging.Logger | None = None,
    interruption: InterruptibleProcess | None = None,
    desc: str | None = None,
) -> list[int]:
    """
        Run ffmpeg with a scene detection filter and extract scene change times.
        Function returns list of scene changes in milliseconds.

        When *desc* is given, it is used as the progress bar description.
        When *logger* is given (and no *desc*), an info message is emitted.
        When *interruption* is given, ctrl+c can cleanly stop the process.
    """

    args = [
        "-i", file_path,
        "-an",                                              # Ignore all audio streams
        "-sn",                                              # Ignore subtitle streams
        "-dn",                                              # Ignore data streams
        "-fps_mode", "auto",
        "-filter_complex", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]

    basename = os.path.basename(file_path)
    bar_desc = desc or f"Detecting scenes: {basename}"

    duration_ms = get_video_duration(file_path)
    duration_s = (duration_ms / 1000.0) if duration_ms else None
    pts_time_re = re.compile(r"pts_time:(\d+\.?\d*)")

    pbar = tqdm(
        total=duration_s,
        desc=bar_desc,
        unit="s",
        **get_tqdm_defaults(),
    )
    last_time = 0.0

    def _on_line(line: str) -> None:
        nonlocal last_time
        m = pts_time_re.search(line)
        if m:
            t = float(m.group(1))
            delta = t - last_time
            if delta > 0:
                pbar.update(delta)
                last_time = t

    proc, stderr_lines = _start_ffmpeg_streaming(args, interruption, on_line=_on_line)
    if duration_s and last_time < duration_s:
        pbar.update(duration_s - last_time)
    pbar.close()

    if proc.returncode != 0:
        logging.warning(f"ffmpeg scene detection exited with code {proc.returncode}")

    # Look for lines with "pts_time:"; these indicate the timestamp of a scene change.
    scene_times = []
    pattern = re.compile(r"pts_time:(\d+\.\d+)")
    for line in stderr_lines:
        match = pattern.search(line)
        if match:
            time_s = float(match.group(1))
            time_ms = int(round(time_s * 1000))
            scene_times.append(time_ms)

    if logger:
        logger.debug(f"Detected {len(scene_times)} scene changes in {basename}")

    return sorted(set(scene_times))


def extract_timestamp_frame_mapping(video_path: str) -> dict[int, int]:
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


def extract_all_frames(
    video_path: str,
    target_dir: str,
    format: str = "jpeg",
    scale: float | tuple[int, int] = 0.5,
    logger: logging.Logger | None = None,
    interruption: InterruptibleProcess | None = None,
    desc: str | None = None,
) -> dict[int, dict]:
    """
        Extract all frames into *target_dir* (should be empty).

        Returns a dict mapping timestamp_ms -> {'path': frame_path, 'frame_id': sequential_number}.

        Frames use sequential numbering (no ``-frame_pts true``) so the
        file count is always reliable.  Timestamps come from the
        ``showinfo`` filter's ``pts_time`` output.  If the two counts
        diverge (extremely rare), the smaller one is used with a warning.

        When *logger* is given, an info message is emitted.
        When *interruption* is given, ctrl+c can cleanly stop the process.
    """

    # Clear target directory
    def clean_target_dir():
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    scale_filter = ""
    if isinstance(scale, float):
        if scale != 1.0:
            rscale = 1 / scale
            scale_filter = f"scale=iw/{rscale}:ih/{rscale}"
    elif isinstance(scale, tuple):
        scale_filter = f"scale={scale[0]}:{scale[1]}"
    else:
        raise RuntimeError("Invalid type for scale")

    # Sequential numbering — always reliable, no PTS-based naming issues
    output_pattern = os.path.join(target_dir, f"frame_%08d.{format}")

    def build_args(extra_args: list[str]) -> list[str]:
        vf_parts = ["showinfo"]
        if scale_filter:
            vf_parts.append(scale_filter)

        return [
            "-i", video_path,
            "-map", "0:v:0",           # Explicitly select only the first video stream
            "-an",                      # Ignore audio
            "-sn",                      # Ignore subtitles
            "-dn",                      # Ignore data streams
            *extra_args,
            "-q:v", "2",
            "-vf", ",".join(vf_parts),
            output_pattern,
        ]

    basename = os.path.basename(video_path)
    bar_desc = desc or f"Extracting frames: {basename}"

    # Get total frame count for a real progress bar
    total_frames = get_video_frames_count(video_path)

    fallback_options: list[list[str]] = [
        ["-fps_mode", "vfr"],
        ["-fps_mode", "cfr"],
        [],
    ]

    frame_pattern = re.compile(r"n: *(\d+).*pts_time:([\d.]+)")
    showinfo_entries: list[tuple[int, int]] = []      # (frame_number, timestamp_ms)

    last_stderr: list[str] = []
    for opts in fallback_options:
        clean_target_dir()
        showinfo_entries.clear()
        args = build_args(opts)

        pbar = tqdm(
            total=total_frames,
            desc=bar_desc,
            unit="frame",
            **get_tqdm_defaults(),
        )

        def _on_line(line: str) -> None:
            match = frame_pattern.search(line)
            if match:
                frame_number = int(match.group(1))
                timestamp_ms = int(round(float(match.group(2)) * 1000))
                showinfo_entries.append((frame_number, timestamp_ms))
                pbar.update(1)

        proc, stderr_lines = _start_ffmpeg_streaming(args, interruption, on_line=_on_line)
        pbar.close()
        last_stderr = stderr_lines

        if proc.returncode == 0:
            break
    else:
        stderr_tail = "".join(last_stderr[-20:]) if last_stderr else "(no output)"
        raise RuntimeError(
            f"ffmpeg frame extraction failed for {basename}. "
            f"Last output:\n{stderr_tail}"
        )

    frame_files = sorted(os.listdir(target_dir))

    # Handle count mismatch gracefully — use the smaller count
    usable = min(len(showinfo_entries), len(frame_files))
    if len(showinfo_entries) != len(frame_files):
        logging.warning(
            f"Frame count mismatch for {basename}: showinfo reported {len(showinfo_entries)} "
            f"frames but {len(frame_files)} files on disk. Using {usable}."
        )

    mapping: dict[int, dict] = {}
    for i in range(usable):
        frame_number, timestamp_ms = showinfo_entries[i]
        fname = frame_files[i]
        mapping[timestamp_ms] = {
            "path": os.path.join(target_dir, fname),
            "frame_id": frame_number,
        }

    if logger:
        logger.info(f"Extracted {len(mapping)} frames from {basename}")

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


def get_video_data(path: str) -> dict:

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

    def get_language(stream) -> str | None:
        if "tags" in stream:
            tags = stream["tags"]
            language = tags.get("language", None)
        else:
            language = None

        try:
            if language:
                language = language_utils.unify_lang(language)
        except Exception:
            language = None

        return language

    output_json = get_video_full_info(path)

    streams = defaultdict(list)
    for stream in output_json["streams"]:
        stream_type = stream["codec_type"]
        tid = stream["index"]
        codec = stream.get("codec_name", None)
        disposition = stream.get("disposition", {})
        is_default = disposition.get("default", 0)
        is_forced = disposition.get("forced", 0)

        stream_data = {
            "default": bool(is_default),
            "forced": bool(is_forced),
            "tid": tid,
            "codec": codec,
        }

        if stream_type == "subtitle":
            language = get_language(stream)
            length = get_length(stream)
            title = stream.get("tags", {}).get("title", None)

            stream_data.update({
                "language": language,
                "length": length,
                "tid": tid,
                "title": title,
                "format": codec})                   # for backward compatibility. TODO: remove when all tools are switched to "codec" for subtitles
        elif stream_type == "video":
            fps = stream["r_frame_rate"]
            length = get_length(stream)
            if length is None:
                length = get_video_duration(path)

            width = stream["width"]
            height = stream["height"]
            bitrate = stream["bitrate"] if "bitrate" in stream else None

            stream_data.update({
                "fps": fps,
                "length": length,
                "width": width,
                "height": height,
                "bitrate": bitrate,
                "tid": tid,
            })
        elif stream_type == "audio":
            language = get_language(stream)
            channels = stream["channels"]
            sample_rate = int(stream["sample_rate"])

            stream_data.update({
                "language": language,
                "channels": channels,
                "sample_rate": sample_rate,
                "tid": tid,
            })

        streams[stream_type].append(stream_data)

    return dict(streams)


def get_video_full_info_mkvmerge(path: str) -> dict:
    """Return file information using ``mkvmerge -J``."""

    result = process_utils.start_process("mkvmerge", ["-J", path])

    if result.returncode != 0:
        raise RuntimeError(f"mkvmerge exited with unexpected error:\n{result.stderr}")

    return json.loads(result.stdout)


def get_video_data_mkvmerge(path: str, enrich: bool = False) -> dict:
    """
        Return stream information parsed from ``mkvmerge -J`` output.
        For non mkv files, mkvmerge does not provide as much information as ffprobe.
        Set 'enrich' to True to enrich mkvmerge's outpput with data from ffprobe.
    """

    def find_ffprobe_track(track_id: int, ffprobe_info: dict | None) -> dict:
        for streams in (ffprobe_info or {}).values():
            for stream in streams:
                if stream.get("tid", None) == track_id:
                    return stream

        return {}

    def merge_properties(initial: dict | None, update: dict) -> dict:
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
                if key != "codec" and key != "format":
                    if key == "fps" and abs(fps_str_to_float(base_value) - fps_str_to_float(value)) > 0.001:
                        logging.warning(f"Inconsistent data provided by mkvmerge ({key}: {value}) and ffprobe ({key}: {base_value})")
                output[key] = value

        return output

    info = get_video_full_info_mkvmerge(path)

    # process streams/tracks
    streams = defaultdict(list)
    ffprobe_info = get_video_data(path) if enrich else None

    for track in info.get("tracks", []):
        track_type = track.get("type")
        tid = track.get("id")
        props = track.get("properties", {})
        uid = props.get("uid", None)

        length_ms = None
        tag_duration = props.get("tag_duration")
        if tag_duration is not None:
            length_ms = time_to_ms(tag_duration)

        language = props.get("language")
        if language == "und":
            language = None
        try:
            if language:
                language = language_utils.unify_lang(language)
        except Exception:
            language = None

        track_initial_data = find_ffprobe_track(tid, ffprobe_info)

        # Prepare common data for all stream types first
        stream_data = {
            "tid": tid,
            "uid": uid,
            "language": language,
            "length": length_ms,
            "codec": track.get("codec"),
            "default": props.get("default_track", track_initial_data.get("default", False)),
            "enabled": props.get("enabled_track", track_initial_data.get("enabled", True)),
            "forced": props.get("forced_track", track_initial_data.get("forced", False)),
        }

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

            fps_str = str(fps) if fps else track_initial_data.get("fps", "0")

            stream_data.update({
                "fps": fps_str,
                "width": width,
                "height": height,
                "bitrate": None,
            })

            streams["video"].append(merge_properties(track_initial_data, stream_data))

        elif track_type == "audio":
            channels = props.get("audio_channels")
            sample_rate = props.get("audio_sampling_frequency")

            stream_data.update({
                "channels": channels,
                "sample_rate": sample_rate,
            })

            streams["audio"].append(merge_properties(track_initial_data, stream_data))

        elif track_type in ("subtitles", "subtitle"):
            # Per-track duration for subtitles as well
            length_ms = None
            tag_duration = props.get("tag_duration")
            if tag_duration is not None:
                try:
                    length_ms = int(round(float(tag_duration) / 1_000_000))
                except (TypeError, ValueError):
                    length_ms = None
            stream_data.update({
                "format": track.get("codec"),                               # for backward compatibility. TODO: remove when all tools are switched to "codec" for subtitles
                "name": props.get("track_name"),
            })

            streams["subtitle"].append(merge_properties(track_initial_data, stream_data))

    # attachments
    attachments = []
    for attachment in info.get("attachments", []):
        content_type = attachment.get("content_type", "")
        if content_type[:5] == "image":
            props = attachment.get("properties", {})
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


def compare_videos(lhs: list[dict], rhs: list[dict]) -> bool:
    if len(lhs) != len(rhs):
        return False

    for lhs_item, rhs_item in zip(lhs, rhs):
        lhs_fps = fps_str_to_float(lhs_item["fps"])
        rhs_fps = fps_str_to_float(rhs_item["fps"])

        diff = abs(lhs_fps - rhs_fps)

        # For videos with fps 1000000/33333 (≈30fps) mkvmerge generates video with 30/1 fps.
        # And videos with fps 29999/500 (≈60fps) it uses 60/1 fps.
        # I'm not sure if this is acceptable but at this moment let it be
        if diff > 0.0021:
            return False

    return True


def collect_video_files(path: str, interruptible) -> list[str]:
    video_files = []
    for cd, _, files in os.walk(path, followlinks=True):
        for file in files:
            interruptible.check_for_stop()
            file_path = os.path.join(cd, file)

            if is_video(file_path):
                video_files.append(file_path)

    return video_files


def extract_subtitle_to_temp(video_path: str, tids: list[int], output_base_path: str, logger: logging.Logger | None = None) -> dict[int, str]:
    """Extract subtitle tracks to temporary files.

    - Determines stream formats internally using video metadata.
    - Appends the track id to the output path: f"{output_base_path}.{tid}.{ext}".
    - Returns a mapping {tid: output_path} for all requested tids.
    """

    tids_list: list[int] = list(tids)
    if logger:
        logger.debug("Extracting subtitles from %s (tids=%s)", video_path, ",".join(str(t) for t in tids_list))

    # Map formats to file extensions
    ext_map = {
        "subrip": ".srt",
        "srt": ".srt",
        "ass": ".ass",
        "ssa": ".ssa",
        "hdmv_pgs_subtitle": ".sup",
        "webvtt": ".vtt",
        "mov_text": ".srt",
        "text": ".srt",
    }

    # Discover formats using video_utils
    try:
        info = get_video_data(video_path)
        stream_fmt = {s.get("tid"): (s.get("format") or "").lower() for s in info.get("subtitle", [])}
    except Exception as e:
        stream_fmt = {}
        if logger:
            logger.debug(f"Failed to get stream info for '{video_path}': {e}")

    # Build mkvextract options
    tid_to_path: dict[int, str] = {}
    options = ["tracks", video_path]
    for tid in tids_list:
        fmt = stream_fmt.get(tid, "")
        suffix = ext_map.get(fmt, ".srt")
        out_path = f"{output_base_path}.{tid}{suffix}"
        tid_to_path[tid] = out_path
        options.append(f"{tid}:{out_path}")
        if logger:
            logger.debug("  tid #%s -> %s (format=%s)", tid, out_path, fmt or "unknown")

    try:
        start = time.perf_counter()
        status = process_utils.start_process("mkvextract", options)
        elapsed = time.perf_counter() - start
        if logger:
            logger.debug("mkvextract finished in %.3fs (rc=%s)", elapsed, status.returncode)
        if status.returncode != 0 and logger:
            logger.error(f"mkvextract failed for {video_path}: {status.stderr}")

    except Exception as e:
        if logger:
            logger.error(f"Subtitle extraction failed for {video_path}: {e}")

    if logger:
        for tid, out_path in tid_to_path.items():
            if os.path.exists(out_path):
                try:
                    size = os.path.getsize(out_path)
                except OSError:
                    size = -1
                logger.debug("  extracted tid #%s -> %s (%s bytes)", tid, out_path, size)
            else:
                logger.debug("  missing output for tid #%s -> %s", tid, out_path)

    return tid_to_path


def generate_mkv(output_path: str, input_video: str, subtitles: list[SubtitleFile] | dict | None = None, audios: list[dict] | None = None, thumbnail: str | None = None):
    # RMVB/RM files cannot be reliably converted to MKV due to RealAudio "cook" codec issues.
    # mkvmerge produces broken files with audio sync problems.
    # See: https://gitlab.com/mbunkus/mkvtoolnix/-/issues/708
    # See: https://forum.videohelp.com/threads/299034-Problem-converting-RMVB-to-MP4
    ext = os.path.splitext(input_video)[1].lower()
    if ext in (".rmvb", ".rm"):
        raise ValueError(
            f"Cannot convert RMVB/RM files to MKV (unsupported RealAudio codec): {input_video}"
        )

    subtitles = subtitles or []
    audios = audios or []

    options = ["-o", output_path]
    options.append(input_video)

    for audio in audios:
        if "language" in audio and audio["language"]:
            options.extend(["--language", f"0:{audio['language']}"])

        name = audio.get("name")
        if name:
            options.extend(["--track-name", f"0:{name}"])

        if audio.get("default", False):
            options.extend(["--default-track", "0:yes"])
        else:
            options.extend(["--default-track", "0:no"])

        options.append(audio["path"])

    if isinstance(subtitles, dict):
        subtitles = [subtitles_utils.build_subtitle_from_dict(path, info) for path, info in subtitles.items()]

    for subtitle in subtitles:
        assert subtitle.path

        lang = subtitle.language
        if lang:
            options.extend(["--language", f"0:{lang}"])

        name = subtitle.name
        if name:
            options.extend(["--track-name", f"0:{name}"])

        is_default = subtitle.default
        if is_default:
            options.extend(["--default-track", "0:yes"])
        else:
            options.extend(["--default-track", "0:no"])

        options.append(subtitle.path)

    if thumbnail:
        options.extend(["--attach-file", thumbnail])

    cmd = "mkvmerge"
    result = process_utils.start_process(cmd, options)

    # validate result and output file
    # mkvmerge returns: 0 = success, 1 = success with warnings, 2 = error
    if result.returncode == 1:
        warnings = (result.stdout or "") + (result.stderr or "")
        if warnings.strip():
            logging.warning(f"{cmd} completed with warnings: {warnings.strip()}")
    elif result.returncode > 1:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"{cmd} exited with unexpected error:\n{result.stderr}\n\nAnd output: {result.stdout}")

    if not os.path.exists(output_path):
        logging.error("Output file was not created")
        raise RuntimeError(f"{cmd} did not create output file")
