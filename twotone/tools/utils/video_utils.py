import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from tqdm import tqdm

from . import language_utils, process_utils, subtitles_utils
from .generic_utils import InterruptibleProcess, fps_str_to_float, get_tqdm_defaults, time_to_ms
from .subtitles_utils import SubtitleFile

DEFAULT_LOGGER = logging.getLogger("TwoTone.utils.video_utils")
_SHOWINFO_PTS_TIME_RE = re.compile(r"pts_time:([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
_SHOWINFO_FRAME_RE = re.compile(r"n: *(\d+).*pts_time:([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
_MEDIA_STREAM_TYPES = ("video", "audio", "subtitle")
_OUTPUT_DURATION_TOLERANCE_MS = 1000
_OUTPUT_END_PACKET_TOLERANCE_MS = 250
_OUTPUT_TAIL_PROBE_LOOKBACK_MS = 30_000
_OUTPUT_TAIL_PROBE_DURATION_SECONDS = 60


def _start_ffmpeg_streaming(
    args: list[str],
    interruption: InterruptibleProcess | None = None,
    on_line: "Callable[[str], None] | None" = None,
    logger: logging.Logger | None = None,
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

    logger = logger or DEFAULT_LOGGER
    command = ["ffmpeg"] + full_args
    logger.debug(f"Starting ffmpeg {' '.join(full_args)}")

    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
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
        proc.stdout.close()

    if proc.stderr:
        proc.stderr.close()

    proc.wait()
    return proc, stderr_lines


def is_video(file: str) -> bool:
    return Path(file).suffix[1:].lower() in ["mkv", "mp4", "avi", "mpg", "mpeg", "mov", "rmvb"]


def get_video_frames_count(video_file: str, logger: logging.Logger | None = None):
    logger = logger or DEFAULT_LOGGER
    result = process_utils.start_process("ffprobe", ["-v", "error", "-select_streams", "v:0", "-count_packets",
                                   "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", video_file],
                                   logger=logger)

    try:
        return int(result.stdout.strip())
    except ValueError:
        logger.error(f"Failed to get frame count for {video_file}")
        return None


def _showinfo_timestamp_correction_ms(video_path: str, logger: logging.Logger | None = None) -> int:
    """Return the correction needed for ffmpeg ``showinfo`` timestamps.

    Some ffmpeg versions expose a negative container start caused by AAC codec
    delay as a positive offset in video-filter ``pts_time`` values.  Matroska
    keeps that delay on the audio track; video blocks can still start at zero.
    Only negative container starts are compensated here so real positive stream
    offsets remain visible to callers.
    """
    logger = logger or DEFAULT_LOGGER
    result = process_utils.start_process(
        "ffprobe",
        [
            "-v", "error",
            "-show_entries", "format=start_time",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        logger=logger,
    )
    if result.returncode != 0:
        logger.warning("Could not probe format start time for %s", video_path)
        return 0

    try:
        start_time = float(result.stdout.strip() or 0.0)
    except ValueError:
        return 0

    return min(0, round(start_time * 1000))


def _showinfo_timestamp_ms(pts_time: str, correction_ms: int = 0) -> int:
    return max(0, round(float(pts_time) * 1000) + correction_ms)


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
    logger = logger or DEFAULT_LOGGER

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

    duration_ms = get_video_duration(file_path, logger=logger)
    duration_s = (duration_ms / 1000.0) if duration_ms else None
    timestamp_correction_ms = _showinfo_timestamp_correction_ms(file_path, logger=logger)

    pbar = tqdm(
        total=duration_s,
        desc=bar_desc,
        unit="s",
        **get_tqdm_defaults(),
    )
    last_time = 0.0

    def _on_line(line: str) -> None:
        nonlocal last_time
        m = _SHOWINFO_PTS_TIME_RE.search(line)
        if m:
            t = _showinfo_timestamp_ms(m.group(1), timestamp_correction_ms) / 1000
            delta = t - last_time
            if delta > 0:
                pbar.update(delta)
                last_time = t

    proc, stderr_lines = _start_ffmpeg_streaming(args, interruption, on_line=_on_line, logger=logger)
    if duration_s and last_time < duration_s:
        pbar.update(duration_s - last_time)
    pbar.close()

    if proc.returncode != 0:
        logger.warning(f"ffmpeg scene detection exited with code {proc.returncode}")

    # Look for lines with "pts_time:"; these indicate the timestamp of a scene change.
    scene_times = []
    for line in stderr_lines:
        match = _SHOWINFO_PTS_TIME_RE.search(line)
        if match:
            time_ms = _showinfo_timestamp_ms(match.group(1), timestamp_correction_ms)
            scene_times.append(time_ms)

    logger.debug(f"Detected {len(scene_times)} scene changes in {basename}")

    return sorted(set(scene_times))


def extract_timestamp_frame_mapping(video_path: str, logger: logging.Logger | None = None) -> dict[int, int]:
    """
    Extracts a mapping of timestamp (seconds) to frame number from a video.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        dict: A dictionary mapping {timestamp in ms (int): frame number (int)}
    """

    logger = logger or DEFAULT_LOGGER
    args = [
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=coded_picture_number,pkt_dts_time",
        "-print_format", "flat",
        video_path
    ]

    # Run the command
    result = process_utils.start_process("ffprobe", args, logger=logger)

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
) -> dict[int, Any]:
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
    provided_logger = logger
    logger = logger or DEFAULT_LOGGER

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
    total_frames = get_video_frames_count(video_path, logger=logger)

    fallback_options: list[list[str]] = [
        ["-fps_mode", "vfr"],
        ["-fps_mode", "cfr"],
        [],
    ]

    timestamp_correction_ms = _showinfo_timestamp_correction_ms(video_path, logger=logger)
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
            match = _SHOWINFO_FRAME_RE.search(line)
            if match:
                frame_number = int(match.group(1))
                timestamp_ms = _showinfo_timestamp_ms(match.group(2), timestamp_correction_ms)
                showinfo_entries.append((frame_number, timestamp_ms))
                pbar.update(1)

        proc, stderr_lines = _start_ffmpeg_streaming(args, interruption, on_line=_on_line, logger=logger)
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
        logger.warning(
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

    if provided_logger:
        logger.info(f"Extracted {len(mapping)} frames from {basename}")

    return mapping


def probe_frame_timestamps(
    video_path: str,
    interruption: InterruptibleProcess | None = None,
    desc: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[int, Any]:
    """Probe all frame timestamps without writing image files.

    Returns ``{timestamp_ms: {"frame_id": N, "path": None}}`` for every
    frame in the video.  Uses ffmpeg's ``showinfo`` filter with null
    output so that frame numbering (``n:``) is guaranteed to match the
    ``n`` variable used by ffmpeg's ``select`` filter.
    """
    logger = logger or DEFAULT_LOGGER
    args = [
        "-i", video_path,
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-vf", "showinfo",
        "-f", "null", "-",
    ]

    basename = os.path.basename(video_path)
    bar_desc = desc or f"Probing frames: {basename}"

    total_frames = get_video_frames_count(video_path, logger=logger)

    timestamp_correction_ms = _showinfo_timestamp_correction_ms(video_path, logger=logger)
    entries: list[tuple[int, int]] = []

    pbar = tqdm(
        total=total_frames,
        desc=bar_desc,
        unit="frame",
        **get_tqdm_defaults(),
    )

    def _on_line(line: str) -> None:
        match = _SHOWINFO_FRAME_RE.search(line)
        if match:
            frame_number = int(match.group(1))
            timestamp_ms = _showinfo_timestamp_ms(match.group(2), timestamp_correction_ms)
            entries.append((frame_number, timestamp_ms))
            pbar.update(1)

    proc, stderr_lines = _start_ffmpeg_streaming(args, interruption, on_line=_on_line, logger=logger)
    pbar.close()

    if proc.returncode != 0:
        stderr_tail = "".join(stderr_lines[-20:]) if stderr_lines else "(no output)"
        raise RuntimeError(
            f"ffmpeg probe failed for {basename}. Last output:\n{stderr_tail}"
        )

    mapping: dict[int, dict] = {}
    for frame_number, timestamp_ms in entries:
        mapping[timestamp_ms] = {"frame_id": frame_number, "path": None}

    return mapping


def _balanced_select_expr(frame_ranges: list[tuple[int, int]]) -> str:
    """Build a balanced binary tree of between() clauses for ffmpeg's select filter.

    A flat ``a+b+c+d`` expression has O(N) parser stack depth and hits
    ffmpeg's internal limit at ~101 terms.  A balanced tree
    ``(a+b)+(c+d)`` has O(log₂ N) depth, supporting thousands of ranges
    in a single invocation.
    """
    parts = [f"between(n\\,{start}\\,{end})" for start, end in frame_ranges]

    def _build(items: list[str]) -> str:
        if len(items) == 1:
            return items[0]
        mid = len(items) // 2
        left = _build(items[:mid])
        right = _build(items[mid:])
        return f"({left}+{right})"

    return _build(parts)


def extract_frames_at_ranges(
    video_path: str,
    target_dir: str,
    frame_ranges: list[tuple[int, int]],
    probed_metadata: dict[int, Any],
    format: str = "jpeg",
    scale: float | tuple[int, int] = 0.5,
    interruption: InterruptibleProcess | None = None,
    desc: str | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Extract frames from specific frame-number ranges and update *probed_metadata* paths.

    *frame_ranges* is a list of ``(first_frame_id, last_frame_id)`` inclusive
    ranges where frame IDs correspond to ffmpeg's sequential ``n`` variable.

    *probed_metadata* is the dict returned by :func:`probe_frame_timestamps`.
    For each extracted frame, the ``"path"`` value at the matching timestamp
    key is set to the written file on disk.

    Extracted files are named ``frame_<timestamp_ms>.<format>`` — unique and
    stable across invocations, so paths recorded in *probed_metadata* by
    earlier extractions into the same directory stay valid, and path-keyed
    caches never see a path re-used with different content.  ffmpeg itself
    writes sequentially-numbered files into a private temporary subdirectory,
    which are renamed once the showinfo timestamps are known.

    Uses ffmpeg's ``select='between(n,a,b)+…'`` filter so only the
    requested frames are encoded and written.  The select expression is
    structured as a balanced binary tree of ``+`` operations so that the
    parser stack depth is O(log₂ N) instead of O(N), allowing thousands
    of ranges in a single ffmpeg invocation.
    """
    if not frame_ranges:
        return

    logger = logger or DEFAULT_LOGGER
    select_expr = _balanced_select_expr(frame_ranges)

    scale_filter = ""
    if isinstance(scale, float):
        if scale != 1.0:
            rscale = 1 / scale
            scale_filter = f"scale=iw/{rscale}:ih/{rscale}"
    elif isinstance(scale, tuple):
        scale_filter = f"scale={scale[0]}:{scale[1]}"

    vf_parts = [f"select='{select_expr}'", "showinfo"]
    if scale_filter:
        vf_parts.append(scale_filter)

    total_frames = sum(end - start + 1 for start, end in frame_ranges)

    basename = os.path.basename(video_path)
    bar_desc = desc or f"Extracting frames: {basename}"

    timestamp_correction_ms = _showinfo_timestamp_correction_ms(video_path, logger=logger)
    showinfo_entries: list[tuple[int, int]] = []  # (output_seq, timestamp_ms)

    pbar = tqdm(
        total=total_frames,
        desc=bar_desc,
        unit="frame",
        **get_tqdm_defaults(),
    )

    def _on_line(line: str) -> None:
        match = _SHOWINFO_FRAME_RE.search(line)
        if match:
            output_seq = int(match.group(1))
            timestamp_ms = _showinfo_timestamp_ms(match.group(2), timestamp_correction_ms)
            showinfo_entries.append((output_seq, timestamp_ms))
            pbar.update(1)

    fallback_options: list[list[str]] = [
        ["-fps_mode", "vfr"],
        ["-fps_mode", "cfr"],
        [],
    ]

    work_dir = tempfile.mkdtemp(prefix=".extract_", dir=target_dir)
    output_pattern = os.path.join(work_dir, f"frame_%08d.{format}")

    def _clean_work_dir():
        for filename in os.listdir(work_dir):
            os.unlink(os.path.join(work_dir, filename))

    try:
        last_stderr: list[str] = []
        for opts in fallback_options:
            _clean_work_dir()
            showinfo_entries.clear()
            pbar.reset()

            args = [
                "-i", video_path,
                "-map", "0:v:0",
                "-an", "-sn", "-dn",
                *opts,
                "-q:v", "2",
                "-vf", ",".join(vf_parts),
                output_pattern,
            ]

            proc, stderr_lines = _start_ffmpeg_streaming(args, interruption, on_line=_on_line, logger=logger)
            last_stderr = stderr_lines

            if proc.returncode == 0:
                break
        else:
            pbar.close()
            stderr_tail = "".join(last_stderr[-20:]) if last_stderr else "(no output)"
            raise RuntimeError(
                f"ffmpeg selective extraction failed for {basename}. "
                f"Last output:\n{stderr_tail}"
            )

        pbar.close()

        frame_files = sorted(os.listdir(work_dir))

        usable = min(len(showinfo_entries), len(frame_files))
        if len(showinfo_entries) != len(frame_files):
            logger.warning(
                f"Frame count mismatch for {basename}: showinfo reported "
                f"{len(showinfo_entries)} frames but {len(frame_files)} files "
                f"on disk. Using {usable}."
            )

        for i in range(usable):
            _, timestamp_ms = showinfo_entries[i]
            if timestamp_ms not in probed_metadata:
                logger.warning(
                    f"Extracted frame at {timestamp_ms}ms has no matching "
                    f"entry in probed metadata — skipping."
                )
                continue

            frame_no = probed_metadata[timestamp_ms]["frame_id"]

            final_path = os.path.join(target_dir, f"frame_{frame_no:06d}_{timestamp_ms:010d}.{format}")
            os.replace(os.path.join(work_dir, frame_files[i]), final_path)
            probed_metadata[timestamp_ms]["path"] = final_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def get_video_duration(video_file, logger: logging.Logger | None = None):
    """Get the duration of a video in milliseconds."""
    logger = logger or DEFAULT_LOGGER
    result = process_utils.start_process("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_file], logger=logger)

    try:
        return int(float(result.stdout.strip())*1000)
    except ValueError:
        logger.error(f"Failed to get duration for {video_file}")
        return None


def get_video_full_info(path: str, logger: logging.Logger | None = None) -> dict:
    logger = logger or DEFAULT_LOGGER
    args = []
    args.extend(["-v", "quiet"])
    args.extend(["-print_format", "json"])
    args.append("-show_format")
    args.append("-show_streams")
    args.append(path)

    result = process_utils.start_process("ffprobe", args, logger=logger)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe exited with unexpected error:\n{result.stderr}")

    output_lines = result.stdout
    output_json = json.loads(output_lines)

    return output_json


def _seconds_to_ms(value: Any) -> int | None:
    try:
        return round(float(value) * 1000)
    except (TypeError, ValueError):
        return None


def _stream_duration_ms(stream: Mapping[str, Any]) -> int | None:
    duration_ms = _seconds_to_ms(stream.get("duration"))
    if duration_ms is not None:
        return duration_ms

    tag_duration = stream.get("tags", {}).get("DURATION")
    if tag_duration is None:
        return None
    try:
        return time_to_ms(tag_duration)
    except (TypeError, ValueError):
        return None


def _last_content_packet_timestamp_ms(
    path: str,
    stream_index: int,
    expected_end_ms: int,
    logger: logging.Logger,
) -> int | None:
    """Return the end timestamp of the last packet in a bounded tail probe."""
    probe_start_ms = max(0, expected_end_ms - _OUTPUT_TAIL_PROBE_LOOKBACK_MS)
    result = process_utils.start_process(
        "ffprobe",
        [
            "-v", "error",
            "-select_streams", str(stream_index),
            "-read_intervals", f"{probe_start_ms / 1000:.6f}%+{_OUTPUT_TAIL_PROBE_DURATION_SECONDS}",
            "-show_entries", "packet=pts_time,dts_time,duration_time",
            "-of", "json",
            path,
        ],
        logger=logger,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe could not read generated output packets:\n{result.stderr}")

    try:
        packets = json.loads(result.stdout).get("packets", [])
    except json.JSONDecodeError as error:
        raise RuntimeError("ffprobe returned invalid packet data for generated output") from error

    packet_ends: list[int] = []
    for packet in packets:
        timestamp_ms = _seconds_to_ms(packet.get("pts_time"))
        if timestamp_ms is None:
            timestamp_ms = _seconds_to_ms(packet.get("dts_time"))
        if timestamp_ms is None:
            continue
        duration_ms = _seconds_to_ms(packet.get("duration_time")) or 0
        packet_ends.append(timestamp_ms + duration_ms)
    return max(packet_ends, default=None)


def validate_media_output(
    path: str,
    logger: logging.Logger | None = None,
    *,
    expected_stream_counts: Mapping[str, int] | None = None,
    expected_duration_ms: int | None = None,
) -> None:
    """Verify a generated media file before it may replace its destination.

    Header probing alone is insufficient because a truncated Matroska file can
    retain valid stream and duration metadata.  In addition to the declared
    layout and duration, this check reads packet metadata from one fixed-size
    window near the expected end.  Runtime is therefore bounded independently
    of the input film's duration and no media samples are decoded.
    """
    logger = logger or DEFAULT_LOGGER
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Generated output is missing or empty: {path}")

    info = get_video_full_info(path, logger=logger)
    media_format = info.get("format")
    streams = info.get("streams")
    if not media_format or not streams:
        raise RuntimeError(f"Generated output is not a valid media file: {path}")

    content_streams = [
        stream for stream in streams
        if stream.get("codec_type") in _MEDIA_STREAM_TYPES
        and not stream.get("disposition", {}).get("attached_pic", 0)
    ]
    actual_stream_counts = Counter(stream.get("codec_type") for stream in content_streams)
    if expected_stream_counts is not None:
        expected_counts = Counter({
            stream_type: count
            for stream_type, count in expected_stream_counts.items()
            if stream_type in _MEDIA_STREAM_TYPES and count
        })
        if actual_stream_counts != expected_counts:
            raise RuntimeError(
                "Generated output stream layout does not match the mux plan: "
                f"expected {dict(expected_counts)}, got {dict(actual_stream_counts)}"
            )

    packet_stream = next(
        (stream for stream_type in ("video", "audio")
         for stream in content_streams if stream.get("codec_type") == stream_type),
        None,
    )
    if packet_stream is None or packet_stream.get("index") is None:
        raise RuntimeError(f"Generated output has no packet-bearing media stream: {path}")

    actual_duration_ms = _seconds_to_ms(media_format.get("duration"))
    if actual_duration_ms is None:
        actual_duration_ms = _stream_duration_ms(packet_stream)
    if actual_duration_ms is None or actual_duration_ms <= 0:
        raise RuntimeError(f"Generated output has no valid duration: {path}")
    if expected_duration_ms is not None \
            and actual_duration_ms + _OUTPUT_DURATION_TOLERANCE_MS < expected_duration_ms:
        raise RuntimeError(
            f"Generated output is too short: {actual_duration_ms} ms, "
            f"expected at least {expected_duration_ms} ms"
        )

    content_end_ms = expected_duration_ms
    if content_end_ms is None:
        content_end_ms = _stream_duration_ms(packet_stream) or actual_duration_ms
    last_packet_ms = _last_content_packet_timestamp_ms(
        path,
        packet_stream["index"],
        content_end_ms,
        logger,
    )
    if last_packet_ms is None \
            or last_packet_ms + _OUTPUT_END_PACKET_TOLERANCE_MS < content_end_ms:
        raise RuntimeError(
            "Generated output has no media packets near its expected end: "
            f"last packet ends at {last_packet_ms} ms, expected {content_end_ms} ms"
        )


def get_video_data(
    path: str,
    logger: logging.Logger | None = None,
    *,
    _probe_info: dict[str, Any] | None = None,
) -> dict:
    logger = logger or DEFAULT_LOGGER

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

    output_json = _probe_info if _probe_info is not None else get_video_full_info(path, logger=logger)

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
                length = get_video_duration(path, logger=logger)

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


def get_video_full_info_mkvmerge(path: str, logger: logging.Logger | None = None) -> dict:
    """Return file information using ``mkvmerge -J``."""

    logger = logger or DEFAULT_LOGGER
    result = process_utils.start_process("mkvmerge", ["-J", path], logger=logger)

    if result.returncode != 0:
        raise RuntimeError(f"mkvmerge exited with unexpected error:\n{result.stderr}")

    return json.loads(result.stdout)


def get_video_data_mkvmerge(
    path: str,
    enrich: bool = False,
    logger: logging.Logger | None = None,
) -> dict:
    """
        Return stream information parsed from ``mkvmerge -J`` output.
        For non mkv files, mkvmerge does not provide as much information as ffprobe.
        Set 'enrich' to True to enrich mkvmerge's output with data from ffprobe.
        In enriched results, ``tid`` remains the mkvmerge track ID while
        ``ffprobe_stream_index`` is the absolute ffprobe/ffmpeg stream index.
    """
    logger = logger or DEFAULT_LOGGER

    def normalized_track_type(track_type: str | None) -> str | None:
        if track_type in ("subtitle", "subtitles"):
            return "subtitle"
        if track_type in ("video", "audio"):
            return track_type
        return None

    def native_track_id(value: object) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError:
                return None
        return None

    def build_track_mapping(mkv_tracks: list[dict], probe_info: dict) -> dict[int, int]:
        """Map mkvmerge track IDs to absolute ffprobe stream indexes.

        The two tools use independent namespaces.  A container-native track
        number is used when both tools expose it; Matroska commonly does not
        expose one through ffprobe, so remaining tracks are correlated by
        order within their already-validated media type.
        """
        mapping: dict[int, int] = {}
        for stream_type in ("video", "audio", "subtitle"):
            typed_tracks = [
                track for track in mkv_tracks
                if normalized_track_type(track.get("type")) == stream_type
            ]
            typed_streams = [
                stream for stream in probe_info.get("streams", [])
                if stream.get("codec_type") == stream_type
                and not stream.get("disposition", {}).get("attached_pic", 0)
            ]
            if len(typed_tracks) != len(typed_streams):
                raise RuntimeError(
                    f"Cannot map {stream_type} tracks between mkvmerge and ffprobe: "
                    f"mkvmerge reported {len(typed_tracks)}, ffprobe reported {len(typed_streams)}."
                )

            unmatched_tracks: list[dict] = []
            used_probe_indexes: set[int] = set()
            for track in typed_tracks:
                track_number = native_track_id(track.get("properties", {}).get("number"))
                native_matches = [
                    stream for stream in typed_streams
                    if stream["index"] not in used_probe_indexes
                    and native_track_id(stream.get("id")) == track_number
                    and track_number is not None
                ]
                if len(native_matches) == 1:
                    probe_index = int(native_matches[0]["index"])
                    mapping[int(track["id"])] = probe_index
                    used_probe_indexes.add(probe_index)
                else:
                    unmatched_tracks.append(track)

            unmatched_streams = [
                stream for stream in typed_streams
                if int(stream["index"]) not in used_probe_indexes
            ]
            if len(unmatched_tracks) != len(unmatched_streams):
                raise RuntimeError(
                    f"Cannot build a one-to-one {stream_type} track mapping "
                    "between mkvmerge and ffprobe."
                )
            for track, stream in zip(unmatched_tracks, unmatched_streams):
                track_id = int(track["id"])
                probe_index = int(stream["index"])
                if track_id in mapping or probe_index in used_probe_indexes:
                    raise RuntimeError(
                        f"Ambiguous {stream_type} track mapping between mkvmerge and ffprobe."
                    )
                mapping[track_id] = probe_index
                used_probe_indexes.add(probe_index)

        supported_track_ids = [
            int(track["id"])
            for track in mkv_tracks
            if normalized_track_type(track.get("type")) is not None
        ]
        supported_probe_indexes = [
            int(stream["index"])
            for stream in probe_info.get("streams", [])
            if normalized_track_type(stream.get("codec_type")) is not None
            and not stream.get("disposition", {}).get("attached_pic", 0)
        ]
        if (
            set(mapping) != set(supported_track_ids)
            or len(mapping) != len(supported_track_ids)
            or set(mapping.values()) != set(supported_probe_indexes)
            or len(mapping) != len(supported_probe_indexes)
        ):
            raise RuntimeError("Cannot build a one-to-one track mapping between mkvmerge and ffprobe.")

        return mapping

    def build_ffprobe_stream_lookup(ffprobe_info: dict | None) -> dict[int, dict] | None:
        if ffprobe_info is not None:
            streams_by_index: dict[int, dict] = {}
            for typed_streams in ffprobe_info.values():
                for stream in typed_streams:
                    stream_index = stream.get("tid")
                    if not isinstance(stream_index, int):
                        raise RuntimeError("ffprobe returned a stream without an integer index.")
                    if stream_index in streams_by_index:
                        raise RuntimeError(f"ffprobe returned duplicate stream index #{stream_index}.")
                    streams_by_index[stream_index] = stream
            return streams_by_index
        else:
            return None

    def find_ffprobe_track(
        track: dict,
        ffprobe_streams_by_index: Mapping[int, dict] | None,
        track_mapping: dict[int, int],
    ) -> dict:
        if ffprobe_streams_by_index is not None:
            track_type = normalized_track_type(track.get("type"))
            probe_index = track_mapping[int(track["id"])]
            stream = ffprobe_streams_by_index.get(probe_index)
            if stream is not None:
                return stream
            raise RuntimeError(
                f"Mapped ffprobe {track_type} stream #{probe_index} was not found "
                f"for mkvmerge track #{track.get('id')}."
            )
        else:
            return {}

    def merge_properties(initial: dict | None, update: dict) -> dict:
        output = initial.copy() if initial is not None else {}
        for key, value in update.items():
            base_value = output.get(key, None)

            if base_value is None:
                output[key] = value
            elif value is None:
                pass
            elif value != base_value:
                if key != "codec" and key != "format":
                    if key == "fps" and abs(fps_str_to_float(base_value) - fps_str_to_float(value)) > 0.001:
                        logger.warning(f"Inconsistent data provided by mkvmerge ({key}: {value}) and ffprobe ({key}: {base_value})")
                output[key] = value

        return output

    info = get_video_full_info_mkvmerge(path, logger=logger)

    # process streams/tracks
    streams = defaultdict(list)
    probe_info = get_video_full_info(path, logger=logger) if enrich else None
    ffprobe_info = (
        get_video_data(path, logger=logger, _probe_info=probe_info)
        if probe_info is not None
        else None
    )
    ffprobe_streams_by_index = build_ffprobe_stream_lookup(ffprobe_info)
    track_mapping = build_track_mapping(info.get("tracks", []), probe_info) if probe_info is not None else {}

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

        track_initial_data = find_ffprobe_track(track, ffprobe_streams_by_index, track_mapping)

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
        if enrich:
            stream_data["ffprobe_stream_index"] = track_mapping[int(tid)]

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

    logger = logger or DEFAULT_LOGGER
    tids_list: list[int] = list(tids)
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
        info = get_video_data(video_path, logger=logger)
        stream_fmt = {s.get("tid"): (s.get("format") or "").lower() for s in info.get("subtitle", [])}
    except Exception as e:
        stream_fmt = {}
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
        logger.debug("  tid #%s -> %s (format=%s)", tid, out_path, fmt or "unknown")

    try:
        start = time.perf_counter()
        status = process_utils.start_process("mkvextract", options, logger=logger)
        elapsed = time.perf_counter() - start
        logger.debug("mkvextract finished in %.3fs (rc=%s)", elapsed, status.returncode)
        if status.returncode != 0:
            logger.error(f"mkvextract failed for {video_path}: {status.stderr}")

    except Exception as e:
        logger.error(f"Subtitle extraction failed for {video_path}: {e}")

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


def generate_mkv(
    output_path: str,
    input_video: str,
    subtitles: list[SubtitleFile] | dict | None = None,
    audios: list[dict] | None = None,
    thumbnail: str | None = None,
    logger: logging.Logger | None = None,
):
    logger = logger or DEFAULT_LOGGER
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
    result = process_utils.start_process(cmd, options, logger=logger)

    # validate result and output file
    # mkvmerge returns: 0 = success, 1 = success with warnings, 2 = error
    if result.returncode == 1:
        warnings = (result.stdout or "") + (result.stderr or "")
        if warnings.strip():
            logger.warning(f"{cmd} completed with warnings: {warnings.strip()}")
    elif result.returncode > 1:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"{cmd} exited with unexpected error:\n{result.stderr}\n\nAnd output: {result.stdout}")

    validate_media_output(output_path, logger=logger)
