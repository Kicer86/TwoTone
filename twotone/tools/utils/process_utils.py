
import logging
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from tqdm import tqdm
from typing import Any

from . import generic_utils

DEFAULT_LOGGER = logging.getLogger("TwoTone.utils.process_utils")

DEFAULT_TOOL_OPTIONS: dict[str, list[str]] = {
    "ffmpeg": ["-hide_banner"],
    "ffprobe": ["-hide_banner"],
    "mkvextract": ["--quiet"],
    "exiftool": ["-q"],
}

@dataclass
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


def start_process(
    process: str,
    args: list[str],
    show_progress: bool = False,
    progress_description: str | None = None,
    logger: logging.Logger | None = None,
    cwd: str | None = None
) -> ProcessResult:
    logger = logger or DEFAULT_LOGGER
    defaults = DEFAULT_TOOL_OPTIONS.get(process, [])
    for opt in reversed(defaults):
        if opt not in args:
            args.insert(0, opt)

    if show_progress and process == "ffmpeg":
        args = ["-progress", "pipe:2", "-nostats", *args]

    command = [process]
    command.extend(args)

    full_cmd = f"{process} {' '.join(args)}"
    logger.debug(f"Starting {full_cmd}")
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
        "cwd": cwd,
    }

    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    sub_process = subprocess.Popen(command, **popen_kwargs)

    captured_stderr: list[str] = []
    stdout: str | None = None
    stderr: str | None = None
    if show_progress:
        if process == "ffmpeg":
            description = progress_description or "Processing video"
            logger.info("%s: started.", description)

            if sub_process.stderr:
                # Use FFmpeg's output timestamp as the progress metric. Reading it
                # from the running process avoids a separate input frame-count scan.
                duration_pattern = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")
                progress_pattern = re.compile(r"out_time_us=(-?\d+)$")
                with tqdm(desc=description, unit="s", total=None, **generic_utils.get_tqdm_defaults()) as pbar:
                    last_time = 0.0
                    duration_found = False
                    for line in sub_process.stderr:
                        line = line.strip()
                        captured_stderr.append(line)
                        duration = duration_pattern.search(line)
                        if duration and not duration_found:
                            hours, minutes, seconds = map(float, duration.groups())
                            pbar.total = hours * 3600 + minutes * 60 + seconds
                            duration_found = True
                            pbar.refresh()
                        match = progress_pattern.fullmatch(line)
                        if match:
                            current_time = int(match.group(1)) / 1_000_000
                            if current_time > last_time:
                                pbar.update(current_time - last_time)
                                last_time = current_time
        elif process == "mkvmerge" and sub_process.stdout:
            progress_pattern = re.compile(r"\w:\s*(\d+)%")
            with tqdm(desc="Muxing", unit="%", total=100, **generic_utils.get_tqdm_defaults()) as pbar:
                last_progress = 0
                for line in sub_process.stdout:
                    line = line.strip()
                    match = progress_pattern.search(line)
                    if match:
                        current_progress = int(match.group(1))
                        delta = current_progress - last_progress
                        pbar.update(delta)
                        last_progress = current_progress
        elif process == "ffprobe":
            # ffprobe does not expose a numerical progress protocol.  Keep an
            # indeterminate progress indicator alive while it reads large or
            # remote containers so the caller still has visible feedback.
            description = progress_description or "Probing media"
            logger.debug("%s: started.", description)
            with tqdm(desc=description, unit="file", total=None, **generic_utils.get_tqdm_defaults()) as pbar:
                while stdout is None or stderr is None:
                    try:
                        stdout, stderr = sub_process.communicate(timeout=0.1)
                        pbar.update(1)
                    except subprocess.TimeoutExpired:
                        pbar.refresh()

    if stdout is None or stderr is None:
        stdout, stderr = sub_process.communicate()

    if captured_stderr:
        stderr = "\n".join(captured_stderr) + (f"\n{stderr}" if stderr else "")

    logger.debug(f"Process finished with {sub_process.returncode}")

    return ProcessResult(sub_process.returncode, str(stdout), str(stderr))


def raise_on_error(status: ProcessResult):
    if status.returncode != 0:
        error = f"Process exited with unexpected error:\n{status.stdout}\n{status.stderr}"
        raise RuntimeError(error)


def ensure_tools_exist(tools: list[str], logger: logging.Logger) -> None:
    """Verify that all required external tools are available."""
    for tool in tools:
        path = shutil.which(tool)
        if path is None:
            raise RuntimeError(f"{tool} not found in PATH")
        logger.debug(f"{tool} path: {path}")
