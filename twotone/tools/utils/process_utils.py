
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import List, Dict

from . import generic_utils
from . import video_utils

DEFAULT_TOOL_OPTIONS: Dict[str, List[str]] = {
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


def start_process(process: str, args: List[str], show_progress = False) -> ProcessResult:
    defaults = DEFAULT_TOOL_OPTIONS.get(process, [])
    for opt in reversed(defaults):
        if opt not in args:
            args.insert(0, opt)

    command = [process]
    command.extend(args)

    logging.debug(f"Starting {process} with options: {' '.join(args)}")
    sub_process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1, preexec_fn=os.setsid)

    if show_progress:
        if process == "ffmpeg":
            index_of_i = args.index("-i")
            input_file = args[index_of_i + 1]

            if video_utils.is_video(input_file) and sub_process.stderr:
                progress_pattern = re.compile(r"frame= *(\d+)")
                frames = video_utils.get_video_frames_count(input_file)
                with logging_redirect_tqdm(), \
                     tqdm(desc="Processing video", unit="frame", total=frames, **generic_utils.get_tqdm_defaults()) as pbar:
                    last_frame = 0
                    for line in sub_process.stderr:
                        line = line.strip()
                        if "frame=" in line:
                            match = progress_pattern.search(line)
                            if match:
                                current_frame = int(match.group(1))
                                delta = current_frame - last_frame
                                pbar.update(delta)
                                last_frame = current_frame
        elif process == "mkvmerge" and sub_process.stdout:
            progress_pattern = re.compile(r"\w:\s*(\d+)%")
            with logging_redirect_tqdm(), \
                 tqdm(desc="Muxing", unit="%", total=100, **generic_utils.get_tqdm_defaults()) as pbar:
                last_progress = 0
                for line in sub_process.stdout:
                    line = line.strip()
                    match = progress_pattern.search(line)
                    if match:
                        current_progress = int(match.group(1))
                        delta = current_progress - last_progress
                        pbar.update(delta)
                        last_progress = current_progress

    stdout, stderr = sub_process.communicate()

    logging.debug(f"Process finished with {sub_process.returncode}")

    return ProcessResult(sub_process.returncode, str(stdout), str(stderr))


def raise_on_error(status: ProcessResult):
    if status.returncode != 0:
        raise RuntimeError(f"Process exited with unexpected error:\n{status.stdout}\n{status.stderr}")


def ensure_tools_exist(tools: List[str], logger: logging.Logger) -> None:
    """Verify that all required external tools are available."""
    for tool in tools:
        path = shutil.which(tool)
        if path is None:
            raise RuntimeError(f"{tool} not found in PATH")
        logger.debug(f"{tool} path: {path}")
