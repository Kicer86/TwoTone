
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import List

from . import generic
from . import video

@dataclass
class ProcessResult:
    returncode: int
    stdout: str | bytes
    stderr: str | bytes


def start_process(process: str, args: List[str], show_progress = False) -> ProcessResult:
    command = [process]
    command.extend(args)

    logging.debug(f"Starting {process} with options: {' '.join(args)}")
    sub_process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1, preexec_fn=os.setsid)

    if show_progress:
        if process == "ffmpeg":
            index_of_i = args.index("-i")
            input_file = args[index_of_i + 1]

            if video.is_video(input_file):
                progress_pattern = re.compile(r"frame= *(\d+)")
                frames = video.get_video_frames_count(input_file)
                with logging_redirect_tqdm(), \
                     tqdm(desc="Processing video", unit="frame", total=frames, **generic.get_tqdm_defaults()) as pbar:
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

    stdout, stderr = sub_process.communicate()

    logging.debug(f"Process finished with {sub_process.returncode}")

    return ProcessResult(sub_process.returncode, stdout, stderr)


def raise_on_error(status: ProcessResult):
    if status.returncode != 0:
        raise RuntimeError(f"Process exited with unexpected error:\n{status.stdout}\n{status.stderr}")

