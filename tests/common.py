
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile

from pathlib import Path
from typing import List

import twotone.twotone as twotone


current_path = os.path.dirname(os.path.abspath(__file__))


class WorkingDirectoryForTest:
    def __init__(self):
        self.directory = None

    @property
    def path(self):
        return self.directory

    def __enter__(self):
        stack_level = inspect.stack()[1]
        frame = stack_level.frame
        cname = ""
        if 'self' in frame.f_locals:
            cname = frame.f_locals['self'].__class__.__name__
        elif 'cls' in frame.f_locals:
            cname = frame.f_locals['cls'].__name__

        self.directory = os.path.join(tempfile.gettempdir(), "twotone_tests", cname, stack_level.function)
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)

        os.makedirs(self.directory, exist_ok=True)
        return self

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.directory)


def list_files(path: str) -> []:
    results = []

    for root, _, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)

            if os.path.isfile(filepath):
                results.append(filepath)

    return results


def add_test_media(filter: str, test_case_path: str, suffixes: [str] = [None], copy: bool = False) -> List[str]:
    filter_regex = re.compile(filter)
    output_files = []

    for media in ["subtitles", "subtitles_txt", "videos"]:
        for root, _, files in os.walk(os.path.join(current_path, media)):
            for file in files:
                if filter_regex.fullmatch(file):
                    for suffix in suffixes:
                        suffix = "" if suffix is None else "-" + suffix + "-"
                        file_path = Path(os.path.join(root, file))
                        dst_file_name = file_path.stem + suffix + file_path.suffix

                        src = os.path.join(current_path, media, file_path)
                        dst = os.path.join(test_case_path, dst_file_name)

                        if copy:
                            shutil.copy2(src, dst)
                        else:
                            os.symlink(src, dst)

                        output_files.append(dst)

    return output_files


def get_video(name: str) -> str:
    return os.path.join(current_path, "videos", name)


def hashes(path: str) -> [()]:
    results = []

    files = list_files(path)

    for filepath in files:
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)

            results.append((filepath, file_hash.hexdigest()))

    return results


def generate_microdvd_subtitles(path: str, length: int, fps: float = 60):
    with open(path, "w") as sf:
        b = 0
        e = 0
        for t in range(length):
            b = int(round(t * fps, 0))
            e = int(round(b + fps/2, 0))
            sf.write(f"{{{b}}}{{{e}}}{t}\n")

        # microdvd requires at least 3 entries for ffmpeg to consume it
        if length < 3:
            # add some empty entries to satisfy ffmpeg
            sf.write(f"{{{e}}}{{{e + 1}}}\n")
            sf.write(f"{{{e + 1}}}{{{e + 2}}}\n")


def run_twotone(tool: str, tool_options = [], global_options = []):
    global_options.append("--quiet")
    twotone.execute([*global_options, tool, *tool_options])
