
import hashlib
import inspect
import os
import re
import shutil
import tempfile

from pathlib import Path
from platformdirs import user_cache_dir
from typing import Dict, List, Union

import twotone.twotone
import twotone.tools.utils.generic_utils


current_path = os.path.dirname(os.path.abspath(__file__))
twotone.tools.utils.generic_utils.DISABLE_PROGRESSBARS = True


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


class FileCache:
    def __init__(self, app_name: str):
        self.base_dir = Path(user_cache_dir(app_name))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_or_generate(self, name: str, version: str, extension: str, generator_fn) -> Path:
        """
        name: logical name (e.g. 'embedding-A')
        version: version string (e.g. 'v1.0')
        extension: file extension without dot (e.g. 'json', 'npy')
        generator_fn(out_path: Path) -> None
        """
        filename = f"{name}_{version}.{extension}"
        out_path = self.base_dir / filename

        if out_path.exists():
            return out_path

        # Clean up older versions with the same name and extension
        for file in self.base_dir.glob(f"{name}_*.{extension}"):
            if file != out_path:
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Warning: could not delete {file}: {e}")

        # Run generator
        generator_fn(out_path)

        if not out_path.exists():
            raise RuntimeError(f"Generator did not produce expected file: {out_path}")

        return out_path


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
                        suffix = "" if suffix is None else "-" + suffix
                        file_path = Path(os.path.join(root, file))
                        dst_file_name = file_path.stem + suffix + file_path.suffix

                        src = os.path.join(current_path, media, file_path)
                        dst = add_to_test_dir(test_case_path, src, copy, dst_file_name)

                        output_files.append(dst)

    return output_files


def add_to_test_dir(test_case_path: str, file_path: str, copy: bool = False, dst_file_name: Union[str, None] = None) -> str:
    basename = os.path.basename(file_path) if dst_file_name is None else dst_file_name
    dst = os.path.join(test_case_path, basename)

    if copy:
        shutil.copy2(file_path, dst)
    else:
        os.symlink(file_path, dst)

    return dst


def get_audio(name: str) -> str:
    return os.path.join(current_path, "audio", name)


def get_video(name: str) -> str:
    return os.path.join(current_path, "videos", name)


def hashes(path: str) -> Dict[str, str]:
    results = {}

    files = list_files(path)

    for filepath in files:
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)

            results[filepath] = file_hash.hexdigest()

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
    twotone.twotone.execute([*global_options, tool, *tool_options])
