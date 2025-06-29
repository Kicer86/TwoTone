
import shutil

from pathlib import Path
from typing import Tuple
import os
import tempfile
import uuid


def split_path(path: str) -> Tuple[str, str, str]:
    info = Path(path)

    return str(info.parent), info.stem, info.suffix[1:]


class ScopedDirectory:
    def __init__(self, path: str):
        self.path = Path(path)

    def __enter__(self):
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=False)
        return self.path  # optional: return the usable Path object

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path, ignore_errors=True)


def get_unique_file_name(directory: str, extension: str) -> str:
    while True:
        file_name = f"{uuid.uuid4().hex}.{extension}"
        full_path = os.path.join(directory, file_name)

        if not os.path.exists(full_path):
            return full_path


class TempFileManager:
    def __init__(self, content: str, extension: str | None = None):
        self.content = content
        self.extension = extension
        self.filepath = None

    def __enter__(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + self.extension, mode='w') as temp_file:
            self.filepath = temp_file.name
            temp_file.write(self.content)

        return self.filepath

    def __exit__(self, exc_type, exc_value, traceback):
        if self.filepath and os.path.exists(self.filepath):
            os.remove(self.filepath)

def get_common_prefix(paths) -> str:
    unified = list(paths)
    return os.path.commonpath(unified)


def get_printable_path(path: str, common_prefix: str) -> str:
    pl = len(common_prefix)
    assert path[:pl] == common_prefix

    # skip '/' or '\' (on windows) if first char
    if path[pl] == os.sep or (os.altsep is not None and path[pl] == os.altsep):
        return path[pl + 1:]
    else:
        return path[pl:]
