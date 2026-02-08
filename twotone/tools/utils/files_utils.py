
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
    def __init__(self, content: str, extension: str | None = None, directory: str | None = None):
        self.content = content
        self.extension = extension
        self.filepath = None
        self.directory = directory

    def __enter__(self):
        suffix = f".{self.extension}" if self.extension else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", dir=self.directory) as temp_file:
            self.filepath = temp_file.name
            temp_file.write(self.content)

        return self.filepath

    def __exit__(self, exc_type, exc_value, traceback):
        if self.filepath and os.path.exists(self.filepath):
            os.remove(self.filepath)


def format_path(path: str, base_path: str | None) -> str:
    """Format path as relative to base_path if possible, otherwise return absolute path."""
    if not base_path:
        return path

    try:
        base = os.path.abspath(base_path)
        target = os.path.abspath(path)
    except OSError:
        return path

    try:
        if os.path.commonpath([base, target]) != base:
            return path
    except ValueError:
        # On Windows, paths on different drives raise ValueError
        return path

    rel = os.path.relpath(target, base)
    return rel or path
