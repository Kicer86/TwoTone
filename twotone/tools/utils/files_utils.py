
import itertools
import shutil

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import uuid


def split_path(path: str) -> tuple[str, str, str]:
    info = Path(path)

    return str(info.parent), info.stem, info.suffix[1:]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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


class _WorkspaceState:
    """State shared between a Workspace and the sub-Workspaces it spawns."""

    def __init__(self) -> None:
        self.counter = itertools.count()
        self.token = uuid.uuid4().hex[:8]
        self.created: list[str] = []


class StagingFile:
    """A temporary file placed next to its target, committed by atomic rename.

    The name is deliberately visible (no leading dot) so that leftovers of
    crashed runs do not go unnoticed.
    """

    def __init__(self, path: str, target: str) -> None:
        self.path = path
        self.target = target
        self.committed = False

    def commit(self) -> None:
        """Atomically replace the target with the staged content."""
        os.replace(self.path, self.target)
        self.committed = True

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path


class Workspace(os.PathLike):
    """Factory for temporary files and directories of a single twotone run.

    All temporary artifacts should be created through this class: it
    guarantees names unique within the run (and against leftovers of
    previous runs sharing the same directory) and honors the keep mode,
    in which nothing inside the working directory is ever deleted.

    The instance is path-like, so it can be passed wherever a directory
    path is expected.
    """

    def __init__(self, root: str, *, keep: bool = False, _state: _WorkspaceState | None = None) -> None:
        self.root = os.fspath(root)
        self.keep = keep
        self._state = _state if _state is not None else _WorkspaceState()

    def __fspath__(self) -> str:
        return self.root

    def __str__(self) -> str:
        return self.root

    def __repr__(self) -> str:
        return f"Workspace({self.root!r}, keep={self.keep})"

    def subdir(self, name: str) -> "Workspace":
        """Return a Workspace rooted at a fixed-name subdirectory, creating it if needed."""
        path = os.path.join(self.root, name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            self._state.created.append(path)
        return Workspace(path, keep=self.keep, _state=self._state)

    def unique_dir(self, label: str) -> str:
        """Create and return a directory with a run-unique, label-based name."""
        path = self._unique_path(label, None)
        os.makedirs(path)
        self._state.created.append(path)
        return path

    def unique_file(self, label: str, extension: str | None = None) -> str:
        """Reserve a run-unique file path (the file itself is not created)."""
        path = self._unique_path(label, extension)
        self._state.created.append(path)
        return path

    @contextmanager
    def scoped_dir(self, label: str) -> Iterator[str]:
        """Create a unique directory removed when the context exits (unless keep)."""
        path = self.unique_dir(label)
        try:
            yield path
        finally:
            if not self.keep:
                shutil.rmtree(path, ignore_errors=True)

    @contextmanager
    def text_file(self, content: str, extension: str | None = None) -> Iterator[str]:
        """Write content to a unique file removed when the context exits (unless keep)."""
        path = self.unique_file("text", extension)
        with open(path, "w") as text_file:
            text_file.write(content)
        try:
            yield path
        finally:
            if not self.keep and os.path.exists(path):
                os.remove(path)

    @contextmanager
    def staging_for(self, target_path: str) -> Iterator[StagingFile]:
        """Yield a StagingFile in the target's directory.

        Living next to the target makes ``commit()`` an atomic same-filesystem
        rename.  When the context exits without a commit (intermediate file,
        rejected result, error), the staged file is removed.  The keep mode
        does not apply here: staging files live among user data, not in the
        working directory.
        """
        directory, stem, extension = split_path(target_path)
        staging = StagingFile(self._staging_path(directory, stem, extension or None), target_path)
        try:
            yield staging
        finally:
            if not staging.committed and os.path.exists(staging.path):
                os.remove(staging.path)

    @contextmanager
    def staging_dir_for(self, target_dir: str) -> Iterator[str]:
        """Create a scratch directory inside the target directory, removed on exit.

        For intermediates that must share a filesystem with their final
        location (so entries can be renamed out instead of copied).
        """
        path = self._staging_path(target_dir, "twotone", None)
        os.makedirs(path)
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def remove_created(self) -> None:
        """Remove everything this run created under the workspace (best effort).

        Entries created by other runs or placed in the directory by the user
        are left untouched.  No-op in keep mode.
        """
        if self.keep:
            return
        for path in reversed(self._state.created):
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        self._state.created.clear()

    def _unique_path(self, label: str, extension: str | None) -> str:
        suffix = f".{extension}" if extension else ""
        while True:
            name = f"{label}-{next(self._state.counter)}{suffix}"
            path = os.path.join(self.root, name)
            if not os.path.exists(path):
                return path

    def _staging_path(self, directory: str, stem: str, extension: str | None) -> str:
        # The token keeps names unique across twotone instances staging in
        # the same user directory, the counter within this instance.
        suffix = f".{extension}" if extension else ""
        while True:
            name = f"{stem}.twotone-tmp-{self._state.token}-{next(self._state.counter)}{suffix}"
            path = os.path.join(directory, name)
            if not os.path.exists(path):
                return path


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
