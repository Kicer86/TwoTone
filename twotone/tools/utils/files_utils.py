
import itertools
import logging
import shutil
import sys
import time

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO
import os
import uuid

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def split_path(path: str) -> tuple[str, str, str]:
    info = Path(path)

    return str(info.parent), info.stem, info.suffix[1:]


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

    The root directory is created on construction when missing.  The
    instance is path-like, so it can be passed wherever a directory path
    is expected.
    """

    def __init__(self, root: str, *, keep: bool = False, _state: _WorkspaceState | None = None) -> None:
        self.root = os.fspath(root)
        self.keep = keep
        self._state = _state if _state is not None else _WorkspaceState()
        os.makedirs(self.root, exist_ok=True)

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


LOCK_FILE_NAME = ".twotone-lock"
KEEP_MARKER_NAME = "twotone-keep"

# Working directories created before locking was introduced carry no lock
# file.  They are considered abandoned only after this much inactivity, so
# that a directory of an instance which is just starting up (created, lock
# not written yet) is never swept.
_LOCKLESS_STALE_AGE_SECONDS = 60 * 60


class DirectoryLock:
    """Advisory lock marking a directory as owned by a live process.

    Held for the whole lifetime of a run.  The lock dies with the process,
    including hard kills, which makes it a reliable liveness signal for
    stale-directory collection.
    """

    def __init__(self, directory: str) -> None:
        self.path = os.path.join(directory, LOCK_FILE_NAME)
        self._handle: IO[str] | None = None

    def try_acquire(self) -> bool:
        handle = open(self.path, "a")
        try:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if sys.platform == "win32":
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _remove_stale_instance_dirs(base_dir: str, own_dir: str, logger: logging.Logger) -> None:
    """Remove working directories abandoned by dead twotone instances.

    A directory is abandoned when its lock can be acquired (its owner is
    gone) or, for lockless legacy directories, when it has not been touched
    for a long time.  Directories with a keep marker are never removed.
    """
    try:
        entries = os.listdir(base_dir)
    except OSError:
        return

    for entry in entries:
        path = os.path.join(base_dir, entry)
        if not os.path.isdir(path) or os.path.abspath(path) == os.path.abspath(own_dir):
            continue
        if os.path.exists(os.path.join(path, KEEP_MARKER_NAME)):
            continue

        if os.path.exists(os.path.join(path, LOCK_FILE_NAME)):
            lock = DirectoryLock(path)
            if not lock.try_acquire():
                continue
            lock.release()
        else:
            try:
                age = time.time() - os.path.getmtime(path)
            except OSError:
                continue
            if age < _LOCKLESS_STALE_AGE_SECONDS:
                continue

        logger.info(f"Removing stale working directory: {path}")
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def open_workspace(
    requested_dir: str | None,
    default_base: str,
    *,
    keep: bool = False,
    logger: logging.Logger,
) -> Iterator[Workspace]:
    """Set up the working directory for a run and yield its Workspace.

    Two modes:

    * default (``requested_dir`` is None): a per-instance subdirectory of
      ``default_base`` is created and removed at exit; other, abandoned
      instance directories are collected at startup.
    * external (``requested_dir`` given): the directory is used exactly as
      provided - no per-instance subdirectory, no scanning, no collection.
      Only entries created by this run are removed at exit, and a second
      instance pointed at the same directory is rejected.

    With ``keep`` nothing is ever deleted from the working directory.
    """
    if requested_dir is None:
        # Never adopt an existing directory: the PID may have been recycled
        # and the directory may hold results of a previous --keep-wd run.
        instance_dir = os.path.join(default_base, str(os.getpid()))
        suffixes = itertools.count(1)
        while os.path.exists(instance_dir):
            instance_dir = os.path.join(default_base, f"{os.getpid()}-{next(suffixes)}")
        os.makedirs(instance_dir)
        lock = DirectoryLock(instance_dir)
        if not lock.try_acquire():
            raise RuntimeError(f"Cannot lock own working directory: {instance_dir}")
        try:
            if keep:
                with open(os.path.join(instance_dir, KEEP_MARKER_NAME), "w"):
                    pass
            else:
                _remove_stale_instance_dirs(default_base, instance_dir, logger)
            yield Workspace(instance_dir, keep=keep)
        finally:
            lock.release()
            if not keep:
                shutil.rmtree(instance_dir, ignore_errors=True)
    else:
        workspace = Workspace(os.fspath(requested_dir), keep=keep)
        lock = DirectoryLock(workspace.root)
        if not lock.try_acquire():
            raise RuntimeError(f"Working directory {workspace.root} is already used by another twotone instance")
        try:
            yield workspace
        finally:
            try:
                workspace.remove_created()
            finally:
                lock.release()
                if not keep:
                    try:
                        os.remove(lock.path)
                    except OSError:
                        pass


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
