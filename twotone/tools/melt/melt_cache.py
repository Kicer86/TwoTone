
import hashlib
import json
import logging
import os
import shutil

from datetime import datetime, timezone


class MeltCache:
    """Persistent cache for expensive per-video PairMatcher operations.

    Caches results of scene detection (phase 1), frame probing (phase 2),
    and frame extraction (phase 3) keyed by individual video file identity
    (path + size + mtime).
    """

    _CODE_FILES = ("pair_matcher.py", "video_utils.py")

    def __init__(self, cache_dir: str, logger: logging.Logger) -> None:
        self.cache_dir = cache_dir
        self.logger = logger
        self._code_hash_value: str | None = None
        os.makedirs(cache_dir, exist_ok=True)

    # -- public API ----------------------------------------------------------

    def load_scene_changes(self, video_path: str) -> list[int] | None:
        data = self._load_json(video_path, "scene_changes.json")
        if data is None:
            return None
        return [int(v) for v in data]

    def save_scene_changes(self, video_path: str, scenes: list[int]) -> None:
        self._save_json(video_path, "scene_changes.json", scenes, indent=2)

    def load_frame_probes(self, video_path: str) -> dict[int, dict] | None:
        data = self._load_json(video_path, "frame_probes.json")
        if data is None:
            return None
        return {int(k): v for k, v in data.items()}

    def save_frame_probes(self, video_path: str, probes: dict[int, dict]) -> None:
        serializable = {str(k): v for k, v in probes.items()}
        self._save_json(video_path, "frame_probes.json", serializable, indent=2)

    def load_scene_frames(
        self,
        video_path: str,
        target_dir: str,
        probed_metadata: dict[int, dict],
    ) -> bool:
        """Restore cached scene frames into *target_dir* via symlinks.

        Updates *probed_metadata* paths to point at the symlinks.
        Returns True on cache hit.
        """
        entry = self._entry_dir(video_path)
        if entry is None:
            return False

        frames_dir = os.path.join(entry, "frames")
        meta_path = os.path.join(entry, "frame_metadata.json")
        if not os.path.isdir(frames_dir) or not os.path.isfile(meta_path):
            return False

        with open(meta_path) as f:
            frame_meta = json.load(f)

        for ts_str, filename in frame_meta.items():
            ts = int(ts_str)
            cached_frame = os.path.join(frames_dir, filename)
            if not os.path.exists(cached_frame):
                self.logger.debug("Cache frame missing: %s", cached_frame)
                return False

            link_path = os.path.join(target_dir, filename)
            os.symlink(cached_frame, link_path)

            if ts in probed_metadata:
                probed_metadata[ts]["path"] = link_path

        self.logger.info("Restored %d cached frames for %s", len(frame_meta), os.path.basename(video_path))
        return True

    def save_scene_frames(
        self,
        video_path: str,
        source_dir: str,
        probed_metadata: dict[int, dict],
    ) -> None:
        """Copy extracted frames from *source_dir* into cache.

        Builds ``frame_metadata.json`` mapping ``{timestamp_ms: filename}``
        from entries in *probed_metadata* that have paths inside *source_dir*.
        """
        entry = self._ensure_entry(video_path)
        frames_dir = os.path.join(entry, "frames")
        if os.path.isdir(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)

        frame_meta: dict[str, str] = {}
        for ts, info in probed_metadata.items():
            path = info.get("path")
            if path is None:
                continue
            if not os.path.abspath(path).startswith(os.path.abspath(source_dir)):
                continue
            filename = os.path.basename(path)
            shutil.copy2(path, os.path.join(frames_dir, filename))
            frame_meta[str(ts)] = filename

        meta_path = os.path.join(entry, "frame_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(frame_meta, f, indent=2)

        self.logger.debug("Cached %d frames for %s", len(frame_meta), os.path.basename(video_path))

    # -- internals -----------------------------------------------------------

    def _cache_key(self, video_path: str) -> str:
        real = os.path.realpath(video_path)
        stat = os.stat(real)
        raw = f"{real}:{stat.st_size}:{stat.st_mtime_ns}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _entry_dir(self, video_path: str) -> str | None:
        """Return the cache entry directory if it exists and metadata is valid."""
        key = self._cache_key(video_path)
        entry = os.path.join(self.cache_dir, key)
        meta_path = os.path.join(entry, "cache_meta.json")
        if not os.path.isfile(meta_path):
            return None

        with open(meta_path) as f:
            meta = json.load(f)

        stored_hash = meta.get("code_hash")
        if stored_hash and stored_hash != self._code_hash():
            self.logger.warning(
                "Cache entry for %s was generated with different code version. "
                "Consider clearing %s if results seem wrong.",
                os.path.basename(video_path),
                self.cache_dir,
            )

        return entry

    def _ensure_entry(self, video_path: str) -> str:
        """Return (and create) the cache entry directory, writing metadata."""
        key = self._cache_key(video_path)
        entry = os.path.join(self.cache_dir, key)
        os.makedirs(entry, exist_ok=True)

        meta_path = os.path.join(entry, "cache_meta.json")
        real = os.path.realpath(video_path)
        stat = os.stat(real)
        meta = {
            "path": real,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "code_hash": self._code_hash(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return entry

    def _load_json(self, video_path: str, filename: str) -> object:
        entry = self._entry_dir(video_path)
        if entry is None:
            return None
        path = os.path.join(entry, filename)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _save_json(self, video_path: str, filename: str, data: object, indent: int | None = None) -> None:
        entry = self._ensure_entry(video_path)
        path = os.path.join(entry, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=indent)

    def _code_hash(self) -> str:
        if self._code_hash_value is not None:
            return self._code_hash_value

        h = hashlib.sha256()
        base = os.path.dirname(__file__)
        utils_dir = os.path.join(os.path.dirname(base), "utils")
        paths = [
            os.path.join(base, "pair_matcher.py"),
            os.path.join(utils_dir, "video_utils.py"),
        ]
        for p in sorted(paths):
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    h.update(f.read())
        self._code_hash_value = h.hexdigest()[:16]
        return self._code_hash_value
