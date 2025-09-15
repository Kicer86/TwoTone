"""Utility helpers to persist TMDB metadata locally."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from . import generic_utils


class TmdbCache:
    """Disk backed cache for TMDB metadata.

    The cache is organised by TMDB identifier.  Each identifier can store
    arbitrary key/value pairs so callers are free to persist multiple pieces of
    metadata for the same movie.
    """

    def __init__(
        self,
        *,
        cache_filename: str = "tmdb_cache.json",
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        cache_dir = generic_utils.get_twotone_config_dir()
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_path = os.path.join(cache_dir, cache_filename)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._cache_path, "r", encoding="utf-8") as cache_file:
                raw_data = json.load(cache_file)
        except FileNotFoundError:
            self._data = {}
            return
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.warning("Could not read TMDB cache: %s", exc)
            self._data = {}
            return

        if not isinstance(raw_data, dict):
            self._logger.warning(
                "TMDB cache file %s had an unexpected format. Starting fresh.",
                self._cache_path,
            )
            self._data = {}
            return

        converted: Dict[str, Dict[str, Any]] = {}
        for tmdb_id, entry in raw_data.items():
            if isinstance(entry, dict):
                converted[str(tmdb_id)] = entry
            else:
                self._logger.warning(
                    "TMDB cache entry for ID %s had an unexpected format. Skipping.",
                    tmdb_id,
                )

        self._data = converted

    def _save(self) -> None:
        """Persist the cache to disk."""

        try:
            with open(self._cache_path, "w", encoding="utf-8") as cache_file:
                json.dump(self._data, cache_file, indent=2, sort_keys=True)
        except OSError as exc:
            self._logger.warning("Could not write TMDB cache: %s", exc)

    def get(self, tmdb_id: str, key: str) -> Any:
        """Return the cached value for ``key`` stored under ``tmdb_id``."""

        return self._data.get(tmdb_id, {}).get(key)

    def get_all(self, tmdb_id: str) -> Dict[str, Any]:
        """Return all cached values for ``tmdb_id``."""

        return dict(self._data.get(tmdb_id, {}))

    def set(self, tmdb_id: str, key: str, value: Any) -> None:
        """Store ``value`` for ``key`` under ``tmdb_id``."""

        self._data.setdefault(tmdb_id, {})[key] = value
        self._save()

    def update(self, tmdb_id: str, values: Dict[str, Any]) -> None:
        """Update multiple values for ``tmdb_id`` at once."""

        if not values:
            return

        self._data.setdefault(tmdb_id, {}).update(values)
        self._save()
