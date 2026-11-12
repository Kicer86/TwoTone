
import os
from collections import defaultdict
from overrides import override
from pathlib import Path

from .duplicates_source import DuplicatesSource


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


class StaticSource(DuplicatesSource):
    def __init__(self, interruption):
        super().__init__(interruption)
        self._entries = defaultdict(list)
        self._metadata = defaultdict(dict)

    def add_entry(self, title: str, path: str):
        self._entries[title].append(_normalize_path(path))

    def add_metadata(self, path: str, key: str, value: str | bool):
        self._metadata[_normalize_path(path)][key] = value

    @override
    def collect_duplicates(self) -> dict[str, tuple]:
        return self._entries

    @override
    def get_metadata_for(self, path: str) -> dict[str, str | bool | None]:
        path_obj = Path(_normalize_path(path))

        while path_obj != path_obj.parent:
            path_str = str(path_obj)
            if path_str in self._metadata:
                return self._metadata[path_str]
            else:
                path_obj = path_obj.parent

        return {}
