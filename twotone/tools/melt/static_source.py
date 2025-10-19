
from collections import defaultdict
from overrides import override
from pathlib import Path
from typing import Dict, Tuple, Union

from .duplicates_source import DuplicatesSource


class StaticSource(DuplicatesSource):
    def __init__(self, interruption):
        super().__init__(interruption)
        self._entries = defaultdict(list)
        self._metadata = defaultdict(dict)

    def add_entry(self, title: str, path: str):
        self._entries[title].append(path)

    def add_metadata(self, path: str, key: str, value: str):
        self._metadata[path][key] = value

    @override
    def collect_duplicates(self) -> Dict[str, Tuple]:
        return self._entries

    @override
    def get_metadata_for(self, path: str) -> Dict[str, Union[str, None]]:
        path_obj = Path(path)

        while path_obj != path_obj.parent:
            path_str = str(path_obj)
            if path_str in self._metadata:
                return self._metadata[path_str]
            else:
                path_obj = path_obj.parent

        return {}
