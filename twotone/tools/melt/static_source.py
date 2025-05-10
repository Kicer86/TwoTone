
from collections import defaultdict
from overrides import override
from typing import Dict, List

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
    def collect_duplicates(self) -> Dict[str, List[str]]:
        return self._entries

    @override
    def get_metadata_for(self, path: str) -> Dict[str, str]:
        return self._metadata.get(path, {})
