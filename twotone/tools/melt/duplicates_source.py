
from typing import Dict, List

from ..utils import generic_utils


class DuplicatesSource:
    def __init__(self, interruption: generic_utils.InterruptibleProcess) -> None:
        self.interruption = interruption

    def collect_duplicates(self) -> Dict[str, List[str]]:
        return {}

    def get_metadata_for(self, path: str) -> Dict[str, str]:
        return {}
