
from typing import Dict, List

from ..utils import generic_utils


class DuplicatesSource:
    def __init__(self, interruption: generic_utils.InterruptibleProcess):
        self.interruption = interruption

    def collect_duplicates(self) -> Dict[str, List[str]]:
        pass

    def get_metadata_for(self, path: str) -> Dict[str, str]:
        pass
