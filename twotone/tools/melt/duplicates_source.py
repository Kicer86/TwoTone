

from ..utils import generic_utils


class DuplicatesSource:
    def __init__(self, interruption: generic_utils.InterruptibleProcess) -> None:
        self.interruption = interruption

    def collect_duplicates(self) -> dict[str, tuple]:
        return {}

    def get_metadata_for(self, path: str) -> dict[str, str | None]:
        return {}
