
from pathlib import Path


def split_path(path: str) -> (str, str, str):
    info = Path(path)

    return str(info.parent), info.stem, info.suffix[1:]
