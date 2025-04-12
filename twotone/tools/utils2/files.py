
import shutil
from pathlib import Path


def split_path(path: str) -> (str, str, str):
    info = Path(path)

    return str(info.parent), info.stem, info.suffix[1:]


class ScopedDirectory:
    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=False)
        return self.path  # optional: return the usable Path object

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path, ignore_errors=True)
