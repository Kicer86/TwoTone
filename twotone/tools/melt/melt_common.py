import os
import re

from typing import Dict, List

FramesInfo = Dict[int, Dict[str, str]]

DEFAULT_TOLERANCE_MS = 100


def _split_path_fix(value: str) -> List[str]:
    pattern = r'"((?:[^"\\]|\\.)*?)"'

    matches = re.findall(pattern, value)
    return [match.replace(r'\"', '"') for match in matches]


def _ensure_working_dir(working_dir: str) -> str:
    wd = os.path.join(working_dir, str(os.getpid()))
    os.makedirs(wd, exist_ok=True)
    return wd


def _is_length_mismatch(base_ms: int | None, other_ms: int | None, tolerance_ms: int) -> bool:
    if base_ms is None or other_ms is None:
        return False
    return abs(base_ms - other_ms) > tolerance_ms
