
import logging
import os

from itertools import permutations
from typing import Iterator

from twotone.tools.utils import generic_utils
from twotone.tools.melt.melt import DEFAULT_TOLERANCE_MS, MeltAnalyzer, MeltPerformer, StaticSource


def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, list):
        return sorted((normalize(item) for item in obj), key=lambda x: repr(x))
    elif isinstance(obj, tuple):
        return tuple(normalize(item) for item in obj)
    else:
        return obj


def all_key_orders(d: dict) -> Iterator[dict]:
    """
    Yield dictionaries with all possible key orderings (same keys and values).
    """
    keys = list(d.keys())
    for perm in permutations(keys):
        yield {k: d[k] for k in perm}


def analyze_duplicates_helper(
    logger: logging.Logger,
    duplicates_source: StaticSource,
    working_dir: str,
    allow_length_mismatch: bool = False,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
):
    os.makedirs(working_dir, exist_ok=True)
    duplicates_raw = duplicates_source.collect_duplicates()
    duplicates = {title: list(files) for title, files in duplicates_raw.items()}
    analyzer = MeltAnalyzer(
        logger,
        duplicates_source,
        working_dir,
        allow_length_mismatch,
        tolerance_ms,
    )
    return analyzer.analyze_duplicates(duplicates)


def process_duplicates_helper(
    logger: logging.Logger,
    interruption: generic_utils.InterruptibleProcess,
    working_dir: str,
    output_dir: str,
    plan,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
):
    performer = MeltPerformer(
        logger,
        interruption,
        working_dir,
        output_dir,
        tolerance_ms,
    )
    performer.process_duplicates(plan)


def build_path_to_id_map(input: dict) -> dict[str, int]:
    return {path: idx for idx, path in enumerate(input.keys())}
