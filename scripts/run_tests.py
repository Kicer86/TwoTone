#!/usr/bin/env python3
"""Run unittest discovery and report the slowest individual tests."""

from __future__ import annotations

import argparse
import sys
import time
import unittest

from pathlib import Path


def _iter_tests(suite: unittest.TestSuite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _iter_tests(test)
        else:
            yield test


def shard_suite(
    suite: unittest.TestSuite,
    shard_count: int,
    shard_index: int,
) -> unittest.TestSuite:
    if shard_count < 1:
        raise ValueError("Shard count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"Shard index must be between 0 and {shard_count - 1}"
        )

    tests = list(_iter_tests(suite))
    return unittest.TestSuite(tests[shard_index::shard_count])


class TimedTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timings: list[tuple[float, str]] = []
        self._started_at: float | None = None

    def startTest(self, test) -> None:
        self._started_at = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test) -> None:
        if self._started_at is not None:
            self.timings.append((time.perf_counter() - self._started_at, test.id()))
            self._started_at = None
        super().stopTest(test)


class TimedTestRunner(unittest.TextTestRunner):
    resultclass = TimedTestResult

    def __init__(self, durations: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.slowest_count = durations

    def run(self, test) -> TimedTestResult:
        result = super().run(test)
        if self.slowest_count:
            self.stream.writeln()
            self.stream.writeln(f"Slowest {self.slowest_count} test(s):")
            for elapsed, test_id in sorted(result.timings, reverse=True)[:self.slowest_count]:
                self.stream.writeln(f"  {elapsed:8.3f}s  {test_id}")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_dir", nargs="?", default="tests")
    parser.add_argument("--durations", type=int, default=50)
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    tests_dir = Path("tests").resolve()
    sys.path.insert(0, str(tests_dir))
    suite = unittest.defaultTestLoader.discover(args.start_dir, pattern=args.pattern)
    discovered_count = suite.countTestCases()
    try:
        suite = shard_suite(suite, args.shard_count, args.shard_index)
    except ValueError as error:
        parser.error(str(error))

    print(
        f"Running shard {args.shard_index + 1}/{args.shard_count}: "
        f"{suite.countTestCases()} of {discovered_count} test(s).",
        file=sys.stderr,
        flush=True,
    )
    result = TimedTestRunner(
        durations=args.durations,
        verbosity=2 if args.verbose else 1,
    ).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
