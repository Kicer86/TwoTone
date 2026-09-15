import importlib.util
import unittest

from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "run_tests",
    Path(__file__).parents[1] / "scripts" / "run_tests.py",
)
assert _SPEC is not None and _SPEC.loader is not None
run_tests = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_tests)


class TestSharding(unittest.TestCase):
    def test_partitions_nested_suite_without_overlap(self):
        tests = [unittest.FunctionTestCase(lambda: None) for _ in range(5)]
        suite = unittest.TestSuite([
            tests[0],
            unittest.TestSuite([
                tests[1],
                tests[2],
            ]),
            tests[3],
            tests[4],
        ])

        first = run_tests.shard_suite(suite, shard_count=2, shard_index=0)
        second = run_tests.shard_suite(suite, shard_count=2, shard_index=1)

        self.assertEqual(list(first), [tests[0], tests[2], tests[4]])
        self.assertEqual(list(second), [tests[1], tests[3]])

    def test_rejects_invalid_shard_selection(self):
        suite = unittest.TestSuite()

        for shard_count, shard_index in ((0, 0), (2, -1), (2, 2)):
            with self.subTest(shard_count=shard_count, shard_index=shard_index):
                with self.assertRaises(ValueError):
                    run_tests.shard_suite(suite, shard_count, shard_index)


if __name__ == "__main__":
    unittest.main()
