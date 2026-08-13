import logging
import unittest
from unittest.mock import patch

from twotone.tools.concatenate import Concatenate
from twotone.tools.utils import files_utils


class ConcatenateAnalyzeUnitTest(unittest.TestCase):
    """Unit tests for Concatenate.analyze() regex and validation logic."""

    def setUp(self):
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.CRITICAL)
        self.workspace = files_utils.Workspace.temporary()
        self.concatenator = Concatenate(self.logger, workspace=self.workspace)

    def _analyze(self, file_paths, **kwargs):
        with patch("twotone.tools.concatenate.video_utils.collect_video_files", return_value=file_paths):
            return self.concatenator.analyze("/fake", **kwargs)

    def test_simple_cd_pair(self):
        files = ["/dir/movie cd1.avi", "/dir/movie cd2.avi"]
        result = self._analyze(files)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        key = next(iter(result))
        self.assertEqual(len(result[key]), 2)
        self.assertEqual(result[key][0][1], 1)
        self.assertEqual(result[key][1][1], 2)

    def test_case_insensitive_matching(self):
        files = ["/dir/movie CD1.avi", "/dir/movie CD2.avi"]
        result = self._analyze(files)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_hyphen_separator(self):
        files = ["/dir/movie-cd1.avi", "/dir/movie-cd2.avi"]
        result = self._analyze(files)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_dot_separator(self):
        files = ["/dir/movie.cd1.avi", "/dir/movie.cd2.avi"]
        result = self._analyze(files)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_directory_based_cd_pattern(self):
        files = ["/dir/movie/cd1.mp4", "/dir/movie/cd2.mp4"]
        result = self._analyze(files)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        key = next(iter(result))
        self.assertIn("movie", key)

    def test_no_match_for_non_cd_files(self):
        files = ["/dir/movie.avi", "/dir/another.avi"]
        result = self._analyze(files)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)

    def test_single_part_returns_none(self):
        files = ["/dir/movie cd1.avi"]
        result = self._analyze(files)
        self.assertIsNone(result)

    def test_single_part_with_ignore_warnings(self):
        files = ["/dir/movie cd1.avi"]
        result = self._analyze(files, ignore_warnings=True)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)

    def test_gap_in_sequence_returns_none(self):
        files = ["/dir/movie cd1.avi", "/dir/movie cd3.avi"]
        result = self._analyze(files)
        self.assertIsNone(result)

    def test_gap_in_sequence_with_ignore_warnings(self):
        files = ["/dir/movie cd1.avi", "/dir/movie cd3.avi"]
        result = self._analyze(files, ignore_warnings=True)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)

    def test_rmvb_extension_converted_to_mkv(self):
        files = ["/dir/movie cd1.rmvb", "/dir/movie cd2.rmvb"]
        result = self._analyze(files)
        self.assertIsNotNone(result)
        key = next(iter(result))
        self.assertTrue(key.endswith(".mkv"))

    def test_parts_sorted_by_number(self):
        files = ["/dir/movie cd3.avi", "/dir/movie cd1.avi", "/dir/movie cd2.avi"]
        result = self._analyze(files)
        self.assertIsNotNone(result)
        key = next(iter(result))
        part_numbers = [part for _, part in result[key]]
        self.assertEqual(part_numbers, [1, 2, 3])

    def test_mixed_valid_and_invalid_groups(self):
        files = [
            "/dir/a cd1.avi", "/dir/a cd2.avi",  # valid pair
            "/dir/b cd1.avi",                      # single part (invalid)
        ]
        result = self._analyze(files)
        self.assertIsNone(result)

    def test_mixed_groups_with_ignore_warnings(self):
        files = [
            "/dir/a cd1.avi", "/dir/a cd2.avi",
            "/dir/b cd1.avi",
        ]
        result = self._analyze(files, ignore_warnings=True)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        key = next(iter(result))
        self.assertIn("a", key)

    def test_explicit_files_preserve_input_order(self):
        files = [self._make_file("second.mp4"), self._make_file("first.mp4")]

        result = self.concatenator.analyze_files(files, self._output_path("combined.mkv"))

        self.assertEqual(list(result.values()), [[(files[0], 1), (files[1], 2)]])

    def test_explicit_files_require_at_least_two_files(self):
        result = self.concatenator.analyze_files([self._make_file("only.mp4")], self._output_path("combined.mkv"))

        self.assertIsNone(result)

    def test_explicit_files_reject_output_that_is_an_input(self):
        input_file = self._make_file("part.mp4")
        other_file = self._make_file("other.mp4")

        result = self.concatenator.analyze_files([input_file, other_file], input_file)

        self.assertIsNone(result)

    def _make_file(self, name):
        stem, extension = name.rsplit(".", maxsplit=1)
        path = self.workspace.unique_file(stem, extension)
        with open(path, "w"):
            pass
        return path

    def _output_path(self, name):
        stem, extension = name.rsplit(".", maxsplit=1)
        return self.workspace.unique_file(stem, extension)


if __name__ == "__main__":
    unittest.main()
