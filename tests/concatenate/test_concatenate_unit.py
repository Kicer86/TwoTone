import logging
import os
import unittest

from unittest.mock import patch

from twotone.tools.concatenate import Concatenate


class ConcatenateAnalyzeUnitTest(unittest.TestCase):
    """Unit tests for Concatenate.analyze() — grouping, regex, validation."""

    def setUp(self):
        self.logger = logging.getLogger("test.ConcatenateAnalyze")
        self.logger.setLevel(logging.CRITICAL)
        self.concatenator = Concatenate(self.logger, working_dir="/tmp/test")

    def _analyze_with_files(self, files: list[str], ignore_warnings: bool = False):
        with patch("twotone.tools.utils.video_utils.collect_video_files", return_value=files):
            return self.concatenator.analyze("/fake/path", ignore_warnings=ignore_warnings)

    def test_basic_cd1_cd2_pair(self):
        files = ["/movies/movie cd1.avi", "/movies/movie cd2.avi"]
        result = self._analyze_with_files(files)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        output = list(result.keys())[0]
        self.assertEqual(output, "/movies/movie.avi")
        self.assertEqual(result[output], [("/movies/movie cd1.avi", 1), ("/movies/movie cd2.avi", 2)])

    def test_case_insensitive_cd(self):
        files = ["/movies/film CD1.mkv", "/movies/film CD2.mkv"]
        result = self._analyze_with_files(files)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_three_parts(self):
        files = [
            "/movies/long cd1.avi",
            "/movies/long cd2.avi",
            "/movies/long cd3.avi",
        ]
        result = self._analyze_with_files(files)

        self.assertIsNotNone(result)
        output = list(result.keys())[0]
        self.assertEqual(len(result[output]), 3)
        parts = [p for _, p in result[output]]
        self.assertEqual(parts, [1, 2, 3])

    def test_no_matching_files_returns_empty(self):
        files = ["/movies/movie.avi", "/movies/other.mkv"]
        result = self._analyze_with_files(files)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)

    def test_single_part_warns_and_returns_none(self):
        files = ["/movies/movie cd1.avi"]
        result = self._analyze_with_files(files)

        self.assertIsNone(result)

    def test_single_part_with_ignore_warnings(self):
        files = ["/movies/movie cd1.avi"]
        result = self._analyze_with_files(files, ignore_warnings=True)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)

    def test_gap_in_part_numbers_warns(self):
        files = ["/movies/movie cd1.avi", "/movies/movie cd3.avi"]
        result = self._analyze_with_files(files)

        self.assertIsNone(result)

    def test_gap_in_parts_with_ignore_warnings(self):
        files = ["/movies/movie cd1.avi", "/movies/movie cd3.avi"]
        result = self._analyze_with_files(files, ignore_warnings=True)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 0)

    def test_rmvb_converted_to_mkv(self):
        files = ["/movies/movie cd1.rmvb", "/movies/movie cd2.rmvb"]
        result = self._analyze_with_files(files)

        self.assertIsNotNone(result)
        output = list(result.keys())[0]
        self.assertTrue(output.endswith(".mkv"), f"Expected .mkv extension, got: {output}")

    def test_directory_based_path(self):
        """When path is like /dir/movie/cd1.mp4, output uses dir name as filename."""
        files = [
            os.path.join("/movies", "Title", "cd1.mp4"),
            os.path.join("/movies", "Title", "cd2.mp4"),
        ]
        result = self._analyze_with_files(files)

        self.assertIsNotNone(result)
        output = list(result.keys())[0]
        self.assertIn("Title", output)
        self.assertTrue(output.endswith(".mp4"))

    def test_hyphen_separator(self):
        files = ["/movies/movie-cd1.avi", "/movies/movie-cd2.avi"]
        result = self._analyze_with_files(files)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_multiple_groups(self):
        files = [
            "/movies/alpha cd1.avi", "/movies/alpha cd2.avi",
            "/movies/beta cd1.mkv", "/movies/beta cd2.mkv",
        ]
        result = self._analyze_with_files(files)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_parts_sorted_by_number(self):
        files = ["/movies/movie cd3.avi", "/movies/movie cd1.avi", "/movies/movie cd2.avi"]
        result = self._analyze_with_files(files)

        self.assertIsNotNone(result)
        output = list(result.keys())[0]
        parts = [p for _, p in result[output]]
        self.assertEqual(parts, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
