
import logging
import tempfile
import unittest

from unittest.mock import patch, MagicMock

from twotone.tools.transcode import Transcoder
from twotone.tools.utils import process_utils, video_utils


class BisectionSearchTest(unittest.TestCase):
    """Unit tests for Transcoder._bisection_search — pure logic, no ffmpeg."""

    def setUp(self):
        self.transcoder = Transcoder(tempfile.mkdtemp(), logging.getLogger("test"), target_ssim=0.98)

    def test_finds_highest_value_meeting_condition(self):
        """Bisection returns the highest CRF where SSIM >= target."""
        # Simulate: CRF 0..25 → SSIM >= 0.98, CRF 26..51 → SSIM < 0.98
        eval_func = lambda crf: 1.0 - crf * 0.001
        best_crf, best_ssim = self.transcoder._bisection_search(
            eval_func, 0, 51, lambda ssim: ssim >= 0.98,
        )
        self.assertEqual(best_crf, 20)

    def test_returns_none_when_no_value_meets_condition(self):
        """When no CRF satisfies the condition, return (None, None)."""
        eval_func = lambda crf: 0.5  # always below any reasonable target
        best_crf, best_ssim = self.transcoder._bisection_search(
            eval_func, 0, 51, lambda ssim: ssim >= 0.98,
        )
        self.assertIsNone(best_crf)
        self.assertIsNone(best_ssim)

    def test_all_values_meet_condition(self):
        """When all values meet the condition, return max_value."""
        eval_func = lambda crf: 1.0
        best_crf, _ = self.transcoder._bisection_search(
            eval_func, 0, 51, lambda ssim: ssim >= 0.98,
        )
        self.assertEqual(best_crf, 51)

    def test_only_min_value_meets_condition(self):
        """When only min_value meets the condition, return min_value."""
        eval_func = lambda crf: 0.99 if crf == 0 else 0.5
        best_crf, _ = self.transcoder._bisection_search(
            eval_func, 0, 51, lambda ssim: ssim >= 0.98,
        )
        self.assertEqual(best_crf, 0)

    def test_eval_returning_none_treated_as_failure(self):
        """When eval_func returns None, the value is skipped."""
        def eval_func(crf):
            if crf > 10:
                return None
            return 0.99
        best_crf, _ = self.transcoder._bisection_search(
            eval_func, 0, 51, lambda ssim: ssim >= 0.98,
        )
        # All CRFs > 10 return None → treated as not meeting condition
        self.assertIsNotNone(best_crf)
        self.assertLessEqual(best_crf, 10)

    def test_single_value_range(self):
        """When min == max, single evaluation decides."""
        best_crf, _ = self.transcoder._bisection_search(
            lambda crf: 0.99, 25, 25, lambda ssim: ssim >= 0.98,
        )
        self.assertEqual(best_crf, 25)


class SelectSegmentsTest(unittest.TestCase):
    """Unit tests for Transcoder._select_segments — mocks get_video_duration."""

    def setUp(self):
        self.transcoder = Transcoder(tempfile.mkdtemp(), logging.getLogger("test"))

    def _select(self, duration_ms, segment_duration=5):
        with patch.object(video_utils, 'get_video_duration', return_value=duration_ms):
            return self.transcoder._select_segments("/fake.mp4", segment_duration)

    def test_short_video_gets_3_segments(self):
        """A 90s video → 3 segments (min)."""
        segments = self._select(90_000)
        self.assertEqual(len(segments), 3)

    def test_long_video_gets_10_segments(self):
        """A 600s video → 10 segments (max)."""
        segments = self._select(600_000)
        self.assertEqual(len(segments), 10)

    def test_segments_cover_full_duration(self):
        """First segment starts near 0, last segment ends near duration."""
        segments = self._select(120_000, segment_duration=5)
        self.assertAlmostEqual(segments[0][0], 0.0, places=1)
        self.assertAlmostEqual(segments[-1][1], 120.0, delta=5.0)

    def test_segments_have_correct_length(self):
        """Each segment should be exactly segment_duration seconds long."""
        segments = self._select(300_000, segment_duration=5)
        for start, end in segments:
            self.assertAlmostEqual(end - start, 5.0, places=1)

    def test_segments_do_not_overlap(self):
        """Segments should be non-overlapping."""
        segments = self._select(300_000, segment_duration=5)
        for i in range(len(segments) - 1):
            self.assertLessEqual(segments[i][1], segments[i + 1][0])

    def test_raises_on_zero_duration(self):
        """Zero duration should raise ValueError."""
        with self.assertRaises(ValueError):
            self._select(0)

    def test_segment_longer_than_video_raises(self):
        """segment_duration > video duration should raise."""
        with self.assertRaises(ValueError):
            self._select(3_000, segment_duration=10)


class SelectScenesParsingTest(unittest.TestCase):
    """Unit tests for Transcoder._select_scenes — mocks ffmpeg output parsing."""

    def setUp(self):
        self.transcoder = Transcoder(tempfile.mkdtemp(), logging.getLogger("test"))

    def test_parses_scene_timestamps_and_merges_overlaps(self):
        """Scene timestamps are parsed from ffmpeg stderr and nearby ones merged."""
        fake_stderr = (
            "[Parsed_showinfo_1 @ 0x1234] n:   0 pts:  10000 pts_time:10.5 ...\n"
            "[Parsed_showinfo_1 @ 0x1234] n:   1 pts:  12000 pts_time:12.0 ...\n"
            "[Parsed_showinfo_1 @ 0x1234] n:   2 pts:  50000 pts_time:50.0 ...\n"
        )
        fake_result = MagicMock()
        fake_result.stderr = fake_stderr

        with patch.object(process_utils, 'start_process', return_value=fake_result):
            segments = self.transcoder._select_scenes("/fake.mp4", segment_duration=5)

        # 10.5 and 12.0 are close → their 5s windows overlap → merged
        # 50.0 is far → separate segment
        self.assertEqual(len(segments), 2)
        # First merged segment: min(10.5-2.5, 12.0-2.5)=8.0, max(10.5+2.5, 12.0+2.5)=14.5
        self.assertAlmostEqual(segments[0][0], 8.0, places=1)
        self.assertAlmostEqual(segments[0][1], 14.5, places=1)
        self.assertAlmostEqual(segments[1][0], 47.5, places=1)
        self.assertAlmostEqual(segments[1][1], 52.5, places=1)

    def test_no_scenes_returns_empty(self):
        """When no scene changes detected, return empty list."""
        fake_result = MagicMock()
        fake_result.stderr = "frame=    0 fps=0.0 q=0.0 Lsize=0kB\n"

        with patch.object(process_utils, 'start_process', return_value=fake_result):
            segments = self.transcoder._select_scenes("/fake.mp4")

        self.assertEqual(segments, [])

    def test_single_scene_returns_one_segment(self):
        """Single scene change → one segment."""
        fake_result = MagicMock()
        fake_result.stderr = "[Parsed_showinfo_1 @ 0x1] n: 0 pts: 5000 pts_time:30.0\n"

        with patch.object(process_utils, 'start_process', return_value=fake_result):
            segments = self.transcoder._select_scenes("/fake.mp4", segment_duration=5)

        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0][0], 27.5, places=1)
        self.assertAlmostEqual(segments[0][1], 32.5, places=1)

    def test_scene_near_zero_clamps_start(self):
        """Scene at 1s with 5s padding → start clamped to 0."""
        fake_result = MagicMock()
        fake_result.stderr = "[Parsed_showinfo_1 @ 0x1] n: 0 pts: 1000 pts_time:1.0\n"

        with patch.object(process_utils, 'start_process', return_value=fake_result):
            segments = self.transcoder._select_scenes("/fake.mp4", segment_duration=5)

        self.assertEqual(segments[0][0], 0.0)


if __name__ == '__main__':
    unittest.main()
