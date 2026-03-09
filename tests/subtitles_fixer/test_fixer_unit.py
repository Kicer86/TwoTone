import logging
import unittest
from functools import partial
from unittest.mock import MagicMock

from twotone.tools.subtitles_fixer import Fixer
from twotone.tools.utils import subtitles_utils


class FixerGetResolverUnitTest(unittest.TestCase):
    """Unit tests for Fixer._get_resolver() — resolver selection logic."""

    def setUp(self):
        self.logger = logging.getLogger("test.Fixer")
        self.logger.setLevel(logging.CRITICAL)
        self.fixer = Fixer(self.logger, working_dir="/tmp/test")

    def _make_content(self, entries: list[tuple[int, int]]):
        """Build a mock SSAFile-like list with .start/.end attributes."""
        items = []
        for start, end in entries:
            item = MagicMock()
            item.start = start
            item.end = end
            items.append(item)
        mock_file = MagicMock()
        mock_file.__len__ = lambda self: len(items)
        mock_file.__getitem__ = lambda self, idx: items[idx]
        return mock_file

    def _make_video_track(self, length: int, fps: str = "24000/1001"):
        return {"length": length, "fps": fps}

    def test_empty_content_returns_no_resolver(self):
        content = self._make_content([])
        video = self._make_video_track(100000)

        resolver = self.fixer._get_resolver(content, video, drop_broken=False)

        self.assertEqual(resolver.__func__, Fixer._no_resolver)

    def test_empty_content_drop_broken_returns_drop(self):
        content = self._make_content([])
        video = self._make_video_track(100000)

        resolver = self.fixer._get_resolver(content, video, drop_broken=True)

        self.assertEqual(resolver.__func__, Fixer._drop_subtitle_resolver)

    def test_long_tail_when_start_before_end_after_length(self):
        """Last subtitle starts before video end but ends after → long_tail."""
        content = self._make_content([(0, 5000), (90000, 120000)])
        video = self._make_video_track(100000)

        resolver = self.fixer._get_resolver(content, video, drop_broken=False)

        self.assertEqual(resolver.__func__, Fixer._long_tail_resolver)

    def test_fps_scale_when_all_beyond_but_rescaleable(self):
        """Last subtitle entirely beyond video length, but fps scaling fixes it."""
        # ffmpeg_default_fps = 23.976, video fps = 25
        # time_from=110000, time_to=120000, both > video_length=100000
        # 120000 * 23.976 / 25 = 114969.6 > 100000 → NOT fixable with fps scaling
        # Need to pick values where scaling actually brings it within video length
        # time_to * default_fps / video_fps < video_length
        # Suppose default_fps=23.976, video_fps=25, video_length=100000
        # time_to * 23.976 / 25 < 100000 → time_to < 104334
        content = self._make_content([(0, 5000), (100500, 104000)])
        video = self._make_video_track(100000, fps="25/1")

        resolver = self.fixer._get_resolver(content, video, drop_broken=False)

        self.assertIsInstance(resolver, partial)
        self.assertEqual(resolver.func.__func__, Fixer._fps_scale_resolver)

    def test_no_resolver_when_subtitle_within_bounds(self):
        """When subtitle fits within video, original analyze shouldn't hit _get_resolver,
        but if it does, the fallback is no_resolver."""
        # This tests the else branch: time_from > video_length but fps scaling doesn't help
        content = self._make_content([(0, 5000), (200000, 300000)])
        video = self._make_video_track(100000)

        resolver = self.fixer._get_resolver(content, video, drop_broken=False)

        self.assertEqual(resolver.__func__, Fixer._no_resolver)

    def test_drop_broken_fallback(self):
        """When no fix possible and drop_broken=True, drop resolver is returned."""
        content = self._make_content([(0, 5000), (200000, 300000)])
        video = self._make_video_track(100000)

        resolver = self.fixer._get_resolver(content, video, drop_broken=True)

        self.assertEqual(resolver.__func__, Fixer._drop_subtitle_resolver)


if __name__ == '__main__':
    unittest.main()
