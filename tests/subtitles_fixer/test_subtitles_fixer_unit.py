import logging
import unittest
from functools import partial

import pysubs2

from twotone.tools.subtitles_fixer import Fixer


def _make_fixer() -> Fixer:
    return Fixer(logger=logging.getLogger("test"), working_dir="/tmp")


def _make_content(events: list[tuple[int, int]]) -> pysubs2.SSAFile:
    """Build a minimal SSAFile with the given (start_ms, end_ms) events."""
    subs = pysubs2.SSAFile()
    for start, end in events:
        subs.append(pysubs2.SSAEvent(start=start, end=end))
    return subs


class GetResolverTest(unittest.TestCase):
    """Tests for Fixer._get_resolver — pure resolver selection logic."""

    def test_empty_content_drop_broken(self) -> None:
        fixer = _make_fixer()
        content = _make_content([])
        resolver = fixer._get_resolver(content, {"length": 100_000, "fps": "24/1"}, drop_broken=True)
        self.assertEqual(resolver.__func__, Fixer._drop_subtitle_resolver)

    def test_empty_content_no_drop(self) -> None:
        fixer = _make_fixer()
        content = _make_content([])
        resolver = fixer._get_resolver(content, {"length": 100_000, "fps": "24/1"}, drop_broken=False)
        self.assertEqual(resolver.__func__, Fixer._no_resolver)

    def test_long_tail_last_event_spans_boundary(self) -> None:
        """Last event starts inside video but ends after — should get long_tail_resolver."""
        fixer = _make_fixer()
        # Video length 10s (10000 ms). Last subtitle: starts at 9s, ends at 12s.
        content = _make_content([(0, 2000), (9000, 12000)])
        resolver = fixer._get_resolver(content, {"length": 10000, "fps": "24/1"}, drop_broken=False)
        self.assertEqual(resolver.__func__, Fixer._long_tail_resolver)

    def test_fps_scale_mismatch(self) -> None:
        """All events are beyond video length but FPS-rescaled they fit → fps_scale_resolver."""
        fixer = _make_fixer()
        # Video: 60000 ms at 25 fps. Default fps = 23.976.
        # Last subtitle starts AND ends beyond video length.
        # After rescale: 62000 * 23.976 / 25 ≈ 59_380 < 60000. Fits.
        content = _make_content([(61000, 62000)])
        resolver = fixer._get_resolver(content, {"length": 60000, "fps": "25/1"}, drop_broken=False)
        # fps_scale_resolver is wrapped in functools.partial
        self.assertIsInstance(resolver, partial)
        self.assertEqual(resolver.func.__func__, Fixer._fps_scale_resolver)

    def test_unrecoverable_drop_broken(self) -> None:
        """Events entirely beyond video and FPS rescale doesn't help → drop."""
        fixer = _make_fixer()
        # Video: 10000 ms at 24 fps. Subtitle at 200000 ms — way beyond any rescale.
        content = _make_content([(200000, 300000)])
        resolver = fixer._get_resolver(content, {"length": 10000, "fps": "24/1"}, drop_broken=True)
        self.assertEqual(resolver.__func__, Fixer._drop_subtitle_resolver)

    def test_unrecoverable_no_drop(self) -> None:
        fixer = _make_fixer()
        content = _make_content([(200000, 300000)])
        resolver = fixer._get_resolver(content, {"length": 10000, "fps": "24/1"}, drop_broken=False)
        self.assertEqual(resolver.__func__, Fixer._no_resolver)

    def test_subtitle_within_video_no_fix_needed(self) -> None:
        """All events within video — falls through to no_resolver / drop."""
        fixer = _make_fixer()
        content = _make_content([(0, 2000), (3000, 5000)])
        # Both events well within 10s video → no condition triggers → falls to default
        resolver = fixer._get_resolver(content, {"length": 10000, "fps": "24/1"}, drop_broken=False)
        self.assertEqual(resolver.__func__, Fixer._no_resolver)


if __name__ == "__main__":
    unittest.main()
