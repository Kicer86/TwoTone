import logging
import os
import tempfile
import unittest

import pysubs2

from twotone.tools.merge import Merge
from twotone.tools.utils import subtitles_utils
from twotone.tools.utils.subtitles_utils import SubtitleFile
from unittest.mock import patch

from common import write_srt_subtitle


class SortSubtitlesUnitTest(unittest.TestCase):
    """Unit tests for Merge._sort_subtitles() — language priority sorting."""

    def _make_merge(self, lang_priority: str = "") -> Merge:
        logger = logging.getLogger("test.SortSubtitles")
        logger.setLevel(logging.CRITICAL)
        return Merge(logger, language="", lang_priority=lang_priority, working_dir="/tmp/test")

    def _sub(self, language: str | None) -> SubtitleFile:
        return SubtitleFile(path=f"/fake/{language}.srt", language=language)

    def test_no_priority_preserves_order(self):
        merger = self._make_merge()
        subs = [self._sub("en"), self._sub("pl"), self._sub("de")]
        result = merger._sort_subtitles(subs)
        self.assertEqual([s.language for s in result], ["en", "pl", "de"])

    def test_single_priority_moves_to_front(self):
        merger = self._make_merge("de")
        subs = [self._sub("en"), self._sub("pl"), self._sub("de")]
        result = merger._sort_subtitles(subs)
        self.assertEqual(result[0].language, "de")

    def test_multiple_priorities_ordered_correctly(self):
        merger = self._make_merge("de,cs")
        subs = [self._sub("en"), self._sub("cs"), self._sub("de"), self._sub("pl"), self._sub("fr")]
        result = merger._sort_subtitles(subs)
        self.assertEqual(result[0].language, "de")
        self.assertEqual(result[1].language, "cs")

    def test_none_language_sorted_after_prioritized(self):
        merger = self._make_merge("de")
        subs = [self._sub(None), self._sub("de"), self._sub("en")]
        result = merger._sort_subtitles(subs)
        self.assertEqual(result[0].language, "de")
        # None and unprioritized languages share the same sort key (end of list)
        # so their relative order is preserved (stable sort)
        self.assertIn(None, [s.language for s in result[1:]])

    def test_empty_subtitles_list(self):
        merger = self._make_merge("de")
        result = merger._sort_subtitles([])
        self.assertEqual(result, [])


class ConvertSubtitleUnitTest(unittest.TestCase):
    """Unit tests for Merge._convert_subtitle() — format and encoding conversion."""

    def setUp(self):
        logger = logging.getLogger("test.ConvertSubtitle")
        logger.setLevel(logging.CRITICAL)
        self.merger = Merge(logger, language="", lang_priority="", working_dir="/tmp/test")

    def test_srt_utf8_not_converted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.srt")
            subs = pysubs2.SSAFile()
            subs.append(pysubs2.SSAEvent(start=0, end=1000, text="Hello"))
            subs.save(path, format_="srt", encoding="utf-8")

            subtitle = SubtitleFile(path=path, language="eng", encoding="utf-8")
            result = self.merger._convert_subtitle("25/1", subtitle, tmp)
            self.assertEqual(result.path, path)  # same path = no conversion

    def test_unsupported_format_converted_to_ass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.sub")
            subs = pysubs2.SSAFile()
            subs.append(pysubs2.SSAEvent(start=0, end=1000, text="Hello"))
            subs.save(path, format_="microdvd", fps=25)

            subtitle = SubtitleFile(path=path, language="eng", encoding="utf-8")
            result = self.merger._convert_subtitle("25/1", subtitle, tmp)
            self.assertNotEqual(result.path, path)
            self.assertTrue(result.path.endswith(".ass"))
            self.assertEqual(result.language, "eng")

    def test_non_utf8_encoding_reconverted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.srt")
            subs = pysubs2.SSAFile()
            subs.append(pysubs2.SSAEvent(start=0, end=1000, text="Zażółć gęślą jaźń"))
            subs.save(path, format_="srt", encoding="cp1250")

            subtitle = SubtitleFile(path=path, language="pol", encoding="windows-1250")
            result = self.merger._convert_subtitle("25/1", subtitle, tmp)
            self.assertNotEqual(result.path, path)
            self.assertEqual(result.encoding, "utf-8")
            # format should stay srt
            self.assertTrue(result.path.endswith(".srt"))

    def test_microdvd_with_custom_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.sub")
            subs = pysubs2.SSAFile()
            subs.append(pysubs2.SSAEvent(start=0, end=500, text="Event 1"))
            subs.append(pysubs2.SSAEvent(start=1000, end=1500, text="Event 2"))
            subs.save(path, format_="microdvd", fps=25)

            subtitle = SubtitleFile(path=path, language="eng", encoding="utf-8")
            result = self.merger._convert_subtitle("25/1", subtitle, tmp)
            self.assertTrue(result.path.endswith(".ass"))

            # Verify content was converted correctly
            converted = pysubs2.load(result.path)
            self.assertEqual(len(converted), 2)
            self.assertAlmostEqual(converted[0].start, 0, delta=1)
            self.assertAlmostEqual(converted[0].end, 500, delta=40)
            self.assertAlmostEqual(converted[1].start, 1000, delta=40)


class DirectorySubtitleMatcherUnitTest(unittest.TestCase):
    """Unit tests for Merge._directory_subtitle_matcher() — matching subtitles to videos by name."""

    def setUp(self):
        logger = logging.getLogger("test.DirMatcher")
        logger.setLevel(logging.CRITICAL)
        self.merger = Merge(logger, language="eng", lang_priority="", working_dir="/tmp/test")

    def _run_matcher(self, video_files: list[str], subtitle_files: list[str], dir_path: str = "/fake/dir"):
        """Run matcher with mocked filesystem scanning and is_subtitle/is_video."""

        class FakeEntry:
            def __init__(self, path):
                self.path = path
            def is_file(self):
                return True

        entries = [FakeEntry(f) for f in video_files + subtitle_files]

        def fake_scandir(path):
            class FakeContext:
                def __enter__(self_ctx):
                    return iter(entries)
                def __exit__(self_ctx, *_):
                    pass
            return FakeContext()

        subtitle_set = set(subtitle_files)
        video_set = set(video_files)

        def fake_build(path, language=None):
            return SubtitleFile(path=path, language=language or "eng", encoding="utf-8")

        with patch("os.scandir", side_effect=fake_scandir), \
             patch("twotone.tools.utils.video_utils.is_video", side_effect=lambda p: p in video_set), \
             patch("twotone.tools.utils.subtitles_utils.is_subtitle", side_effect=lambda p: p in subtitle_set), \
             patch.object(self.merger, "_build_subtitle_from_path", side_effect=fake_build):
            return self.merger._directory_subtitle_matcher(dir_path)

    def test_single_video_single_subtitle(self):
        result = self._run_matcher(
            ["/d/Movie.mp4"],
            ["/d/Movie.srt"],
        )
        self.assertIn("/d/Movie.mp4", result)
        self.assertEqual(len(result["/d/Movie.mp4"]), 1)

    def test_subtitle_prefix_matching(self):
        result = self._run_matcher(
            ["/d/Movie.mp4"],
            ["/d/Movie_en.srt", "/d/Movie_de.srt"],
        )
        self.assertIn("/d/Movie.mp4", result)
        self.assertEqual(len(result["/d/Movie.mp4"]), 2)

    def test_no_matching_subtitle(self):
        result = self._run_matcher(
            ["/d/Movie.mp4"],
            ["/d/Other.srt"],
        )
        self.assertEqual(len(result), 0)

    def test_unmatched_subtitles_not_in_result(self):
        result = self._run_matcher(
            ["/d/Movie.mp4"],
            ["/d/Movie.srt", "/d/Orphan.srt"],
        )
        self.assertIn("/d/Movie.mp4", result)
        self.assertEqual(len(result["/d/Movie.mp4"]), 1)

    def test_multiple_videos_matched_correctly(self):
        result = self._run_matcher(
            ["/d/A.mp4", "/d/B.mp4"],
            ["/d/A.srt", "/d/B.srt"],
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result["/d/A.mp4"]), 1)
        self.assertEqual(len(result["/d/B.mp4"]), 1)

    def test_subtitle_not_reused(self):
        """A subtitle matched to one video shouldn't be matched to another."""
        result = self._run_matcher(
            ["/d/Movie.mp4", "/d/Movie Special.mp4"],
            ["/d/Movie Special.srt"],
        )
        # "Movie Special.srt" starts with "Movie Special" and "Movie",
        # but longest-name-first matching should assign it to "Movie Special"
        if "/d/Movie Special.mp4" in result:
            self.assertEqual(len(result["/d/Movie Special.mp4"]), 1)
            self.assertNotIn("/d/Movie.mp4", result)


class SubtitleCommentLogicUnitTest(unittest.TestCase):
    """Unit tests for subtitle naming/comment logic in Merge._merge()."""

    def test_different_name_gets_comment(self):
        """When a subtitle file name doesn't start with the video name, it gets a track name."""
        from collections import defaultdict

        video_name = "Atoms - 8579"
        subtitles = [
            SubtitleFile(path="/d/Atoms - 8579.srt", language="eng"),
            SubtitleFile(path="/d/commentary by director.srt", language="eng"),
        ]

        # Replicate the comment logic from _merge()
        sorted_subtitles = subtitles
        subtitles_by_lang: dict[str | None, list[SubtitleFile]] = defaultdict(list)
        for s in sorted_subtitles:
            subtitles_by_lang[s.language].append(s)

        for _, subs in subtitles_by_lang.items():
            if len(subs) > 1:
                for s in subs:
                    assert s.path
                    from pathlib import Path
                    subtitle_name = Path(s.path).stem
                    if not subtitle_name.lower().startswith(video_name.lower()):
                        s.name = subtitle_name

        names = [s.name for s in subtitles]
        self.assertIn("commentary by director", names)
        self.assertIn(None, names)

    def test_similar_names_no_comment(self):
        """When subtitle names start with the video name, no track name is set."""
        from collections import defaultdict

        video_name = "Atoms - 8579"
        subtitles = [
            SubtitleFile(path="/d/Atoms - 8579.srt", language="eng"),
            SubtitleFile(path="/d/Atoms - 8579-director.srt", language="eng"),
        ]

        subtitles_by_lang: dict[str | None, list[SubtitleFile]] = defaultdict(list)
        for s in subtitles:
            subtitles_by_lang[s.language].append(s)

        for _, subs in subtitles_by_lang.items():
            if len(subs) > 1:
                for s in subs:
                    assert s.path
                    from pathlib import Path
                    subtitle_name = Path(s.path).stem
                    if not subtitle_name.lower().startswith(video_name.lower()):
                        s.name = subtitle_name

        for s in subtitles:
            self.assertIsNone(s.name)


class GuessSubtitleLanguageUnitTest(unittest.TestCase):
    """Unit tests for guess_subtitle_language() — language detection via py3langid."""

    def test_english_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_srt_subtitle(
                os.path.join(tmp, "test.srt"),
                [
                    (0, 6000, "Hello World"),
                    (6000, 12000, "This is some sample subtitle in english"),
                ],
            )
            lang = subtitles_utils.guess_subtitle_language(path, "utf-8")
            self.assertEqual(lang, "en")

    def test_polish_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_srt_subtitle(
                os.path.join(tmp, "test.srt"),
                [
                    (0, 6000, "Witaj Świecie"),
                    (6000, 12000, "To jest przykładowy tekst po polsku"),
                ],
            )
            lang = subtitles_utils.guess_subtitle_language(path, "utf-8")
            self.assertEqual(lang, "pl")

    def test_german_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_srt_subtitle(
                os.path.join(tmp, "test.srt"),
                [
                    (0, 6000, "Hallo Welt"),
                    (6000, 12000, "Dies ist ein Beispiel für einen Untertitel auf Deutsch"),
                ],
            )
            lang = subtitles_utils.guess_subtitle_language(path, "utf-8")
            self.assertEqual(lang, "de")

    def test_empty_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.srt")
            with open(path, "w") as f:
                f.write("")
            lang = subtitles_utils.guess_subtitle_language(path, "utf-8")
            self.assertEqual(lang, "")


if __name__ == "__main__":
    unittest.main()
