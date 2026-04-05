import logging
import os
import unittest

from twotone.tools.merge import Merge
from twotone.tools.utils.subtitles_utils import SubtitleFile
from common import TwoToneTestCase, write_srt_subtitle


class SortSubtitlesUnitTest(unittest.TestCase):
    """Unit tests for Merge._sort_subtitles()."""

    def _merger(self, lang_priority=""):
        logger = logging.getLogger("test")
        logger.setLevel(logging.CRITICAL)
        return Merge(logger, language="und", lang_priority=lang_priority, working_dir="/tmp")

    def _sub(self, lang):
        return SubtitleFile(path="/fake.srt", language=lang)

    def test_no_priority_preserves_order(self):
        merger = self._merger()
        subs = [self._sub("pol"), self._sub("eng"), self._sub("deu")]
        result = merger._sort_subtitles(subs)
        langs = [s.language for s in result]
        self.assertEqual(langs, ["pol", "eng", "deu"])

    def test_single_priority_moves_to_front(self):
        merger = self._merger("eng")
        subs = [self._sub("pol"), self._sub("eng"), self._sub("deu")]
        result = merger._sort_subtitles(subs)
        self.assertEqual(result[0].language, "eng")

    def test_multiple_priorities_respected(self):
        merger = self._merger("deu,eng")
        subs = [self._sub("pol"), self._sub("eng"), self._sub("deu")]
        result = merger._sort_subtitles(subs)
        langs = [s.language for s in result]
        self.assertEqual(langs[0], "deu")
        self.assertEqual(langs[1], "eng")

    def test_none_language_sorted_before_unknown(self):
        merger = self._merger("eng")
        subs = [self._sub("pol"), self._sub("eng"), self._sub(None)]
        result = merger._sort_subtitles(subs)
        self.assertEqual(result[0].language, "eng")
        # None is in priority list (last slot), "pol" is not — so None comes before pol
        langs = [s.language for s in result]
        self.assertLess(langs.index(None), langs.index("pol"))

    def test_unknown_languages_after_priorities(self):
        merger = self._merger("eng")
        subs = [self._sub("jpn"), self._sub("eng"), self._sub("kor")]
        result = merger._sort_subtitles(subs)
        self.assertEqual(result[0].language, "eng")


class DirectorySubtitleMatcherUnitTest(TwoToneTestCase):
    """Unit tests for Merge._directory_subtitle_matcher()."""

    def _merger(self):
        return Merge(self.logger, language="und", lang_priority="", working_dir=self.wd.path)

    def test_simple_match(self):
        # Create video stub (extension makes it a "video")
        video = os.path.join(self.wd.path, "movie.mp4")
        with open(video, "wb") as f:
            f.write(b"\x00" * 100)

        write_srt_subtitle(
            os.path.join(self.wd.path, "movie.srt"),
            [(0, 1000, "Hello")],
        )

        merger = self._merger()
        result = merger._directory_subtitle_matcher(self.wd.path)
        self.assertEqual(len(result), 1)
        self.assertIn(video, result)
        self.assertEqual(len(result[video]), 1)

    def test_multiple_subtitles_for_one_video(self):
        video = os.path.join(self.wd.path, "movie.mp4")
        with open(video, "wb") as f:
            f.write(b"\x00" * 100)

        write_srt_subtitle(os.path.join(self.wd.path, "movie_en.srt"), [(0, 1000, "Hello")])
        write_srt_subtitle(os.path.join(self.wd.path, "movie_pl.srt"), [(0, 1000, "Witaj")])

        merger = self._merger()
        result = merger._directory_subtitle_matcher(self.wd.path)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[video]), 2)

    def test_unmatched_subtitles_not_in_result(self):
        video = os.path.join(self.wd.path, "movie.mp4")
        with open(video, "wb") as f:
            f.write(b"\x00" * 100)

        write_srt_subtitle(os.path.join(self.wd.path, "other.srt"), [(0, 1000, "Text")])

        merger = self._merger()
        result = merger._directory_subtitle_matcher(self.wd.path)
        self.assertEqual(len(result), 0)

    def test_two_videos_each_with_subtitle(self):
        for name in ["alpha", "beta"]:
            video = os.path.join(self.wd.path, f"{name}.mp4")
            with open(video, "wb") as f:
                f.write(b"\x00" * 100)
            write_srt_subtitle(os.path.join(self.wd.path, f"{name}.srt"), [(0, 1000, "Sub")])

        merger = self._merger()
        result = merger._directory_subtitle_matcher(self.wd.path)
        self.assertEqual(len(result), 2)

    def test_longer_video_name_matches_first(self):
        # "movie_extended.mp4" should match "movie_extended_en.srt", not "movie.mp4"
        short_video = os.path.join(self.wd.path, "movie.mp4")
        long_video = os.path.join(self.wd.path, "movie_extended.mp4")
        for v in [short_video, long_video]:
            with open(v, "wb") as f:
                f.write(b"\x00" * 100)

        write_srt_subtitle(os.path.join(self.wd.path, "movie_extended_en.srt"), [(0, 1000, "Hello")])

        merger = self._merger()
        result = merger._directory_subtitle_matcher(self.wd.path)
        self.assertEqual(len(result), 1)
        self.assertIn(long_video, result)

    def test_suffix_appended_subtitles_match(self):
        """Subtitles renamed with a language suffix (e.g. movie_de.srt) match movie.mp4."""
        for name in ["moon_dark", "Woman - 58142"]:
            video = os.path.join(self.wd.path, f"{name}.mp4")
            with open(video, "wb") as f:
                f.write(b"\x00" * 100)
            write_srt_subtitle(os.path.join(self.wd.path, f"{name}_de.srt"), [(0, 1000, "Hallo")])

        merger = self._merger()
        result = merger._directory_subtitle_matcher(self.wd.path)
        self.assertEqual(len(result), 2)
        for video_path, subs in result.items():
            self.assertEqual(len(subs), 1)


if __name__ == "__main__":
    unittest.main()
