import unittest

from twotone.tools.utils.subtitles_utils import (
    MKVMERGE_UNSUPPORTED_FORMATS,
    subtitle_format_from_extension,
)


class MkvmergeFormatSupportUnitTest(unittest.TestCase):
    """Unit tests verifying which subtitle formats are marked unsupported by mkvmerge."""

    def test_srt_supported(self):
        fmt = subtitle_format_from_extension("movie.srt")
        self.assertEqual(fmt, "srt")
        self.assertNotIn(fmt, MKVMERGE_UNSUPPORTED_FORMATS)

    def test_ass_supported(self):
        fmt = subtitle_format_from_extension("movie.ass")
        self.assertEqual(fmt, "ass")
        self.assertNotIn(fmt, MKVMERGE_UNSUPPORTED_FORMATS)

    def test_ssa_supported(self):
        fmt = subtitle_format_from_extension("movie.ssa")
        self.assertEqual(fmt, "ssa")
        self.assertNotIn(fmt, MKVMERGE_UNSUPPORTED_FORMATS)

    def test_vtt_supported(self):
        fmt = subtitle_format_from_extension("movie.vtt")
        self.assertEqual(fmt, "vtt")
        self.assertNotIn(fmt, MKVMERGE_UNSUPPORTED_FORMATS)

    def test_sub_microdvd_unsupported(self):
        fmt = subtitle_format_from_extension("movie.sub")
        self.assertEqual(fmt, "microdvd")
        self.assertIn(fmt, MKVMERGE_UNSUPPORTED_FORMATS)

    def test_json_unsupported(self):
        fmt = subtitle_format_from_extension("movie.json")
        self.assertEqual(fmt, "json")
        self.assertIn(fmt, MKVMERGE_UNSUPPORTED_FORMATS)

    def test_txt_tmp_unsupported(self):
        fmt = subtitle_format_from_extension("movie.txt")
        self.assertEqual(fmt, "tmp")
        self.assertIn(fmt, MKVMERGE_UNSUPPORTED_FORMATS)

    def test_ttml_unsupported(self):
        fmt = subtitle_format_from_extension("movie.ttml")
        self.assertEqual(fmt, "ttml")
        self.assertIn(fmt, MKVMERGE_UNSUPPORTED_FORMATS)

    def test_unknown_extension_returns_none(self):
        fmt = subtitle_format_from_extension("movie.xyz")
        self.assertIsNone(fmt)


if __name__ == "__main__":
    unittest.main()
