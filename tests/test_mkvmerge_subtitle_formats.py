import os
import unittest

from parameterized import parameterized

from common import TwoToneTestCase, add_test_media, generate_subtitles, list_files
from twotone.tools.utils import subtitles_utils, video_utils


class MkvmergeSubtitleFormats(TwoToneTestCase):
    cases = [
        ("srt_supported", "srt", True),
        ("ass_supported", "ass", True),
        ("ssa_supported", "ssa", True),
        ("vtt_supported", "vtt", True),
        ("sub_unsupported", "sub", False),
        ("json_unsupported", "json", False),
        ("txt_unsupported", "txt", False),
        ("ttml_unsupported", "ttml", False),
    ]

    @parameterized.expand(cases)
    def test_mkvmerge_subtitle_formats(self, _name: str, ext: str, should_work: bool):
        add_test_media(r"moon\.mp4", self.wd.path)
        video_path = next(path for path in list_files(self.wd.path) if path.lower().endswith(".mp4"))

        subtitle_path = os.path.join(self.wd.path, f"subtitle.{ext}")
        if ext == "sub":
            generate_subtitles(subtitle_path, length=3, unit="seconds", fps=25, interval_ms=1000, event_ms=500)
        else:
            generate_subtitles(subtitle_path, length=2000, unit="ms", interval_ms=1000, event_ms=100)

        output_path = os.path.join(self.wd.path, f"out_{ext}.mkv")
        subtitle = subtitles_utils.SubtitleFile(path=subtitle_path, language="eng")

        if should_work:
            video_utils.generate_mkv(output_path=output_path, input_video=video_path, subtitles=[subtitle])
            self.assertTrue(os.path.exists(output_path))
        else:
            with self.assertRaises(RuntimeError):
                video_utils.generate_mkv(output_path=output_path, input_video=video_path, subtitles=[subtitle])


if __name__ == "__main__":
    unittest.main()
