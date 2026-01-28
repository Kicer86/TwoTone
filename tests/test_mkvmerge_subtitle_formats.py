import os
import unittest

import pysubs2
from parameterized import parameterized

from common import TwoToneTestCase, add_test_media, generate_microdvd_subtitles, generate_subrip_subtitles, list_files
from twotone.tools.utils import subtitles_utils, video_utils


class MkvmergeSubtitleFormats(TwoToneTestCase):
    cases = [
        ("srt_supported", "srt", True),
        ("ass_supported", "ass", True),
        ("sub_unsupported", "sub", False),
    ]

    @parameterized.expand(cases)
    def test_mkvmerge_subtitle_formats(self, _name: str, ext: str, should_work: bool):
        add_test_media(r"moon\.mp4", self.wd.path)
        video_path = next(path for path in list_files(self.wd.path) if path.lower().endswith(".mp4"))

        subtitle_path = os.path.join(self.wd.path, f"subtitle.{ext}")
        if ext == "srt":
            generate_subrip_subtitles(subtitle_path, length=2000)
        elif ext == "ass":
            subs = pysubs2.SSAFile()
            subs.append(pysubs2.SSAEvent(start=0, end=1000, text="Hello"))
            subs.save(subtitle_path, format_="ass")
        elif ext == "sub":
            generate_microdvd_subtitles(subtitle_path, length=3, fps=25)
        else:
            self.fail(f"Unhandled subtitle extension in test: {ext}")

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
