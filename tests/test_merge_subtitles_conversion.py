
import os
import unittest

import pysubs2
from parameterized import parameterized

from twotone.tools.utils import subtitles_utils
from common import TwoToneTestCase, list_files, add_test_media, generate_microdvd_subtitles, run_twotone, extract_subtitles


class SubtitlesConversion(TwoToneTestCase):
    subtitle_cases = [
        (
            "microdvd_subtitles_with_nondefault_fps",
            "sea-waves-crashing-on-beach-shore.*mp4",
            "sea-waves.txt",
            25,
            None,
            25,
        ),
        (
            "microdvd_subtitles_with_default_fps",
            "moon_23.976.mp4",
            "moon_23.976.txt",
            1,
            subtitles_utils.ffmpeg_default_fps,
            1,
        ),
    ]

    @parameterized.expand(subtitle_cases)
    def test_microdvd_subtitles_conversion(
        self,
        name: str,
        media_filter: str,
        subtitle_filename: str,
        length: int,
        fps: float | None,
        expected_events: int,
    ):
        add_test_media(media_filter, self.wd.path)

        subtitle_path = os.path.join(self.wd.path, subtitle_filename)
        if fps is None:
            generate_microdvd_subtitles(subtitle_path, length = length)
        else:
            generate_microdvd_subtitles(subtitle_path, length = length, fps = fps)

        run_twotone("merge", [self.wd.path, "-l", "auto"], ["--no-dry-run"])

        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]

        subtitles_path = os.path.join(self.wd.path, "subtitles.srt")
        extract_subtitles(video, subtitles_path)

        subs = pysubs2.load(subtitles_path)
        self.assertEqual(len(subs), expected_events)

        ms_time = 0
        for event in subs:
            # one millisecond difference is acceptable (hence delta = 1)
            self.assertAlmostEqual(event.start, ms_time, delta=1)
            self.assertAlmostEqual(event.end, ms_time + 500, delta=1)
            ms_time += 1000


if __name__ == '__main__':
    unittest.main()
