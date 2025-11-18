
import os
import unittest

import pysubs2

from twotone.tools.utils import subtitles_utils
from common import TwoToneTestCase, list_files, add_test_media, generate_microdvd_subtitles, run_twotone, extract_subtitles

class SubtitlesConversion(TwoToneTestCase):
    def test_microdvd_subtitles_with_nondefault_fps(self):
        add_test_media("sea-waves-crashing-on-beach-shore.*mp4", self.wd.path)
        generate_microdvd_subtitles(os.path.join(self.wd.path, "sea-waves.txt"), 25)

        run_twotone("merge", [self.wd.path, "-l", "auto"], ["--no-dry-run"])

        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]

        subtitles_path = os.path.join(self.wd.path, "subtitles.srt")
        extract_subtitles(video, subtitles_path)

        subs = pysubs2.load(subtitles_path)
        self.assertEqual(len(subs), 25)

        ms_time = 0
        for event in subs:
            # one millisecond difference is acceptable (hence delta = 1)
            self.assertAlmostEqual(event.start, ms_time, delta=1)
            self.assertAlmostEqual(event.end, ms_time + 500, delta=1)
            ms_time += 1000


    def test_microdvd_subtitles_with_default_fps(self):
        add_test_media("moon_23.976.mp4", self.wd.path)
        generate_microdvd_subtitles(os.path.join(self.wd.path, "moon_23.976.txt"), length = 1, fps = subtitles_utils.ffmpeg_default_fps)

        run_twotone("merge", [self.wd.path, "-l", "auto"], ["--no-dry-run"])

        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]

        subtitles_path = os.path.join(self.wd.path, "subtitles.srt")
        extract_subtitles(video, subtitles_path)

        subs = pysubs2.load(subtitles_path)
        self.assertEqual(len(subs), 1)

        ms_time = 0
        for event in subs:
            # one millisecond difference is acceptable (hence delta = 1)
            self.assertAlmostEqual(event.start, ms_time, delta=1)
            self.assertAlmostEqual(event.end, ms_time + 500, delta=1)
            ms_time += 1000


if __name__ == '__main__':
    unittest.main()
