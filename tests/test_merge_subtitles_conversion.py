
import os
import unittest

from twotone.tools.utils import process_utils, subtitles_utils, generic_utils
from common import WorkingDirectoryForTest, list_files, add_test_media, generate_microdvd_subtitles, run_twotone


class SubtitlesConversion(unittest.TestCase):

    def test_microdvd_subtitles_with_nondefault_fps(self):

        with WorkingDirectoryForTest() as td:
            add_test_media("sea-waves-crashing-on-beach-shore.*mp4", td.path)
            generate_microdvd_subtitles(os.path.join(td.path, "sea-waves.txt"), 25)

            run_twotone("merge", [td.path, "-l", "auto"], ["--no-dry-run"])

            files_after = list_files(td.path)
            self.assertEqual(len(files_after), 1)

            video = files_after[0]

            subtitles_path = os.path.join(td.path, "subtitles.srt")
            process_utils.start_process("ffmpeg", ["-i", video, "-map", "0:s:0", subtitles_path])

            lines = 0
            with open(subtitles_path, mode='r') as subtitles_file:
                ms_time = 0
                for line in subtitles_file:
                    match = subtitles_utils.subrip_time_pattern.match(line.strip())
                    if match:
                        lines += 1
                        start_time, end_time = match.groups()
                        start_ms = generic_utils.time_to_ms(start_time)
                        end_ms = generic_utils.time_to_ms(end_time)

                        # one millisecond difference is acceptable (hence delta = 1)
                        self.assertAlmostEqual(start_ms, ms_time, delta = 1)
                        self.assertAlmostEqual(end_ms, ms_time + 500, delta = 1)
                        ms_time += 1000

            self.assertEqual(lines, 25)


    def test_microdvd_subtitles_with_default_fps(self):
        with WorkingDirectoryForTest() as td:
            add_test_media("moon_23.976.mp4", td.path)
            generate_microdvd_subtitles(os.path.join(td.path, "moon_23.976.txt"), length = 1, fps = subtitles_utils.ffmpeg_default_fps)

            run_twotone("merge", [td.path, "-l", "auto"], ["--no-dry-run"])

            files_after = list_files(td.path)
            self.assertEqual(len(files_after), 1)

            video = files_after[0]

            subtitles_path = os.path.join(td.path, "subtitles.srt")
            process_utils.start_process("ffmpeg", ["-i", video, "-map", "0:s:0", subtitles_path])

            lines = 0
            with open(subtitles_path, mode='r') as subtitles_file:
                ms_time = 0
                for line in subtitles_file:
                    match = subtitles_utils.subrip_time_pattern.match(line.strip())
                    if match:
                        lines += 1
                        start_time, end_time = match.groups()
                        start_ms = generic_utils.time_to_ms(start_time)
                        end_ms = generic_utils.time_to_ms(end_time)

                        # one millisecond difference is acceptable (hence delta = 1)
                        self.assertAlmostEqual(start_ms, ms_time, delta = 1)
                        self.assertAlmostEqual(end_ms, ms_time + 500, delta = 1)
                        ms_time += 1000

            self.assertEqual(lines, 1)


if __name__ == '__main__':
    unittest.main()
