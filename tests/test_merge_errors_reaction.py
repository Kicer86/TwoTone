import unittest

from common import TwoToneTestCase, add_test_media, hashes, run_twotone, simulate_process_failure

class SimpleSubtitlesMerge(TwoToneTestCase):
    def test_no_changes_when_mkvmerge_exits_with_error(self):
        with simulate_process_failure("mkvmerge") as mock_start_process:
            add_test_media("Blue_Sky_and_Clouds_Timelapse.*(?:mov|srt)", self.wd.path)

            hashes_before = hashes(self.wd.path)
            self.assertEqual(len(hashes_before), 2)
            try:
                run_twotone("merge", [self.wd.path], ["--no-dry-run"])
            except RuntimeError:
                pass

            hashes_after = hashes(self.wd.path)

            self.assertEqual(hashes_before, hashes_after)
            self.assertEqual(mock_start_process.call_count, 3)


    def test_no_changes_when_ffprobe_exits_with_error(self):
        with simulate_process_failure("ffprobe") as mock_start_process:
            add_test_media("Blue_Sky_and_Clouds_Timelapse.*(?:mov|srt)", self.wd.path)

            hashes_before = hashes(self.wd.path)
            self.assertEqual(len(hashes_before), 2)
            try:
                run_twotone("merge", [self.wd.path], ["--no-dry-run"])
            except RuntimeError:
                pass

            hashes_after = hashes(self.wd.path)

            self.assertEqual(hashes_before, hashes_after)
            self.assertEqual(mock_start_process.call_count, 1)

    def test_no_changes_when_ffmpeg_exits_with_error(self):
        with simulate_process_failure("ffmpeg") as mock_start_process:
            add_test_media("Blue_Sky_and_Clouds_Timelapse.*(?:mov|srt)", self.wd.path)

            hashes_before = hashes(self.wd.path)
            self.assertEqual(len(hashes_before), 2)
            try:
                run_twotone("merge", [self.wd.path], ["--no-dry-run"])
            except RuntimeError:
                pass

            hashes_after = hashes(self.wd.path)

            self.assertEqual(hashes_before, hashes_after)
            self.assertEqual(mock_start_process.call_count, 2)


if __name__ == '__main__':
    unittest.main()
