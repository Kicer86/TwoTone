
import unittest
import logging

import twotone.twotone as twotone
from twotone.tools.transcode import Transcoder
from common import TestDataWorkingDirectory, get_video, add_test_media, hashes


class TranscoderTests(unittest.TestCase):

    def test_video_1_for_best_crf(self):
        test_video = get_video("big_buck_bunny_720p_2mb.mp4")
        best_enc = Transcoder().find_optimal_crf(test_video, allow_segments=False)

        self.assertEqual(best_enc, 28)

    def test_video_with_segments_and_no_segments(self):
        transcoder = Transcoder()
        for test_video, crf in [(get_video("10189155-hd_1920_1080_25fps.mp4"), 27),
                                (get_video("big_buck_bunny_720p_10mb.mp4"), 29)]:
            best_enc_segments = transcoder.find_optimal_crf(
                test_video, allow_segments=True)
            best_enc_no_segments = transcoder.find_optimal_crf(
                test_video, allow_segments=False)

            self.assertEqual(best_enc_no_segments, crf)
            self.assertTrue(abs(best_enc_no_segments - best_enc_segments) < 2)

    def test_transcoding_on_short_videos_dry_run(self):
        with TestDataWorkingDirectory() as td:
            add_test_media("VID_20240412_1815.*", td.path, copy = True)
            hashes_before = hashes(td.path)
            self.assertEqual(len(hashes_before), 4)

            try:
                twotone.execute(["transcode", td.path])
            except:
                self.assertTrue(False)

            hashes_after = hashes(td.path)
            self.assertEqual(hashes_after, hashes_before)

    def test_transcoding_on_short_videos_live_run(self):
        with TestDataWorkingDirectory() as td:
            add_test_media("VID_20240412_1815.*", td.path, copy = True)
            hashes_before = hashes(td.path)
            self.assertEqual(len(hashes_before), 4)

            try:
                twotone.execute(["-r", "transcode", td.path])
            except:
                self.assertTrue(False)

            hashes_after = hashes(td.path)
            self.assertNotEqual(hashes_after, hashes_before)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.ERROR)
    unittest.main()
