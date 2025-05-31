
import unittest
import logging

import twotone.twotone as twotone
from twotone.tools.transcode import Transcoder
from common import WorkingDirectoryForTest, get_video, add_test_media, hashes, run_twotone


class TranscoderTests(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger("TranscoderTests")

    def test_video_1_for_best_crf(self):
        test_video = get_video("big_buck_bunny_720p_2mb.mp4")
        best_enc = Transcoder(self.logger).find_optimal_crf(test_video, allow_segments=False)

        self.assertEqual(best_enc, 28)

    def test_video_with_segments_and_no_segments(self):
        transcoder = Transcoder(self.logger)
        for test_video, crf in [(get_video("10189155-hd_1920_1080_25fps.mp4"), 27),
                                (get_video("big_buck_bunny_720p_10mb.mp4"), 29)]:
            best_enc_segments = transcoder.find_optimal_crf(test_video, allow_segments=True)
            best_enc_no_segments = transcoder.find_optimal_crf(test_video, allow_segments=False)

            self.assertEqual(best_enc_no_segments, crf)
            self.assertTrue(abs(best_enc_no_segments - best_enc_segments) < 2)

    def test_transcoding_on_short_videos_dry_run(self):
        with WorkingDirectoryForTest() as td:
            add_test_media("VID_20240412_1815.*", td.path, copy = True)
            hashes_before = hashes(td.path)
            self.assertEqual(len(hashes_before), 4)

            try:
                run_twotone("transcode", [td.path])
            except:
                self.assertTrue(False)

            hashes_after = hashes(td.path)
            self.assertEqual(hashes_after, hashes_before)

    def test_transcoding_on_short_videos_live_run(self):
        with WorkingDirectoryForTest() as td:

            # VID_20240412_181520.mp4 does not transcode properly with ffmpeg 7.0. With 7.1 it works.
            # Disable as of now
            add_test_media("VID_20240412_1815.[^0]\\.mp4", td.path, copy = True)
            hashes_before = hashes(td.path)
            self.assertEqual(len(hashes_before), 3)

            try:
                run_twotone("transcode", [td.path], ["-r"])
            except:
                self.assertTrue(False)

            hashes_after = hashes(td.path)
            self.assertEqual(len(hashes_after), 3)
            self.assertEqual(hashes_before.keys(), hashes_after.keys())
            self.assertTrue(all(hashes_after[k] != hashes_before[k] for k in hashes_before))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)
    unittest.main()
