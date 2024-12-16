
import unittest
import logging

from transcode import Transcoder
from common import TestDataWorkingDirectory, get_video


class TranscoderTests(unittest.TestCase):

    def test_video_1_for_best_crf(self):
        test_video = get_video("big_buck_bunny_720p_2mb.mp4")
        best_enc = Transcoder().find_optimal_crf(test_video, allow_segments=False)

        self.assertEqual(best_enc, 28)

    def test_video_with_segments_and_no_segments(self):
        encoder = Transcoder()
        for test_video, crf in [(get_video("10189155-hd_1920_1080_25fps.mp4"), 27),
                                (get_video("big_buck_bunny_720p_10mb.mp4"), 29)]:
            best_enc_segments = encoder.find_optimal_crf(test_video, allow_segments=True)
            best_enc_no_segments = encoder.find_optimal_crf(test_video, allow_segments=False)

            self.assertEqual(best_enc_no_segments, crf)
            self.assertTrue(abs(best_enc_no_segments - best_enc_segments) < 2)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)
    unittest.main()
