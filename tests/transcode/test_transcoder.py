
import unittest

from twotone.tools.transcode import Transcoder
from common import TwoToneTestCase, get_video, add_test_media, hashes, run_twotone


class TranscoderTests(TwoToneTestCase):
    def test_video_1_for_best_crf(self):
        test_video = get_video("big_buck_bunny_720p_2mb.mp4")
        best_enc = Transcoder(self.wd.path, self.logger).find_optimal_crf(test_video, allow_segments=False)

        self.assertEqual(best_enc, 28)

    def test_transcoding_on_short_videos_dry_run(self):
        add_test_media("VID_20240412_1815.*", self.wd.path, copy = True)
        hashes_before = hashes(self.wd.path)
        self.assertEqual(len(hashes_before), 4)

        run_twotone("transcode", [self.wd.path])

        hashes_after = hashes(self.wd.path)
        self.assertEqual(hashes_after, hashes_before)

    def test_transcoding_on_short_videos_live_run(self):
        # VID_20240412_181520.mp4 does not transcode properly with ffmpeg 7.0. With 7.1 it works.
        # Disable as of now
        add_test_media("VID_20240412_1815.[^0]\\.mp4", self.wd.path, copy = True)
        hashes_before = hashes(self.wd.path)
        self.assertEqual(len(hashes_before), 3)

        run_twotone("transcode", [self.wd.path], ["-r"])

        hashes_after = hashes(self.wd.path)
        self.assertEqual(len(hashes_after), 3)
        self.assertEqual(hashes_before.keys(), hashes_after.keys())
        self.assertTrue(all(hashes_after[k] != hashes_before[k] for k in hashes_before))


if __name__ == '__main__':
    unittest.main()
