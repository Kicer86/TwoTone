
import unittest

import utils

import sys
sys.path.append("..")

import mod.vof_algo as vof_algo


class TestVOFAlgorithms(unittest.TestCase):

    def test_adjust_same_videos(self):
        video1_frames = [11.50, 15.49, 20.01, 31.23]
        video2_frames = [11.51, 15.51, 20.00, 31.22]

        result = vof_algo.adjust_videos(video1_frames, video2_frames,
                                        video1_fps=30, video2_fps=24,
                                        video1_length=35.45, video2_length=35.44)

        expected_result = {
            "segments": [
                {
                    "#1": {
                        "begin": utils.Around(0.0),
                        "end": utils.Around(35.45)
                    },
                    "#2": {
                        "begin": utils.Around(0.0),
                        "end": utils.Around(35.44)
                    }
                }
            ]
        }

        self.assertEqual(result, expected_result)

    def test_adjust_videos_w_and_wo_intro(self):
        video1_frames = [21.50, 25.49, 30.01, 41.23]     # 10s offset due to an intro + extra frame (scene change after intro)
        video2_frames = [11.51, 15.51, 20.00, 31.22]

        result = vof_algo.adjust_videos(video1_frames, video2_frames,
                                        video1_fps=30, video2_fps=24,
                                        video1_length=45.45, video2_length=35.44)

        expected_result = {
            "segments": [
                {
                    "#1": {
                        "begin": utils.Around(0.0),
                        "end": utils.Around(45.45)
                    },
                    "#2": {
                        "begin": utils.Around(10.0),
                        "end": utils.Around(45.44)
                    }
                }
            ]
        }

        self.assertEqual(result, expected_result)

    def test_speedup_video(self):
        video1_frames = [1.0, 10.0]
        video2_frames = [0.8, 8.0]              # second video goes faster than first one

        result = vof_algo.adjust_videos(video1_frames, video2_frames,
                                        video1_fps=30, video2_fps=24,
                                        video1_length=20, video2_length=16)

        expected_result = {
            "segments": [
                {
                    "#1": {
                        "begin": utils.Around(0.0),
                        "end": utils.Around(20)
                    },
                    "#2": {
                        "begin": utils.Around(0.0),
                        "end": utils.Around(20)             # video 2 was expanded
                    }
                }
            ]
        }

        self.assertEqual(result, expected_result)


    def test_speedup_video_without_intro(self):
        video1_frames = [11.0, 20.0]
        video2_frames = [0.8, 8.0]              # second video goes faster than first one (25%) but has no intro (10 seconds long)

        result = vof_algo.adjust_videos(video1_frames, video2_frames,
                                        video1_fps=30, video2_fps=24,
                                        video1_length=30, video2_length=16)

        expected_result = {
            "segments": [
                {
                    "#1": {
                        "begin": utils.Around(0.0),
                        "end": utils.Around(30)
                    },
                    "#2": {
                        "begin": utils.Around(10.0),
                        "end": utils.Around(30)             # video 2 was expanded and moved by 10 seconds
                    }
                }
            ]
        }

        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
