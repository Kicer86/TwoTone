
import unittest
import numpy as np

import sys
sys.path.append("..")

import mod.vof_algo as vof_algo


class TestVOFAlgorithms(unittest.TestCase):

    def test_match_scenes(self):
        scene1 = np.array([1, 2, 3, 4])
        scene2 = np.array([2, 3, 4, 5])
        scene3 = np.array([3, 4, 5, 6])
        scene4 = np.array([4, 5, 6, 7])
        scene5 = np.array([5, 6, 7, 8])
        scene6 = np.array([6, 7, 8, 9])
        scene7 = np.array([0, 1, 2, 3])

        video1_scenes = { 0: {"hash": scene1}, 1: {"hash": scene3}, 2: {"hash": scene4}, 3: {"hash": scene5}, 4: {"hash": scene6} }
        video2_scenes = { 0: {"hash": scene1}, 1: {"hash": scene2}, 2: {"hash": scene3}, 3: {"hash": scene4}, 4: {"hash": scene6}, 5: {"hash": scene7} }

        matches = vof_algo.match_scenes(video1_scenes, video2_scenes, lambda l, r: np.array_equal(l, r))

        expected_matches = [(0, 0), (1, 2), (2, 3), (4, 4)]

        self.assertEqual(set(matches), set(expected_matches))


if __name__ == '__main__':
    unittest.main()
