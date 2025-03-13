
import unittest
import logging
import os

import twotone.twotone as twotone
from twotone.tools.utilities import extract_scenes
from common import WorkingDirectoryForTest, get_video


def collect_files(directory: str):
    """Returns a list of all files with relative paths inside the given directory."""
    file_list = []

    for root, _, files in os.walk(directory):
        for file in files:
            # Compute relative path
            relative_path = os.path.relpath(os.path.join(root, file), start=directory)
            file_list.append(relative_path)

    return file_list


class UtilitiesScenesTests(unittest.TestCase):

    def test_video_1_for_scenes_extraction(self):
        with WorkingDirectoryForTest() as wd:
            test_video = get_video("big_buck_bunny_720p_10mb.mp4")
            best_enc = extract_scenes(video_path = test_video, output_dir = wd.path, format = "png", scale = 10)

            files = collect_files(wd.path)

            expected_files = [
                "scene_0/frame_0.000.png",
                "scene_0/frame_8.360.png",
                "scene_1/frame_8.400.png",
                "scene_1/frame_12.920.png",
                "scene_2/frame_12.960.png",
                "scene_2/frame_16.040.png",
                "scene_3/frame_16.080.png",
                "scene_3/frame_21.680.png",
                "scene_4/frame_21.720.png",
                "scene_4/frame_23.560.png",
                "scene_5/frame_23.600.png",
                "scene_5/frame_25.720.png",
                "scene_6/frame_25.760.png",
                "scene_6/frame_27.800.png",
                "scene_7/frame_27.840.png",
                "scene_7/frame_32.200.png",
                "scene_8/frame_32.240.png",
                "scene_8/frame_42.000.png",
                "scene_9/frame_42.040.png",
                "scene_9/frame_43.680.png",
                "scene_10/frame_43.720.png",
                "scene_10/frame_52.000.png",
                "scene_11/frame_52.040.png",
                "scene_11/frame_56.160.png",
                "scene_12/frame_56.200.png",
                "scene_12/frame_62.240.png",
            ]

            for file in expected_files:
                self.assertTrue(file in files)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)
    unittest.main()
