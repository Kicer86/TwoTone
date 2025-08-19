
import unittest
import os
import re

from pathlib import Path
from typing import List

from twotone.tools.utilities import extract_scenes
from common import TwoToneTestCase, WorkingDirectoryForTest, get_video


def collect_files(directory: str):
    """Returns a list of all files with relative paths inside the given directory."""
    file_list = []

    for root, _, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), start=directory)
            file_list.append(relative_path)

    return file_list


def extract_number_from_filename(filename: str) -> float:
    match = re.search(r'frame_(\d+\.\d+)\.png$', filename)
    if match:
        return float(match.group(1))
    raise ValueError(f"Filename {filename} does not match expected pattern")

def extract_scene_number(path: str) -> int:
    match = re.search(r'scene_(\d+)', path)
    if match:
        return int(match.group(1))
    raise ValueError(f"Path {path} does not contain a scene number")

def pick_first_last_sorted(files: List[str]) -> List[str]:
    from collections import defaultdict

    grouped = defaultdict(list)

    # Group files by scene
    for path in files:
        scene = os.path.dirname(path)
        grouped[scene].append(path)

    result = []
    # Sort by scene number
    for scene in sorted(grouped.keys(), key=extract_scene_number):
        scene_files = grouped[scene]
        sorted_files = sorted(scene_files, key=extract_number_from_filename)
        result.append(sorted_files[0])
        result.append(sorted_files[-1])

    return result


class UtilitiesScenesTests(TwoToneTestCase):
    def test_video_1_for_scenes_extraction(self):
        test_video = get_video("big_buck_bunny_720p_10mb.mp4")
        extract_scenes(video_path = test_video, output_dir = self.wd.path, format = "png", scale = 10)

        files = pick_first_last_sorted(collect_files(self.wd.path))
        files = [Path(file).as_posix() for file in files]

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

        self.assertEqual(files, expected_files)


if __name__ == '__main__':
    unittest.main()
