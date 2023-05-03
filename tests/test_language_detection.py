
import sys
sys.path.append("..")

import os
import unittest

import twotone
from common import TestDataWorkingDirectory, file_tracks, list_files


current_path = os.path.dirname(os.path.abspath(__file__))


class SimpleSubtitlesMerge(unittest.TestCase):

    def test_english_recognition(self):
        with TestDataWorkingDirectory() as td:
            os.symlink(os.path.join(current_path, "videos", "Frog - 113403.mp4"),
                       os.path.join(td.path, "Frog.mp4"))

            with open(os.path.join(td.path, "Frog.txt"), "w") as sf:
                sf.write("00:00:00:Hello World")
                sf.write("00:00:06:This is some sample subtitle in english")

            twotone.run([td.path, "-l", "auto"])

            files_after = list_files(td.path)
            self.assertEqual(len(files_after), 1)

            video = files_after[0]
            self.assertEqual(video[-4:], ".mkv")
            tracks = file_tracks(video)
            self.assertEqual(len(tracks["video"]), 1)
            self.assertEqual(len(tracks["subtitles"]), 1)
            self.assertEqual(tracks["subtitles"][0]["properties"]["language"], "eng")

    def test_polish_recognition(self):
        with TestDataWorkingDirectory() as td:
            os.symlink(os.path.join(current_path, "videos", "Frog - 113403.mp4"),
                       os.path.join(td.path, "Frog.mp4"))

            with open(os.path.join(td.path, "Frog.txt"), "w") as sf:
                sf.write("00:00:00:Witaj Świecie")
                sf.write("00:00:06:To jest przykładowy tekst po polsku")

            twotone.run([td.path, "-l", "auto"])

            files_after = list_files(td.path)
            self.assertEqual(len(files_after), 1)

            video = files_after[0]
            self.assertEqual(video[-4:], ".mkv")
            tracks = file_tracks(video)
            self.assertEqual(len(tracks["video"]), 1)
            self.assertEqual(len(tracks["subtitles"]), 1)
            self.assertEqual(tracks["subtitles"][0]["properties"]["language"], "pol")


if __name__ == '__main__':
    unittest.main()
