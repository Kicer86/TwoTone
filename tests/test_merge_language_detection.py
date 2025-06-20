
import os
import unittest

from twotone.tools.utils import video_utils
from common import WorkingDirectoryForTest, list_files, add_test_media, run_twotone


class SimpleSubtitlesMerge(unittest.TestCase):

    def test_english_recognition(self):
        with WorkingDirectoryForTest() as td:
            add_test_media("Frog.*mp4", td.path)

            with open(os.path.join(td.path, "Frog.txt"), "w") as sf:
                sf.write("00:00:00:Hello World\n")
                sf.write("00:00:06:This is some sample subtitle in english\n")

            run_twotone("merge", [td.path, "-l", "auto"], ["--no-dry-run"])

            files_after = list_files(td.path)
            self.assertEqual(len(files_after), 1)

            video = files_after[0]
            self.assertEqual(video[-4:], ".mkv")
            tracks = video_utils.get_video_data(video)
            self.assertEqual(len(tracks.video_tracks), 1)
            self.assertEqual(len(tracks.subtitles), 1)
            self.assertEqual(tracks.subtitles[0].language, "eng")

    def test_polish_recognition(self):
        with WorkingDirectoryForTest() as td:
            add_test_media("Frog.*mp4", td.path)

            with open(os.path.join(td.path, "Frog.txt"), "w") as sf:
                sf.write("00:00:00:Witaj Świecie\n")
                sf.write("00:00:06:To jest przykładowy tekst po polsku\n")

            run_twotone("merge", [td.path, "-l", "auto"], ["--no-dry-run"])

            files_after = list_files(td.path)
            self.assertEqual(len(files_after), 1)

            video = files_after[0]
            self.assertEqual(video[-4:], ".mkv")
            tracks = video_utils.get_video_data(video)
            self.assertEqual(len(tracks.video_tracks), 1)
            self.assertEqual(len(tracks.subtitles), 1)
            self.assertEqual(tracks.subtitles[0].language, "pol")

    def test_language_priority(self):
        with WorkingDirectoryForTest() as td:
            add_test_media("close-up-of-flowers.*mp4", td.path)
            with open(os.path.join(td.path, "close-up-of-flowers_en.srt"), "w") as sf:
                sf.write("00:00:00:Hello World\n")
                sf.write("00:00:06:This is some sample subtitle in english\n")

            with open(os.path.join(td.path, "close-up-of-flowers_pl.srt"), "w") as sf:
                sf.write("00:00:00:Witaj Świecie\n")
                sf.write("00:00:06:To jest przykładowy tekst po polsku\n")

            with open(os.path.join(td.path, "close-up-of-flowers_de.srt"), "w") as sf:
                sf.write("00:00:00:Hallo Welt\n")
                sf.write("00:00:06:Dies ist ein Beispiel für einen Untertitel auf Deutsch\n")

            with open(os.path.join(td.path, "close-up-of-flowers_cz.srt"), "w") as sf:
                sf.write("00:00:00:Ahoj světe\n")
                sf.write("00:00:06:Toto je ukázka titulků v češtině\n")

            with open(os.path.join(td.path, "close-up-of-flowers_fr.srt"), "w") as sf:
                sf.write("00:00:00:Bonjour le monde\n")
                sf.write("00:00:06:Ceci est un exemple de sous-titre en français\n")

            run_twotone("merge", [td.path, "-l", "auto", "-p" "de,cs"], ["--no-dry-run"])

            files_after = list_files(td.path)
            self.assertEqual(len(files_after), 1)

            video = files_after[0]
            tracks = video_utils.get_video_data(video)
            self.assertEqual(len(tracks.subtitles), 5)
            self.assertEqual(tracks.subtitles[0].language, "ger")
            self.assertEqual(tracks.subtitles[1].language, "cze")
            self.assertEqual(tracks.subtitles[0].default, 1)
            self.assertEqual(tracks.subtitles[1].default, 0)


if __name__ == '__main__':
    unittest.main()
