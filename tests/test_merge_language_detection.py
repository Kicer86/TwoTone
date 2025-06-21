
import os
import unittest

from twotone.tools.utils import video_utils
from common import WorkingDirectoryForTest, list_files, add_test_media, run_twotone, write_subtitle


class SimpleSubtitlesMerge(unittest.TestCase):

    def test_english_recognition(self):
        with WorkingDirectoryForTest() as td:
            add_test_media("Frog.*mp4", td.path)

            write_subtitle(
                os.path.join(td.path, "Frog.txt"),
                [
                    "00:00:00:Hello World",
                    "00:00:06:This is some sample subtitle in english",
                ],
            )

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

            write_subtitle(
                os.path.join(td.path, "Frog.txt"),
                [
                    "00:00:00:Witaj Świecie",
                    "00:00:06:To jest przykładowy tekst po polsku",
                ],
            )

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
            write_subtitle(
                os.path.join(td.path, "close-up-of-flowers_en.srt"),
                [
                    "00:00:00:Hello World",
                    "00:00:06:This is some sample subtitle in english",
                ],
            )

            write_subtitle(
                os.path.join(td.path, "close-up-of-flowers_pl.srt"),
                [
                    "00:00:00:Witaj Świecie",
                    "00:00:06:To jest przykładowy tekst po polsku",
                ],
            )

            write_subtitle(
                os.path.join(td.path, "close-up-of-flowers_de.srt"),
                [
                    "00:00:00:Hallo Welt",
                    "00:00:06:Dies ist ein Beispiel für einen Untertitel auf Deutsch",
                ],
            )

            write_subtitle(
                os.path.join(td.path, "close-up-of-flowers_cz.srt"),
                [
                    "00:00:00:Ahoj světe",
                    "00:00:06:Toto je ukázka titulků v češtině",
                ],
            )

            write_subtitle(
                os.path.join(td.path, "close-up-of-flowers_fr.srt"),
                [
                    "00:00:00:Bonjour le monde",
                    "00:00:06:Ceci est un exemple de sous-titre en français",
                ],
            )

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
