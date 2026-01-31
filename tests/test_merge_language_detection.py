
import os
import unittest

from common import TwoToneTestCase, assert_video_info, list_files, add_test_media, run_twotone, write_srt_subtitle


class SimpleSubtitlesMerge(TwoToneTestCase):
    def test_english_recognition(self):
        add_test_media("Frog.*mp4", self.wd.path)

        write_srt_subtitle(
            os.path.join(self.wd.path, "Frog.srt"),
            [
                (0, 6000, "Hello World"),
                (6000, 12000, "This is some sample subtitle in english"),
            ],
        )

        run_twotone("merge", [self.wd.path, "-l", "auto"], ["--no-dry-run"])

        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]
        tracks = assert_video_info(self, video, expected_subtitles=1)
        self.assertEqual(tracks["subtitle"][0]["language"], "eng")

    def test_polish_recognition(self):
        add_test_media("Frog.*mp4", self.wd.path)

        write_srt_subtitle(
            os.path.join(self.wd.path, "Frog.srt"),
            [
                (0, 6000, "Witaj Świecie"),
                (6000, 12000, "To jest przykładowy tekst po polsku"),
            ],
        )

        run_twotone("merge", [self.wd.path, "-l", "auto"], ["--no-dry-run"])

        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]
        tracks = assert_video_info(self, video, expected_subtitles=1)
        self.assertEqual(tracks["subtitle"][0]["language"], "pol")

    def test_language_priority(self):
        add_test_media("close-up-of-flowers.*mp4", self.wd.path)
        write_srt_subtitle(
            os.path.join(self.wd.path, "close-up-of-flowers_en.srt"),
            [
                (0, 6000, "Hello World"),
                (6000, 12000, "This is some sample subtitle in english"),
            ],
        )

        write_srt_subtitle(
            os.path.join(self.wd.path, "close-up-of-flowers_pl.srt"),
            [
                (0, 6000, "Witaj Świecie"),
                (6000, 12000, "To jest przykładowy tekst po polsku"),
            ],
        )

        write_srt_subtitle(
            os.path.join(self.wd.path, "close-up-of-flowers_de.srt"),
            [
                (0, 6000, "Hallo Welt"),
                (6000, 12000, "Dies ist ein Beispiel für einen Untertitel auf Deutsch"),
            ],
        )

        write_srt_subtitle(
            os.path.join(self.wd.path, "close-up-of-flowers_cz.srt"),
            [
                (0, 6000, "Ahoj světe"),
                (6000, 12000, "Toto je ukázka titulků v češtině"),
            ],
        )

        write_srt_subtitle(
            os.path.join(self.wd.path, "close-up-of-flowers_fr.srt"),
            [
                (0, 6000, "Bonjour le monde"),
                (6000, 12000, "Ceci est un exemple de sous-titre en français"),
            ],
        )

        run_twotone("merge", [self.wd.path, "-l", "auto", "-p" "de,cs"], ["--no-dry-run"])

        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]
        tracks = assert_video_info(self, video, expected_subtitles=5)
        self.assertEqual(tracks["subtitle"][0]["language"], "deu")
        self.assertEqual(tracks["subtitle"][1]["language"], "ces")
        self.assertEqual(tracks["subtitle"][0]["default"], 0)
        self.assertEqual(tracks["subtitle"][1]["default"], 0)


if __name__ == '__main__':
    unittest.main()
