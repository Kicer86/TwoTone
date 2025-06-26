
import logging
import os
import re
import unittest

from twotone.tools.utils import files_utils, video_utils
from common import TwoToneTestCase, assert_video_info, list_files, add_test_media, hashes, run_twotone, write_subtitle


default_video_set = [
    "Atoms - 8579.mp4",
    "Blue_Sky_and_Clouds_Timelapse_0892__Videvo.mov",
    "close-up-of-flowers-13554420.mp4",
    "DSC_8073.MP4",
    "fog-over-mountainside-13008647.mp4",
    "Frog - 113403.mp4",
    "Grass - 66810.mp4",
    "herd-of-horses-in-fog-13642605.mp4",
    "moon_23.976.mp4",
    "moon_dark.mp4",
    "moon.mp4",
    "sea-waves-crashing-on-beach-shore-4793288.mp4",
    "Woman - 58142.mp4"
]


def get_default_media_set_regex():
    media = []
    for video in default_video_set:
        video_escaped = re.escape(video)
        media.append(video_escaped)

        subtitle = files_utils.split_path(video)[1] + ".srt"
        subtitle_escaped = re.escape(subtitle)
        media.append(subtitle_escaped)

    filter = "|".join(media)
    return filter


class SubtitlesMerge(TwoToneTestCase):
    def test_dry_run_is_respected(self):
        add_test_media(get_default_media_set_regex(), self.wd.path)

        hashes_before = hashes(self.wd.path)
        self.assertEqual(len(hashes_before), 2 * 13)        # 13 videos and 13 subtitles expected
        run_twotone("merge", [self.wd.path])

        hashes_after = hashes(self.wd.path)

        self.assertEqual(hashes_before, hashes_after)

    def test_dry_run_with_conversion_is_respected(self):
        add_test_media("herd-of-horses-in-fog.*(mp4|txt)", self.wd.path)

        hashes_before = hashes(self.wd.path)
        self.assertEqual(len(hashes_before), 2)
        run_twotone("merge", [self.wd.path])

        hashes_after = hashes(self.wd.path)

        self.assertEqual(hashes_before, hashes_after)

    def test_many_videos_conversion(self):
        add_test_media(get_default_media_set_regex(), self.wd.path)

        files_before = list_files(self.wd.path)
        self.assertEqual(len(files_before), 2 * 13)         # 13 videos and 13 subtitles expected

        run_twotone("merge", [self.wd.path], ["--no-dry-run"])

        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1 * 13)          # 13 mkv videos expected

        for video in files_after:
            assert_video_info(self, video, expected_subtitles=1)

    def test_subtitles_language(self):
        # combine mp4 with srt into mkv
        add_test_media("Atoms.*(mp4|srt)", self.wd.path)

        run_twotone("merge", [self.wd.path, "-l", "pol"], ["--no-dry-run"])

        # verify results
        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]
        tracks = assert_video_info(self, video, expected_subtitles=1)
        self.assertEqual(tracks["subtitle"][0]["language"], "pol")

    def test_subtitles_with_a_bit_different_names(self):
        add_test_media("moon_dark.*|Woman.*", self.wd.path)
        os.rename(os.path.join(self.wd.path, "moon_dark.srt"), os.path.join(self.wd.path, "moon_dark_de.srt"))
        os.rename(os.path.join(self.wd.path, "Woman - 58142.srt"), os.path.join(self.wd.path, "Woman - 58142_de.srt"))

        run_twotone("merge", [self.wd.path, "-l", "deu"], ["--no-dry-run"])

        # verify results
        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 2)

        for video in files_after:
            tracks = assert_video_info(self, video, expected_subtitles=1)
            self.assertEqual(tracks["subtitle"][0]["language"], "deu")

    def test_multiple_subtitles_for_single_file(self):
        # one file in directory with many subtitles
        add_test_media("Atoms.*mp4", self.wd.path)
        add_test_media("Atoms.*srt", self.wd.path, ["PL", "EN", "DE"])

        run_twotone("merge", [self.wd.path], ["--no-dry-run"])

        # verify results: all subtitle-like files should be sucked in
        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]
        assert_video_info(self, video, expected_subtitles=3)

    def test_raw_txt_subtitles_conversion(self):
        # Allow automatic txt to srt conversion
        add_test_media("herd-of-horses-in-fog.*(mp4|txt)", self.wd.path)

        run_twotone("merge", [self.wd.path], ["--no-dry-run"])

        # verify results
        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]
        assert_video_info(self, video, expected_subtitles=1)

    def test_invalid_subtitle_extension(self):
        add_test_media("Frog.*mp4", self.wd.path)

        write_subtitle(
                os.path.join(self.wd.path, "Frog_en.srt"),
                [
                    "00:00:00:Hello World",
                    "00:00:06:This is some sample subtitle in english",
                ],
            )

        write_subtitle(
                os.path.join(self.wd.path, "Frog_pl.srt"),
                [
                    "00:00:00:Witaj Świecie",
                    "00:00:06:To jest przykładowy tekst po polsku",
                ],
                encoding="cp1250",
            )

        run_twotone("merge", [self.wd.path], ["--no-dry-run"])

        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]
        assert_video_info(self, video, expected_subtitles=2)

    def test_multilevel_structure(self):
        add_test_media("sea-waves-crashing-on-beach-shore.*mp4", self.wd.path)
        add_test_media("sea-waves-crashing-on-beach-shore.*srt", self.wd.path, ["PL", "EN"])

        subdir = os.path.join(self.wd.path, "subdir")
        os.mkdir(subdir)

        add_test_media("Grass.*mp4", subdir)
        add_test_media("Grass.*srt", subdir, ["PL", "EN"])

        run_twotone("merge", [self.wd.path], ["--no-dry-run"])

        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 2)

        for video in files_after:
            assert_video_info(self, video, expected_subtitles=2)

    def test_subtitles_in_subdirectory(self):
        add_test_media("sea-waves-crashing-on-beach-shore.*mp4", self.wd.path)
        add_test_media("sea-waves-crashing-on-beach-shore.*srt", self.wd.path, ["PL", "EN"])

        subdir = os.path.join(self.wd.path, "subdir")
        os.mkdir(subdir)

        add_test_media("sea-waves-crashing-on-beach-shore.*srt", subdir, ["DE", "CS"])

        run_twotone("merge", [self.wd.path], ["--no-dry-run"])

        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]
        assert_video_info(self, video, expected_subtitles=4)

    def test_appending_subtitles_to_mkv_with_subtitles(self):
        # combine mp4 with srt into mkv
        add_test_media("fog-over-mountainside.*(mp4|srt)", self.wd.path)

        run_twotone("merge", [self.wd.path, "-l", "de"], ["--no-dry-run"])

        # combine mkv with srt into mkv with 2 subtitles
        add_test_media("fog-over-mountainside.*srt", self.wd.path)

        run_twotone("merge", [self.wd.path, "-l", "pl"], ["--no-dry-run"])

        # verify results
        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]
        tracks = assert_video_info(self, video, expected_subtitles=2)
        self.assertEqual(tracks["subtitle"][0]["language"], "deu")
        self.assertEqual(tracks["subtitle"][1]["language"], "pol")

    def test_two_videos_one_subtitle(self):
        # create mkv file
        add_test_media("Woman.*(mp4|srt)", self.wd.path)
        run_twotone("merge", [self.wd.path], ["--no-dry-run"])

        # copy original file one again
        add_test_media("Woman.*(mp4|srt)", self.wd.path)

        # now there are two movies with the same name but different extension and one subtitle.
        # twotone should panic as this is not supported
        files_before = list_files(self.wd.path)
        run_twotone("merge", [self.wd.path], ["--no-dry-run"])

        # verify results
        files_after = list_files(self.wd.path)
        self.assertEqual(files_after, files_before)

    def test_collects_subtitles_from_subdirs(self):
        add_test_media("Grass.*mp4", self.wd.path)
        add_test_media("Grass.*srt", self.wd.path, ["EN"])

        subdir = os.path.join(self.wd.path, "sub")
        os.mkdir(subdir)
        add_test_media("Grass.*srt", subdir, ["DE"])

        subsubdir = os.path.join(subdir, "sub")
        os.mkdir(subsubdir)
        add_test_media("Grass.*srt", subsubdir, ["NE"])

        run_twotone("merge", [self.wd.path], ["--no-dry-run"])

        # verify results
        files_after = list_files(self.wd.path)
        self.assertEqual(len(files_after), 1)

        video = files_after[0]
        assert_video_info(self, video, expected_subtitles=3)


if __name__ == '__main__':
    unittest.main()
