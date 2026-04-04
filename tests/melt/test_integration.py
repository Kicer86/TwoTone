
import logging
import os

from twotone.tools.utils import generic_utils, video_utils
from twotone.tools.melt.melt import StaticSource
from common import (
    add_test_media,
    add_to_test_dir,
    build_test_video,
    hashes,
)
from melt.helpers import (
    MeltTestBase,
    analyze_duplicates_helper,
    process_duplicates_helper,
)


class MeltIntegrationTest(MeltTestBase):

    def test_simple_duplicate_detection(self):
        file1 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = ["v1"])[0]
        file2 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = ["v2"])[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Grass", file1)
        duplicates.add_entry("Grass", file2)

        input_file_hashes = hashes(self.wd.path)
        self.assertEqual(len(input_file_hashes), 2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # expect output to be equal to the first of files
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        # check if file was not altered
        self.assertEqual(list(output_file_hash.values())[0], input_file_hashes[file1])


    def test_dry_run_is_being_respected(self):
        file1 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = ["v1"])[0]
        file2 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = ["v2"])[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Grass", file1)
        duplicates.add_entry("Grass", file2)

        input_file_hashes = hashes(self.wd.path)
        self.assertEqual(len(input_file_hashes), 2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        # Dry run: only prepare plan, do not execute
        _ = analyze_duplicates_helper(logger, duplicates, self.wd.path)

        # expect output to be empty
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 0)

    def test_inputs_are_kept_by_default(self):
        file1 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes=["r1"])[0]
        file2 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes=["r2"])[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Grass", file1)
        duplicates.add_entry("Grass", file2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        self.assertTrue(os.path.exists(file1))
        self.assertTrue(os.path.exists(file2))
        self.assertEqual(len(hashes(output_dir)), 1)


    def test_skip_on_length_mismatch(self):
        file1 = add_test_media("DSC_8073.MP4", self.wd.path)[0]
        file2 = add_test_media("moon.mp4", self.wd.path)[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)
        duplicates.add_metadata(file1, "audio_lang", "eng")
        duplicates.add_metadata(file2, "audio_lang", "de")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 0)

    def test_allow_length_mismatch(self):
        file1 = add_to_test_dir(self.wd.path, str(self.sample_video_file))
        file2 = add_to_test_dir(self.wd.path, str(self.sample_vhs_video_file))

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)
        duplicates.add_metadata(file1, "audio_lang", "eng")
        duplicates.add_metadata(file2, "audio_lang", "de")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(
            logger,
            duplicates,
            self.wd.path,
            allow_length_mismatch=True,
        )
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash.keys())[0]
        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["audio"]), 2)

    def test_mismatch_unused_file_ignored(self):
        file1 = build_test_video(
            os.path.join(self.wd.path, "rich.mkv"),
            self.wd.path,
            None,
            duration=1,
            width=1280,
            height=720,
            audio_name=(True, "eng"),
        )
        file2 = build_test_video(
            os.path.join(self.wd.path, "unused.mkv"),
            self.wd.path,
            None,
            duration=2,
            width=640,
            height=480,
        )

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)
        duplicates.add_metadata(file1, "audio_lang", "eng")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash.keys())[0]
        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["audio"]), 1)


    def test_mismatch_unused_third_input(self):
        file1 = build_test_video(
            os.path.join(self.wd.path, "a.mkv"),
            self.wd.path,
            None,
            duration=1,
            width=1280,
            height=720,
            audio_name=(True, "eng"),
        )
        file2 = build_test_video(
            os.path.join(self.wd.path, "b.mkv"),
            self.wd.path,
            None,
            duration=1,
            width=640,
            height=480,
            audio_name=(True, "de"),
        )
        file3 = build_test_video(
            os.path.join(self.wd.path, "c.mkv"),
            self.wd.path,
            None,
            duration=2,
            width=320,
            height=240,
        )

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)
        duplicates.add_entry("Video", file3)
        duplicates.add_metadata(file1, "audio_lang", "eng")
        duplicates.add_metadata(file2, "audio_lang", "de")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash.keys())[0]
        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["audio"]), 2)
        self.assertEqual({a["language"] for a in output_file_data["tracks"]["audio"]}, {"eng", "deu"})


    def test_same_multiscene_video_duplicate_detection(self):
        file1 = add_to_test_dir(self.wd.path, self.sample_video_file)
        file2 = add_to_test_dir(self.wd.path, self.sample_vhs_video_file)

        files = [file1, file2]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("video", file1)
        duplicates.add_entry("video", file2)
        duplicates.add_metadata(file1, "audio_lang", "eng")
        duplicates.add_metadata(file2, "audio_lang", "pol")

        input_file_hashes = hashes(self.wd.path)
        self.assertEqual(len(input_file_hashes), 2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(
            logger,
            duplicates,
            self.wd.path,
            allow_length_mismatch=True,
        )
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data(output_file)
        self.assertEqual(len(output_file_data["video"]), 1)
        self.assertEqual(output_file_data["video"][0]["height"], 1080)
        self.assertEqual(output_file_data["video"][0]["width"], 1920)

        self.assertEqual(len(output_file_data["audio"]), 2)
        self.assertEqual(output_file_data["audio"][0]["language"], "eng")
        self.assertEqual(output_file_data["audio"][1]["language"], "pol")


    def test_series_duplication(self):
        series1_dir = os.path.join(self.wd.path, "series1")
        series2_dir = os.path.join(self.wd.path, "series2")

        os.makedirs(series1_dir)
        os.makedirs(series2_dir)

        for episode in range(5):
            add_test_media("Grass - 66810.mp4", series1_dir, suffixes = [f"suf-S1E{episode}"])[0]
            add_test_media("Grass - 66810.mp4", series2_dir, suffixes = [f"S1E{episode}"])[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Grass", series1_dir)
        duplicates.add_entry("Grass", series2_dir)
        duplicates.add_metadata(series1_dir, "audio_lang", "nor")
        duplicates.add_metadata(series2_dir, "audio_lang", "ger")

        input_file_hashes = hashes(self.wd.path)
        self.assertEqual(len(input_file_hashes), 10)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate output
        output_file_hash = hashes(output_dir)
        output_files = sorted(list(output_file_hash))

        for i, output_file in enumerate(output_files):
            output_file_name = os.path.basename(output_file)
            self.assertEqual(output_file_name, f"Grass - 66810-suf-S1E{i}.mkv")

            output_file_data = video_utils.get_video_data(output_file)
            self.assertEqual(len(output_file_data["video"]), 1)
            self.assertEqual(output_file_data["video"][0]["height"], 2160)
            self.assertEqual(output_file_data["video"][0]["width"], 3840)

            self.assertEqual(len(output_file_data["audio"]), 2)
            self.assertEqual(output_file_data["audio"][0]["language"], "deu")
            self.assertEqual(output_file_data["audio"][1]["language"], "nor")

    def test_languages_ordering(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        langs = ["pol", "en", "ger", "ja", "nor"]

        for i in range(5):
            file = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = [f"v{i}"])[0]
            duplicates.add_entry("Grass", file)
            duplicates.add_metadata(file, "audio_lang", langs[i])

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate alphabetical order
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash)[0]
        output_file_data = video_utils.get_video_data(output_file)
        self.assertEqual(output_file_data["audio"][0]["language"], "deu")
        self.assertEqual(output_file_data["audio"][1]["language"], "eng")
        self.assertEqual(output_file_data["audio"][2]["language"], "jpn")
        self.assertEqual(output_file_data["audio"][3]["language"], "nor")
        self.assertEqual(output_file_data["audio"][4]["language"], "pol")

    def test_unknown_language_streams_sorted_last(self):
        """Streams with unknown language (from --force-all-streams) should appear after all known-language streams."""
        video1 = build_test_video(os.path.join(self.wd.path, "o1.mkv"), self.wd.path, "sea-waves-crashing-on-beach-shore-4793288.mp4", subtitle = True)
        video2 = build_test_video(os.path.join(self.wd.path, "o2.mkv"), self.wd.path, "sea-waves-crashing-on-beach-shore-4793288.mp4", subtitle = True)

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Sea Waves", video1)
        duplicates.add_entry("Sea Waves", video2)
        duplicates.add_metadata(video1, "audio_lang", "eng")
        duplicates.add_metadata(video2, "audio_lang", "pol")
        duplicates.add_metadata(video1, "subtitle_lang", "pol")
        # video2 subtitle: unknown language, kept via force_all_streams
        duplicates.add_metadata(video2, "force_all_streams", True)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data(output_file)
        subtitles = output_file_data["subtitle"]
        self.assertEqual(len(subtitles), 2)
        # Known language (pol) should come first, unknown last
        self.assertEqual(subtitles[0]["language"], "pol")
        self.assertIsNone(subtitles[1]["language"])

    def test_default_language(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        langs = ["pol", "en", "ger", "ja", "nor"]

        for i in range(5):
            file = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = [f"v{i}"])[0]
            duplicates.add_entry("Grass", file)
            duplicates.add_metadata(file, "audio_lang", langs[i])
            duplicates.add_metadata(file, "audio_prod_lang", "ja")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate alphabetical order
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash)[0]
        output_file_data = video_utils.get_video_data(output_file)
        self.assertEqual(output_file_data["audio"][0]["language"], "deu")
        self.assertEqual(output_file_data["audio"][1]["language"], "eng")
        self.assertEqual(output_file_data["audio"][2]["language"], "jpn")
        self.assertEqual(output_file_data["audio"][2]["default"], True)
        self.assertEqual(output_file_data["audio"][3]["language"], "nor")
        self.assertEqual(output_file_data["audio"][4]["language"], "pol")


    def test_subtitle_streams(self):
        video1 = build_test_video(os.path.join(self.wd.path, "o1.mkv"), self.wd.path, "sea-waves-crashing-on-beach-shore-4793288.mp4", subtitle = True)
        video2 = build_test_video(os.path.join(self.wd.path, "o2.mkv"), self.wd.path, "sea-waves-crashing-on-beach-shore-4793288.mp4", subtitle = True)

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Sea Waves", video1)
        duplicates.add_entry("Sea Waves", video2)
        duplicates.add_metadata(video1, "audio_lang", "eng")
        duplicates.add_metadata(video2, "audio_lang", "eng")
        duplicates.add_metadata(video1, "subtitle_lang", "jpn")
        duplicates.add_metadata(video2, "subtitle_lang", "br")

        input_file_hashes = hashes(self.wd.path)
        self.assertEqual(len(input_file_hashes), 2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = logging.getLogger("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data(output_file)
        self.assertEqual(len(output_file_data["video"]), 1)
        self.assertEqual(output_file_data["video"][0]["height"], 1080)
        self.assertEqual(output_file_data["video"][0]["width"], 1920)

        self.assertEqual(len(output_file_data["subtitle"]), 2)
        languages = { output_file_data["subtitle"][0]["language"],
                      output_file_data["subtitle"][1]["language"] }
        self.assertEqual(languages, {"jpn", "bre"})


    def test_additional_attachements(self):
        video1 = build_test_video(os.path.join(self.wd.path, "o1.mkv"), self.wd.path, "fog-over-mountainside-13008647.mp4", subtitle = True, thumbnail_name = "parrot.jpeg")
        video2 = build_test_video(os.path.join(self.wd.path, "o2.mkv"), self.wd.path, "fog-over-mountainside-13008647.mp4", subtitle = True)

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Fog", video1)
        duplicates.add_entry("Fog", video2)
        duplicates.add_metadata(video1, "audio_lang", "eng")
        duplicates.add_metadata(video2, "audio_lang", "eng")
        duplicates.add_metadata(video1, "subtitle_lang", "pol")
        duplicates.add_metadata(video2, "subtitle_lang", "eng")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = logging.getLogger("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["video"]), 1)
        self.assertEqual(len(output_file_data["attachments"]), 1)


    def test_attachement_in_file_with_useless_streams(self):
        # video #1 comes with all interesting data. the only thing video #2 can offer is an attachment.
        video1 = build_test_video(os.path.join(self.wd.path, "o1.mkv"), self.wd.path, "fog-over-mountainside-13008647.mp4", subtitle = True)
        video2 = build_test_video(os.path.join(self.wd.path, "o2.mkv"), self.wd.path, "fog-over-mountainside-13008647.mp4", thumbnail_name = "parrot.jpeg")

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_metadata(video1, "subtitle_lang", "eng")
        duplicates.add_metadata(video1, "audio_lang", "eng")
        duplicates.add_metadata(video2, "audio_lang", "eng")
        duplicates.add_entry("Fog", video1)
        duplicates.add_entry("Fog", video2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = logging.getLogger("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["video"]), 1)
        self.assertEqual(len(output_file_data["attachments"]), 1)
