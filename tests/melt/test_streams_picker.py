
import argparse
import contextlib
import io
import os
import platform

from parameterized import parameterized

from twotone.tools.utils import generic_utils
from twotone.tools.melt.melt import MeltTool, StaticSource, StreamsPicker
from common import TwoToneTestCase
from melt.helpers import (
    MeltTestBase,
    normalize,
    all_key_orders,
    _build_path_to_id_map,
)


class StreamsPickerTest(MeltTestBase):

    def test_streams_picker_prefers_higher_sample_rate_audio(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file1 = os.path.join(self.wd.path, "audio_48k.mkv")
        file2 = os.path.join(self.wd.path, "audio_24k.mkv")

        files_details = {
            file1: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 6, "sample_rate": 48000}],
                "subtitle": [],
            },
            file2: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 6, "sample_rate": 24000}],
                "subtitle": [],
            },
        }
        ids = {file1: 1, file2: 2}

        _, audio_streams, _ = sp.pick_streams(files_details, ids)

        self.assertEqual(audio_streams[0][0], file1)

    def test_melt_tool_parses_force_all_streams_as_per_input_flag(self):
        parser = argparse.ArgumentParser()
        MeltTool().setup_parser(parser)

        args = parser.parse_args([
            "-o", "/tmp/out",
            "-t", "Example",
            "-i", "/tmp/a.mkv",
            "--force-all-streams",
            "-i", "/tmp/b.mkv",
        ])

        self.assertTrue(args.input_entries[0]["force_all_streams"])
        self.assertNotIn("force_all_streams", args.input_entries[1])

    def test_streams_picker_keeps_forced_streams_including_unknown_language(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file_forced = os.path.join(self.wd.path, "forced.mkv")
        file_other = os.path.join(self.wd.path, "other.mkv")

        duplicates.add_metadata(file_forced, "force_all_streams", True)

        files_details = {
            file_forced: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [
                    {"tid": 1, "language": None, "channels": 2, "sample_rate": 24000},
                    {"tid": 2, "language": "eng", "channels": 2, "sample_rate": 24000},
                ],
                "subtitle": [
                    {"tid": 3, "language": None},
                ],
            },
            file_other: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [
                    {"tid": 5, "language": "pol", "channels": 2, "sample_rate": 48000},
                    {"tid": 6, "language": "eng", "channels": 2, "sample_rate": 96000},
                ],
                "subtitle": [
                    {"tid": 8, "language": "deu"},
                ],
            },
        }
        ids = {file_forced: 1, file_other: 2}

        _, audio_streams, subtitle_streams = sp.pick_streams(files_details, ids)

        self.assertEqual(audio_streams, [
            (file_forced, 1, None),
            (file_forced, 2, "eng"),
            (file_other, 5, "pol"),
        ])
        self.assertEqual(subtitle_streams, [
            (file_forced, 3, None),
            (file_other, 8, "deu"),
        ])

    def test_streams_picker_raises_on_unknown_language_without_force_flag(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file1 = os.path.join(self.wd.path, "unknown_audio_1.mkv")
        file2 = os.path.join(self.wd.path, "unknown_audio_2.mkv")

        files_details = {
            file1: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [{"tid": 1, "language": None, "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
            file2: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
        }
        ids = {file1: 1, file2: 2}

        with self.assertRaises(RuntimeError):
            sp.pick_streams(files_details, ids)

    def test_force_all_streams_does_not_affect_video_selection(self):
        """Force flag only applies to audio/subtitle — video uses normal preference."""
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file_forced = os.path.join(self.wd.path, "forced_lo.mkv")
        file_other = os.path.join(self.wd.path, "other_hi.mkv")

        duplicates.add_metadata(file_forced, "force_all_streams", True)

        files_details = {
            file_forced: {
                "video": [{"tid": 0, "width": 640, "height": 480, "fps": "25"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
            file_other: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
        }
        ids = {file_forced: 1, file_other: 2}

        video_streams, _, _ = sp.pick_streams(files_details, ids)

        # Higher resolution from non-forced file should be preferred
        self.assertEqual(video_streams[0][0], file_other)

    def test_force_all_streams_treats_und_as_unknown(self):
        """'und' language is normalized to None, then treated as undefined for forced inputs."""
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file_forced = os.path.join(self.wd.path, "forced_und.mkv")

        duplicates.add_metadata(file_forced, "force_all_streams", True)

        files_details = {
            file_forced: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [{"tid": 1, "language": "und", "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
        }
        ids = {file_forced: 1}

        _, audio_streams, _ = sp.pick_streams(files_details, ids)

        # 'und' → None in output (normalized through undefined bucket)
        self.assertEqual(len(audio_streams), 1)
        self.assertIsNone(audio_streams[0][2])

    def test_force_all_streams_parser_requires_preceding_input(self):
        """--force-all-streams before any -i should fail."""
        parser = argparse.ArgumentParser()
        MeltTool().setup_parser(parser)

        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["--force-all-streams", "-i", "/tmp/a.mkv", "-o", "/out", "-t", "X"])

    def test_force_all_streams_both_inputs_forced_same_language(self):
        """Two forced inputs with the same language keep all streams from both."""
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file_a = os.path.join(self.wd.path, "forced_a.mkv")
        file_b = os.path.join(self.wd.path, "forced_b.mkv")

        duplicates.add_metadata(file_a, "force_all_streams", True)
        duplicates.add_metadata(file_b, "force_all_streams", True)

        files_details = {
            file_a: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
            file_b: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [{"tid": 2, "language": "eng", "channels": 2, "sample_rate": 96000}],
                "subtitle": [],
            },
        }
        ids = {file_a: 1, file_b: 2}

        _, audio_streams, _ = sp.pick_streams(files_details, ids)

        # Both forced — both eng streams kept
        self.assertEqual(len(audio_streams), 2)
        paths = {s[0] for s in audio_streams}
        self.assertEqual(paths, {file_a, file_b})

    def test_force_all_streams_covers_all_languages_non_forced_skipped(self):
        """When forced input covers all unique keys, non-forced contributes nothing."""
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file_forced = os.path.join(self.wd.path, "forced_full.mkv")
        file_other = os.path.join(self.wd.path, "other.mkv")

        duplicates.add_metadata(file_forced, "force_all_streams", True)

        files_details = {
            file_forced: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [
                    {"tid": 1, "language": "eng", "channels": 6, "sample_rate": 48000},
                    {"tid": 2, "language": "pol", "channels": 6, "sample_rate": 48000},
                ],
                "subtitle": [{"tid": 3, "language": "eng"}],
            },
            file_other: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [
                    {"tid": 4, "language": "eng", "channels": 6, "sample_rate": 96000},
                    {"tid": 5, "language": "pol", "channels": 6, "sample_rate": 96000},
                ],
                "subtitle": [{"tid": 6, "language": "eng"}],
            },
        }
        ids = {file_forced: 1, file_other: 2}

        _, audio_streams, subtitle_streams = sp.pick_streams(files_details, ids)

        # All from forced, nothing from other (same language+channels = same key)
        forced_audio = [s for s in audio_streams if s[0] == file_forced]
        other_audio = [s for s in audio_streams if s[0] == file_other]
        self.assertEqual(len(forced_audio), 2)
        self.assertEqual(len(other_audio), 0)

        forced_subs = [s for s in subtitle_streams if s[0] == file_forced]
        other_subs = [s for s in subtitle_streams if s[0] == file_other]
        self.assertEqual(len(forced_subs), 1)
        self.assertEqual(len(other_subs), 0)

    def test_streams_picker_prefers_higher_resolution_video(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file1 = os.path.join(self.wd.path, "video_800.mkv")
        file2 = os.path.join(self.wd.path, "video_796.mkv")

        files_details = {
            file1: {
                "video": [{"tid": 0, "width": 1920, "height": 800, "fps": "24000/1001"}],
                "audio": [],
                "subtitle": [],
            },
            file2: {
                "video": [{"tid": 0, "width": 1920, "height": 796, "fps": "24000/1001"}],
                "audio": [],
                "subtitle": [],
            },
        }
        ids = {file1: 1, file2: 2}

        video_streams, _, _ = sp.pick_streams(files_details, ids)

        self.assertEqual(video_streams[0][0], file1)

    def test_static_source_production_audio_language_metadata(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        path = "/tmp/fake.mkv" if platform.system() != "Windows" else r"c:\tmp\fake.mkv"
        duplicates.add_entry("Some title", path)
        duplicates.add_metadata(path, "audio_prod_lang", "eng")

        self.assertEqual(
            "eng",
            duplicates.get_metadata_for(path)["audio_prod_lang"],
        )
        self.assertIsNone(
            duplicates.get_metadata_for("/not/exists").get("audio_prod_lang")
        )

    sample_streams = [
        # case: merge all audio tracks
        (
            "mix audios",
            # input
            {
                "fileA": {
                    "video": [{"height": "1024", "width": "1024", "fps": "24", "tid": 0}],
                    "audio": [{"language": "jp", "channels": "2", "sample_rate": "32000", "tid": 2},
                              {"language": "de", "channels": "2", "sample_rate": "32000", "tid": 4}]
                },
                "fileB": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30", "tid": 6}],
                    "audio": [{"language": "br", "channels": "2", "sample_rate": "32000", "tid": 8},
                              {"language": "nl", "channels": "2", "sample_rate": "32000", "tid": 10}]
                }
            },
            # expected output
            (
                [("fileB", 6, None)],
                [("fileA", 2, "jp"), ("fileA", 4, "de"), ("fileB", 8, "br"), ("fileB", 10, "nl")],
                []
            )
        ),
        # case: pick one file whenever possible

        (
            "prefer one file",
            # input (fileB is a superset of fileA, so prefer it)
            {
                "fileA": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30", "tid": 1}],
                    "audio": [{"language": "cz", "channels": "2", "sample_rate": "32000", "tid": 2}],
                    "subtitle": [{"language": "pl", "tid": 3}]
                },
                "fileB": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30", "tid": 1}],
                    "audio": [{"language": "cz", "channels": "2", "sample_rate": "32000", "tid": 2}],
                    "subtitle": [{"language": "pl", "tid": 4}, {"language": "br", "tid": 3}]
                }
            },
            # expected output
            # Explanation: fileB is a superset of fileA, so no need to pick any streams from fileA
            (
                [("fileB", 1, None)],
                [("fileB", 2, "cz")],
                [("fileB", 4, "pl"), ("fileB", 3, "br")]
            )
        ),

        (
            "same but different",
            # input
            {
                "fileA": {
                    "video": [{"height": "1024", "width": "1024", "fps": "24", "tid": 1}],
                    "audio": [{"language": "jp", "channels": "2", "sample_rate": "32000", "tid": 4},
                              {"language": "jp", "channels": "2", "sample_rate": "32000", "tid": 6}],
                    "subtitle": [{"language": "de", "tid": 15}, {"language": "de", "tid": 8}]
                },
                "fileB": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30", "tid": 2}],
                    "audio": [{"language": "jp", "channels": "2", "sample_rate": "32000", "tid": 1},
                              {"language": "jp", "channels": "6", "sample_rate": "32000", "tid": 0}],
                    "subtitle": [{"language": "pl", "tid": 15}, {"language": "pl", "tid": 17}]
                }
            },
            # expected output
            # Explanation:
            # There are two identical (basing on parameters) audio inputs in file A.
            # Consider them different (why would there be two identical audio stracks?) and include both in output.
            #
            # Include 6 channel audio track from file B (as best one) but ignore 2 channel one (assume it's a duplicate of tracks from file A).
            #
            # Same logic goes for subtitles. Include both (most likely different) subtitle tracks from file A and
            # both subtitle tracks from file B
            (
                [("fileB", 2, None)],
                [("fileA", 4, "jp"), ("fileA", 6, "jp"), ("fileB", 0, "jp")],
                [("fileA", 15, "de"), ("fileA", 8, "de"), ("fileB", 15, "pl"), ("fileB", 17, "pl")]
            )
        ),
    ]

    @parameterized.expand(sample_streams)
    def test_streams_pick_decision(self, name, input, expected_streams):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        streams_picker = StreamsPicker(self.logger.getChild("Melter"), duplicates, self.wd.path)

        ids = _build_path_to_id_map(input)

        # Test all possible combinations of order of input files. Output should be stable
        for video_info in all_key_orders(input):
            picked_streams = streams_picker.pick_streams(video_info, ids)
            picked_streams_normalized = normalize(picked_streams)
            expected_streams_normalized = normalize(expected_streams)

            self.assertEqual(picked_streams_normalized, expected_streams_normalized)
