
import logging
import tempfile
import unittest
import os
import platform
import argparse

from functools import partial
from itertools import permutations
from parameterized import parameterized
from pathlib import Path
from typing import Iterator

from twotone.tools.utils import generic_utils, process_utils, video_utils
from twotone.tools.melt.melt import DEFAULT_TOLERANCE_MS, MeltAnalyzer, MeltPerformer, MeltTool, StaticSource, StreamsPicker
from twotone.tools.utils.files_utils import ScopedDirectory
from common import (
    TwoToneTestCase,
    FileCache,
    add_test_media,
    add_to_test_dir,
    build_test_video,
    get_audio,
    get_video,
    hashes,
)


def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, list):
        return sorted((normalize(item) for item in obj), key=lambda x: repr(x))
    elif isinstance(obj, tuple):
        return tuple(normalize(item) for item in obj)
    else:
        return obj


def all_key_orders(d: dict) -> Iterator[dict]:
    """
    Yield dictionaries with all possible key orderings (same keys and values).
    """
    keys = list(d.keys())
    for perm in permutations(keys):
        yield {k: d[k] for k in perm}


def analyze_duplicates_helper(
    logger: logging.Logger,
    duplicates_source: StaticSource,
    working_dir: str,
    allow_length_mismatch: bool = False,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
):
    os.makedirs(working_dir, exist_ok=True)
    duplicates_raw = duplicates_source.collect_duplicates()
    duplicates = {title: list(files) for title, files in duplicates_raw.items()}
    analyzer = MeltAnalyzer(
        logger,
        duplicates_source,
        working_dir,
        allow_length_mismatch,
        tolerance_ms,
    )
    return analyzer.analyze_duplicates(duplicates)


def process_duplicates_helper(
    logger: logging.Logger,
    interruption: generic_utils.InterruptibleProcess,
    working_dir: str,
    output_dir: str,
    plan,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
):
    performer = MeltPerformer(
        logger,
        interruption,
        working_dir,
        output_dir,
        tolerance_ms,
    )
    performer.process_duplicates(plan)


def _build_path_to_id_map(input: dict) -> dict[str, int]:
    return {path: idx for idx, path in enumerate(input.keys())}


class MeltingTest(TwoToneTestCase):

    def setUp(self):
        super().setUp()

        def run_ffmpeg(args, expected_path: str | None = None):
            status = process_utils.start_process("ffmpeg", args)
            if status.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {status.stderr}")
            if expected_path and not os.path.exists(expected_path):
                raise RuntimeError(f"ffmpeg did not produce expected file: {expected_path}")
            return status

        def gen_sample(out_path: Path):
            videos = ["Atoms - 8579.mp4",
                      "Blue_Sky_and_Clouds_Timelapse_0892__Videvo.mov",
                      "Frog - 113403.mp4", "sea-waves-crashing-on-beach-shore-4793288.mp4",
                      "Woman - 58142.mp4"]
            audios = ["806912__kevp888__250510_122217_fr_large_crowd_in_palais_garnier.wav",
                      "807385__josefpres__piano-loops-066-efect-4-octave-long-loop-120-bpm.wav",
                      "807184__logicmoon__mirrors.wav",
                      "807419__kvgarlic__light-spring-rain-and-kids-and-birds-may-13-2025-vtwo.wav"]

            #unify fps and add audio path
            output_dir = os.path.join(self.wd.path, "gen_sample")

            with ScopedDirectory(output_dir) as sd:
                output_files = []
                for video, audio in zip(videos, audios):
                    video_input_path = get_video(video)
                    audio_input_path = get_audio(audio)
                    output_path = os.path.join(output_dir, video + ".mp4")
                    run_ffmpeg(
                        [
                            "-i", video_input_path,
                            "-i", audio_input_path,
                            "-r", "25",
                            "-vf", "fps=25",
                            "-c:v", "libx264",
                            "-preset", "veryfast",
                            "-crf", "18",
                            "-pix_fmt", "yuv420p",
                            "-shortest",
                            "-map", "0:v:0",
                            "-map", "1:a:0",
                            "-c:a", "aac",
                            "-ar", "44100",
                            output_path,
                        ],
                        expected_path=output_path,
                    )
                    output_files.append(output_path)

                # concatenate
                files_list_path = os.path.join(output_dir, "filelist.txt")
                with open(files_list_path, "w", encoding="utf-8") as f:
                    for path in output_files:
                        # Escape single quotes if needed
                        safe_path = path.replace("'", "'\\''")
                        f.write(f"file '{safe_path}'\n")

                run_ffmpeg(
                    [
                        "-f", "concat",
                        "-safe", "0",
                        "-i", files_list_path,
                        "-c", "copy",
                        str(out_path),
                    ],
                    expected_path=str(out_path),
                )

        def gen_vhs(out_path: Path, input: str):
            """
                Process input file and worse its quality
            """
            duration = video_utils.get_video_duration(input) / 1000                                     # duration of original video

            vf = ",".join([
                "fps=26.5",                                                                             # use non standard fps
                "setpts=PTS/1.05",                                                                      # speed it up by 5%
                "boxblur=enable='between(t,5,10)':lr=2",                                                # add some blur
                f"crop=w=iw*0.9:h=ih*0.9:x='(iw-iw*0.9)*t/{duration}':y='(ih-ih*0.9)*t/{duration}'",    # add a crop (90% H and W) which moves from top left corner to bottom right
                f"scale={960}:{720}"                                                                    # scale to 4:3
            ])

            af = "atempo=1.05"

            args = [
                "-i", input,
                "-filter_complex", vf,
                "-filter:a", af,
                "-c:v", "libx264",
                "-crf", "40",                                                                           # use low quality
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(out_path)
            ]

            run_ffmpeg(args, expected_path=str(out_path))

        file_cache = FileCache("TwoToneTests")

        self.sample_video_file = str(file_cache.get_or_generate("melter_tests_sample", "1", "mp4", gen_sample))
        self.sample_vhs_video_file = str(file_cache.get_or_generate("melter_tests_vhs", "1", "mp4", partial(gen_vhs, input = self.sample_video_file)))


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


    def test_static_source_production_audio_language_metadata(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        path = "/tmp/fake.mkv" if platform.system() != "Windows" else "c:\tmp\fake.mkv"
        duplicates.add_entry("Some title", path)
        duplicates.add_metadata(path, "audio_prod_lang", "eng")

        self.assertEqual(
            "eng",
            duplicates.get_metadata_for(path)["audio_prod_lang"],
        )
        self.assertIsNone(
            duplicates.get_metadata_for("/not/exists").get("audio_prod_lang")
        )


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
        file1 = add_to_test_dir(self.wd.path, str(self.sample_video_file))
        file2 = add_to_test_dir(self.wd.path, str(self.sample_vhs_video_file))

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



class MeltPerformerUnitTest(unittest.TestCase):
    """Unit tests for MeltPerformer internal methods."""

    def _make_performer(self) -> MeltPerformer:
        performer = object.__new__(MeltPerformer)
        performer.logger = logging.getLogger("test.MeltPerformer")
        performer.wd = tempfile.mkdtemp()
        performer.output_dir = tempfile.mkdtemp()
        performer.tolerance_ms = DEFAULT_TOLERANCE_MS
        performer.interruption = generic_utils.InterruptibleProcess()
        return performer

    def test_stream_sorting_puts_unknown_languages_last(self):
        streams = [
            ("audio", 1, "/a.mkv", None),
            ("audio", 2, "/a.mkv", "eng"),
            ("subtitle", 3, "/a.mkv", None),
            ("subtitle", 4, "/a.mkv", "pol"),
            ("subtitle", 5, "/a.mkv", "deu"),
        ]

        sort_key = lambda stream: (stream[3] is None, stream[3] or "")
        result = sorted(streams, key=sort_key)

        languages = [s[3] for s in result]
        self.assertEqual(languages, ["deu", "eng", "pol", None, None])

    def test_stream_sorting_alphabetical_when_all_known(self):
        streams = [
            ("subtitle", 1, "/a.mkv", "pol"),
            ("subtitle", 2, "/a.mkv", "eng"),
            ("subtitle", 3, "/a.mkv", "deu"),
            ("audio", 4, "/a.mkv", "jpn"),
        ]

        sort_key = lambda stream: (stream[3] is None, stream[3] or "")
        result = sorted(streams, key=sort_key)

        languages = [s[3] for s in result]
        self.assertEqual(languages, ["deu", "eng", "jpn", "pol"])

    def test_build_mkvmerge_args_track_order_respects_unknown_last(self):
        performer = self._make_performer()

        file_a = "/tmp/a.mkv"
        file_b = "/tmp/b.mkv"

        streams_list_sorted = [
            ("video", 0, file_a, None),
            ("audio", 1, file_a, "eng"),
            ("subtitle", 3, file_a, "deu"),
            ("subtitle", 4, file_a, "pol"),
            ("subtitle", 5, file_a, None),
        ]

        args = performer._build_mkvmerge_args(
            "/tmp/out.mkv",
            streams_list_sorted,
            attachments=[],
            preferred_audio=None,
            required_input_files=[file_a],
        )

        # Track order should preserve the sorted order
        track_order_idx = args.index("--track-order")
        track_order = args[track_order_idx + 1]
        self.assertEqual(track_order, "0:0,0:1,0:3,0:4,0:5")

if __name__ == '__main__':
    unittest.main()
