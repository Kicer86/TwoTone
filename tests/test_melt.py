
import logging
import unittest
import os

from functools import partial
from itertools import permutations
from parameterized import parameterized
from pathlib import Path
from typing import Dict, Iterator

from twotone.tools.utils import generic_utils, process_utils, video_utils
from twotone.tools.melt import Melter
from twotone.tools.melt.melt import StaticSource, StreamsPicker
from twotone.tools.utils.files_utils import ScopedDirectory
from common import (
    TwoToneTestCase,
    FileCache,
    add_test_media,
    add_to_test_dir,
    build_test_video,
    current_path,
    get_audio,
    get_video,
    get_font,
    get_chapter,
    hashes,
    list_files,
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


def all_key_orders(d: Dict) -> Iterator[Dict]:
    """
    Yield dictionaries with all possible key orderings (same keys and values).
    """
    keys = list(d.keys())
    for perm in permutations(keys):
        yield {k: d[k] for k in perm}


class MeltingTest(TwoToneTestCase):

    def setUp(self):
        super().setUp()

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
                    process_utils.start_process("ffmpeg", ["-i", video_input_path, "-i", audio_input_path, "-r", "25", "-vf", "fps=25", "-c:v", "libx265", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                                                "-shortest", "-map", "0:v:0", "-map", "1:a:0", "-c:a", "libvorbis", "-ar", "44100",
                                                output_path])
                    output_files.append(output_path)

                # concatenate
                files_list_path = os.path.join(output_dir, "filelist.txt")
                with open(files_list_path, "w", encoding="utf-8") as f:
                    for path in output_files:
                        # Escape single quotes if needed
                        safe_path = path.replace("'", "'\\''")
                        f.write(f"file '{safe_path}'\n")

                process_utils.start_process("ffmpeg", ["-f", "concat", "-safe", "0", "-i", files_list_path, "-c", "copy", str(out_path)])

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
                "-c:v", "libx265",
                "-crf", "40",                                                                           # use low quality
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                str(out_path)
            ]

            process_utils.start_process("ffmpeg", args)

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

        melter = Melter(self.logger.getChild("Melter"), interruption, duplicates, live_run = True, wd = self.wd.path, output = output_dir)
        melter.melt()

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

        melter = Melter(self.logger.getChild("Melter"), interruption, duplicates, live_run = False, wd = self.wd.path, output = output_dir)
        melter.melt()

        # expect output to be empty
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 0)

    def test_inputs_are_removed_by_default(self):
        file1 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes=["r1"])[0]
        file2 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes=["r2"])[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Grass", file1)
        duplicates.add_entry("Grass", file2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        melter = Melter(self.logger.getChild("Melter"), interruption, duplicates,
                        live_run=True, wd=self.wd.path, output=output_dir)
        melter.melt()

        self.assertFalse(os.path.exists(file1))
        self.assertFalse(os.path.exists(file2))
        self.assertEqual(len(hashes(output_dir)), 1)

    def test_keep_input_files_flag(self):
        file1 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes=["k1"])[0]
        file2 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes=["k2"])[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Grass", file1)
        duplicates.add_entry("Grass", file2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        melter = Melter(self.logger.getChild("Melter"), interruption, duplicates,
                        live_run=True, wd=self.wd.path, output=output_dir,
                        keep_input_files=True)
        melter.melt()

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

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        melter = Melter(self.logger.getChild("Melter"), interruption, duplicates, live_run = True, wd = self.wd.path, output = output_dir)
        melter.melt()

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 0)

    def test_allow_length_mismatch(self):
        file1 = add_test_media("DSC_8073.MP4", self.wd.path)[0]
        file2 = add_test_media("moon.mp4", self.wd.path)[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        melter = Melter(self.logger.getChild("Melter"), interruption, duplicates, live_run = True, wd = self.wd.path, output = output_dir, allow_length_mismatch = True)
        melter.melt()

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)


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

        melter = Melter(self.logger.getChild("Melter"), interruption, duplicates, live_run = True, wd = self.wd.path, output = output_dir, allow_length_mismatch = True)
        melter.melt()

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

        melter = Melter(self.logger.getChild("Melter"), interruption, duplicates, live_run = True, wd = self.wd.path, output = output_dir)
        melter.melt()

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
            self.assertEqual(output_file_data["audio"][0]["language"], "nor")
            self.assertEqual(output_file_data["audio"][1]["language"], "deu")


    def test_languages_prioritization(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        langs = ["pol", "en", "ger", "ja", "nor"]

        for i in range(5):
            file = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = [f"v{i}"])[0]
            duplicates.add_entry("Grass", file)
            duplicates.add_metadata(file, "audio_lang", langs[i])

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        melter = Melter(self.logger.getChild("Melter"), interruption, duplicates, live_run = True, wd = self.wd.path, output = output_dir, languages_priority = ["de", "jpn", "eng", "no", "pl"])
        melter.melt()

        # validate order
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash)[0]
        output_file_data = video_utils.get_video_data(output_file)
        self.assertEqual(output_file_data["audio"][0]["language"], "deu")
        self.assertEqual(output_file_data["audio"][1]["language"], "jpn")
        self.assertEqual(output_file_data["audio"][2]["language"], "eng")
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

        melter = Melter(logging.getLogger("Melter"), interruption, duplicates, live_run = True, wd = self.wd.path, output = output_dir)
        melter.melt()

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data(output_file)
        self.assertEqual(len(output_file_data["video"]), 1)
        self.assertEqual(output_file_data["video"][0]["height"], 1080)
        self.assertEqual(output_file_data["video"][0]["width"], 1920)

        self.assertEqual(len(output_file_data["audio"]), 1)

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

        melter = Melter(logging.getLogger("Melter"), interruption, duplicates, live_run = True, wd = self.wd.path, output = output_dir)
        melter.melt()

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["video"]), 1)
        self.assertEqual(len(output_file_data["attachments"]), 1)

    def test_chapters(self):
        video1 = build_test_video(
            os.path.join(self.wd.path, "c1.mkv"),
            self.wd.path,
            "fog-over-mountainside-13008647.mp4",
            subtitle=True,
            chapter_name="simple_chapters.txt",
        )
        video2 = build_test_video(
            os.path.join(self.wd.path, "c2.mkv"),
            self.wd.path,
            "fog-over-mountainside-13008647.mp4",
            subtitle=True,
        )

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Fog", video1)
        duplicates.add_entry("Fog", video2)
        duplicates.add_metadata(video1, "audio_lang", "eng")
        duplicates.add_metadata(video2, "audio_lang", "eng")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        melter = Melter(
            logging.getLogger("Melter"),
            interruption,
            duplicates,
            live_run=True,
            wd=self.wd.path,
            output=output_dir,
        )
        melter.melt()

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertTrue(output_file_data["chapters"])


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

        melter = Melter(logging.getLogger("Melter"), interruption, duplicates, live_run = True, wd = self.wd.path, output = output_dir)
        melter.melt()

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["video"]), 1)
        self.assertEqual(len(output_file_data["attachments"]), 1)

    def test_font_attachment(self):
        font = get_font("dummy.ttf")
        video1 = build_test_video(
            os.path.join(self.wd.path, "o1.mkv"),
            self.wd.path,
            "fog-over-mountainside-13008647.mp4",
            subtitle=True,
            attachments=[font],
        )
        video2 = build_test_video(
            os.path.join(self.wd.path, "o2.mkv"),
            self.wd.path,
            "fog-over-mountainside-13008647.mp4",
            subtitle=True,
        )

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Fog", video1)
        duplicates.add_entry("Fog", video2)
        duplicates.add_metadata(video1, "audio_lang", "eng")
        duplicates.add_metadata(video2, "audio_lang", "eng")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        melter = Melter(
            logging.getLogger("Melter"),
            interruption,
            duplicates,
            live_run=True,
            wd=self.wd.path,
            output=output_dir,
        )
        melter.melt()

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
        streams_picker = StreamsPicker(self.logger.getChild("Melter"), duplicates)

        # Test all possible combinations of order of input files. Output should be stable
        for video_info in all_key_orders(input):
            picked_streams = streams_picker.pick_streams(video_info)
            picked_streams_normalized = normalize(picked_streams)
            expected_streams_normalized = normalize(expected_streams)

            self.assertEqual(picked_streams_normalized, expected_streams_normalized)


if __name__ == '__main__':
    unittest.main()
