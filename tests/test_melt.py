
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
from common import TwoToneTestCase, FileCache, add_test_media, add_to_test_dir, build_test_video, current_path, get_audio, get_video, hashes, list_files


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


    def test_same_multiscene_video_duplicate_detection(self):
        file_cache = FileCache("TwoToneTests")

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
                duration = video_utils.get_video_duration(input) / 1000                                           # duration of original video

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

        file1 = file_cache.get_or_generate("melter_tests_sample", "1", "mp4", gen_sample)
        file1 = add_to_test_dir(self.wd.path, str(file1))
        file2 = file_cache.get_or_generate("melter_tests_vhs", "1", "mp4", partial(gen_vhs, input=file1))
        file2 = add_to_test_dir(self.wd.path, str(file2))

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

        melter = Melter(self.logger.getChild("Melter"), interruption, duplicates, live_run = True, wd = self.wd.path, output = output_dir)
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
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data2(output_file)
        self.assertEqual(len(output_file_data["video"]), 1)
        self.assertEqual(output_file_data["video"][0]["height"], 1080)
        self.assertEqual(output_file_data["video"][0]["width"], 1920)

        self.assertEqual(len(output_file_data["audio"]), 1)

        self.assertEqual(len(output_file_data["subtitle"]), 2)
        languages = { output_file_data["subtitle"][0]["language"],
                      output_file_data["subtitle"][1]["language"] }
        self.assertEqual(languages, {"jpn", "bre"})


    sample_streams = [
        # case: merge all audio tracks
        (
            "mix audios",
            # input
            {
                "fileA": {
                    "video": [{"height": "1024", "width": "1024", "fps": "24"}],
                    "audio": [{"language": "jp", "channels": "2", "sample_rate": "32000"},
                              {"language": "de", "channels": "2", "sample_rate": "32000"}]
                },
                "fileB": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30"}],
                    "audio": [{"language": "br", "channels": "2", "sample_rate": "32000"},
                              {"language": "nl", "channels": "2", "sample_rate": "32000"}]
                }
            },
            # expected output
            (
                [("fileB", 0, None)],
                [("fileA", 0, "jp"), ("fileA", 1, "de"), ("fileB", 0, "br"), ("fileB", 1, "nl")],
                []
            )
        ),
        # case: pick one file whenever possible

        (
            "prefer one file",
            # input (fileB is a superset of fileA, so prefer it)
            {
                "fileA": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30"}],
                    "audio": [{"language": "cz", "channels": "2", "sample_rate": "32000"}],
                    "subtitle": [{"language": "pl"}]
                },
                "fileB": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30"}],
                    "audio": [{"language": "cz", "channels": "2", "sample_rate": "32000"}],
                    "subtitle": [{"language": "pl"}, {"language": "br"}]
                }
            },
            # expected output
            # Explanation: fileB is a superset of fileA, so no need to pick any streams from fileA
            (
                [("fileB", 0, None)],
                [("fileB", 0, "cz")],
                [("fileB", 0, "pl"), ("fileB", 1, "br")]
            )
        ),

        (
            "same but different",
            # input
            {
                "fileA": {
                    "video": [{"height": "1024", "width": "1024", "fps": "24"}],
                    "audio": [{"language": "jp", "channels": "2", "sample_rate": "32000"},
                              {"language": "jp", "channels": "2", "sample_rate": "32000"}],
                    "subtitle": [{"language": "de"}, {"language": "de"}]
                },
                "fileB": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30"}],
                    "audio": [{"language": "jp", "channels": "2", "sample_rate": "32000"},
                              {"language": "jp", "channels": "6", "sample_rate": "32000"}],
                    "subtitle": [{"language": "pl"}, {"language": "pl"}]
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
                [("fileB", 0, None)],
                [("fileA", 0, "jp"), ("fileA", 1, "jp"), ("fileB", 1, "jp")],
                [("fileA", 0, "de"), ("fileA", 1, "de"), ("fileB", 0, "pl"), ("fileB", 1, "pl")]
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
