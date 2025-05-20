
import cv2
import imagehash
import logging
import unittest
import numpy as np
import os

from PIL import Image
from functools import partial
from overrides import override
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from typing import Dict, List

import twotone.tools.utils as utils
from twotone.tools.melt import Melter
from twotone.tools.melt.melt import StaticSource
from twotone.tools.utils2 import process, video
from common import WorkingDirectoryForTest, FileCache, add_test_media, add_to_test_dir, current_path, get_audio, get_video, hashes, list_files


class MeltingTest(unittest.TestCase):

    def setUp(self):
        logging.getLogger("Melter").setLevel(logging.DEBUG)

    def test_simple_duplicate_detection(self):
        with WorkingDirectoryForTest() as td:
            file1 = add_test_media("Grass - 66810.mp4", td.path, suffixes = ["v1"])[0]
            file2 = add_test_media("Grass - 66810.mp4", td.path, suffixes = ["v2"])[0]

            interruption = utils.InterruptibleProcess()
            duplicates = StaticSource(interruption)
            duplicates.add_entry("Grass", file1)
            duplicates.add_entry("Grass", file2)

            input_file_hashes = hashes(td.path)
            self.assertEqual(len(input_file_hashes), 2)

            output_dir = os.path.join(td.path, "output")
            os.makedirs(output_dir)

            melter = Melter(logging.getLogger("Melter"), interruption, duplicates, live_run = True, wd = td.path, output = output_dir)
            melter.melt()

            # expect output to be equal to the first of files
            output_file_hash = hashes(output_dir)
            self.assertEqual(len(output_file_hash), 1)

            # check if file was not altered
            self.assertEqual(list(output_file_hash.values())[0], input_file_hashes[file1])


    def test_dry_run_is_being_respected(self):
        with WorkingDirectoryForTest() as td:
            file1 = add_test_media("Grass - 66810.mp4", td.path, suffixes = ["v1"])[0]
            file2 = add_test_media("Grass - 66810.mp4", td.path, suffixes = ["v2"])[0]

            interruption = utils.InterruptibleProcess()
            duplicates = StaticSource(interruption)
            duplicates.add_entry("Grass", file1)
            duplicates.add_entry("Grass", file2)

            input_file_hashes = hashes(td.path)
            self.assertEqual(len(input_file_hashes), 2)

            output_dir = os.path.join(td.path, "output")
            os.makedirs(output_dir)

            melter = Melter(logging.getLogger("Melter"), interruption, duplicates, live_run = False, wd = td.path, output = output_dir)
            melter.melt()

            # expect output to be empty
            output_file_hash = hashes(output_dir)
            self.assertEqual(len(output_file_hash), 0)


    def test_same_multiscene_video_duplicate_detection(self):
        file_cache = FileCache("TwoToneTests")

        with WorkingDirectoryForTest() as td:
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
                output_files = []
                for video, audio in zip(videos, audios):
                    video_input_path = get_video(video)
                    audio_input_path = get_audio(audio)
                    output_path = os.path.join(td.path, video + ".mp4")
                    process.start_process("ffmpeg", ["-i", video_input_path, "-i", audio_input_path, "-r", "25", "-vf", "fps=25", "-c:v", "libx265", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                                                     "-shortest", "-map", "0:v:0", "-map", "1:a:0", "-c:a", "libvorbis", "-ar", "44100",
                                                     output_path])
                    output_files.append(output_path)

                # concatenate
                files_list_path = os.path.join(td.path, "filelist.txt")
                with open(files_list_path, "w", encoding="utf-8") as f:
                    for path in output_files:
                        # Escape single quotes if needed
                        safe_path = path.replace("'", "'\\''")
                        f.write(f"file '{safe_path}'\n")

                process.start_process("ffmpeg", ["-f", "concat", "-safe", "0", "-i", files_list_path, "-c", "copy", str(out_path)])

            def gen_vhs(out_path: Path, input: str):
                """
                    Process input file and worse its quality
                """
                duration = video.get_video_duration(input) / 1000                                           # duration of original video

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

                process.start_process("ffmpeg", args)

            file1 = file_cache.get_or_generate("melter_tests_sample", "1", "mp4", gen_sample)
            file1 = add_to_test_dir(td.path, str(file1))
            file2 = file_cache.get_or_generate("melter_tests_vhs", "1", "mp4", partial(gen_vhs, input=file1))
            file2 = add_to_test_dir(td.path, str(file2))

            files = [file1, file2]

            interruption = utils.InterruptibleProcess()
            duplicates = StaticSource(interruption)
            duplicates.add_entry("video", file1)
            duplicates.add_entry("video", file2)
            duplicates.add_metadata(file1, "audio_lang", "eng")
            duplicates.add_metadata(file2, "audio_lang", "pol")

            input_file_hashes = hashes(td.path)
            self.assertEqual(len(input_file_hashes), 2)

            output_dir = os.path.join(td.path, "output")
            os.makedirs(output_dir)

            melter = Melter(logging.getLogger("Melter"), interruption, duplicates, live_run = True, wd = td.path, output = output_dir)
            melter.melt()

            # validate output
            output_file_hash = hashes(output_dir)
            self.assertEqual(len(output_file_hash), 1)

            output_file = list(output_file_hash)[0]

            output_file_data = video.get_video_data2(output_file)
            self.assertEqual(len(output_file_data["video"]), 1)
            self.assertEqual(output_file_data["video"][0]["height"], 1080)
            self.assertEqual(output_file_data["video"][0]["width"], 1920)

            self.assertEqual(len(output_file_data["audio"]), 2)
            self.assertEqual(output_file_data["audio"][0]["language"], "eng")
            self.assertEqual(output_file_data["audio"][1]["language"], "pol")

if __name__ == '__main__':
    unittest.main()
