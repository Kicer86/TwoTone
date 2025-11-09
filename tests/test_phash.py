
import numpy as np
import unittest
import os
from functools import partial
from pathlib import Path

from twotone.tools.utils.files_utils import ScopedDirectory
from twotone.tools.utils import process_utils, video_utils
from twotone.tools.melt.phash_cache import PhashCache

from common import (
    TwoToneTestCase,
    FileCache,
    get_audio,
    get_video,
)

class MeltingTest(TwoToneTestCase):

    def setUp(self):
        super().setUp()

        def gen_sample(out_path: Path):
            videos = ["Atoms - 8579.mp4",
                      "Blue_Sky_and_Clouds_Timelapse_0892__Videvo.mov",
                      "Frog - 113403.mp4",
                      "sea-waves-crashing-on-beach-shore-4793288.mp4",
                      "Woman - 58142.mp4"]
            audios = ["806912__kevp888__250510_122217_fr_large_crowd_in_palais_garnier.wav",
                      "807385__josefpres__piano-loops-066-efect-4-octave-long-loop-120-bpm.wav",
                      "807184__logicmoon__mirrors.wav",
                      "807419__kvgarlic__light-spring-rain-and-kids-and-birds-may-13-2025-vtwo.wav",
                      "807385__josefpres__piano-loops-066-efect-4-octave-long-loop-120-bpm.wav",
                      ]

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

        def gen_lower_resolution(out_path: Path, input: str):
            """
                Process input file and worse its quality
            """

            args = [
                "-i", input,
                "-filter:v", "scale=iw/2:ih/2",
                "-c:v", "libx265",
                "-crf", "18",
                "-preset", "slow",
                str(out_path)
            ]

            process_utils.start_process("ffmpeg", args)

        file_cache = FileCache("TwoToneTests")

        self.sample_video_file = str(file_cache.get_or_generate("phash_tests_sample", "1", "mp4", gen_sample))
        self.sample_lower_res = str(file_cache.get_or_generate("phash_tests_lower_res", "1", "mp4", partial(gen_lower_resolution, input = self.sample_video_file)))


    def test_phash_distance(self):
        """
            Test that pHash distance is low for similar videos and high for different videos
        """

        lhs_wd = os.path.join(self.wd.path, "lhs_all")
        rhs_wd = os.path.join(self.wd.path, "rhs_all")

        os.makedirs(lhs_wd)
        os.makedirs(rhs_wd)

        phash1 = video_utils.extract_all_frames(self.sample_video_file, lhs_wd, scale = 0.5, format = "png")
        phash2 = video_utils.extract_all_frames(self.sample_lower_res, rhs_wd, scale = 0.5, format = "png")

        assert(len(phash1) == len(phash2))
        length = len(phash1)

        phash = PhashCache()
        points = np.linspace(0, length - 1, num=10, dtype=int)

        for p in points:
            p_lhs_key = list(phash1.keys())[p]
            p_rhs_key = list(phash2.keys())[p]

            lhs_hash = phash.get(phash1[p_lhs_key]["path"])
            rhs_hash = phash.get(phash2[p_rhs_key]["path"])

            distance = int(abs(lhs_hash - rhs_hash))

            self.assertLessEqual(distance, 20, f"pHash distance too high for similar videos at frame {p}: {distance}")


if __name__ == '__main__':
    unittest.main()