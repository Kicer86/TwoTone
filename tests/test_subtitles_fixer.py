
import os
import unittest
import tempfile

import subtitles_fixer
import utils

from common import TestDataWorkingDirectory, add_test_media, hashes, current_path, generate_microdvd_subtitles


def create_broken_video(output_video_path: str, input_video: str):
    with tempfile.TemporaryDirectory() as subtitle_dir:
        input_video_info = utils.get_video_data(input_video)
        default_video_track = input_video_info.video_tracks[0]
        fps = utils.fps_str_to_float(default_video_track.fps)

        if abs(fps - utils.ffmpeg_default_fps) < 1:
            raise RuntimeError("source video is not suitable, has nearly default fps")

        length = default_video_track.length

        subtitle_path = f"{subtitle_dir}/sub.sub"
        generate_microdvd_subtitles(subtitle_path, int(length), fps)

        utils.start_process("ffmpeg", ["-hide_banner", "-i", input_video, "-i", subtitle_path, "-map", "0", "-map", "1", "-c:v", "copy", "-c:a", "copy", output_video_path])


class SubtitlesFixer(unittest.TestCase):

    def test_dry_run_is_respected(self):
        with TestDataWorkingDirectory() as td:
            output_video_path = f"{td.path}/test_video.mkv"
            create_broken_video(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

            hashes_before = hashes(td.path)
            subtitles_fixer.run([td.path])
            hashes_after = hashes(td.path)

            self.assertEqual(hashes_before, hashes_after)

    def test_video_fixing(self):
        with TestDataWorkingDirectory() as td:
            output_video_path = f"{td.path}/test_video.mkv"
            create_broken_video(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

            hashes_before = hashes(td.path)
            subtitles_fixer.run(["-r", td.path])
            hashes_after = hashes(td.path)

            self.assertNotEqual(hashes_before, hashes_after)

if __name__ == '__main__':
    unittest.main()