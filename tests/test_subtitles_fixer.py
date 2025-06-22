
import logging
import unittest
import tempfile

from twotone.tools.utils import subtitles_utils, video_utils

from common import (
    WorkingDirectoryTestCase,
    add_test_media,
    hashes,
    current_path,
    generate_microdvd_subtitles,
    run_twotone,
    write_subtitle,
)
from twotone.tools.utils import generic_utils, process_utils


def create_broken_video_with_scaled_subtitle_timings(output_video_path: str, input_video: str):
    with tempfile.TemporaryDirectory() as subtitle_dir:
        input_video_info = video_utils.get_video_data(input_video)
        default_video_track = input_video_info.video_tracks[0]
        fps = generic_utils.fps_str_to_float(default_video_track.fps)

        if abs(fps - subtitles_utils.ffmpeg_default_fps) < 1:
            raise RuntimeError("source video is not suitable, has nearly default fps")

        length = default_video_track.length / 1000

        subtitle_path = f"{subtitle_dir}/sub.sub"
        generate_microdvd_subtitles(subtitle_path, int(length), fps)

        # convert to srt format
        srt_subtitle_path = f"{subtitle_dir}/sub.srt"
        status = process_utils.start_process("ffmpeg", ["-hide_banner", "-y", "-i", subtitle_path, srt_subtitle_path])

        video_utils.generate_mkv(input_video = input_video, output_path = output_video_path, subtitles = [subtitles_utils.SubtitleFile(srt_subtitle_path, "eng", "utf8")])


def create_broken_video_with_too_long_last_subtitle(output_video_path: str, input_video: str):
    with tempfile.TemporaryDirectory() as subtitle_dir:
        input_video_info = video_utils.get_video_data(input_video)
        default_video_track = input_video_info.video_tracks[0]
        length = default_video_track.length

        subtitle_path = f"{subtitle_dir}/sub.srt"
        write_subtitle(
            subtitle_path,
            [
                "1",
                f"{generic_utils.ms_to_time(0)} --> {generic_utils.ms_to_time(1000)}",
                "1",
                "",
                "2",
                f"{generic_utils.ms_to_time(1000)} --> {generic_utils.ms_to_time((length + 10) * 1000)}",
                "2",
            ],
        )

        video_utils.generate_mkv(input_video = input_video, output_path = output_video_path, subtitles = [subtitles_utils.SubtitleFile(subtitle_path, "eng", "utf8")])


def create_broken_video_with_incompatible_subtitles(output_video_path: str, input_video: str):
    with tempfile.TemporaryDirectory() as subtitle_dir:
        input_video_info = video_utils.get_video_data(input_video)
        default_video_track = input_video_info.video_tracks[0]
        fps = generic_utils.fps_str_to_float(default_video_track.fps)

        if abs(fps - subtitles_utils.ffmpeg_default_fps) < 1:
            raise RuntimeError("source video is not suitable, has nearly default fps")

        length = default_video_track.length

        subtitle_path = f"{subtitle_dir}/sub.sub"
        generate_microdvd_subtitles(subtitle_path, int(length), fps)

        process_utils.start_process("ffmpeg", ["-hide_banner", "-i", input_video, "-i", subtitle_path, "-map", "0", "-map", "1", "-c:v", "copy", "-c:a", "copy", output_video_path])


class SubtitlesFixer(WorkingDirectoryTestCase):

    def setUp(self):
        super().setUp()
        logging.getLogger().setLevel(logging.ERROR)

    def test_dry_run_is_respected(self):
        output_video_path = f"{self.wd.path}/test_video.mkv"
        create_broken_video_with_scaled_subtitle_timings(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

        hashes_before = hashes(self.wd.path)
        run_twotone("subtitles_fix", [self.wd.path])
        hashes_after = hashes(self.wd.path)

        self.assertEqual(hashes_before, hashes_after)

    def test_video_with_scaled_subtitle_timings_fixing(self):
        output_video_path = f"{self.wd.path}/test_video.mkv"
        create_broken_video_with_scaled_subtitle_timings(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

        hashes_before = hashes(self.wd.path)
        run_twotone("subtitles_fix", [self.wd.path], ["-r"])
        hashes_after = hashes(self.wd.path)

        self.assertNotEqual(hashes_before, hashes_after)

        # run again - there should be no changes
        run_twotone("subtitles_fix", [self.wd.path], ["-r"])
        hashes_after_after = hashes(self.wd.path)
        self.assertEqual(hashes_after, hashes_after_after)

    def test_video_with_too_long_last_subtitle_fixing(self):
        output_video_path = f"{self.wd.path}/test_video.mkv"
        create_broken_video_with_too_long_last_subtitle(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

        hashes_before = hashes(self.wd.path)
        run_twotone("subtitles_fix", [self.wd.path], ["-r"])
        hashes_after = hashes(self.wd.path)

        self.assertNotEqual(hashes_before, hashes_after)

        # run again - there should be no changes
        run_twotone("subtitles_fix", [self.wd.path], ["-r"])
        hashes_after_after = hashes(self.wd.path)
        self.assertEqual(hashes_after, hashes_after_after)

    def test_deal_with_incompatible_videos(self):
        output_video_path = f"{self.wd.path}/test_video.mkv"
        create_broken_video_with_incompatible_subtitles(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

        hashes_before = hashes(self.wd.path)
        run_twotone("subtitles_fix", [self.wd.path], ["-r"])
        hashes_after = hashes(self.wd.path)

        self.assertEqual(hashes_before, hashes_after)

if __name__ == '__main__':
    unittest.main()
