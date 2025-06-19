
import logging
import unittest
import tempfile

from twotone.tools.utils2 import subtitles as subs, video

from common import WorkingDirectoryForTest, add_test_media, hashes, current_path, generate_microdvd_subtitles, run_twotone
from twotone.tools.utils2 import generic, process


def create_broken_video_with_scaled_subtitle_timings(output_video_path: str, input_video: str):
    with tempfile.TemporaryDirectory() as subtitle_dir:
        input_video_info = video.get_video_data(input_video)
        default_video_track = input_video_info.video_tracks[0]
        fps = generic.fps_str_to_float(default_video_track.fps)

        if abs(fps - subs.ffmpeg_default_fps) < 1:
            raise RuntimeError("source video is not suitable, has nearly default fps")

        length = default_video_track.length / 1000

        subtitle_path = f"{subtitle_dir}/sub.sub"
        generate_microdvd_subtitles(subtitle_path, int(length), fps)

        # convert to srt format
        srt_subtitle_path = f"{subtitle_dir}/sub.srt"
        status = process.start_process("ffmpeg", ["-hide_banner", "-y", "-i", subtitle_path, srt_subtitle_path])

        video.generate_mkv(input_video = input_video, output_path = output_video_path, subtitles = [subs.SubtitleFile(srt_subtitle_path, "eng", "utf8")])


def create_broken_video_with_too_long_last_subtitle(output_video_path: str, input_video: str):
    with tempfile.TemporaryDirectory() as subtitle_dir:
        input_video_info = video.get_video_data(input_video)
        default_video_track = input_video_info.video_tracks[0]
        length = default_video_track.length

        subtitle_path = f"{subtitle_dir}/sub.srt"
        with open(subtitle_path, 'w', encoding='utf-8') as file:
            file.write(f"1\n")
            file.write(f"{generic.ms_to_time(0)} --> {generic.ms_to_time(1000)}\n")
            file.write(f"1\n\n")

            file.write(f"2\n")
            file.write(f"{generic.ms_to_time(1000)} --> {generic.ms_to_time((length + 10) * 1000)}\n")
            file.write(f"2\n")

        video.generate_mkv(input_video = input_video, output_path = output_video_path, subtitles = [subs.SubtitleFile(subtitle_path, "eng", "utf8")])


def create_broken_video_with_incompatible_subtitles(output_video_path: str, input_video: str):
    with tempfile.TemporaryDirectory() as subtitle_dir:
        input_video_info = video.get_video_data(input_video)
        default_video_track = input_video_info.video_tracks[0]
        fps = generic.fps_str_to_float(default_video_track.fps)

        if abs(fps - subs.ffmpeg_default_fps) < 1:
            raise RuntimeError("source video is not suitable, has nearly default fps")

        length = default_video_track.length

        subtitle_path = f"{subtitle_dir}/sub.sub"
        generate_microdvd_subtitles(subtitle_path, int(length), fps)

        process.start_process("ffmpeg", ["-hide_banner", "-i", input_video, "-i", subtitle_path, "-map", "0", "-map", "1", "-c:v", "copy", "-c:a", "copy", output_video_path])


class SubtitlesFixer(unittest.TestCase):

    def setUp(self):
        logging.getLogger().setLevel(logging.ERROR)

    def test_dry_run_is_respected(self):
        with WorkingDirectoryForTest() as td:
            output_video_path = f"{td.path}/test_video.mkv"
            create_broken_video_with_scaled_subtitle_timings(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

            hashes_before = hashes(td.path)
            run_twotone("subtitles_fix", [td.path])
            hashes_after = hashes(td.path)

            self.assertEqual(hashes_before, hashes_after)

    def test_video_with_scaled_subtitle_timings_fixing(self):
        with WorkingDirectoryForTest() as td:
            output_video_path = f"{td.path}/test_video.mkv"
            create_broken_video_with_scaled_subtitle_timings(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

            hashes_before = hashes(td.path)
            run_twotone("subtitles_fix", [td.path], ["-r"])
            hashes_after = hashes(td.path)

            self.assertNotEqual(hashes_before, hashes_after)

            # run again - there should be no changes
            run_twotone("subtitles_fix", [td.path], ["-r"])
            hashes_after_after = hashes(td.path)
            self.assertEqual(hashes_after, hashes_after_after)

    def test_video_with_too_long_last_subtitle_fixing(self):
        with WorkingDirectoryForTest() as td:
            output_video_path = f"{td.path}/test_video.mkv"
            create_broken_video_with_too_long_last_subtitle(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

            hashes_before = hashes(td.path)
            run_twotone("subtitles_fix", [td.path], ["-r"])
            hashes_after = hashes(td.path)

            self.assertNotEqual(hashes_before, hashes_after)

            # run again - there should be no changes
            run_twotone("subtitles_fix", [td.path], ["-r"])
            hashes_after_after = hashes(td.path)
            self.assertEqual(hashes_after, hashes_after_after)

    def test_deal_with_incompatible_videos(self):
        with WorkingDirectoryForTest() as td:
            output_video_path = f"{td.path}/test_video.mkv"
            create_broken_video_with_incompatible_subtitles(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

            hashes_before = hashes(td.path)
            run_twotone("subtitles_fix", [td.path], ["-r"])
            hashes_after = hashes(td.path)

            self.assertEqual(hashes_before, hashes_after)

if __name__ == '__main__':
    unittest.main()
