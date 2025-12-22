
import os
import unittest
import tempfile

import pysubs2

from twotone.tools.utils import subtitles_utils, video_utils

from common import (
    TwoToneTestCase,
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
        default_video_track = input_video_info["video"][0]
        fps = generic_utils.fps_str_to_float(default_video_track["fps"])

        if abs(fps - subtitles_utils.ffmpeg_default_fps) < 1:
            raise RuntimeError("source video is not suitable, has nearly default fps")

        length = default_video_track["length"] / 1000

        subtitle_path = f"{subtitle_dir}/sub.sub"
        generate_microdvd_subtitles(subtitle_path, int(length), fps)

        # convert to srt format
        srt_subtitle_path = f"{subtitle_dir}/sub.srt"
        status = process_utils.start_process("ffmpeg", ["-y", "-i", subtitle_path, srt_subtitle_path])

        video_utils.generate_mkv(input_video = input_video, output_path = output_video_path, subtitles = [subtitles_utils.SubtitleFile(path = srt_subtitle_path, language = "eng", encoding = "utf8")])


def create_broken_video_with_too_long_last_subtitle(output_video_path: str, input_video: str):
    with tempfile.TemporaryDirectory() as subtitle_dir:
        input_video_info = video_utils.get_video_data(input_video)
        default_video_track = input_video_info["video"][0]
        length = default_video_track["length"]

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

        video_utils.generate_mkv(input_video = input_video, output_path = output_video_path, subtitles = [subtitles_utils.SubtitleFile(path = subtitle_path, language = "eng", encoding = "utf8")])


def create_broken_video_with_some_incompatible_subtitles(output_video_path: str, input_video: str):
    with tempfile.TemporaryDirectory() as subtitle_dir:
        input_video_info = video_utils.get_video_data(input_video)
        default_video_track = input_video_info["video"][0]
        fps = generic_utils.fps_str_to_float(default_video_track["fps"])

        if abs(fps - subtitles_utils.ffmpeg_default_fps) < 1:
            raise RuntimeError("source video is not suitable, has nearly default fps")

        length = default_video_track["length"]

        # Create 'broken' subtitles 10x longer than video lenght.
        # Using subtitles_utils.ffmpeg_default_fps as fps no matter the original video's fps due to ffmpeg bug (https://trac.ffmpeg.org/ticket/10929)
        subtitle_path1 = f"{subtitle_dir}/sub1.sub"
        generate_microdvd_subtitles(subtitle_path1, int(length/1000 * 10), subtitles_utils.ffmpeg_default_fps)

        # generate another, valid subtitles
        subtitle_path2 = f"{subtitle_dir}/sub2.sub"
        generate_microdvd_subtitles(subtitle_path2, int(length/ 1000), subtitles_utils.ffmpeg_default_fps)

        process_utils.start_process("ffmpeg", ["-i", input_video, "-i", subtitle_path1, "-i", subtitle_path2, "-map", "0", "-map", "1", "-map", "2", "-c:v", "copy", "-c:a", "copy", output_video_path])


def create_broken_video_with_metadata_rich_subtitles(output_video_path: str, input_video: str):
    with tempfile.TemporaryDirectory() as subtitle_dir:
        input_video_info = video_utils.get_video_data(input_video)
        video_length = input_video_info["video"][0]["length"]

        last_start = max(0, video_length - 2000)
        last_end = video_length * 2

        def _write(idx: int, lang: str | None, title: str, default: bool) -> subtitles_utils.SubtitleFile:
            path = os.path.join(subtitle_dir, f"sub_{idx}.srt")
            subs = pysubs2.SSAFile()
            subs.append(pysubs2.SSAEvent(start = 0, end = 1000, text = f"Intro {idx}"))
            subs.append(pysubs2.SSAEvent(start = last_start, end = last_end, text = f"Tail {idx}"))
            subs.save(path, format = "srt")
            return subtitles_utils.SubtitleFile(path = path, language = lang, encoding = "utf8", name = title, default = default)

        subtitles = [
            _write(1, "eng", "English commentary", False),
            _write(2, "pol", "Polish closed captions", True),
            _write(3, None, "No language track", False),
        ]

        video_utils.generate_mkv(input_video=input_video, output_path=output_video_path, subtitles=subtitles)


class SubtitlesFixer(TwoToneTestCase):
    def test_dry_run_is_respected(self):
        output_video_path = f"{self.wd.path}/test_video.mkv"
        create_broken_video_with_scaled_subtitle_timings(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

        hashes_before = hashes(self.wd.path)
        run_twotone("subtitles_fix", [self.wd.path])
        hashes_after = hashes(self.wd.path)

        self.assertEqual(hashes_before, hashes_after)

    def test_metadata_preserved(self):
        output_video_path = f"{self.wd.path}/test_video.mkv"
        create_broken_video_with_metadata_rich_subtitles(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

        def collect_subtitle_metadata(video_path: str):
            info = video_utils.get_video_data(video_path)
            subtitles = info.get("subtitle", [])
            video_length = info["video"][0]["length"]
            metadata = [{k: v for k, v in subtitle.items() if k not in ["length"]} for subtitle in subtitles]

            return metadata, video_length

        metadata_before, video_length = collect_subtitle_metadata(output_video_path)

        run_twotone("subtitles_fix", [self.wd.path], ["-r"])

        metadata_after, _ = collect_subtitle_metadata(output_video_path)

        self.assertEqual(len(metadata_before), len(metadata_after))
        for before, after in zip(metadata_before, metadata_after):
            self.assertEqual(before, after)

        # Ensure subtitles are no longer longer than the video itself
        fixed_info = video_utils.get_video_data(output_video_path)
        for track in fixed_info.get("subtitle", []):
            length = track.get("length")
            if length is not None:
                self.assertLessEqual(length, video_length * 1.001)

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

    def test_leave_unfixable_subtitles(self):
        output_video_path = f"{self.wd.path}/test_video.mkv"
        create_broken_video_with_some_incompatible_subtitles(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

        hashes_before = hashes(self.wd.path)
        run_twotone("subtitles_fix", [self.wd.path], ["-r"])
        hashes_after = hashes(self.wd.path)

        # by default unfixable subtitle should leave source file untouched
        self.assertEqual(hashes_before, hashes_after)

    def test_drop_unfixable_subtitles(self):
        output_video_path = f"{self.wd.path}/test_video.mkv"
        create_broken_video_with_some_incompatible_subtitles(output_video_path, f"{current_path}/videos/sea-waves-crashing-on-beach-shore-4793288.mp4")

        hashes_before = hashes(self.wd.path)
        run_twotone("subtitles_fix", ["-d", self.wd.path], ["-r"])
        hashes_after = hashes(self.wd.path)

        # -d option should drop unfixable subtitles
        self.assertEqual(len(hashes_after), 1)

        output_file = list(hashes_after.keys())[0]
        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["subtitle"]), 1)


if __name__ == '__main__':
    unittest.main()
