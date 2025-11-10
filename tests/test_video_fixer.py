import os
import unittest
import tempfile

from twotone.tools.utils import video_utils, process_utils, subtitles_utils

from common import (
    TwoToneTestCase,
    hashes,
    run_twotone,
    generate_subrip_subtitles,
)


def create_video_with_mismatched_subtitles(wd: str, *, video_duration_ms: int = 2000, long_subtitle_ms: int = 5000):
    """
    Create a simple video with two subtitle tracks: one matching video duration and one much longer.
    This ensures VideoFixer flags and removes the abnormal subtitle track.
    """

    base_video = os.path.join(wd, "base.mp4")

    # Generate a plain black video of the requested duration
    process_utils.start_process(
        "ffmpeg",
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=128x128:d={video_duration_ms/1000}",
            "-pix_fmt",
            "yuv420p",
            base_video,
        ],
    )

    # Create two SRT subtitle files: one aligned with the video and one much longer
    srt_ok = os.path.join(wd, "ok.srt")
    srt_long = os.path.join(wd, "long.srt")
    generate_subrip_subtitles(srt_ok, video_duration_ms)
    generate_subrip_subtitles(srt_long, long_subtitle_ms)

    # Build final MKV with both subtitle tracks
    output_path = os.path.join(wd, "test_with_long_subs.mkv")
    video_utils.generate_mkv(
        output_path = output_path,
        input_video = base_video,
        subtitles=[
            subtitles_utils.SubtitleFile(srt_ok, "eng", "utf8"),
            subtitles_utils.SubtitleFile(srt_long, "pol", "utf8"),
        ],
    )

    return output_path


class VideoFixerTests(TwoToneTestCase):
    def test_dry_run_is_respected(self):
        create_video_with_mismatched_subtitles(self.wd.path)

        before = hashes(self.wd.path)
        run_twotone("video_fix", [self.wd.path])
        after = hashes(self.wd.path)

        self.assertEqual(before, after)

    def test_removes_abnormal_subtitles(self):
        output_video_path = create_video_with_mismatched_subtitles(self.wd.path)

        # Sanity: ensure we start with two subtitle tracks
        info_before = video_utils.get_video_data(output_video_path)
        self.assertIn("subtitle", info_before)
        self.assertEqual(len(info_before["subtitle"]), 2)

        run_twotone("video_fix", [self.wd.path], ["-r"])  # live run

        info_after = video_utils.get_video_data(output_video_path)
        self.assertIn("subtitle", info_after)
        self.assertEqual(len(info_after["subtitle"]), 1)


if __name__ == '__main__':
    unittest.main()
