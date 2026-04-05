
import os
import re

from twotone.tools.utils import generic_utils, video_utils
from twotone.tools.melt.melt import StaticSource
from common import (
    TwoToneTestCase,
    FileCache,
    add_to_test_dir,
    get_video,
    hashes,
    run_ffmpeg,
)
from melt.helpers import (
    analyze_duplicates_helper,
    process_duplicates_helper,
)


class AudioAlignmentTest(TwoToneTestCase):
    """Verify that melt's audio patching produces correctly time-aligned output.

    Generates two test videos from Big Buck Bunny (trimmed to 15 s):
      - base: original with audio replaced by short beeps at known timestamps
      - degraded: sped up by 4 %, lower resolution and quality

    After melt merges them (allow_length_mismatch), the patched audio track
    from the degraded file should have its beeps aligned back to the base timeline.
    """

    BEEP_TIMES = [3.0, 7.0, 12.0]
    BEEP_DURATION = 0.3
    VIDEO_DURATION = 15
    SPEED = 1.04
    ALIGNMENT_TOLERANCE = 0.5  # seconds

    def setUp(self):
        super().setUp()
        file_cache = FileCache("TwoToneTests")
        bbb = get_video("big_buck_bunny_720p_10mb.mp4")
        beep_times = self.BEEP_TIMES
        beep_dur = self.BEEP_DURATION
        video_dur = self.VIDEO_DURATION
        speed = self.SPEED

        def gen_beep_video(out_path):
            beep_expr = "+".join(
                f"between(t\\,{t:.1f}\\,{t + beep_dur:.1f})"
                for t in beep_times
            )
            run_ffmpeg([
                "-y",
                "-t", str(video_dur),
                "-i", bbb,
                "-f", "lavfi", "-i",
                f"aevalsrc=0.5*sin(2*PI*440*t)*({beep_expr}):s=44100:d={video_dur}",
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "44100",
                str(out_path),
            ], expected_path=str(out_path))

        self.base_video = str(file_cache.get_or_generate(
            "audio_align_base", "1", "mp4", gen_beep_video,
        ))

        def gen_degraded(out_path):
            vf_parts = ["fps=26.5", f"setpts=PTS/{speed}", "scale=640:480", "boxblur=lr=1"]
            run_ffmpeg([
                "-y", "-i", self.base_video,
                "-vf", ",".join(vf_parts),
                "-af", f"atempo={speed}",
                "-c:v", "libx264", "-crf", "35", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(out_path),
            ], expected_path=str(out_path))

        self.degraded_video = str(file_cache.get_or_generate(
            "audio_align_deg", "1", "mp4", gen_degraded,
        ))

    @staticmethod
    def _detect_beep_times(audio_file: str) -> list[float]:
        """Detect non-silent events using ffmpeg silencedetect.

        Returns the start time of each detected beep.
        """
        result = run_ffmpeg([
            "-i", audio_file,
            "-af", "silencedetect=noise=-30dB:d=0.05",
            "-f", "null", "-",
        ])
        silence_ends = [
            float(m.group(1))
            for m in re.finditer(r'silence_end:\s*([\d.]+)', result.stderr)
        ]
        silence_starts = [
            float(m.group(1))
            for m in re.finditer(r'silence_start:\s*([\d.]+)', result.stderr)
        ]
        # A beep spans from a silence_end to the next silence_start.
        # The final silence_end (at EOF) has no following silence_start — filter it out.
        return [se for se in silence_ends if any(ss > se for ss in silence_starts)]

    def test_audio_alignment_after_melt(self):
        """Patched audio track beeps must land at the original base-timeline positions."""
        file1 = add_to_test_dir(self.wd.path, self.base_video)
        file2 = add_to_test_dir(self.wd.path, self.degraded_video)

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)
        duplicates.add_metadata(file1, "audio_lang", "eng")
        duplicates.add_metadata(file2, "audio_lang", "pol")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(
            logger, duplicates, self.wd.path, allow_length_mismatch=True,
        )
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_files = list(hashes(output_dir).keys())
        self.assertEqual(len(output_files), 1)
        output_file = output_files[0]

        output_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_data["tracks"]["audio"]), 2)

        # Extract patched audio (pol — from the degraded file, sorted as 2nd track)
        patched_audio = os.path.join(self.wd.path, "patched_audio.wav")
        run_ffmpeg(
            ["-y", "-i", output_file, "-map", "0:a:1", patched_audio],
            expected_path=patched_audio,
        )

        detected = self._detect_beep_times(patched_audio)
        self.assertEqual(
            len(detected), len(self.BEEP_TIMES),
            f"Expected {len(self.BEEP_TIMES)} beeps in patched audio, "
            f"detected {len(detected)}: {detected}",
        )

        for expected, actual in zip(self.BEEP_TIMES, detected):
            self.assertAlmostEqual(
                expected, actual, delta=self.ALIGNMENT_TOLERANCE,
                msg=f"Beep expected at {expected}s, detected at {actual}s "
                    f"(offset: {actual - expected:+.3f}s)",
            )
