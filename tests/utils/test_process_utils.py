import subprocess
import unittest
from io import StringIO

from unittest.mock import Mock, call, patch

from twotone.tools.utils import process_utils, video_utils


class StartProcessTest(unittest.TestCase):
    def test_ffmpeg_audio_progress_uses_timestamps_without_probing_input(self):
        process = Mock(returncode=0)
        process.stderr = StringIO(
            "  Duration: 00:01:00.00, start: 0.000000, bitrate: 128 kb/s\n"
            "out_time_us=N/A\nout_time_us=-1000\n"
            "out_time_us=1500000\nout_time_us=1500000\n"
            "out_time_us=3000000\nwarning: example\n"
        )
        process.communicate.return_value = ("", "")
        logger = Mock()

        with patch.object(process_utils.subprocess, "Popen", return_value=process) as popen, \
             patch.object(process_utils, "tqdm") as progress, \
             patch.object(video_utils, "is_video", side_effect=AssertionError("Extra probe")), \
             patch.object(video_utils, "get_video_frames_count", side_effect=AssertionError("Extra scan")):
            result = process_utils.start_process(
                "ffmpeg", ["-i", "source.mkv", "-map", "0:a:0", "audio.flac"],
                show_progress=True, progress_description="Decoding audio", logger=logger,
            )

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-progress") + 1], "pipe:2")
        self.assertIn("-nostats", command)
        progress.assert_called_once_with(
            desc="Decoding audio", unit="s", total=None,
            **process_utils.generic_utils.get_tqdm_defaults(),
        )
        bar = progress.return_value.__enter__.return_value
        self.assertEqual(bar.total, 60.0)
        self.assertEqual(bar.update.call_args_list, [call(1.5), call(1.5)])
        self.assertIn("warning: example", result.stderr)
        logger.info.assert_any_call("%s: started.", "Decoding audio")

    def test_ffmpeg_video_progress_uses_output_time(self):
        process = Mock(returncode=0)
        process.stderr = StringIO("frame=24\nout_time_us=1000000\nprogress=end\n")
        process.communicate.return_value = ("", "")

        with patch.object(process_utils.subprocess, "Popen", return_value=process), \
             patch.object(process_utils, "tqdm") as progress:
            process_utils.start_process(
                "ffmpeg", ["-i", "source.mkv", "output.mkv"], show_progress=True,
            )

        progress.return_value.__enter__.return_value.update.assert_called_once_with(1.0)

    def test_ffmpeg_concat_progress_without_known_duration(self):
        process = Mock(returncode=1)
        process.stderr = StringIO("Duration: N/A\nout_time_us=2000000\nencoding failed\n")
        process.communicate.return_value = ("", "")

        with patch.object(process_utils.subprocess, "Popen", return_value=process), \
             patch.object(process_utils, "tqdm") as progress:
            result = process_utils.start_process(
                "ffmpeg", ["-f", "concat", "-i", "parts.txt", "audio.mka"],
                show_progress=True,
            )

        progress.return_value.__enter__.return_value.update.assert_called_once_with(2.0)
        self.assertEqual(result.returncode, 1)
        self.assertIn("encoding failed", result.stderr)

    def test_ffprobe_progress_drains_output_while_process_is_running(self):
        process = Mock(returncode=0)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["ffprobe"], 0.1),
            ('{"streams": []}', ""),
        ]
        logger = Mock()

        with patch.object(process_utils.subprocess, "Popen", return_value=process), \
             patch.object(process_utils, "tqdm"):
            result = process_utils.start_process(
                "ffprobe",
                [],
                show_progress=True,
                logger=logger,
            )

        self.assertEqual(result, process_utils.ProcessResult(0, '{"streams": []}', ""))
        self.assertEqual(process.communicate.call_args_list, [call(timeout=0.1), call(timeout=0.1)])
        logger.debug.assert_any_call("%s: started.", "Probing media")


if __name__ == "__main__":
    unittest.main()
