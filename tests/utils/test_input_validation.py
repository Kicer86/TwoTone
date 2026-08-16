import logging
import os
import tempfile
import unittest

from unittest.mock import patch

from twotone.tools.utils import input_validation, process_utils


class InputValidatorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = os.path.join(self.temp_dir.name, "input.mkv")
        with open(self.path, "wb") as file:
            file.write(b"media")
        self.logger = logging.getLogger("InputValidatorTest")

    def test_full_validation_decodes_audio_and_video(self):
        validator = input_validation.InputValidator(
            input_validation.ValidationMode.FULL,
            self.logger,
            self.temp_dir.name,
        )
        probe_success = process_utils.ProcessResult(0, '{"streams": [{"codec_type": "audio", "codec_name": "ac3"}]}', "")
        decode_success = process_utils.ProcessResult(
            0,
            "",
            "frame=10 fps=0.0 q=-0.0 Lsize=N/A time=00:00:01.00 bitrate=N/A",
        )

        with patch.object(process_utils, "start_process", side_effect=[probe_success, decode_success]) as start_process:
            report = validator.validate([self.path])

        self.assertTrue(report.is_valid)
        self.assertEqual(report.checked_count, 1)
        self.assertEqual(report.cached_count, 0)
        self.assertEqual([call.args[0] for call in start_process.call_args_list], ["ffprobe", "ffmpeg"])
        self.assertTrue(start_process.call_args_list[0].kwargs["show_progress"])
        self.assertIn("-xerror", start_process.call_args_list[1].args[1])

    def test_cached_failure_is_reported_without_running_media_tools_again(self):
        validator = input_validation.InputValidator(
            input_validation.ValidationMode.FULL,
            self.logger,
            self.temp_dir.name,
        )
        probe_success = process_utils.ProcessResult(
            0,
            '{"streams": [{"index": 1, "codec_type": "audio", "codec_name": "ac3", "sample_rate": "48000", "channels": 2}]}',
            "",
        )
        decode_failure = process_utils.ProcessResult(
            1,
            "",
            "[ac3] incomplete frame\nError submitting packet to decoder: Invalid data found when processing input\nNothing was written into output file\n",
        )

        with patch.object(process_utils, "start_process", side_effect=[probe_success, decode_failure]):
            first = validator.validate([self.path])
        with patch.object(process_utils, "start_process") as start_process:
            second = validator.validate([self.path])

        self.assertFalse(first.is_valid)
        self.assertIn("audio #1: ac3 48000 Hz 2 channels", first.issues[0].message)
        self.assertIn("[ac3] incomplete frame", first.issues[0].message)
        self.assertIn("Invalid data found", first.issues[0].message)
        with self.assertLogs(self.logger, "ERROR") as logs:
            first.render(self.logger)
        self.assertNotIn("Suggested repair", "\n".join(logs.output))
        self.assertFalse(second.is_valid)
        self.assertEqual(second.checked_count, 0)
        self.assertEqual(second.cached_count, 1)
        start_process.assert_not_called()

    def test_full_validation_reports_ffmpeg_error_even_when_process_succeeds(self):
        validator = input_validation.InputValidator(
            input_validation.ValidationMode.FULL,
            self.logger,
            self.temp_dir.name,
        )
        probe_success = process_utils.ProcessResult(
            0,
            '{"streams": [{"codec_type": "video", "codec_name": "h264"}]}',
            "",
        )
        decode_with_error = process_utils.ProcessResult(
            0,
            "",
            "[in#0/matroska,webm] File ended prematurely\nframe=10 fps=0.0",
        )

        with patch.object(process_utils, "start_process", side_effect=[probe_success, decode_with_error]):
            report = validator.validate([self.path])

        self.assertFalse(report.is_valid)
        self.assertIn("File ended prematurely", report.issues[0].message)

    def test_fast_validation_does_not_decode_payload(self):
        validator = input_validation.InputValidator(
            input_validation.ValidationMode.FAST,
            self.logger,
            self.temp_dir.name,
        )
        success = process_utils.ProcessResult(0, '{"streams": []}', "")

        with patch.object(process_utils, "start_process", return_value=success) as start_process:
            report = validator.validate([self.path])

        self.assertTrue(report.is_valid)
        start_process.assert_called_once()
        self.assertEqual(start_process.call_args.args[0], "ffprobe")

    def test_validation_logs_progress_and_summary(self):
        validator = input_validation.InputValidator(
            input_validation.ValidationMode.FAST,
            self.logger,
            self.temp_dir.name,
        )
        success = process_utils.ProcessResult(0, '{"streams": []}', "")

        with patch.object(process_utils, "start_process", return_value=success), \
             self.assertLogs(self.logger, "INFO") as logs:
            validator.validate([self.path])

        self.assertIn("Validating 1 input file(s) with fast validation.", logs.output[0])
        self.assertIn("Input validation 1/1: checking", logs.output[1])
        self.assertIn("Input validation complete: 1 checked, 0 cached, 0 issue(s).", logs.output[2])
