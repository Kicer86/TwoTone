import logging
import os
import tempfile
import unittest
from unittest.mock import Mock

from twotone.tools.utils import input_validation, media_analysis


class InputValidationPolicyTest(unittest.TestCase):
    def test_mode_defines_validation_requirements(self):
        cases = (
            (input_validation.ValidationMode.OFF, False, set(), False),
            (input_validation.ValidationMode.FAST, True, {"ffprobe"}, False),
            (input_validation.ValidationMode.FULL, True, {"ffmpeg", "ffprobe"}, True),
        )

        for mode, enabled, required_tools, validate_all_streams in cases:
            with self.subTest(mode=mode):
                policy = input_validation.InputValidationPolicy(mode)

                self.assertEqual(policy.enabled, enabled)
                self.assertEqual(policy.required_tools(), required_tools)
                self.assertEqual(policy.validate_all_streams, validate_all_streams)


class InputValidatorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = os.path.join(self.temp_dir.name, "input.mkv")
        with open(self.path, "wb") as file:
            file.write(b"media")
        self.logger = logging.getLogger("InputValidatorTest")
        self.media_analysis = Mock(spec=media_analysis.MediaAnalysisSession)
        self.media_analysis.result_for.return_value = None

    def _validator(
        self,
        mode: input_validation.ValidationMode,
    ) -> input_validation.InputValidator:
        return input_validation.InputValidator(
            input_validation.InputValidationPolicy(mode),
            self.logger,
            self.temp_dir.name,
            media_analysis_session=self.media_analysis,
        )

    def _probe(
        self,
        streams: list[dict] | None = None,
        error: str | None = None,
    ) -> media_analysis.MediaProbeResult:
        result = media_analysis.MediaProbeResult(
            path=os.path.realpath(self.path),
            data={"streams": streams or []},
            error=error,
        )
        self.media_analysis.probe.return_value = result
        return result

    def _scan(self, decode_error: str | None = None) -> media_analysis.VideoScanResult:
        result = media_analysis.VideoScanResult(
            path=os.path.realpath(self.path),
            features=media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
            frames={},
            scene_changes=(),
            identity_samples=(),
            decode_error=decode_error,
        )
        self.media_analysis.fulfill.return_value = result
        return result

    def test_full_validation_requests_probe_and_stream_decode_from_media_analysis(self):
        self._probe([{"codec_type": "audio", "codec_name": "ac3"}])
        self._scan()

        report = self._validator(input_validation.ValidationMode.FULL).validate([self.path])

        self.assertTrue(report.is_valid)
        self.assertEqual(report.checked_count, 1)
        self.assertEqual(report.cached_count, 0)
        self.media_analysis.probe.assert_called_once_with(os.path.realpath(self.path))
        self.media_analysis.fulfill.assert_called_once_with(
            media_analysis.MediaAnalysisRequest(
                path=os.path.realpath(self.path),
                label=self.path,
                features=media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
            ),
        )

    def test_full_validation_accepts_successful_decode_from_media_analysis(self):
        scan = self._scan()
        self.media_analysis.result_for.return_value = scan
        self._probe([{"codec_type": "video", "codec_name": "h264"}])

        report = self._validator(input_validation.ValidationMode.FULL).validate([self.path])

        self.assertTrue(report.is_valid)
        self.media_analysis.fulfill.assert_called_once()

    def test_full_validation_reports_decode_error_from_media_analysis(self):
        self._scan("Invalid data found when processing input")
        self._probe([{"index": 0, "codec_type": "video", "codec_name": "h264"}])

        report = self._validator(input_validation.ValidationMode.FULL).validate([self.path])

        self.assertFalse(report.is_valid)
        self.assertIn("Invalid data found", report.issues[0].message)

    def test_full_validation_does_not_treat_partial_analysis_as_cached_validation(self):
        self.media_analysis.result_for.return_value = media_analysis.VideoScanResult(
            path=self.path,
            features=media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            frames={},
            scene_changes=(),
            identity_samples=(),
            decode_error=None,
        )
        self._probe([{"codec_type": "video", "codec_name": "h264"}])
        self._scan()

        report = self._validator(input_validation.ValidationMode.FULL).validate([self.path])

        self.assertTrue(report.is_valid)
        self.media_analysis.fulfill.assert_called_once()

    def test_analysis_decode_error_replaces_stale_cached_success(self):
        successful_scan = self._scan()
        self.media_analysis.result_for.return_value = successful_scan
        self._probe([{"index": 0, "codec_type": "video", "codec_name": "h264"}])
        validator = self._validator(input_validation.ValidationMode.FULL)

        first = validator.validate([self.path])

        failed_scan = self._scan("Invalid data found when processing input")
        self.media_analysis.result_for.return_value = failed_scan
        second = validator.validate([self.path])

        self.assertTrue(first.is_valid)
        self.assertFalse(second.is_valid)
        self.assertEqual(second.cached_count, 0)
        self.assertIn("Invalid data found", second.issues[0].message)

    def test_cached_failure_is_reported_without_querying_media_analysis_again(self):
        self._probe([{
            "index": 1,
            "codec_type": "audio",
            "codec_name": "ac3",
            "sample_rate": "48000",
            "channels": 2,
        }])
        self._scan(
            "[ac3] incomplete frame | Error submitting packet to decoder: "
            "Invalid data found when processing input"
        )
        validator = self._validator(input_validation.ValidationMode.FULL)

        with self.assertLogs(self.logger, "INFO") as fresh_logs:
            first = validator.validate([self.path])
        self.media_analysis.reset_mock()
        with self.assertLogs(self.logger, "INFO") as saved_logs:
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
        self.media_analysis.probe.assert_not_called()
        self.media_analysis.fulfill.assert_not_called()
        self.assertIn(f"Input {self.path} is invalid.", "\n".join(fresh_logs.output))
        saved_output = "\n".join(saved_logs.output)
        self.assertIn(
            "Loaded validation results from an earlier run for 1 unchanged input file.",
            saved_output,
        )
        self.assertIn(f"Input {self.path} is invalid.", saved_output)
        self.assertNotIn("previously checked", saved_output)

    def test_probe_error_is_reported_without_stream_decode(self):
        self._probe(error="Invalid data found when processing input")

        report = self._validator(input_validation.ValidationMode.FULL).validate([self.path])

        self.assertFalse(report.is_valid)
        self.assertIn("Invalid data found", report.issues[0].message)
        self.media_analysis.fulfill.assert_not_called()

    def test_fast_validation_only_requests_probe(self):
        self._probe()

        report = self._validator(input_validation.ValidationMode.FAST).validate([self.path])

        self.assertTrue(report.is_valid)
        self.media_analysis.probe.assert_called_once()
        self.media_analysis.fulfill.assert_not_called()

    def test_validation_logs_progress_and_summary(self):
        self._probe()
        validator = self._validator(input_validation.ValidationMode.FAST)

        with self.assertLogs(self.logger, "INFO") as logs:
            validator.validate([self.path])

        self.assertIn("Checking metadata for 1 input file.", logs.output[0])
        self.assertIn(f"Checking input {self.path}.", logs.output[1])
        self.assertIn(f"Input {self.path} is valid.", logs.output[2])
        self.assertIn("All input files are valid.", logs.output[3])

    def test_cached_success_loads_the_previous_result_before_reporting_it(self):
        self._probe()
        validator = self._validator(input_validation.ValidationMode.FAST)
        validator.validate([self.path])

        with self.assertLogs(self.logger, "INFO") as logs:
            report = validator.validate([self.path])

        self.assertTrue(report.is_valid)
        self.assertEqual(report.checked_count, 0)
        self.assertEqual(report.cached_count, 1)
        output = "\n".join(logs.output)
        self.assertIn(
            "Loaded validation results from an earlier run for 1 unchanged input file.",
            output,
        )
        self.assertIn(f"Input {self.path} is valid.", output)
        self.assertNotIn("previously checked", output)
        self.assertNotIn("using cached result", output)

    def test_mixed_sources_are_announced_before_results_are_reported(self):
        second_path = os.path.join(self.temp_dir.name, "second.mkv")
        with open(second_path, "wb") as file:
            file.write(b"other media")
        self._probe()
        validator = self._validator(input_validation.ValidationMode.FAST)
        validator.validate([self.path])

        with self.assertLogs(self.logger, "INFO") as logs:
            report = validator.validate([self.path, second_path])

        self.assertTrue(report.is_valid)
        self.assertEqual(report.checked_count, 1)
        self.assertEqual(report.cached_count, 1)
        messages = [record.getMessage() for record in logs.records]
        self.assertEqual(
            messages[:3],
            [
                "Loaded validation results from an earlier run for 1 unchanged input file.",
                "Checking metadata for 1 input file.",
                f"Checking input {second_path}.",
            ],
        )
        self.assertEqual(
            messages[3:],
            [
                f"Input {self.path} is valid.",
                f"Input {second_path} is valid.",
                "All input files are valid.",
            ],
        )

    def test_failed_validation_logs_the_outcome_without_a_success_summary(self):
        self._probe(error="Invalid data found when processing input")
        validator = self._validator(input_validation.ValidationMode.FULL)

        with self.assertLogs(self.logger, "INFO") as logs:
            report = validator.validate([self.path])

        self.assertFalse(report.is_valid)
        output = "\n".join(logs.output)
        self.assertIn("Checking 1 input file for media errors.", output)
        self.assertIn(f"Input {self.path} is invalid.", output)
        self.assertNotIn("All input files are valid.", output)

    def test_uses_the_supplied_reference_in_logs_and_media_analysis(self):
        self._probe([{"codec_type": "audio", "codec_name": "ac3"}])
        self._scan()
        target = input_validation.InputValidationTarget(self.path, "#7")

        with self.assertLogs(self.logger, "INFO") as logs:
            report = self._validator(input_validation.ValidationMode.FULL).validate([target])

        self.assertTrue(report.is_valid)
        output = "\n".join(logs.output)
        self.assertIn("Checking input #7.", output)
        self.assertIn("Input #7 is valid.", output)
        self.assertNotIn(self.path, output)
        self.media_analysis.fulfill.assert_called_once_with(
            media_analysis.MediaAnalysisRequest(
                path=os.path.realpath(self.path),
                label="#7",
                features=media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
            ),
        )

    def test_uses_the_supplied_reference_in_error_reports(self):
        self._probe(error="Invalid data found when processing input")
        target = input_validation.InputValidationTarget(self.path, "#7")

        report = self._validator(input_validation.ValidationMode.FULL).validate([target])

        with self.assertLogs(self.logger, "ERROR") as logs:
            report.render(self.logger)
        output = "\n".join(logs.output)
        self.assertIn("Input validation failed for #7", output)
        self.assertNotIn(self.path, output)


if __name__ == "__main__":
    unittest.main()
