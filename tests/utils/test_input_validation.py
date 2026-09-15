import logging
import os
import tempfile
import unittest

from unittest.mock import Mock

from twotone.tools.utils import input_validation, media_analysis


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
            mode,
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
        self.media_analysis.validate_streams.return_value = result
        return result

    def test_full_validation_requests_probe_and_stream_decode_from_media_analysis(self):
        self._probe([{"codec_type": "audio", "codec_name": "ac3"}])
        self._scan()

        report = self._validator(input_validation.ValidationMode.FULL).validate([self.path])

        self.assertTrue(report.is_valid)
        self.assertEqual(report.checked_count, 1)
        self.assertEqual(report.cached_count, 0)
        self.media_analysis.probe.assert_called_once_with(os.path.realpath(self.path))
        self.media_analysis.validate_streams.assert_called_once_with(
            os.path.realpath(self.path),
            label=os.path.realpath(self.path),
        )

    def test_full_validation_accepts_successful_decode_from_media_analysis(self):
        scan = self._scan()
        self.media_analysis.result_for.return_value = scan
        self._probe([{"codec_type": "video", "codec_name": "h264"}])

        report = self._validator(input_validation.ValidationMode.FULL).validate([self.path])

        self.assertTrue(report.is_valid)
        self.media_analysis.validate_streams.assert_called_once()

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
        self.media_analysis.validate_streams.assert_called_once()

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

        first = validator.validate([self.path])
        self.media_analysis.reset_mock()
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
        self.media_analysis.validate_streams.assert_not_called()

    def test_probe_error_is_reported_without_stream_decode(self):
        self._probe(error="Invalid data found when processing input")

        report = self._validator(input_validation.ValidationMode.FULL).validate([self.path])

        self.assertFalse(report.is_valid)
        self.assertIn("Invalid data found", report.issues[0].message)
        self.media_analysis.validate_streams.assert_not_called()

    def test_fast_validation_only_requests_probe(self):
        self._probe()

        report = self._validator(input_validation.ValidationMode.FAST).validate([self.path])

        self.assertTrue(report.is_valid)
        self.media_analysis.probe.assert_called_once()
        self.media_analysis.validate_streams.assert_not_called()

    def test_validation_logs_progress_and_summary(self):
        self._probe()
        validator = self._validator(input_validation.ValidationMode.FAST)

        with self.assertLogs(self.logger, "INFO") as logs:
            validator.validate([self.path])

        self.assertIn("Validating 1 input file(s) with fast validation.", logs.output[0])
        self.assertIn("Input validation 1/1: checking", logs.output[1])
        self.assertIn("Input validation complete: 1 checked, 0 cached, 0 issue(s).", logs.output[2])


if __name__ == "__main__":
    unittest.main()
