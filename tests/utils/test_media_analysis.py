import logging
import os
import tempfile
import unittest

from types import SimpleNamespace
from unittest.mock import Mock, patch

from twotone.tools.utils import files_utils, generic_utils, media_analysis, process_utils, video_utils


class MediaAnalysisSessionTest(unittest.TestCase):
    def setUp(self):
        self.workspace = files_utils.Workspace.temporary()
        self.addCleanup(self.workspace.close)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = os.path.join(self.temp_dir.name, "input.mkv")
        with open(self.path, "wb") as file:
            file.write(b"media")

        self.session = media_analysis.MediaAnalysisSession(
            self.workspace,
            generic_utils.InterruptibleProcess(),
            logging.getLogger("MediaAnalysisSessionTest"),
            validate_all_streams=True,
        )

    def _probe_result(self, streams: list[dict] | None = None) -> media_analysis.MediaProbeResult:
        return media_analysis.MediaProbeResult(
            path=os.path.realpath(self.path),
            data={
                "streams": [{"codec_type": "video"}] if streams is None else streams,
            },
            error=None,
        )

    def test_probe_reuses_result_for_the_same_unchanged_file(self):
        success = process_utils.ProcessResult(
            0,
            '{"streams": [{"codec_type": "video", "codec_name": "h264"}]}',
            "",
        )

        with patch.object(process_utils, "start_process", return_value=success) as start_process:
            first = self.session.probe(self.path)
            second = self.session.probe(self.path)

        self.assertIs(first, second)
        self.assertTrue(first.has_video)
        start_process.assert_called_once()

    def test_validation_only_decode_is_run_by_the_media_session(self):
        def fake_start(args, _interruption, on_line, logger):
            del on_line, logger
            self.assertIn("-xerror", args)
            self.assertIn("0:a?", args)
            self.assertNotIn("-filter_complex", args)
            return SimpleNamespace(returncode=0), []

        with patch.object(self.session, "probe", return_value=self._probe_result()), \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_start) as start, \
             patch.object(video_utils, "_showinfo_timestamp_correction_ms") as timestamp_correction:
            result = self.session.validate_streams(self.path)

        start.assert_called_once()
        timestamp_correction.assert_not_called()
        self.assertTrue(result.validated_all_streams)
        self.assertIsNone(result.decode_error)

    def test_validation_skips_decode_when_probe_reports_no_audio_or_video(self):
        with patch.object(self.session, "probe", return_value=self._probe_result([])), \
             patch.object(video_utils, "_start_ffmpeg_streaming") as start:
            result = self.session.validate_streams(self.path)

        start.assert_not_called()
        self.assertTrue(result.validated_all_streams)
        self.assertIsNone(result.decode_error)

    def test_reuses_one_scan_for_the_same_unchanged_file(self):
        result = media_analysis.VideoScanResult(
            path=os.path.realpath(self.path),
            features=(
                media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES
                | media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS
            ),
            frames={},
            scene_changes=(),
            identity_samples=(),
            decode_error=None,
        )

        with patch.object(self.session, "_scan", return_value=result) as scan:
            first = self.session.scan(
                self.path,
                duration_ms=1000,
                fps=25.0,
                label="#1",
                features=media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            )
            second = self.session.scan(
                self.path,
                duration_ms=1000,
                fps=25.0,
                label="#2",
                features=media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            )

        self.assertIs(first, second)
        scan.assert_called_once()

    def test_upgrades_cached_scan_with_only_missing_features(self):
        identity = media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES
        matching = media_analysis.MediaAnalysisFeature.MATCHING
        validation = media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS

        def scan(_path, _duration_ms, _fps, _label, features):
            return media_analysis.VideoScanResult(
                path=os.path.realpath(self.path),
                features=features,
                frames={40: {"frame_id": 1, "path": None}} if features & matching else {},
                scene_changes=(40,) if features & matching else (),
                identity_samples=(
                    media_analysis.VideoSample(0, 0, 0, "/sample.png"),
                ) if features & identity else (),
                decode_error=None,
            )

        with patch.object(self.session, "_scan", side_effect=scan) as scan_mock:
            first = self.session.scan(
                self.path,
                duration_ms=1000,
                fps=25.0,
                label="#1",
                features=identity,
            )
            upgraded = self.session.scan(
                self.path,
                duration_ms=1000,
                fps=25.0,
                label="#1",
                features=matching,
            )
            restored = self.session.scan(
                self.path,
                duration_ms=1000,
                fps=25.0,
                label="#2",
                features=identity | matching,
            )

        self.assertEqual(first.features, identity | validation)
        self.assertEqual(upgraded.features, identity | matching | validation)
        self.assertEqual(upgraded.identity_samples, first.identity_samples)
        self.assertEqual(upgraded.scene_changes, (40,))
        self.assertEqual(list(upgraded.frames), [40])
        self.assertIs(restored, upgraded)
        self.assertEqual(
            [call.args[-1] for call in scan_mock.call_args_list],
            [identity | validation, matching],
        )

    def test_collects_frames_scenes_samples_and_validation_in_one_ffmpeg_call(self):
        def fake_start(args, _interruption, on_line, logger):
            stats_options = [
                index for index, value in enumerate(args)
                if value == "-stats_enc_pre:v:0"
            ]
            self.assertEqual(len(stats_options), 2)
            frame_stats = args[stats_options[0] + 1]
            sample_stats = args[stats_options[1] + 1]

            with open(frame_stats, "w", encoding="utf-8") as file:
                file.write("0 0.000\n1 0.040\n2 0.080\n")
            with open(sample_stats, "w", encoding="utf-8") as file:
                file.write("0 0.000\n1 0.080\n")

            output_pattern = next(value for value in args if "identity_%08d.png" in value)
            for index in (1, 2):
                with open(output_pattern.replace("%08d", f"{index:08d}"), "wb") as file:
                    file.write(b"png")

            on_line("frame:0 pts:2 pts_time:0.080\n")
            on_line("lavfi.scene_score=0.75\n")
            on_line("out_time_ms=80000\n")
            on_line("progress=end\n")
            return SimpleNamespace(returncode=0), []

        with patch.object(self.session, "probe", return_value=self._probe_result()), \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_start) as start, \
             patch.object(video_utils, "_showinfo_timestamp_correction_ms", return_value=0):
            result = self.session.scan(
                self.path,
                duration_ms=120,
                fps=25.0,
                label="#1",
                features=(
                    media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES
                    | media_analysis.MediaAnalysisFeature.MATCHING
                ),
            )

        start.assert_called_once()
        args = start.call_args.args[0]
        self.assertIn("-xerror", args)
        self.assertIn("0:a?", args)
        self.assertIn("split=3", " ".join(args))
        stats_formats = [
            args[index + 1]
            for index, value in enumerate(args)
            if value == "-stats_enc_pre_fmt:v:0"
        ]
        self.assertEqual(stats_formats, ["{ni} {ti}", "{ni} {ti}"])
        self.assertEqual(result.scene_changes, (80,))
        self.assertEqual(list(result.frames), [0, 40, 80])
        self.assertEqual(
            [(sample.timestamp_ms, sample.frame_id) for sample in result.identity_samples],
            [(0, 0)] * 4 + [(80, 2)] * 3,
        )
        self.assertTrue(result.validated_all_streams)
        self.assertIsNone(result.decode_error)

    def test_scene_only_scan_has_a_mapped_filter_output(self):
        session = media_analysis.MediaAnalysisSession(
            self.workspace,
            generic_utils.InterruptibleProcess(),
            logging.getLogger("SceneOnlyMediaAnalysisTest"),
            validate_all_streams=False,
        )

        def fake_start(args, _interruption, on_line, logger):
            del on_line, logger
            self.assertIn("[scanout]", args)
            self.assertIn("split=2[vscenes][voutput]", " ".join(args))
            return SimpleNamespace(returncode=0), []

        with patch.object(session, "probe", return_value=self._probe_result()), \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_start):
            result = session.scan(
                self.path,
                duration_ms=1000,
                fps=25.0,
                label="#1",
                features=media_analysis.MediaAnalysisFeature.SCENE_CHANGES,
            )

        self.assertEqual(result.scene_changes, ())
        self.assertIsNone(result.decode_error)

    def test_identity_samples_can_reuse_a_sparse_frame(self):
        samples = media_analysis.MediaAnalysisSession._build_samples(
            (0, 250, 500, 750, 1000),
            [(0, 0), (1, 1000)],
            ["/first.png", "/last.png"],
            {},
        )

        self.assertEqual(len(samples), 5)
        self.assertEqual(
            [(sample.timestamp_ms, sample.path) for sample in samples],
            [
                (0, "/first.png"),
                (0, "/first.png"),
                (0, "/first.png"),
                (1000, "/last.png"),
                (1000, "/last.png"),
            ],
        )

    def test_matching_scan_restores_and_updates_persistent_cache(self):
        persistent = Mock()
        persistent.load_scene_changes.return_value = [120]
        persistent.load_frame_probes.return_value = None
        self.session.set_persistent_cache(persistent)
        scanned = media_analysis.VideoScanResult(
            path=os.path.realpath(self.path),
            features=media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
            frames={0: {"frame_id": 0, "path": "/temporary/sample.png"}},
            scene_changes=(),
            identity_samples=(),
            decode_error=None,
        )

        with patch.object(self.session, "_scan", return_value=scanned) as scan:
            result = self.session.scan(
                self.path,
                duration_ms=1000,
                fps=25.0,
                label="#1",
                features=media_analysis.MediaAnalysisFeature.MATCHING,
            )

        scan.assert_called_once_with(
            os.path.realpath(self.path),
            1000,
            25.0,
            "#1",
            media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS
            | media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
        )
        self.assertEqual(result.scene_changes, (120,))
        self.assertEqual(list(result.frames), [0])
        persistent.save_frame_probes.assert_called_once_with(
            os.path.realpath(self.path),
            {0: {"frame_id": 0, "path": None}},
        )
        persistent.save_scene_changes.assert_not_called()

    def test_complete_persistent_matching_cache_avoids_a_scan(self):
        session = media_analysis.MediaAnalysisSession(
            self.workspace,
            generic_utils.InterruptibleProcess(),
            logging.getLogger("PersistentMediaAnalysisTest"),
            validate_all_streams=False,
        )
        persistent = Mock()
        persistent.load_scene_changes.return_value = [120]
        persistent.load_frame_probes.return_value = {
            0: {"frame_id": 0, "path": None},
        }
        session.set_persistent_cache(persistent)

        with patch.object(session, "_scan") as scan:
            result = session.scan(
                self.path,
                duration_ms=1000,
                fps=25.0,
                label="#1",
                features=media_analysis.MediaAnalysisFeature.MATCHING,
            )

        scan.assert_not_called()
        self.assertEqual(result.scene_changes, (120,))
        self.assertEqual(list(result.frames), [0])

    def test_identity_scan_does_not_collect_matching_data(self):
        def fake_start(args, _interruption, on_line, logger):
            del on_line, logger
            stats_options = [
                index for index, value in enumerate(args)
                if value == "-stats_enc_pre:v:0"
            ]
            self.assertEqual(len(stats_options), 1)
            sample_stats = args[stats_options[0] + 1]
            with open(sample_stats, "w", encoding="utf-8") as file:
                file.write("0 0.000\n1 0.080\n")

            output_pattern = next(value for value in args if "identity_%08d.png" in value)
            for index in (1, 2):
                with open(output_pattern.replace("%08d", f"{index:08d}"), "wb") as file:
                    file.write(b"png")

            return SimpleNamespace(returncode=0), []

        with patch.object(self.session, "probe", return_value=self._probe_result()), \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_start) as start, \
             patch.object(video_utils, "_showinfo_timestamp_correction_ms", return_value=0):
            result = self.session.scan(
                self.path,
                duration_ms=120,
                fps=25.0,
                label="#1",
                features=media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            )

        args = start.call_args.args[0]
        self.assertIn("-xerror", args)
        self.assertIn("split=2[vvalidate][vsamples]", " ".join(args))
        self.assertIn("[vvalidate]", args)
        self.assertNotIn("frames.txt", " ".join(args))
        self.assertNotIn("gt(scene", " ".join(args))
        self.assertEqual(
            result.features,
            media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES
            | media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
        )
        self.assertEqual(result.frames, {})
        self.assertEqual(result.scene_changes, ())
        self.assertEqual(len(result.identity_samples), 7)


if __name__ == "__main__":
    unittest.main()
