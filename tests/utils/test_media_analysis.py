import logging
import os
import tempfile
import unittest

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

    def test_scan_reuses_probe_for_negative_timestamp_correction(self):
        probe_result = process_utils.ProcessResult(
            0,
            '{"format": {"start_time": "-0.020"}, "streams": [{"codec_type": "video"}]}',
            "",
        )

        def fake_ffmpeg(_args, _interruption, on_line, logger):
            del logger
            on_line("frame:0 pts:2 pts_time:0.080\n")
            return Mock(returncode=0), []

        with patch.object(process_utils, "start_process", return_value=probe_result) as start_process, \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_ffmpeg), \
             patch.object(video_utils, "_showinfo_timestamp_correction_ms") as old_probe:
            result = self.session.scan(
                self.path,
                duration_ms=120,
                fps=25.0,
                label="#1",
                features=media_analysis.MediaAnalysisFeature.SCENE_CHANGES,
            )

        self.assertEqual(result.scene_changes, (60,))
        start_process.assert_called_once()
        old_probe.assert_not_called()

    def test_fulfill_scans_requested_media_features(self):
        request = media_analysis.MediaAnalysisRequest(
            path=self.path,
            duration_ms=1000,
            fps=25.0,
            label="#1",
            features=media_analysis.MediaAnalysisFeature.MATCHING,
        )
        expected = object()

        with patch.object(self.session, "scan", return_value=expected) as scan:
            result = self.session.fulfill(request)

        self.assertIs(result, expected)
        scan.assert_called_once_with(
            self.path,
            duration_ms=1000,
            fps=25.0,
            label="#1",
            features=media_analysis.MediaAnalysisFeature.MATCHING,
        )

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

    def test_failed_scan_does_not_update_persistent_cache(self):
        persistent = Mock()
        persistent.load_frame_probes.return_value = None
        self.session.set_persistent_cache(persistent)
        scanned = media_analysis.VideoScanResult(
            path=os.path.realpath(self.path),
            features=media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
            frames={0: {"frame_id": 0, "path": None}},
            scene_changes=(),
            identity_samples=(),
            decode_error="corrupt input",
        )

        with patch.object(self.session, "_scan", return_value=scanned):
            self.session.scan(
                self.path,
                duration_ms=1000,
                fps=25.0,
                label="#1",
                features=media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
            )

        persistent.save_frame_probes.assert_not_called()
        persistent.save_scene_changes.assert_not_called()

    def test_scanned_scenes_are_saved_to_persistent_cache(self):
        persistent = Mock()
        persistent.load_scene_changes.return_value = None
        self.session.set_persistent_cache(persistent)
        scanned = media_analysis.VideoScanResult(
            path=os.path.realpath(self.path),
            features=media_analysis.MediaAnalysisFeature.SCENE_CHANGES,
            frames={},
            scene_changes=(120,),
            identity_samples=(),
            decode_error=None,
        )

        with patch.object(self.session, "_scan", return_value=scanned):
            self.session.scan(
                self.path,
                duration_ms=1000,
                fps=25.0,
                label="#1",
                features=media_analysis.MediaAnalysisFeature.SCENE_CHANGES,
            )

        persistent.save_scene_changes.assert_called_once_with(os.path.realpath(self.path), [120])
        persistent.save_frame_probes.assert_not_called()

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

    def test_persistent_cache_does_not_replace_fresh_session_frames(self):
        session = media_analysis.MediaAnalysisSession(
            self.workspace,
            generic_utils.InterruptibleProcess(),
            logging.getLogger("PersistentMediaAnalysisTest"),
            validate_all_streams=False,
        )
        persistent = Mock()
        persistent.load_frame_probes.return_value = None
        session.set_persistent_cache(persistent)
        scanned = media_analysis.VideoScanResult(
            path=os.path.realpath(self.path),
            features=media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
            frames={0: {"frame_id": 0, "path": "/session/frame.png"}},
            scene_changes=(),
            identity_samples=(),
            decode_error=None,
        )

        with patch.object(session, "_scan", return_value=scanned) as scan:
            first = session.scan(
                self.path, duration_ms=1000, fps=25.0, label="#1",
                features=media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
            )
            persistent.load_frame_probes.return_value = {0: {"frame_id": 0, "path": None}}
            second = session.scan(
                self.path, duration_ms=1000, fps=25.0, label="#1",
                features=media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
            )

        self.assertIs(second, first)
        self.assertEqual(second.frames[0]["path"], "/session/frame.png")
        scan.assert_called_once()
        persistent.load_frame_probes.assert_called_once_with(os.path.realpath(self.path))


if __name__ == "__main__":
    unittest.main()
