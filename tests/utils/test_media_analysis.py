import logging
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from twotone.tools.utils import (
    files_utils,
    generic_utils,
    media_analysis,
    video_utils,
)


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
        )

    def _probe_result(self, streams: list[dict] | None = None) -> media_analysis.MediaProbeResult:
        raw_streams = [{"index": 0, "codec_type": "video"}] if streams is None else streams
        normalized_data = (
            {"video": [{"tid": 0, "length": 120, "fps": "25/1"}]}
            if any(stream.get("codec_type") == "video" for stream in raw_streams)
            else {}
        )
        return media_analysis.MediaProbeResult(
            path=os.path.realpath(self.path),
            data={"streams": raw_streams},
            error=None,
            normalized_data=normalized_data,
        )

    def _request(
        self,
        features: media_analysis.MediaAnalysisFeature,
        label: str = "#1",
    ) -> media_analysis.MediaAnalysisRequest:
        return media_analysis.MediaAnalysisRequest(
            path=self.path,
            label=label,
            features=features,
        )

    @staticmethod
    def _stats_path(args: list[str], option: str, occurrence: int = 0) -> str:
        options = [index for index, value in enumerate(args) if value == option]
        return args[options[occurrence] + 1]

    def test_probe_reuses_result_for_the_same_unchanged_file(self):
        data = {"streams": [{"codec_type": "video", "codec_name": "h264"}]}
        normalized_data = {
            "video": [{"tid": 0, "length": 1000, "fps": "25/1"}],
        }

        with self.assertLogs("MediaAnalysisSessionTest", level="DEBUG") as captured, \
             patch.object(video_utils, "get_video_full_info", return_value=data) as probe, \
             patch.object(
                 video_utils,
                 "normalize_video_data",
                 return_value=normalized_data,
             ) as normalize:
            first = self.session.probe(self.path)
            second = self.session.probe(self.path)

        self.assertIs(first, second)
        self.assertTrue(first.has_video)
        self.assertIs(first.normalized_data, normalized_data)
        probe.assert_called_once_with(
            os.path.realpath(self.path),
            logger=self.session.logger,
            show_progress=True,
            progress_description="Reading media metadata",
        )
        normalize.assert_called_once_with(data)
        logs = "\n".join(captured.output)
        self.assertIn("Media probe requested", logs)
        self.assertIn("Running ffprobe", logs)
        self.assertIn("streams=1, video=True, audio=False, error=none", logs)
        self.assertIn("Media probe cache hit", logs)
        cache_restore = next(
            record
            for record in captured.records
            if "Media probe restored from this run's cache" in record.getMessage()
        )
        self.assertEqual(cache_restore.levelno, logging.DEBUG)

    def test_probe_caches_error_reported_by_video_utils(self):
        error = RuntimeError("ffprobe failed for input.mkv: corrupt header")

        with patch.object(video_utils, "get_video_full_info", side_effect=error) as probe, \
             patch.object(video_utils, "normalize_video_data") as normalize:
            first = self.session.probe(self.path)
            second = self.session.probe(self.path)

        self.assertIs(first, second)
        self.assertEqual(first.data, {})
        self.assertEqual(first.normalized_data, {})
        self.assertEqual(first.error, str(error))
        probe.assert_called_once()
        normalize.assert_not_called()

    def test_probe_result_ignores_attached_picture_when_selecting_primary_video(self):
        result = media_analysis.MediaProbeResult(
            path=os.path.realpath(self.path),
            data={"streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "disposition": {"attached_pic": 1},
                },
                {"index": 1, "codec_type": "video"},
            ]},
            error=None,
            normalized_data={"video": [
                {"tid": 0, "length": 1, "fps": "0/0"},
                {"tid": 1, "length": 1000, "fps": "25/1"},
            ]},
        )

        self.assertEqual(
            result.primary_video_track,
            {"tid": 1, "length": 1000, "fps": "25/1"},
        )

    def test_scan_reuses_probe_for_negative_timestamp_correction(self):
        probe_result = {
            "format": {"start_time": "-0.020", "duration": "0.120"},
            "streams": [{
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "r_frame_rate": "25/1",
                "width": 16,
                "height": 16,
            }],
        }

        def fake_ffmpeg(args, _interruption, on_line, logger):
            del logger
            scene_stats = self._stats_path(args, "-stats_enc_pre:v:0")
            with open(scene_stats, "w", encoding="utf-8") as file:
                file.write("0 2 1/25\n")
            return Mock(returncode=0), []

        with patch.object(video_utils, "get_video_full_info", return_value=probe_result) as probe, \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_ffmpeg), \
             patch.object(video_utils, "_showinfo_timestamp_correction_ms") as old_probe:
            result = self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.SCENE_CHANGES,
            ))

        self.assertEqual(result.scene_changes, (60,))
        probe.assert_called_once()
        old_probe.assert_not_called()

    def test_fulfill_scans_requested_media_features(self):
        request = media_analysis.MediaAnalysisRequest(
            path=self.path,
            label="#1",
            features=media_analysis.MediaAnalysisFeature.MATCHING,
        )
        expected = media_analysis.VideoScanResult(
            path=os.path.realpath(self.path),
            features=media_analysis.MediaAnalysisFeature.MATCHING,
            frames={},
            scene_changes=(),
            identity_samples=(),
            decode_error=None,
        )

        with self.assertLogs("MediaAnalysisSessionTest", level="DEBUG") as captured, \
             patch.object(self.session, "_scan", return_value=expected) as scan:
            result = self.session.fulfill(request)

        self.assertIs(result, expected)
        scan.assert_called_once_with(
            os.path.realpath(self.path),
            "#1",
            media_analysis.MediaAnalysisFeature.MATCHING,
        )
        self.assertIn(
            "Fulfilling media analysis request for #1",
            "\n".join(captured.output),
        )
        self.assertIn(
            "features=[scene_changes, frame_timestamps]",
            "\n".join(captured.output),
        )

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
            result = self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
                label=self.path,
            ))

        start.assert_called_once()
        timestamp_correction.assert_not_called()
        self.assertTrue(result.validated_all_streams)
        self.assertIsNone(result.decode_error)

    def test_validation_excludes_attached_pictures_from_video_timeline(self):
        streams = [
            {"index": 0, "codec_type": "video"},
            {
                "index": 1,
                "codec_type": "video",
                "disposition": {"attached_pic": 1},
            },
        ]

        with patch.object(self.session, "probe", return_value=self._probe_result(streams)), \
             patch.object(
                 video_utils,
                 "_start_ffmpeg_streaming",
                 return_value=(SimpleNamespace(returncode=0), []),
             ) as start:
            self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
            ))

        args = start.call_args.args[0]
        self.assertIn("0:V?", args)
        self.assertNotIn("0:v?", args)

    def test_failed_scan_does_not_complete_progress_bar(self):
        progress = Mock()

        with patch.object(self.session, "probe", return_value=self._probe_result()), \
             patch.object(media_analysis, "tqdm", return_value=progress), \
             patch.object(
                 video_utils,
                 "_start_ffmpeg_streaming",
                 return_value=(SimpleNamespace(returncode=-9), []),
             ):
            result = self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
            ), raise_on_error=False)

        progress.update.assert_not_called()
        progress.close.assert_called_once_with()
        self.assertEqual(result.decode_error, "ffmpeg exited with code -9")

    def test_required_scan_raises_its_decode_error(self):
        failed = media_analysis.VideoScanResult(
            path=os.path.realpath(self.path),
            features=media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            frames={},
            scene_changes=(),
            identity_samples=(),
            decode_error="scan stopped after the first frame",
        )

        with patch.object(self.session, "_scan", return_value=failed):
            with self.assertRaisesRegex(
                RuntimeError,
                "Media analysis failed for #1: scan stopped after the first frame",
            ):
                self.session.fulfill(
                    self._request(media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES),
                    raise_on_error=True,
                )

    def test_progress_description_explains_analysis_purpose(self):
        cases = (
            (
                media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
                "Checking input #1 for decoding errors",
            ),
            (
                media_analysis.MediaAnalysisFeature.MATCHING,
                "Analyzing input #1 for timeline matching",
            ),
            (
                media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
                "Sampling input #1 for comparison",
            ),
            (
                media_analysis.MediaAnalysisFeature.NONE,
                "Analyzing input #1",
            ),
            (
                (
                    media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES
                    | media_analysis.MediaAnalysisFeature.MATCHING
                    | media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS
                ),
                "Analyzing input #1 (decode validation, timeline matching, sample comparison)",
            ),
        )

        for features, expected in cases:
            with self.subTest(features=features):
                self.assertEqual(
                    self.session._progress_description("#1", features),
                    expected,
                )

    def test_validation_skips_decode_when_probe_reports_no_audio_or_video(self):
        with self.assertLogs("MediaAnalysisSessionTest", level="DEBUG") as captured, \
             patch.object(self.session, "probe", return_value=self._probe_result([])), \
             patch.object(video_utils, "_start_ffmpeg_streaming") as start:
            result = self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
                label=self.path,
            ))

        start.assert_not_called()
        self.assertTrue(result.validated_all_streams)
        self.assertIsNone(result.decode_error)
        self.assertIn(
            "FFmpeg media scan skipped for",
            "\n".join(captured.output),
        )

    def test_identity_samples_require_metadata_from_probe(self):
        probe = media_analysis.MediaProbeResult(
            path=os.path.realpath(self.path),
            data={"streams": [{"index": 0, "codec_type": "video"}]},
            error=None,
            normalized_data={"video": [{"tid": 0, "length": None, "fps": "0/0"}]},
        )

        with patch.object(self.session, "probe", return_value=probe), \
             patch.object(video_utils, "_start_ffmpeg_streaming") as start:
            with self.assertRaisesRegex(
                ValueError,
                "probe did not report a positive video duration and frame rate",
            ):
                self.session.fulfill(self._request(
                    media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
                ))

        start.assert_not_called()

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

        with self.assertLogs("MediaAnalysisSessionTest", level="DEBUG") as captured, \
             patch.object(self.session, "_scan", return_value=result) as scan:
            first = self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            ))
            second = self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
                label="#2",
            ))

        self.assertIs(first, second)
        scan.assert_called_once()
        cache_restore = next(
            record
            for record in captured.records
            if "Media scan for #2 restored from cache" in record.getMessage()
        )
        self.assertEqual(cache_restore.levelno, logging.DEBUG)

    def test_upgrades_cached_scan_with_only_missing_features(self):
        identity = media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES
        matching = media_analysis.MediaAnalysisFeature.MATCHING
        def scan(_path, _label, features):
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

        with self.assertLogs("MediaAnalysisSessionTest", level="DEBUG") as captured, \
             patch.object(self.session, "_scan", side_effect=scan) as scan_mock:
            first = self.session.fulfill(self._request(identity))
            upgraded = self.session.fulfill(self._request(matching))
            restored = self.session.fulfill(self._request(
                identity | matching,
                label="#2",
            ))

        self.assertEqual(first.features, identity)
        self.assertEqual(upgraded.features, identity | matching)
        self.assertEqual(upgraded.identity_samples, first.identity_samples)
        self.assertEqual(upgraded.scene_changes, (40,))
        self.assertEqual(list(upgraded.frames), [40])
        self.assertIs(restored, upgraded)
        self.assertEqual(
            [call.args[-1] for call in scan_mock.call_args_list],
            [identity, matching],
        )
        logs = "\n".join(captured.output)
        self.assertIn(
            "features=[identity_samples]",
            logs,
        )
        self.assertIn(
            "requires a fresh scan: missing=[identity_samples]",
            logs,
        )
        self.assertIn(
            "features=[scene_changes, frame_timestamps]",
            logs,
        )
        self.assertIn(
            "requires a fresh scan: missing=[scene_changes, frame_timestamps]",
            logs,
        )
        self.assertIn(
            "Fresh media analysis collected for #1: features=[scene_changes, frame_timestamps], "
            "frames=1, scene_changes=1, "
            "identity_samples=0, decode_error=none",
            logs,
        )
        self.assertIn("satisfied without FFmpeg", logs)

    def test_collects_frames_scenes_samples_and_validation_in_one_ffmpeg_call(self):
        def fake_start(args, _interruption, on_line, logger):
            stats_options = [
                index for index, value in enumerate(args)
                if value == "-stats_enc_pre:v:0"
            ]
            self.assertEqual(len(stats_options), 2)
            frame_stats = args[stats_options[0] + 1]
            sample_stats = args[stats_options[1] + 1]
            scene_stats = self._stats_path(args, "-stats_enc_pre:v:1")

            with open(frame_stats, "w", encoding="utf-8") as file:
                file.write("0 0 1/1000\n1 40 1/1000\n2 80 1/1000\n")
            with open(sample_stats, "w", encoding="utf-8") as file:
                file.write("0 0 1/1000\n1 80 1/1000\n")
            with open(scene_stats, "w", encoding="utf-8") as file:
                file.write("0 80 1/1000\n")

            output_pattern = next(value for value in args if "identity_%08d.png" in value)
            for index in (1, 2):
                with open(output_pattern.replace("%08d", f"{index:08d}"), "wb") as file:
                    file.write(b"png")

            on_line("out_time_ms=80000\n")
            on_line("progress=end\n")
            return SimpleNamespace(returncode=0), []

        with self.assertLogs("MediaAnalysisSessionTest", level="DEBUG") as captured, \
             patch.object(self.session, "probe", return_value=self._probe_result()), \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_start) as start, \
             patch.object(video_utils, "_showinfo_timestamp_correction_ms", return_value=0):
            result = self.session.fulfill(self._request(
                (
                    media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES
                    | media_analysis.MediaAnalysisFeature.MATCHING
                    | media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS
                ),
            ))

        start.assert_called_once()
        args = start.call_args.args[0]
        self.assertIn("-xerror", args)
        self.assertIn("0:a?", args)
        self.assertIn("split=3", " ".join(args))
        self.assertNotIn("file='pipe\\:2'", " ".join(args))
        self.assertNotIn("metadata=mode=print", " ".join(args))
        self.assertIn("[scenes]", args)
        stats_formats = [
            args[index + 1]
            for index, value in enumerate(args)
            if value == "-stats_enc_pre_fmt:v:0"
        ]
        self.assertEqual(
            stats_formats,
            ["{ni} {ptsi} {tbi}", "{ni} {ptsi} {tbi}"],
        )
        scene_stats_format = self._stats_path(args, "-stats_enc_pre_fmt:v:1")
        self.assertEqual(scene_stats_format, "{ni} {ptsi} {tbi}")
        self.assertEqual(result.scene_changes, (80,))
        self.assertEqual(list(result.frames), [0, 40, 80])
        self.assertEqual(
            [(sample.timestamp_ms, sample.frame_id) for sample in result.identity_samples],
            [(0, 0), (80, 2)],
        )
        self.assertTrue(result.validated_all_streams)
        self.assertIsNone(result.decode_error)
        logs = "\n".join(captured.output)
        self.assertIn(
            "FFmpeg media scan pipeline for #1: "
            "features=[identity_samples, scene_changes, frame_timestamps, validate_streams]",
            logs,
        )
        self.assertIn(
            "video_branches=[vframes, vscenes, vsamples], identity_targets=7, "
            "validates_all_streams=True",
            logs,
        )

    def test_scene_only_scan_has_a_mapped_filter_output(self):
        session = media_analysis.MediaAnalysisSession(
            self.workspace,
            generic_utils.InterruptibleProcess(),
            logging.getLogger("SceneOnlyMediaAnalysisTest"),
        )

        def fake_start(args, _interruption, on_line, logger):
            del on_line, logger
            self.assertIn("[scenes]", args)
            self.assertNotIn("voutput", " ".join(args))
            scene_stats = self._stats_path(args, "-stats_enc_pre:v:0")
            with open(scene_stats, "w", encoding="utf-8"):
                pass
            return Mock(returncode=0), []

        probe = media_analysis.MediaProbeResult(
            path=os.path.realpath(self.path),
            data={"streams": [{"codec_type": "video"}]},
            error=None,
        )
        with patch.object(session, "probe", return_value=probe), \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_start):
            result = session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.SCENE_CHANGES,
            ))

        self.assertEqual(result.scene_changes, ())
        self.assertIsNone(result.decode_error)

    def test_identity_samples_do_not_reuse_sparse_frames(self):
        samples = media_analysis.MediaAnalysisSession._build_samples(
            (0, 250, 500, 750, 1000),
            [(0, 0), (1, 1000)],
            ["/first.png", "/last.png"],
            {},
        )

        self.assertEqual(
            [(sample.target_ms, sample.timestamp_ms, sample.path) for sample in samples],
            [
                (0, 0, "/first.png"),
                (1000, 1000, "/last.png"),
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

        with self.assertLogs("MediaAnalysisSessionTest", level="DEBUG") as captured, \
             patch.object(self.session, "_scan", return_value=scanned) as scan:
            result = self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.MATCHING,
            ))

        scan.assert_called_once_with(
            os.path.realpath(self.path),
            "#1",
            media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS
        )
        self.assertEqual(result.scene_changes, (120,))
        self.assertEqual(list(result.frames), [0])
        persistent.save_frame_probes.assert_called_once_with(
            os.path.realpath(self.path),
            {0: {"frame_id": 0, "path": None}},
        )
        persistent.save_scene_changes.assert_not_called()
        self.assertIn(
            "Persistent media analysis cache restored data for #1: features=[scene_changes]",
            "\n".join(captured.output),
        )

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
            self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
            ), raise_on_error=False)

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
            self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.SCENE_CHANGES,
            ))

        persistent.save_scene_changes.assert_called_once_with(os.path.realpath(self.path), [120])
        persistent.save_frame_probes.assert_not_called()

    def test_complete_persistent_matching_cache_avoids_a_scan(self):
        session = media_analysis.MediaAnalysisSession(
            self.workspace,
            generic_utils.InterruptibleProcess(),
            logging.getLogger("PersistentMediaAnalysisTest"),
        )
        persistent = Mock()
        persistent.load_scene_changes.return_value = [120]
        persistent.load_frame_probes.return_value = {
            0: {"frame_id": 0, "path": None},
        }
        session.set_persistent_cache(persistent)

        with patch.object(session, "_scan") as scan:
            result = session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.MATCHING,
            ))

        scan.assert_not_called()
        self.assertEqual(result.scene_changes, (120,))
        self.assertEqual(list(result.frames), [0])

    def test_persistent_cache_does_not_replace_fresh_session_frames(self):
        session = media_analysis.MediaAnalysisSession(
            self.workspace,
            generic_utils.InterruptibleProcess(),
            logging.getLogger("PersistentMediaAnalysisTest"),
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
            first = session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
            ))
            persistent.load_frame_probes.return_value = {0: {"frame_id": 0, "path": None}}
            second = session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.FRAME_TIMESTAMPS,
            ))

        self.assertIs(second, first)
        self.assertEqual(second.frames[0]["path"], "/session/frame.png")
        scan.assert_called_once()
        persistent.load_frame_probes.assert_called_once_with(os.path.realpath(self.path))

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
                file.write("0 0 1/1000\n1 80 1/1000\n")

            output_pattern = next(value for value in args if "identity_%08d.png" in value)
            for index in (1, 2):
                with open(output_pattern.replace("%08d", f"{index:08d}"), "wb") as file:
                    file.write(b"png")

            return SimpleNamespace(returncode=0), []

        with patch.object(self.session, "probe", return_value=self._probe_result()), \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_start) as start, \
             patch.object(video_utils, "_showinfo_timestamp_correction_ms", return_value=0):
            result = self.session.fulfill(self._request(
                media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
            ))

        args = start.call_args.args[0]
        self.assertNotIn("-xerror", args)
        self.assertNotIn("vvalidate", " ".join(args))
        self.assertNotIn("frames.txt", " ".join(args))
        self.assertNotIn("gt(scene", " ".join(args))
        self.assertEqual(
            result.features,
            media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES,
        )
        self.assertEqual(result.frames, {})
        self.assertEqual(result.scene_changes, ())
        self.assertEqual(len(result.identity_samples), 2)

    def test_scene_and_identity_scan_maps_scenes_to_a_null_output(self):
        session = media_analysis.MediaAnalysisSession(
            self.workspace,
            generic_utils.InterruptibleProcess(),
            logging.getLogger("MediaAnalysisSessionTest.combined"),
        )

        def fake_start(args, _interruption, on_line, logger):
            del on_line, logger
            stats_options = [
                index for index, value in enumerate(args)
                if value == "-stats_enc_pre:v:0"
            ]
            self.assertEqual(len(stats_options), 2)
            scene_stats = args[stats_options[0] + 1]
            sample_stats = args[stats_options[1] + 1]
            with open(scene_stats, "w", encoding="utf-8") as file:
                file.write("0 0 1/1000\n")
            with open(sample_stats, "w", encoding="utf-8") as file:
                file.write("0 0 1/1000\n")

            output_pattern = next(value for value in args if "identity_%08d.png" in value)
            with open(output_pattern.replace("%08d", "00000001"), "wb") as file:
                file.write(b"png")
            return SimpleNamespace(returncode=0), []

        with patch.object(session, "probe", return_value=self._probe_result()), \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_start) as start, \
             patch.object(video_utils, "_showinfo_timestamp_correction_ms", return_value=0):
            result = session.fulfill(self._request(
                (
                    media_analysis.MediaAnalysisFeature.SCENE_CHANGES
                    | media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES
                ),
            ))

        args = start.call_args.args[0]
        output_triplets = [args[index:index + 3] for index in range(len(args) - 2)]
        self.assertIn(["-f", "null", "-"], output_triplets)
        self.assertIn("[scenes]", args)
        self.assertTrue(result.supports(media_analysis.MediaAnalysisFeature.SCENE_CHANGES))
        self.assertTrue(result.supports(media_analysis.MediaAnalysisFeature.IDENTITY_SAMPLES))
        self.assertEqual(result.scene_changes, (0,))
        self.assertEqual(len(result.identity_samples), 1)

    def test_frame_stats_preserve_millisecond_precision_for_long_timestamps(self):
        stats_path = os.path.join(self.temp_dir.name, "frames.txt")
        with open(stats_path, "w", encoding="utf-8") as file:
            file.write(
                "26629 1110651 1/1000\n"
                "26630 26655664 1/24000\n"
            )

        entries = self.session._read_frame_entries(stats_path, correction_ms=-21)

        self.assertEqual(entries, [(26629, 1110630), (26630, 1110632)])


if __name__ == "__main__":
    unittest.main()
