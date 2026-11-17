import logging
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from parameterized import parameterized

from melt.helpers import MeltComponentFactories, _FAKE_PROCESS_OK
from twotone.tools.melt import audio_strategy, debug_routines, mapping_validation, matching_analysis
from twotone.tools.melt.audio_patcher import (
    AudioPatcher,
    AudioPatchResult,
    AudioStrategy,
    TimelineInterval,
    VideoToAudioTimeline,
)
from twotone.tools.melt.melt_common import AudioStreamRef, FrameSpan, VideoStreamRef
from twotone.tools.melt.pair_matcher import MappingRelation, PairMatcher, SegmentsMappingResult
from twotone.tools.melt.source_aligner import AlignmentGroup, SourceAligner
from twotone.tools.melt.track_timeline import TrackTimeline
from twotone.tools.tool import ToolRuntimeContext
from twotone.tools.utils import (
    files_utils,
    generic_utils,
    media_analysis,
    process_utils,
    video_utils,
)


def _video_details(path: str, length_ms: int | None = None) -> dict:
    """Plan data naming the single video track of *path*."""
    track: dict = {"tid": 0, "ffprobe_stream_index": 0}
    if length_ms is not None:
        track["length"] = length_ms
    return {path: {"tracks": {"video": [track]}}}


def _global_linear_matching(mapping: list[tuple[int, int]]) -> SegmentsMappingResult:
    return SegmentsMappingResult(
        mapping=mapping,
        lhs_all_frames={},
        rhs_all_frames={},
        relation=MappingRelation.GLOBAL_LINEAR,
        lhs_fps=25.0,
        rhs_fps=25.0,
    )


class SourceAlignerUnitTest(MeltComponentFactories, unittest.TestCase):
    """Unit tests for matching, validating and fitting aligned sources."""

    def test_pair_match_reuses_media_analysis_session(self):
        media_analysis_session = Mock(spec=media_analysis.MediaAnalysisSession)
        workspace = files_utils.Workspace.temporary()
        self.addCleanup(workspace.close)
        interruption = generic_utils.InterruptibleProcess()
        context = ToolRuntimeContext(workspace, interruption, media_analysis_session)
        logger = logging.getLogger("test.SourceAligner")
        timeline = TrackTimeline(workspace, logger)
        aligner = SourceAligner(
            logger,
            context,
            timeline,
            AudioPatcher(logger, timeline),
            max_temporary_frame_offset=5.0,
        )
        base_path = "/media/base.mkv"
        source_path = "/media/source.mkv"

        with patch("twotone.tools.melt.source_aligner.PairMatcher") as matcher_type:
            matcher_type.return_value.create_segments_mapping.return_value = Mock()
            aligner._pair_match(
                AlignmentGroup(
                    file_ids={base_path: 1, source_path: 2},
                    base_video=VideoStreamRef(base_path, 0, 0, None),
                ),
                "/tmp/matching",
                source_path,
            )

        self.assertIs(
            matcher_type.call_args.kwargs["media_analysis_session"],
            media_analysis_session,
        )
        # The tolerance is a validation policy; matching never sees it.
        self.assertNotIn("max_temporary_frame_offset", matcher_type.call_args.kwargs)

    def test_align_source_applies_the_temporary_offset_tolerance(self):
        aligner = self._make_source_aligner()
        aligner.max_temporary_frame_offset = 5.0
        base_path = "/media/base.mkv"
        source_path = "/media/source.mkv"
        group = AlignmentGroup(
            file_ids={base_path: 1, source_path: 2},
            base_video=VideoStreamRef(base_path, 0, 0, None),
            base_duration_ms=6000,
            files_details=_video_details(source_path, 6000),
        )
        matching = SegmentsMappingResult(
            mapping=[(0, 0), (6000, 6000)],
            lhs_all_frames={},
            rhs_all_frames={},
            relation=MappingRelation.GLOBAL_LINEAR,
            lhs_fps=25.0,
            rhs_fps=25.0,
        )
        with patch.object(aligner, "_pair_match", return_value=matching), \
             patch.object(mapping_validation, "validate_mapping") as validate:
            aligner._align_source(group, source_path, consumer="subtitle")

        validate.assert_called_once_with(
            matching,
            consumer="subtitle",
            lhs_id=1,
            rhs_id=2,
            max_temporary_frame_offset=5.0,
            logger=aligner.logger,
        )
        self.assertIs(group.aligned_sources[source_path], matching)

    def test_rejects_temporary_offset_tolerance_below_default(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            SourceAligner(
                logging.getLogger("test.SourceAligner"),
                Mock(),
                Mock(),
                Mock(),
                max_temporary_frame_offset=1.0,
            )

    def test_align_source_reports_coverage_with_the_plan_track_length(self):
        # The coverage report compares the two video tracks, so the source
        # length comes from the same plan data as the base length, not from
        # a container that also spans a longer audio track.
        aligner = self._make_source_aligner()
        base_path = "/media/base.mkv"
        source_path = "/media/source.mkv"
        group = AlignmentGroup(
            file_ids={base_path: 1, source_path: 2},
            base_video=VideoStreamRef(base_path, 0, 0, None),
            base_duration_ms=6000,
            files_details=_video_details(source_path, 5960),
        )

        with patch.object(aligner, "_pair_match", return_value=_global_linear_matching([(0, 0), (5960, 5960)])), \
             patch.object(video_utils, "get_video_duration", return_value=6000), \
             patch.object(debug_routines, "log_coverage") as log_coverage:
            aligner._align_source(group, source_path, consumer="audio")

        self.assertEqual(log_coverage.call_args.args[2:4], (6000, 5960))

    def test_mapping_to_video_measures_rebased_frame_times(self):
        aligner = self._make_source_aligner()
        fake_info = {"streams": [{"codec_type": "video", "index": 0, "start_time": "0.500"}]}

        with patch.object(video_utils, "get_video_full_info", return_value=fake_info):
            rebased = aligner._mapping_to_video_ms(
                VideoStreamRef("/tmp/a.mkv", 0, 0, None), FrameSpan(0, 41, 2),
            )
            preserved = aligner._mapping_to_video_ms(
                VideoStreamRef("/tmp/a.mp4", 0, 0, None), FrameSpan(500, 541, 2),
            )

        self.assertEqual(rebased, 500)
        self.assertEqual(preserved, 0)

    @parameterized.expand([
        ("missing", None),
        ("invalid", "unknown"),
    ])
    def test_effective_video_fps_falls_back_to_collected_frame_count(
        self,
        _name,
        nb_frames,
    ):
        aligner = self._make_source_aligner()
        stream = {
            "codec_type": "video",
            "start_time": "0.500000",
            "duration": "1.000000",
        }
        if nb_frames is not None:
            stream["nb_frames"] = nb_frames

        with patch.object(
            video_utils,
            "get_video_full_info",
            return_value={"streams": [stream]},
        ):
            effective_fps = aligner._effective_video_fps(
                "/tmp/source.mkv",
                FrameSpan(0, 1000, 3),
            )

        self.assertEqual(effective_fps, 2.0)

    def test_effective_video_fps_requires_video_stream(self):
        aligner = self._make_source_aligner()
        with patch.object(
            video_utils,
            "get_video_full_info",
            return_value={"streams": [{"codec_type": "audio"}]},
        ):
            effective_fps = aligner._effective_video_fps(
                "/tmp/source.mkv",
                FrameSpan(0, 1000, 2),
            )

        self.assertIsNone(effective_fps)

    def test_video_to_audio_timeline_accounts_for_rebased_video_and_delayed_audio(self):
        aligner = self._make_source_aligner()
        source_info = {
            "streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.500000"},
                {"codec_type": "audio", "index": 1, "start_time": "0.488000"},
            ],
        }

        with patch.object(video_utils, "get_video_full_info", return_value=source_info):
            timeline = aligner._video_to_audio_timeline(
                VideoStreamRef("/tmp/source.mkv", 0, 0, None),
                AudioStreamRef("/tmp/source.mkv", 1, 1, "pol"),
                FrameSpan(0, 41, 2),
            )

        self.assertEqual(timeline.mapping_to_video_ms, 500)
        self.assertEqual(timeline.video_to_stream_ms(0), 500)
        self.assertEqual(timeline.video_to_rebased_audio_ms(0), 12)
        self.assertEqual(timeline.rebased_audio_to_video_ms(12), 0)

    def test_video_to_audio_timeline_uses_priming_aware_aac_content_start(self):
        aligner = self._make_source_aligner()
        source_info = {
            "streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.478000"},
                {
                    "codec_type": "audio",
                    "index": 1,
                    "codec_name": "aac",
                    "initial_padding": 1024,
                    "sample_rate": "48000",
                    "start_time": "0.500000",
                },
            ],
        }

        with patch.object(video_utils, "get_video_full_info", return_value=source_info), \
             patch.object(aligner.timeline, "aac_priming_exposed", return_value=True):
            timeline = aligner._video_to_audio_timeline(
                VideoStreamRef("/tmp/source.mkv", 0, 0, None),
                AudioStreamRef("/tmp/source.mkv", 1, 1, "pol"),
                FrameSpan(3, 3, 1),
            )

        self.assertEqual(timeline.mapping_to_video_ms, 475)
        self.assertEqual(timeline.audio_content_start_ms, 521)
        self.assertEqual(timeline.video_to_rebased_audio_ms(3), -43)

    def test_patch_audio_uses_unique_temp_paths_for_same_stream_index(self):
        aligner = self._make_source_aligner()
        base_video = "/tmp/base.mkv"
        audio_a = "/tmp/audio-a.mkv"
        audio_b = "/tmp/audio-b.avi"
        base_duration = 60000
        # A real speed change (spans differ) so the GLOBAL_LINEAR relation
        # routes to the global-time-scale patcher whose output path we probe.
        mapping = [(0, 0), (base_duration, 58000)]

        def make_matching() -> SegmentsMappingResult:
            return SegmentsMappingResult(
                mapping=mapping,
                lhs_all_frames={},
                rhs_all_frames={},
                relation=MappingRelation.GLOBAL_LINEAR,
                lhs_fps=24.0,
                rhs_fps=24.0,
            )

        group = AlignmentGroup(
            file_ids={base_video: 1, audio_a: 2, audio_b: 3},
            base_video=VideoStreamRef(base_video, 0, 0, None),
            base_duration_ms=base_duration,
            files_details={**_video_details(audio_a), **_video_details(audio_b)},
        )
        group.aligned_sources[audio_a] = make_matching()
        group.aligned_sources[audio_b] = make_matching()

        patch_outputs = []

        def fake_request(
            _wd,
            _base_video,
            _source_video,
            source_audio,
            _base_audio,
            output_path,
            _mapping,
            _base_duration,
            **_kwargs,
        ):
            self.assertEqual(source_audio.ffprobe_stream_index, 1)
            return Mock(source_audio=source_audio, output_path=output_path)

        def fake_execute(request):
            patch_outputs.append(request.output_path)
            return AudioPatchResult(
                AudioStreamRef(request.output_path, 0, 0, request.source_audio.language),
                0,
                base_duration,
                "global-time-scale",
            )

        with patch.object(debug_routines, "log_coverage", lambda *args, **kwargs: None), \
             patch.object(aligner, "_create_audio_patch_request", side_effect=fake_request), \
            patch.object(aligner.audio_patcher, "execute", side_effect=fake_execute):
            first = aligner.patch_audio(group, AudioStreamRef(audio_a, 1, 1, "pol"))
            second = aligner.patch_audio(group, AudioStreamRef(audio_b, 1, 1, "pol"))

        self.assertEqual(first.stream.mkvmerge_track_id, 0)
        self.assertEqual(first.stream.ffprobe_stream_index, 0)
        self.assertEqual(second.stream.mkvmerge_track_id, 0)
        self.assertEqual(second.stream.ffprobe_stream_index, 0)
        self.assertEqual(first.stream.language, "pol")
        self.assertEqual(second.stream.language, "pol")
        self.assertEqual(first.timeline_start_ms, 0)
        self.assertEqual(second.timeline_start_ms, 0)
        self.assertEqual(patch_outputs, [first.stream.path, second.stream.path])
        self.assertNotEqual(first.stream.path, second.stream.path)

    def test_align_source_rejects_absolute_global_linear_frame_step(self):
        aligner = self._make_source_aligner()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        mapping = [
            (0, 0),
            (100000, 100000),
            (200000, 200000),
            (300000, 299400),
            (400000, 399400),
            (500000, 499400),
        ]
        lhs_frames = {
            timestamp: {"frame_id": frame, "path": None}
            for timestamp, frame in [
                (0, 0),
                (100000, 2500),
                (200000, 5000),
                (300000, 7500),
                (400000, 10000),
                (500000, 12500),
            ]
        }
        rhs_frames = {
            timestamp: {"frame_id": frame, "path": None}
            for timestamp, frame in [
                (0, 0),
                (100000, 2500),
                (200000, 5000),
                (299400, 7485),
                (399400, 9985),
                (499400, 12485),
            ]
        }
        group = AlignmentGroup(
            file_ids={base_video.path: 1, source_video.path: 2},
            base_video=base_video,
            base_duration_ms=500000,
            files_details=_video_details(source_video.path, 500000),
        )
        matching = SegmentsMappingResult(
            mapping=mapping,
            lhs_all_frames=lhs_frames,
            rhs_all_frames=rhs_frames,
            relation=MappingRelation.GLOBAL_LINEAR,
            lhs_fps=25.0,
            rhs_fps=25.0,
            frame_slope=1.0,
            content_discontinuities=tuple(
                matching_analysis.classify_global_linear_discontinuities(
                    mapping, lhs_frames, rhs_frames,
                    frame_slope=1.0, lhs_fps=25.0, rhs_fps=25.0,
                ).content
            ),
        )

        with patch.object(aligner, "_pair_match", return_value=matching), \
             patch.object(debug_routines, "log_coverage"), \
             self.assertRaisesRegex(RuntimeError, "local frame-sequence difference"):
            aligner._align_source(group, source_video.path, consumer="audio")
        self.assertNotIn(source_video.path, group.aligned_sources)

    def test_align_source_rejects_sparse_global_linear_frame_step(self):
        aligner = self._make_source_aligner()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        mapping = [(0, 0), (200000, 200000), (400000, 399400), (600000, 599400)]
        group = AlignmentGroup(
            file_ids={base_video.path: 1, source_video.path: 2},
            base_video=base_video,
            base_duration_ms=600000,
            files_details=_video_details(source_video.path, 600000),
        )
        matching = SegmentsMappingResult(
            mapping=mapping,
            lhs_all_frames={lhs: {"frame_id": lhs // 40, "path": None} for lhs, _ in mapping},
            rhs_all_frames={rhs: {"frame_id": rhs // 40, "path": None} for _, rhs in mapping},
            relation=MappingRelation.GLOBAL_LINEAR,
            lhs_fps=25.0, rhs_fps=25.0, frame_slope=1.0,
            content_discontinuities=tuple(
                matching_analysis.classify_global_linear_discontinuities(
                    mapping,
                    {lhs: {"frame_id": lhs // 40, "path": None} for lhs, _ in mapping},
                    {rhs: {"frame_id": rhs // 40, "path": None} for _, rhs in mapping},
                    frame_slope=1.0, lhs_fps=25.0, rhs_fps=25.0,
                ).content
            ),
        )
        with patch.object(aligner, "_pair_match", return_value=matching), \
             patch.object(debug_routines, "log_coverage"), \
             self.assertRaisesRegex(RuntimeError, "local frame-sequence difference"):
            aligner._align_source(group, source_video.path, consumer="audio")
        self.assertNotIn(source_video.path, group.aligned_sources)

    @parameterized.expand([
        ("global", MappingRelation.GLOBAL_LINEAR),
        ("generic", MappingRelation.GENERIC),
    ])
    def test_align_source_rejects_unsupported_pts_step(self, _case, relation):
        aligner = self._make_source_aligner()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        timing_step = (1600000, 1700000, 1600000, 1703000, -3000)
        group = AlignmentGroup(
            file_ids={base_video.path: 1, source_video.path: 2},
            base_video=base_video,
            base_duration_ms=2000000,
            files_details=_video_details(source_video.path, 2000000),
        )
        matching = SegmentsMappingResult(
            mapping=[(0, 0), (1600000, 1600000), (1700000, 1703000)],
            lhs_all_frames={},
            rhs_all_frames={},
            relation=relation,
            lhs_fps=25.0, rhs_fps=25.0, frame_slope=1.0,
            timing_discontinuities=(timing_step,),
            timing_discontinuity_debug_path="/tmp/timing-debug",
        )

        with patch.object(aligner, "_pair_match", return_value=matching), \
             self.assertLogs(aligner.logger, level="ERROR") as captured, \
             self.assertRaisesRegex(RuntimeError, "Unsupported audio timeline discontinuity"):
            aligner._align_source(group, source_video.path, consumer="audio")
        self.assertNotIn(source_video.path, group.aligned_sources)
        self.assertTrue(any(
            "00:26:40,000-00:28:20,000 (1600000-1700000 ms)" in message
            and "00:26:40,000-00:28:23,000 (1600000-1703000 ms)" in message
            for message in captured.output
        ))
        details_path = os.path.join("/tmp/timing-debug", "details.txt")
        self.assertTrue(any(
            f"Full frame diagnostics: {details_path}" in message
            for message in captured.output
        ))

    def test_align_source_rejects_pts_step_with_real_global_fit(self):
        aligner = self._make_source_aligner()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        lhs_frames = {i * 40: {"frame_id": i, "path": None} for i in range(48000)}
        rhs_frames = {
            i * 40 + (3000 if i >= 40000 else 0): {"frame_id": i, "path": None}
            for i in range(48000)
        }
        pairs = [(j * 100000, j * 100000 + (3000 if j >= 16 else 0)) for j in range(20)]
        analysis = Mock(spec=media_analysis.MediaAnalysisSession)
        analysis.probe.side_effect = lambda path: media_analysis.MediaProbeResult(
            path,
            {},
            None,
            {"video": [{"fps": "25/1"}]},
        )
        matcher = PairMatcher(
            generic_utils.InterruptibleProcess(), aligner.workspace.root,
            base_video.path, source_video.path, aligner.logger,
            media_analysis_session=analysis,
        )
        fit = matcher.detect_global_linear(pairs, lhs_frames, rhs_frames)
        self.assertIsNotNone(fit)
        self.assertEqual(matching_analysis.find_global_linear_content_discontinuities(
            pairs, lhs_frames, rhs_frames,
            frame_slope=fit.slope, lhs_fps=25.0, rhs_fps=25.0,
        ), [])
        timing_discontinuities = matching_analysis.classify_global_linear_discontinuities(
            pairs, lhs_frames, rhs_frames,
            frame_slope=fit.slope, lhs_fps=25.0, rhs_fps=25.0,
        ).timing
        self.assertEqual(len(timing_discontinuities), 1)
        # The aligner enforces PairMatcher's final timing verdict without
        # rerunning detection or reaching an audio strategy.
        group = AlignmentGroup(
            file_ids={base_video.path: 1, source_video.path: 2},
            base_video=base_video,
            base_duration_ms=1920000,
            files_details=_video_details(source_video.path, 1923000),
        )
        matching = SegmentsMappingResult(
            mapping=pairs,
            lhs_all_frames=lhs_frames,
            rhs_all_frames=rhs_frames,
            relation=MappingRelation.GLOBAL_LINEAR,
            lhs_fps=25.0,
            rhs_fps=25.0,
            frame_slope=fit.slope,
            timing_discontinuities=tuple(timing_discontinuities),
        )
        with patch.object(aligner, "_pair_match", return_value=matching), \
             patch.object(video_utils, "get_video_full_info", return_value={}), \
             self.assertRaisesRegex(RuntimeError, "Unsupported audio timeline discontinuity"):
            aligner._align_source(group, source_video.path, consumer="audio")
        self.assertNotIn(source_video.path, group.aligned_sources)

    def test_patch_audio_uses_pair_matcher_timing_verdict_without_reclassification(self):
        aligner = self._make_source_aligner()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        mapping = [(0, 0), (60000, 60000)]
        group = AlignmentGroup(
            file_ids={base_video.path: 1, source_video.path: 2},
            base_video=base_video,
            base_duration_ms=60000,
            files_details=_video_details(source_video.path),
        )
        group.aligned_sources[source_video.path] = SegmentsMappingResult(
            mapping=mapping,
            lhs_all_frames={
                0: {"frame_id": 0, "path": None},
                60000: {"frame_id": 1440, "path": None},
            },
            rhs_all_frames={
                0: {"frame_id": 0, "path": None},
                60000: {"frame_id": 1440, "path": None},
            },
            relation=MappingRelation.GLOBAL_LINEAR,
            lhs_fps=24.0,
            rhs_fps=24.0,
            frame_slope=1.0,
        )
        request = Mock(output_path="/tmp/patched.mka", source_audio=source_audio)
        result = AudioPatchResult(
            AudioStreamRef("/tmp/patched.mka", 0, 0, "pol"),
            0,
            60000,
            "unscaled-normalization",
        )

        with patch.object(
                 matching_analysis,
                 "classify_global_linear_discontinuities",
                 side_effect=AssertionError("aligner must use PairMatcher's final verdict"),
             ), \
             patch.object(
                 aligner,
                 "_plan_audio",
                 return_value=audio_strategy.AudioPlan(AudioStrategy.UNSCALED, mapping),
             ), \
             patch.object(aligner, "_create_audio_patch_request", return_value=request), \
             patch.object(aligner.audio_patcher, "execute", return_value=result):
            actual = aligner.patch_audio(group, source_audio)

        self.assertEqual(actual, result)

    def test_unscaled_audio_patch_routes_through_shared_executor(self):
        aligner = self._make_source_aligner()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        mapping = [(0, 0), (60000, 60000)]
        group = AlignmentGroup(
            file_ids={base_video.path: 1, source_audio.path: 2},
            base_video=base_video,
            base_duration_ms=60000,
            files_details=_video_details(source_video.path),
        )
        group.aligned_sources[source_audio.path] = SegmentsMappingResult(
            mapping=mapping,
            lhs_all_frames={},
            rhs_all_frames={},
            relation=MappingRelation.GLOBAL_LINEAR,
            lhs_fps=24.0,
            rhs_fps=24.0,
        )
        request = Mock(output_path="/tmp/patched.mka", source_audio=source_audio)
        result = AudioPatchResult(
            AudioStreamRef("/tmp/patched.mka", 0, 0, "pol"),
            500,
            59500,
            "unscaled-normalization",
        )

        with patch.object(
                 aligner,
                 "_plan_audio",
                 return_value=audio_strategy.AudioPlan(
                     AudioStrategy.UNSCALED, mapping, unscaled_shift_ms=500,
                 ),
             ), \
             patch.object(aligner, "_create_audio_patch_request", return_value=request) as create, \
             patch.object(aligner.audio_patcher, "execute", return_value=result) as execute:
            actual = aligner.patch_audio(group, source_audio)

        self.assertEqual(actual, result)
        self.assertEqual(create.call_args.kwargs["output_end_ms"], 60000)
        self.assertEqual(
            create.call_args.args[6],
            audio_strategy.AudioPlan(AudioStrategy.UNSCALED, mapping, unscaled_shift_ms=500),
        )
        execute.assert_called_once_with(request)

    def test_audio_patch_request_carries_the_plan(self):
        aligner = self._make_source_aligner()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        plan = audio_strategy.AudioPlan(
            AudioStrategy.SUBSEGMENT,
            [(0, 0), (30000, 30100), (60000, 60900)],
        )

        with patch.object(
                 aligner,
                 "_video_to_audio_timeline",
                 return_value=VideoToAudioTimeline(0, 0),
             ), \
             patch.object(aligner, "_mapping_to_video_ms", return_value=0):
            request = aligner._create_audio_patch_request(
                "/tmp/work",
                base_video,
                source_video,
                source_audio,
                None,
                "/tmp/out.mka",
                plan,
                60000,
                fill_gaps_from_base=False,
            )

        self.assertEqual(request.strategy, AudioStrategy.SUBSEGMENT)
        self.assertEqual(request.segment_count, audio_strategy.SUBSEGMENT_COUNT)
        self.assertEqual(request.unscaled_shift_ms, 0)
        self.assertEqual(request.mapping, tuple(plan.mapping))
        self.assertEqual(request.target_interval, TimelineInterval(0, 60000))

    def test_subsegment_audio_patch_preserves_selected_stream_identity(self):
        aligner = self._make_source_aligner()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        base_audio = AudioStreamRef("/tmp/base.mkv", 3, 3, "eng")
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        mapping = [(0, 0), (60000, 60000)]
        group = AlignmentGroup(
            file_ids={base_video.path: 1, source_audio.path: 2},
            base_video=base_video,
            base_audio=base_audio,
            base_duration_ms=60000,
            base_audio_end_ms=60000,
            files_details=_video_details(source_video.path),
        )
        group.aligned_sources[source_audio.path] = SegmentsMappingResult(
            mapping=mapping,
            lhs_all_frames={},
            rhs_all_frames={},
            relation=MappingRelation.GENERIC,
            lhs_fps=24.0,
            rhs_fps=24.0,
        )

        def fake_request(
            _wd,
            received_base_video,
            received_source_video,
            received_source_audio,
            received_base_audio,
            output_path,
            plan,
            _base_duration,
            **_kwargs,
        ):
            self.assertEqual(plan.strategy, AudioStrategy.SUBSEGMENT)
            self.assertEqual(received_base_video, base_video)
            self.assertEqual(received_source_video, source_video)
            self.assertEqual(received_source_audio, source_audio)
            self.assertEqual(received_base_audio, base_audio)
            return Mock(source_audio=received_source_audio, output_path=output_path)

        def fake_execute(request):
            return AudioPatchResult(
                AudioStreamRef(request.output_path, 0, 0, request.source_audio.language),
                125,
                59875,
                "subsegment-atempo (1 segments)",
            )

        with patch.object(
                 aligner,
                 "_plan_audio",
                 return_value=audio_strategy.AudioPlan(AudioStrategy.SUBSEGMENT, mapping),
             ), \
             patch.object(aligner, "_create_audio_patch_request", side_effect=fake_request), \
             patch.object(aligner.audio_patcher, "execute", side_effect=fake_execute):
            patched = aligner.patch_audio(group, source_audio)

        self.assertEqual(patched.stream.mkvmerge_track_id, 0)
        self.assertEqual(patched.stream.ffprobe_stream_index, 0)
        self.assertEqual(patched.stream.language, "pol")
        self.assertEqual(patched.timeline_start_ms, 125)
        self.assertEqual(patched.duration_ms, 59875)

    @staticmethod
    def _execute_global_time_scale(
        aligner,
        working_dir,
        base_video,
        source_video,
        source_audio,
        base_audio,
        output_path,
        segment_pairs,
        *,
        fill_gaps_from_base,
    ):
        base_duration_ms = video_utils.get_video_duration(
            base_video.path,
            logger=aligner.logger,
        )
        if base_duration_ms is None:
            raise RuntimeError(
                f"Could not determine base video duration for {base_video.path}."
            )
        request = aligner._create_audio_patch_request(
            working_dir,
            base_video,
            source_video,
            source_audio,
            base_audio,
            output_path,
            audio_strategy.AudioPlan(AudioStrategy.GLOBAL_TIME_SCALE, segment_pairs),
            base_duration_ms,
            fill_gaps_from_base=fill_gaps_from_base,
        )
        return aligner.audio_patcher.execute(request).timeline_start_ms

    def test_patch_audio_fps_mismatch_with_audio_start_offset(self):
        """Exact scenario: 23.976fps base + 25fps AVI source with audio start offset.

        Real-world case:
        - Base (4K MKV, 23.976fps): mapping lhs range [2420 … 6961247] ms
        - Source (AVI, 25fps): mapping rhs range [40 … 6673840] ms
        - source_dur = 6673800 ms, target_dur = 6958827 ms
        - fps_ratio ≈ 0.9590 → needs asetrate scaling
        - Audio in the AVI container starts ~488 ms after the video.
        - The common segment starts at 40 ms on the source timeline, so sync
          offset must shift from 2420 to ~2887 ms
          (correction = round((488 - 40) × 6958827/6673800) = 467 ms).

        Without the start-offset correction, audio arrives ~450 ms too early.
        """
        aligner = self._make_source_aligner()

        # Mapping: first pair at 2420/40, last pair at 6961247/6673840
        seg1_start, seg2_start = 2420, 40
        seg1_end, seg2_end = 6961247, 6673840
        pairs = [(seg1_start, seg2_start), (seg1_end, seg2_end)]

        source_dur = seg2_end - seg2_start           # 6673800
        target_dur = seg1_end - seg1_start           # 6958827
        video_ratio = target_dur / source_dur        # ≈ 1.042714
        audio_start_ms = 488
        start_gap_ms = audio_start_ms - seg2_start
        actual_trimmed_dur = source_dur - start_gap_ms

        calls = []
        returned_sync_offset = None

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_get_duration(path, **_kwargs):
            if "source_trimmed" in path:
                return actual_trimmed_dur
            if "source_scaled" in path:
                return round(actual_trimmed_dur * video_ratio)
            if "out." in path:
                return round(actual_trimmed_dur * video_ratio)
            return 7000000  # base video duration (>seg1_end → has tail)

        fake_full_info = {"streams": [
            {"codec_type": "video", "index": 0, "start_time": "0.0"},
            {"codec_type": "audio", "index": 1, "channels": 2, "sample_rate": "48000",
             "sample_fmt": "s16", "start_time": "0.488"},
        ]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process', side_effect=fake_start_process), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                 patch.object(video_utils, 'get_video_full_info', return_value=fake_full_info):
                returned_sync_offset = self._execute_global_time_scale(
                    aligner, wd, VideoStreamRef("/base.mkv", 0, 0, None),
                    VideoStreamRef("/source.avi", 0, 0, None),
                    AudioStreamRef("/source.avi", 1, 1, "pol"), None, output_path, pairs,
                    fill_gaps_from_base=False,
                )

        expected_correction = round(start_gap_ms * video_ratio)
        expected_sync = seg1_start + expected_correction
        self.assertEqual(returned_sync_offset, expected_sync,
                         f"Sync offset should be {seg1_start} + {expected_correction} = {expected_sync}, "
                         f"not {seg1_start} (the uncorrected value)")
        self.assertGreater(returned_sync_offset, seg1_start,
                           "Audio stream start must push sync offset forward")

        # --- asetrate must be applied (fps differ) ---
        filter_calls = [c for c in calls if any("asetrate" in str(a) for a in c[1])]
        self.assertEqual(len(filter_calls), 1, "asetrate should be used exactly once")
        filter_arg = next(a for a in filter_calls[0][1] if "asetrate" in str(a))
        # fps_ratio = source_dur / target_dur ≈ 0.959035
        source_per_base = source_dur / target_dur
        self.assertEqual(
            filter_arg,
            aligner.audio_patcher._global_time_scale_filter(
                actual_trimmed_dur,
                round(actual_trimmed_dur / source_per_base),
                48000,
            ),
        )
        self.assertNotIn("atempo", filter_arg)
        self.assertNotIn("atrim", filter_arg)

        # --- Trim timestamps use the decoded, rebased audio timeline ---
        ffmpeg_args = [c[1] for c in calls if c[0] == "ffmpeg"]
        trim_call = next(a for a in ffmpeg_args if "source_trimmed" in str(a))
        self.assertNotIn("-ss", trim_call)
        self.assertNotIn("-to", trim_call)
        self.assertLess(trim_call.index("-i"), trim_call.index("-filter:a"))
        trim_filter = trim_call[trim_call.index("-filter:a") + 1]
        self.assertIn(
            f"atrim=start=0.000000:end={(seg2_end - audio_start_ms) / 1000:.6f}",
            trim_filter,
        )
        self.assertIn("asetpts=PTS-STARTPTS", trim_filter)

        # --- No head/tail extraction (fill_gaps_from_base=False) ---
        head_tail_calls = [c for c in calls if any("head" in str(a) or "tail" in str(a) for a in c[1])]
        self.assertEqual(head_tail_calls, [], "fill_gaps_from_base=False should skip head/tail")

    def test_patch_audio_no_start_offset_keeps_original_sync_offset(self):
        """When the audio track has no start offset, sync offset equals seg1_start."""
        aligner = self._make_source_aligner()

        seg1_start, seg2_start = 2420, 40
        seg1_end, seg2_end = 6961247, 6673840
        pairs = [(seg1_start, seg2_start), (seg1_end, seg2_end)]
        source_dur = seg2_end - seg2_start

        def fake_get_duration(path, **_kwargs):
            if "source_trimmed" in path:
                return source_dur
            if "source_scaled" in path or "out." in path:
                return seg1_end - seg1_start
            return 7000000

        fake_full_info = {"streams": [{"codec_type": "audio", "index": 1, "channels": 2,
                                       "sample_rate": "48000", "sample_fmt": "s16"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process',
                              side_effect=lambda t, a, **kw: _FAKE_PROCESS_OK), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                 patch.object(video_utils, 'get_video_full_info', return_value=fake_full_info):
                sync = self._execute_global_time_scale(
                    aligner, wd, VideoStreamRef("/base.mkv", 0, 0, None),
                    VideoStreamRef("/source.mkv", 0, 0, None),
                    AudioStreamRef("/source.mkv", 1, 1, "pol"), None, output_path, pairs,
                    fill_gaps_from_base=False,
                )

        self.assertEqual(sync, seg1_start,
                         "Without audio start offset, sync offset should be the raw seg1_start")

    def test_patch_audio_fill_gaps_raises_on_start_offset(self):
        """In fill-audio-gaps mode, audio start correction cannot be compensated."""
        aligner = self._make_source_aligner()

        seg1_start, seg2_start = 2420, 40
        seg1_end, seg2_end = 6961247, 6673840
        pairs = [(seg1_start, seg2_start), (seg1_end, seg2_end)]
        source_dur = seg2_end - seg2_start
        audio_start_ms = 488
        start_gap_ms = audio_start_ms - seg2_start

        def fake_get_duration(path, **_kwargs):
            if "source_trimmed" in path:
                return source_dur - start_gap_ms
            if "source_scaled" in path:
                return round((source_dur - start_gap_ms) * ((seg1_end - seg1_start) / source_dur))
            return 7000000

        fake_full_info = {"streams": [{"codec_type": "audio", "index": 1, "channels": 2,
                                       "sample_rate": "48000", "sample_fmt": "s16",
                                       "start_time": "0.488"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process',
                              side_effect=lambda t, a, **kw: _FAKE_PROCESS_OK), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                 patch.object(video_utils, 'get_video_full_info', return_value=fake_full_info):
                with self.assertRaises(RuntimeError, msg="Should raise on start offset in fill-audio-gaps mode") as ctx:
                    self._execute_global_time_scale(
                        aligner, wd, VideoStreamRef("/base.mkv", 0, 0, None),
                        VideoStreamRef("/source.avi", 0, 0, None),
                        AudioStreamRef("/source.avi", 1, 1, "pol"),
                        AudioStreamRef("/base.mkv", 1, 1, "eng"), output_path, pairs,
                        fill_gaps_from_base=True,
                    )
                self.assertIn("fill-audio-gaps", str(ctx.exception))
                expected_correction = round(
                    start_gap_ms * ((seg1_end - seg1_start) / source_dur)
                )
                self.assertIn(str(expected_correction), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
