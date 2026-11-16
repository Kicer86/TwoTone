import logging
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from common import run_ffmpeg
from parameterized import parameterized

from melt.helpers import MeltComponentFactories, _FAKE_PROCESS_OK
from twotone.tools.melt import audio_strategy, debug_routines, matching_analysis
from twotone.tools.melt.static_source import StaticSource
from twotone.tools.melt.streams_picker import StreamsPicker
from twotone.tools.melt.melt_common import (
    AttachmentRef,
    AudioStreamRef,
    MeltInputFiles,
    SubtitleStreamRef,
    VideoStreamRef,
)
from twotone.tools.melt.melt_performer import (
    AudioPatchResult,
    MeltPerformer,
    TimelineInterval,
    VideoToAudioTimeline,
    _AudioStrategy,
    _GroupContext,
    _PairMatchResult,
    _StreamEntry,
)
from twotone.tools.melt.melt_plan import MeltGroupPlan, MeltTitlePlan
from twotone.tools.melt.pair_matcher import MappingRelation, PairMatcher, SegmentsMappingResult
from twotone.tools.tool import ToolRuntimeContext
from twotone.tools.utils import (
    files_utils,
    generic_utils,
    media_analysis,
    process_utils,
    video_utils,
)


class MeltPerformerUnitTest(MeltComponentFactories, unittest.TestCase):
    """Unit tests for MeltPerformer internal methods."""

    def _process_single_source_plan(
        self,
        source_path: str,
        selected_streams: dict[str, list[tuple[str, int, int, str | None]]],
        prepared_entries: list[_StreamEntry],
    ) -> list[str]:
        performer = self._make_performer()
        plan = (MeltTitlePlan(
            title="Title",
            groups=(MeltGroupPlan(
                input_files=MeltInputFiles((source_path,)),
                output_name="Movie",
                video_streams=tuple(
                    VideoStreamRef(*stream)
                    for stream in selected_streams.get("video", [])
                ),
                audio_streams=tuple(
                    AudioStreamRef(*stream)
                    for stream in selected_streams.get("audio", [])
                ),
                subtitle_streams=tuple(
                    SubtitleStreamRef(*stream)
                    for stream in selected_streams.get("subtitle", [])
                ),
            ),),
        ),)
        prepared = Mock(entries=prepared_entries, input_files={source_path})
        captured_args: list[str] = []

        def run_mkvmerge(_tool, args, **_kwargs):
            captured_args.extend(args)
            with open(args[1], "wb") as output_file:
                output_file.write(b"muxed")
            return _FAKE_PROCESS_OK

        with patch.object(performer, "_prepare_stream_entries", return_value=prepared), \
             patch.object(process_utils, "start_process", side_effect=run_mkvmerge), \
             patch.object(video_utils, "validate_media_output"):
            performer.process_duplicates(plan)

        return captured_args

    def test_pair_match_reuses_media_analysis_session(self):
        media_analysis_session = Mock(spec=media_analysis.MediaAnalysisSession)
        output = files_utils.Workspace.temporary()
        workspace = files_utils.Workspace.temporary()
        self.addCleanup(output.close)
        self.addCleanup(workspace.close)
        interruption = generic_utils.InterruptibleProcess()
        context = ToolRuntimeContext(workspace, interruption, media_analysis_session)
        performer = MeltPerformer(
            logger=logging.getLogger("test.MeltPerformer"),
            context=context,
            output_dir=output.root,
            max_temporary_frame_offset=5.0,
        )
        base_path = "/media/base.mkv"
        source_path = "/media/source.mkv"

        with patch(
            "twotone.tools.melt.melt_performer.PairMatcher",
        ) as matcher_type, patch.object(
            video_utils,
            "get_video_duration",
            return_value=6000,
        ):
            matcher_type.return_value.create_segments_mapping.return_value = Mock()
            performer._pair_match(
                _GroupContext(
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
        self.assertEqual(
            matcher_type.call_args.kwargs["max_temporary_frame_offset"],
            5.0,
        )

    def test_pair_match_takes_source_duration_from_plan_track_length(self):
        # The coverage report compares the two video tracks, so the source
        # length comes from the same plan data as the base length, not from
        # a container that also spans a longer audio track.
        performer = self._make_performer()
        base_path = "/media/base.mkv"
        source_path = "/media/source.mkv"
        group = _GroupContext(
            file_ids={base_path: 1, source_path: 2},
            base_video=VideoStreamRef(base_path, 0, 0, None),
            files_details={
                source_path: {"tracks": {"video": [{"tid": 0, "length": 5960}]}},
            },
        )

        with patch("twotone.tools.melt.melt_performer.PairMatcher") as matcher_type, \
             patch.object(video_utils, "get_video_duration", return_value=6000):
            matcher_type.return_value.create_segments_mapping.return_value = Mock()
            result = performer._pair_match(group, "/tmp/matching", source_path)

        self.assertEqual(result.source_duration, 5960)

    def test_process_duplicates_rejects_output_aliasing_input(self):
        performer = self._make_performer()
        title_dir = os.path.join(performer.output_dir, "Title")
        os.makedirs(title_dir)
        input_path = os.path.join(title_dir, "Movie.mkv")
        with open(input_path, "wb") as input_file:
            input_file.write(b"source")
        input_alias = os.path.join(performer.output_dir, "movie-source.mkv")
        os.symlink(input_path, input_alias)
        plan = (MeltTitlePlan(
            title="Title",
            groups=(MeltGroupPlan(
                input_files=MeltInputFiles((input_alias,)),
                output_name="Movie",
                video_streams=(VideoStreamRef(input_alias, 0, 0, None),),
            ),),
        ),)

        with patch.object(process_utils, "start_process") as start_process:
            with self.assertRaisesRegex(RuntimeError, "aliases input"):
                performer.process_duplicates(plan)

        start_process.assert_not_called()
        with open(input_path, "rb") as input_file:
            self.assertEqual(input_file.read(), b"source")

    def test_process_duplicates_remuxes_single_mp4_input(self):
        source_path = "/library/Movie.mp4"

        args = self._process_single_source_plan(
            source_path,
            {"video": [(source_path, 0, 0, None)]},
            [_StreamEntry("video", 0, source_path, None)],
        )

        self.assertEqual(args[-1], "0:0")
        self.assertIn(source_path, args)
        self.assertEqual(args[args.index("--video-tracks") + 1], "0")

    def test_process_duplicates_remuxes_single_mkv_with_unselected_tracks(self):
        source_path = "/library/Movie.mkv"

        args = self._process_single_source_plan(
            source_path,
            {
                "video": [(source_path, 0, 0, None)],
                "audio": [(source_path, 2, 2, "eng")],
            },
            [
                _StreamEntry("video", 0, source_path, None),
                _StreamEntry("audio", 2, source_path, "eng"),
            ],
        )

        self.assertEqual(args[args.index("--audio-tracks") + 1], "2")
        self.assertNotIn("1", args[args.index("--audio-tracks") + 1].split(","))

    def test_process_duplicates_remuxes_single_mkv_with_metadata_and_sync_changes(self):
        source_path = "/library/Movie.mkv"

        args = self._process_single_source_plan(
            source_path,
            {"audio": [(source_path, 1, 1, "pol")]},
            [_StreamEntry("audio", 1, source_path, "pol", 125)],
        )

        self.assertEqual(args[args.index("--language") + 1], "1:pol")
        self.assertEqual(args[args.index("--sync") + 1], "1:125")

    def test_process_duplicates_extracts_selected_chapters(self):
        performer = self._make_performer()
        source_path = "/library/Movie.mkv"
        plan = (MeltTitlePlan(
            title="Title",
            groups=(MeltGroupPlan(
                input_files=MeltInputFiles((source_path,)),
                output_name="Movie",
                chapter_source=source_path,
                video_streams=(VideoStreamRef(source_path, 0, 0, None),),
            ),),
        ),)
        prepared = Mock(
            entries=[_StreamEntry("video", 0, source_path, None)],
            input_files={source_path},
        )
        captured_mkvmerge_args: list[str] = []

        def run_process(tool, args, **_kwargs):
            if tool == "mkvextract":
                with open(args[-1], "w") as chapters_file:
                    chapters_file.write("<Chapters />")
            else:
                captured_mkvmerge_args.extend(args)
                with open(args[1], "wb") as output_file:
                    output_file.write(b"muxed")
            return _FAKE_PROCESS_OK

        with patch.object(performer, "_prepare_stream_entries", return_value=prepared), \
             patch.object(process_utils, "start_process", side_effect=run_process), \
             patch.object(video_utils, "validate_media_output"):
            performer.process_duplicates(plan)

        self.assertEqual(captured_mkvmerge_args[captured_mkvmerge_args.index(source_path) - 1], "--no-chapters")
        chapter_index = captured_mkvmerge_args.index("--chapters")
        self.assertTrue(captured_mkvmerge_args[chapter_index + 1].endswith(".xml"))

    def test_process_duplicates_failed_mux_preserves_existing_output(self):
        performer = self._make_performer()
        input_a = os.path.join(performer.output_dir, "a.mkv")
        input_b = os.path.join(performer.output_dir, "b.mkv")
        for path in (input_a, input_b):
            with open(path, "wb") as input_file:
                input_file.write(b"input")
        output_path = os.path.join(performer.output_dir, "Title", "Movie.mkv")
        os.makedirs(os.path.dirname(output_path))
        with open(output_path, "wb") as output_file:
            output_file.write(b"previous output")
        plan = (MeltTitlePlan(
            title="Title",
            groups=(MeltGroupPlan(
                input_files=MeltInputFiles((input_a, input_b)),
                output_name="Movie",
                video_streams=(VideoStreamRef(input_a, 0, 0, None),),
                audio_streams=(AudioStreamRef(input_b, 1, 1, "eng"),),
            ),),
        ),)
        prepared = Mock(entries=[
            _StreamEntry("video", 0, input_a, None),
            _StreamEntry("audio", 1, input_b, "eng"),
        ], input_files={input_a, input_b})

        with patch.object(performer, "_prepare_stream_entries", return_value=prepared), \
             patch.object(performer, "build_mkvmerge_args", return_value=[]), \
             patch.object(process_utils, "start_process", return_value=Mock(returncode=2, stderr="failed")):
            with self.assertRaises(RuntimeError):
                performer.process_duplicates(plan)

        with open(output_path, "rb") as output_file:
            self.assertEqual(output_file.read(), b"previous output")

    def test_process_duplicates_failed_validation_preserves_output_and_removes_staging(self):
        performer = self._make_performer()
        input_path = os.path.join(performer.output_dir, "input.mkv")
        with open(input_path, "wb") as input_file:
            input_file.write(b"input")
        output_path = os.path.join(performer.output_dir, "Title", "Movie.mkv")
        os.makedirs(os.path.dirname(output_path))
        with open(output_path, "wb") as output_file:
            output_file.write(b"previous output")
        plan = (MeltTitlePlan(
            title="Title",
            groups=(MeltGroupPlan(
                input_files=MeltInputFiles((input_path,)),
                output_name="Movie",
                video_streams=(VideoStreamRef(input_path, 0, 0, None),),
                files_details={
                    input_path: {
                        "tracks": {"video": [{"tid": 0, "length": 2000}]},
                    },
                },
            ),),
        ),)
        prepared = Mock(
            entries=[_StreamEntry("video", 0, input_path, None)],
            input_files={input_path},
        )

        def create_truncated_output(_tool, args, **_kwargs):
            with open(args[1], "wb") as staged_file:
                staged_file.write(b"probeable but truncated")
            return _FAKE_PROCESS_OK

        with patch.object(performer, "_prepare_stream_entries", return_value=prepared), \
             patch.object(process_utils, "start_process", side_effect=create_truncated_output), \
             patch.object(
                 video_utils,
                 "validate_media_output",
                 side_effect=RuntimeError("missing content near its expected end"),
             ) as validate:
            with self.assertRaisesRegex(RuntimeError, "missing content"):
                performer.process_duplicates(plan)

        staged_path = validate.call_args.args[0]
        self.assertEqual(validate.call_args.kwargs["expected_stream_counts"], {"video": 1})
        self.assertEqual(validate.call_args.kwargs["expected_duration_ms"], 2000)
        self.assertFalse(os.path.exists(staged_path))
        with open(output_path, "rb") as output_file:
            self.assertEqual(output_file.read(), b"previous output")

    def test_stream_sorting_puts_unknown_languages_last(self):
        streams = [
            _StreamEntry("audio", 1, "/a.mkv", None),
            _StreamEntry("audio", 2, "/a.mkv", "eng"),
            _StreamEntry("subtitle", 3, "/a.mkv", None),
            _StreamEntry("subtitle", 4, "/a.mkv", "pol"),
            _StreamEntry("subtitle", 5, "/a.mkv", "deu"),
        ]

        sort_key = lambda stream: (stream.language is None, stream.language or "")
        result = sorted(streams, key=sort_key)

        languages = [s.language for s in result]
        self.assertEqual(languages, ["deu", "eng", "pol", None, None])

    def test_stream_sorting_alphabetical_when_all_known(self):
        streams = [
            _StreamEntry("subtitle", 1, "/a.mkv", "pol"),
            _StreamEntry("subtitle", 2, "/a.mkv", "eng"),
            _StreamEntry("subtitle", 3, "/a.mkv", "deu"),
            _StreamEntry("audio", 4, "/a.mkv", "jpn"),
        ]

        sort_key = lambda stream: (stream.language is None, stream.language or "")
        result = sorted(streams, key=sort_key)

        languages = [s.language for s in result]
        self.assertEqual(languages, ["deu", "eng", "jpn", "pol"])

    def test_build_mkvmerge_args_track_order_respects_unknown_last(self):
        performer = self._make_performer()

        file_a = "/tmp/a.mkv"
        file_b = "/tmp/b.mkv"

        streams_list_sorted = [
            _StreamEntry("video", 0, file_a, None),
            _StreamEntry("audio", 1, file_a, "eng"),
            _StreamEntry("subtitle", 3, file_a, "deu"),
            _StreamEntry("subtitle", 4, file_a, "pol"),
            _StreamEntry("subtitle", 5, file_a, None),
        ]

        args = performer.build_mkvmerge_args(
            "/tmp/out.mkv",
            streams_list_sorted,
            attachments=[],
            preferred_audio=None,
            required_input_files=[file_a],
        )

        # Track order should preserve the sorted order
        track_order_idx = args.index("--track-order")
        track_order = args[track_order_idx + 1]
        self.assertEqual(track_order, "0:0,0:1,0:3,0:4,0:5")

    def test_build_mkvmerge_args_uses_only_explicit_chapter_file(self):
        performer = self._make_performer()
        first_file = "/tmp/first.mkv"
        second_file = "/tmp/second.mkv"

        args = performer.build_mkvmerge_args(
            "/tmp/out.mkv",
            [
                _StreamEntry("video", 0, first_file, None),
                _StreamEntry("audio", 1, second_file, "eng"),
            ],
            attachments=[],
            preferred_audio=None,
            required_input_files=[first_file, second_file],
            chapters_file="/tmp/chapters.xml",
        )

        self.assertEqual(args[:4], ["-o", "/tmp/out.mkv", "--chapters", "/tmp/chapters.xml"])
        for file_path in (first_file, second_file):
            self.assertEqual(args[args.index(file_path) - 1], "--no-chapters")

    def test_mapping_stream_bias_measures_rebased_frame_times(self):
        performer = self._make_performer()
        fake_info = {"streams": [{"codec_type": "video", "start_time": "0.500"}]}

        with patch.object(video_utils, "get_video_full_info", return_value=fake_info):
            rebased = performer._mapping_stream_bias_ms("/tmp/a.mkv", {0: {}, 41: {}})
            preserved = performer._mapping_stream_bias_ms("/tmp/a.mp4", {500: {}, 541: {}})

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
        performer = self._make_performer()
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
            effective_fps = performer._effective_video_fps(
                "/tmp/source.mkv",
                {0: {}, 500: {}, 1000: {}},
            )

        self.assertEqual(effective_fps, 2.0)

    def test_effective_video_fps_requires_video_stream(self):
        performer = self._make_performer()
        with patch.object(
            video_utils,
            "get_video_full_info",
            return_value={"streams": [{"codec_type": "audio"}]},
        ):
            effective_fps = performer._effective_video_fps(
                "/tmp/source.mkv",
                {0: {}, 1000: {}},
            )

        self.assertIsNone(effective_fps)

    @staticmethod
    def _execute_performer_global_time_scale(
        performer,
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
            logger=performer.logger,
        )
        if base_duration_ms is None:
            raise RuntimeError(
                f"Could not determine base video duration for {base_video.path}."
            )
        request = performer._create_audio_patch_request(
            working_dir,
            base_video,
            source_video,
            source_audio,
            base_audio,
            output_path,
            audio_strategy.AudioPlan(_AudioStrategy.GLOBAL_TIME_SCALE, segment_pairs),
            base_duration_ms,
            fill_gaps_from_base=fill_gaps_from_base,
        )
        return performer.audio_patcher.execute(request).timeline_start_ms

    def test_video_to_audio_timeline_accounts_for_rebased_video_and_delayed_audio(self):
        performer = self._make_performer()
        source_info = {
            "streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.500000"},
                {"codec_type": "audio", "index": 1, "start_time": "0.488000"},
            ],
        }

        with patch.object(video_utils, "get_video_full_info", return_value=source_info):
            timeline = performer._video_to_audio_timeline(
                VideoStreamRef("/tmp/source.mkv", 0, 0, None),
                AudioStreamRef("/tmp/source.mkv", 1, 1, "pol"),
                {0: {"path": None}, 41: {"path": None}},
            )

        self.assertEqual(timeline.mapping_to_video_ms, 500)
        self.assertEqual(timeline.video_to_stream_ms(0), 500)
        self.assertEqual(timeline.video_to_rebased_audio_ms(0), 12)
        self.assertEqual(timeline.rebased_audio_to_video_ms(12), 0)

    def test_video_to_audio_timeline_uses_priming_aware_aac_content_start(self):
        performer = self._make_performer()
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
             patch.object(performer.timeline, "aac_priming_exposed", return_value=True):
            timeline = performer._video_to_audio_timeline(
                VideoStreamRef("/tmp/source.mkv", 0, 0, None),
                AudioStreamRef("/tmp/source.mkv", 1, 1, "pol"),
                {3: {"path": None}},
            )

        self.assertEqual(timeline.mapping_to_video_ms, 475)
        self.assertEqual(timeline.audio_content_start_ms, 521)
        self.assertEqual(timeline.video_to_rebased_audio_ms(3), -43)

    def test_build_mkvmerge_args_applies_sync_offset(self):
        performer = self._make_performer()

        patched_file = "/tmp/patched.mka"
        base_file = "/tmp/base.mkv"

        streams = [
            _StreamEntry("video", 0, base_file, None),
            _StreamEntry("audio", 0, patched_file, "pol", sync_offset_ms=3000),
        ]

        args = performer.build_mkvmerge_args(
            "/tmp/out.mkv",
            streams,
            attachments=[],
            preferred_audio=None,
            required_input_files=[base_file, patched_file],
        )

        self.assertIn("--sync", args, "Should contain --sync flag")
        sync_idx = args.index("--sync")
        self.assertEqual(args[sync_idx + 1], "0:3000", "Sync should be TID:offset")

    def test_build_mkvmerge_args_applies_sync_offset_zero(self):
        performer = self._make_performer()

        patched_file = "/tmp/patched.mka"
        base_file = "/tmp/base.mkv"

        streams = [
            _StreamEntry("video", 0, base_file, None),
            _StreamEntry("audio", 0, patched_file, "pol", sync_offset_ms=0),
        ]

        args = performer.build_mkvmerge_args(
            "/tmp/out.mkv",
            streams,
            attachments=[],
            preferred_audio=None,
            required_input_files=[base_file, patched_file],
        )

        self.assertIn("--sync", args, "Should contain --sync flag even for offset 0")
        sync_idx = args.index("--sync")
        self.assertEqual(args[sync_idx + 1], "0:0", "Sync should be TID:0")

    def test_build_mkvmerge_args_applies_stream_sync_offsets_independently(self):
        performer = self._make_performer()

        base_file = "/tmp/base.mov"
        patched_file = "/tmp/patched.mka"

        streams = [
            _StreamEntry("video", 0, base_file, None, sync_offset_ms=492),
            _StreamEntry("audio", 1, base_file, "pol", sync_offset_ms=471),
            _StreamEntry("subtitle", 2, base_file, "eng", sync_offset_ms=250),
            _StreamEntry("audio", 0, patched_file, "eng", sync_offset_ms=500),
        ]

        args = performer.build_mkvmerge_args(
            "/tmp/out.mkv",
            streams,
            attachments=[],
            preferred_audio=None,
            required_input_files=[base_file, patched_file],
        )

        sync_values = [
            args[index + 1]
            for index, value in enumerate(args)
            if value == "--sync"
        ]
        self.assertIn("0:492", sync_values)
        self.assertIn("1:471", sync_values)
        self.assertIn("2:250", sync_values)
        self.assertIn("0:500", sync_values)

    def test_build_mkvmerge_args_preserves_rewritten_subtitle_metadata(self):
        performer = self._make_performer()
        subtitle_path = "/tmp/aligned.ass"

        args = performer.build_mkvmerge_args(
            "/tmp/out.mkv",
            [
                _StreamEntry(
                    "subtitle",
                    0,
                    subtitle_path,
                    "pol",
                    name="Forced signs",
                    forced=True,
                    default=False,
                ),
            ],
            attachments=[],
            preferred_audio=None,
            required_input_files=[subtitle_path],
        )

        self.assertIn("0:Forced signs", args)
        forced_index = args.index("--forced-display-flag")
        self.assertEqual(args[forced_index + 1], "0:yes")
        default_index = args.index("--default-track")
        self.assertEqual(args[default_index + 1], "0:no")

    def test_build_mkvmerge_args_applies_track_sync_offset_only_to_selected_track(self):
        performer = self._make_performer()

        base_file = "/tmp/base.mov"

        streams = [
            _StreamEntry("video", 0, base_file, None, sync_offset_ms=492),
            _StreamEntry("audio", 1, base_file, "pol"),
        ]

        args = performer.build_mkvmerge_args(
            "/tmp/out.mkv",
            streams,
            attachments=[],
            preferred_audio=None,
            required_input_files=[base_file],
        )

        sync_values = [
            args[index + 1]
            for index, value in enumerate(args)
            if value == "--sync"
        ]
        self.assertIn("0:492", sync_values)
        self.assertNotIn("1:492", sync_values)

    def test_decode_audio_uses_ffprobe_index_when_mov_has_intervening_data_stream(self):
        performer = self._make_performer()
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mov")
            ordered_path = os.path.join(tmpdir, "video-data-audio.mov")
            decoded_path = os.path.join(tmpdir, "decoded.flac")
            expected_pcm = os.path.join(tmpdir, "expected.pcm")
            actual_pcm = os.path.join(tmpdir, "actual.pcm")
            run_ffmpeg([
                "-y",
                "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=2",
                "-map", "0:v", "-map", "1:a",
                "-c:v", "mpeg4", "-c:a", "pcm_s16le",
                "-timecode", "00:00:00:00",
                source_path,
            ], expected_path=source_path)
            run_ffmpeg([
                "-y", "-i", source_path,
                "-map", "0:v", "-map", "0:d", "-map", "0:a",
                "-c", "copy",
                ordered_path,
            ], expected_path=ordered_path)

            details = video_utils.get_video_data_mkvmerge(
                ordered_path,
                enrich=True,
                logger=performer.logger,
            )
            audio = details["tracks"]["audio"][0]
            self.assertEqual((audio["tid"], audio["ffprobe_stream_index"]), (1, 2))
            duplicates = StaticSource(performer.interruption)
            duplicates.add_metadata(ordered_path, "audio_lang", "eng")
            picker = StreamsPicker(
                performer.logger.getChild("StreamsPicker"),
                duplicates,
            )
            selected_video, selected_audio, _ = picker.pick_streams(
                {ordered_path: details["tracks"]},
                {ordered_path: 1},
            )
            self.assertEqual(
                selected_audio,
                [AudioStreamRef(ordered_path, 1, 2, "eng")],
            )
            prepared = performer._prepare_stream_entries(
                selected_video,
                selected_audio,
                [],
                [],
                {ordered_path: 1},
                {ordered_path: details},
            )
            mux_args = performer.build_mkvmerge_args(
                os.path.join(tmpdir, "output.mkv"),
                prepared.entries,
                attachments=[],
                preferred_audio=None,
                required_input_files=prepared.input_files,
            )
            self.assertEqual(mux_args[mux_args.index("--audio-tracks") + 1], "1")
            performer.timeline.decode_audio_stream_to_flac(
                selected_audio[0],
                decoded_path,
            )

            for input_path, output_path, stream_index in (
                (ordered_path, expected_pcm, 2),
                (decoded_path, actual_pcm, 0),
            ):
                run_ffmpeg([
                    "-y", "-i", input_path,
                    "-map", f"0:{stream_index}",
                    "-c:a", "pcm_s32le", "-f", "s32le",
                    output_path,
                ], expected_path=output_path)

            with open(expected_pcm, "rb") as expected, open(actual_pcm, "rb") as actual:
                self.assertEqual(actual.read(), expected.read())

    def test_length_matching_audio_uses_direct_passthrough_when_safe(self):
        performer = self._make_performer()
        base_video = "/tmp/base.mkv"
        source_audio = "/tmp/source.mkv"

        with patch.object(performer, "_video_track_duration", return_value=6000), \
             patch.object(performer, "_base_output_end_ms", return_value=6000), \
             patch.object(performer, "_base_audio_end_ms", return_value=6000), \
             patch.object(performer.timeline, "source_stream_start_offset_ms", return_value=0), \
             patch.object(performer.timeline, "source_stream_end_offset_ms", return_value=5900), \
             patch.object(performer.timeline, "audio_content_start_ms", return_value=0), \
             patch.object(
                 performer.timeline,
                 "stream_info",
                 return_value={"codec_name": "aac", "initial_padding": 0},
             ), \
             patch.object(performer.timeline, "track_sync_offset_ms", return_value=None), \
             patch.object(performer.audio_patcher, "prepare_normalized_unscaled_audio", side_effect=AssertionError):
            prepared = performer._prepare_stream_entries(
                video_streams=[VideoStreamRef(base_video, 0, 0, None)],
                audio_streams=[AudioStreamRef(source_audio, 1, 1, "eng")],
                subtitle_streams=[],
                attachments=[],
                file_ids={base_video: 1, source_audio: 2},
                files_details={},
            )

        self.assertIn(_StreamEntry("audio", 1, source_audio, "eng", None), prepared.entries)
        self.assertEqual(prepared.input_files, {base_video, source_audio})

    def test_base_matroska_audio_with_aac_priming_uses_direct_passthrough(self):
        performer = self._make_performer()
        base_video = "/tmp/base.mkv"

        with patch.object(performer, "_video_track_duration", return_value=6000), \
             patch.object(performer, "_base_output_end_ms", return_value=6000), \
             patch.object(performer, "_base_audio_end_ms", return_value=5900), \
             patch.object(performer.timeline, "audio_content_start_ms", return_value=0), \
             patch.object(performer.timeline, "source_stream_start_offset_ms", return_value=0), \
             patch.object(
                 performer.timeline,
                 "stream_info",
                 return_value={"codec_name": "aac", "initial_padding": 1024},
             ), \
             patch.object(performer.timeline, "track_sync_offset_ms", return_value=None), \
             patch.object(performer.audio_patcher, "prepare_normalized_unscaled_audio", side_effect=AssertionError):
            prepared = performer._prepare_stream_entries(
                video_streams=[VideoStreamRef(base_video, 0, 0, None)],
                audio_streams=[AudioStreamRef(base_video, 1, 1, "pol")],
                subtitle_streams=[],
                attachments=[],
                file_ids={base_video: 1},
                files_details={},
            )

        self.assertIn(_StreamEntry("audio", 1, base_video, "pol", None), prepared.entries)
        self.assertEqual(prepared.input_files, {base_video})

    def test_prepare_stream_entries_includes_final_attachment_sources(self):
        performer = self._make_performer()
        base_video = "/tmp/base.mkv"
        attachment_source = "/tmp/cover.mkv"

        with patch.object(performer, "_video_track_duration", return_value=6000), \
             patch.object(performer, "_base_output_end_ms", return_value=6000), \
             patch.object(performer, "_base_audio_end_ms", return_value=None), \
             patch.object(performer.timeline, "source_stream_start_offset_ms", return_value=0), \
             patch.object(performer.timeline, "track_sync_offset_ms", return_value=None):
            prepared = performer._prepare_stream_entries(
                video_streams=[VideoStreamRef(base_video, 0, 0, None)],
                audio_streams=[],
                subtitle_streams=[],
                attachments=[AttachmentRef(attachment_source, 3)],
                file_ids={base_video: 1, attachment_source: 2},
                files_details={},
            )

        self.assertEqual(prepared.input_files, {base_video, attachment_source})

    @parameterized.expand([
        ("normalized_then_passthrough", [1, 2]),
        ("passthrough_then_normalized", [2, 1]),
    ])
    def test_prepare_stream_entries_keeps_source_used_by_other_audio(
        self,
        _name,
        stream_order,
    ):
        performer = self._make_performer()
        base_video = "/tmp/base.mkv"
        source_path = "/tmp/source.mkv"
        normalized_path = "/tmp/normalized.mka"
        audio_streams = [
            AudioStreamRef(
                source_path,
                stream_index,
                stream_index,
                "eng" if stream_index == 1 else "pol",
            )
            for stream_index in stream_order
        ]

        def normalization_reason(stream):
            if stream.mkvmerge_track_id == 1:
                return "normalizing AAC timing"
            return None

        def normalize(stream, **_kwargs):
            return AudioStreamRef(normalized_path, 0, 0, stream.language)

        with self.assertLogs(performer.logger, level=logging.INFO) as logs, \
             patch.object(performer, "_video_track_duration", return_value=6000), \
             patch.object(performer, "_base_output_end_ms", return_value=6000), \
             patch.object(performer, "_base_audio_end_ms", return_value=None), \
             patch.object(performer.timeline, "source_stream_end_offset_ms", return_value=5900), \
             patch.object(performer.timeline, "audio_content_start_ms", return_value=0), \
             patch.object(performer.timeline, "stream_info", return_value={}), \
             patch.object(performer.timeline, "track_sync_offset_ms", return_value=None), \
             patch.object(performer.timeline, "audio_mkvmerge_normalization_reason", side_effect=normalization_reason), \
             patch.object(performer.audio_patcher, "prepare_normalized_unscaled_audio", side_effect=normalize):
            prepared = performer._prepare_stream_entries(
                video_streams=[VideoStreamRef(base_video, 0, 0, None)],
                audio_streams=audio_streams,
                subtitle_streams=[],
                attachments=[],
                file_ids={base_video: 1, source_path: 2},
                files_details={},
            )

        self.assertIn(
            "Preparing audio track #1 from file #2 for muxing: "
            "normalizing AAC timing.",
            "\n".join(logs.output),
        )

        self.assertEqual(prepared.input_files, {base_video, source_path, normalized_path})
        self.assertIn(_StreamEntry("audio", 0, normalized_path, "eng", None), prepared.entries)
        self.assertIn(_StreamEntry("audio", 2, source_path, "pol", None), prepared.entries)
        args = performer.build_mkvmerge_args(
            "/tmp/output.mkv",
            prepared.entries,
            attachments=[],
            preferred_audio=None,
            required_input_files=prepared.input_files,
        )
        self.assertEqual(args.count(source_path), 1)
        self.assertEqual(args.count(normalized_path), 1)

    def test_patch_mismatched_audio_uses_unique_temp_paths_for_same_stream_index(self):
        performer = self._make_performer()
        base_video = "/tmp/base.mkv"
        audio_a = "/tmp/audio-a.mkv"
        audio_b = "/tmp/audio-b.avi"
        base_duration = 60000
        # A real speed change (spans differ) so the GLOBAL_LINEAR relation
        # routes to the global-time-scale patcher whose output path we probe.
        mapping = [(0, 0), (base_duration, 58000)]

        def make_match_result(source_duration: int) -> _PairMatchResult:
            return _PairMatchResult(
                matching=SegmentsMappingResult(
                    mapping=mapping,
                    lhs_all_frames={},
                    rhs_all_frames={},
                    relation=MappingRelation.GLOBAL_LINEAR,
                    lhs_fps=24.0,
                    rhs_fps=24.0,
                ),
                source_duration=source_duration,
            )

        group = _GroupContext(
            file_ids={base_video: 1, audio_a: 2, audio_b: 3},
            base_video=VideoStreamRef(base_video, 0, 0, None),
            base_duration_ms=base_duration,
        )
        group.pair_matches[audio_a] = make_match_result(base_duration)
        group.pair_matches[audio_b] = make_match_result(base_duration)

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
             patch.object(performer, "_create_audio_patch_request", side_effect=fake_request), \
            patch.object(performer.audio_patcher, "execute", side_effect=fake_execute):
            first = performer._patch_mismatched_audio(
                group, VideoStreamRef(audio_a, 0, 0, None), AudioStreamRef(audio_a, 1, 1, "pol"),
            )
            second = performer._patch_mismatched_audio(
                group, VideoStreamRef(audio_b, 0, 0, None), AudioStreamRef(audio_b, 1, 1, "pol"),
            )

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

    def test_patch_mismatched_audio_rejects_absolute_global_linear_frame_step(self):
        performer = self._make_performer()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        source_audio = AudioStreamRef(source_video.path, 1, 1, "pol")
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
        group = _GroupContext(
            file_ids={base_video.path: 1, source_video.path: 2},
            base_video=base_video,
            base_duration_ms=500000,
        )
        group.pair_matches[source_video.path] = _PairMatchResult(
            matching=SegmentsMappingResult(
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
            ),
            source_duration=500000,
        )

        with patch.object(debug_routines, "log_coverage"), \
             self.assertRaisesRegex(RuntimeError, "holes in the scene sequence"):
            performer._patch_mismatched_audio(
                group, source_video, source_audio,
            )

    def test_patch_mismatched_audio_rejects_sparse_global_linear_frame_step(self):
        performer = self._make_performer()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        mapping = [(0, 0), (200000, 200000), (400000, 399400), (600000, 599400)]
        group = _GroupContext(
            file_ids={base_video.path: 1, source_video.path: 2},
            base_video=base_video,
            base_duration_ms=600000,
        )
        group.pair_matches[source_video.path] = _PairMatchResult(
            matching=SegmentsMappingResult(
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
            ),
            source_duration=600000,
        )
        with patch.object(debug_routines, "log_coverage"), \
             patch.object(performer, "_plan_audio", side_effect=AssertionError("unsafe mapping reached audio strategy")) as choose, \
             patch.object(performer.audio_patcher, "execute") as execute, \
             self.assertRaisesRegex(RuntimeError, "discontinuit.*not supported yet"):
            performer._patch_mismatched_audio(
                group, source_video, AudioStreamRef(source_video.path, 1, 1, "pol"),
            )
        choose.assert_not_called()
        execute.assert_not_called()

    @parameterized.expand([
        ("global", MappingRelation.GLOBAL_LINEAR),
        ("generic", MappingRelation.GENERIC),
    ])
    def test_patch_mismatched_audio_rejects_unsupported_pts_step_before_strategy(self, _case, relation):
        performer = self._make_performer()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        timing_step = (1600000, 1700000, 1600000, 1703000, -3000)
        group = _GroupContext(
            file_ids={base_video.path: 1, source_video.path: 2},
            base_video=base_video,
            base_duration_ms=2000000,
        )
        group.pair_matches[source_video.path] = _PairMatchResult(
            matching=SegmentsMappingResult(
                mapping=[(0, 0), (1600000, 1600000), (1700000, 1703000)],
                lhs_all_frames={},
                rhs_all_frames={},
                relation=relation,
                lhs_fps=25.0, rhs_fps=25.0, frame_slope=1.0,
                timing_discontinuities=(timing_step,),
                timing_discontinuity_debug_path="/tmp/timing-debug",
            ),
            source_duration=2000000,
        )

        with patch.object(performer, "_plan_audio", side_effect=AssertionError("unsafe mapping reached audio strategy")) as choose, \
             self.assertLogs(performer.logger, level="ERROR") as captured, \
             self.assertRaisesRegex(RuntimeError, "Unsupported audio timeline discontinuity"):
            performer._patch_mismatched_audio(
                group, source_video, AudioStreamRef(source_video.path, 1, 1, "pol"),
            )
        choose.assert_not_called()
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

    def test_patch_mismatched_audio_rejects_pts_step_with_real_global_fit(self):
        performer = self._make_performer()
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
            generic_utils.InterruptibleProcess(), performer.workspace.root,
            base_video.path, source_video.path, performer.logger,
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
        # The performer enforces PairMatcher's final timing verdict without
        # rerunning detection or reaching an audio strategy.
        group = _GroupContext(
            file_ids={base_video.path: 1, source_video.path: 2},
            base_video=base_video,
            base_duration_ms=1920000,
        )
        group.pair_matches[source_video.path] = _PairMatchResult(
            matching=SegmentsMappingResult(
                mapping=pairs,
                lhs_all_frames=lhs_frames,
                rhs_all_frames=rhs_frames,
                relation=MappingRelation.GLOBAL_LINEAR,
                lhs_fps=25.0,
                rhs_fps=25.0,
                frame_slope=fit.slope,
                timing_discontinuities=tuple(timing_discontinuities),
            ),
            source_duration=1923000,
        )
        with patch.object(video_utils, "get_video_full_info", return_value={}), \
             patch.object(performer, "_create_audio_patch_request") as request, \
             patch.object(performer.audio_patcher, "execute") as execute, \
             self.assertRaisesRegex(RuntimeError, "Unsupported audio timeline discontinuity"):
            performer._patch_mismatched_audio(
                group, source_video, AudioStreamRef(source_video.path, 1, 1, "pol"),
            )
        request.assert_not_called()
        execute.assert_not_called()

    def test_patch_mismatched_audio_uses_pair_matcher_timing_verdict_without_reclassification(self):
        performer = self._make_performer()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        mapping = [(0, 0), (60000, 60000)]
        group = _GroupContext(
            file_ids={base_video.path: 1, source_video.path: 2},
            base_video=base_video,
            base_duration_ms=60000,
        )
        group.pair_matches[source_video.path] = _PairMatchResult(
            matching=SegmentsMappingResult(
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
            ),
            source_duration=60000,
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
                 side_effect=AssertionError("performer must use PairMatcher's final verdict"),
             ), \
             patch.object(
                 performer,
                 "_plan_audio",
                 return_value=audio_strategy.AudioPlan(_AudioStrategy.UNSCALED, mapping),
             ), \
             patch.object(performer, "_create_audio_patch_request", return_value=request), \
             patch.object(performer.audio_patcher, "execute", return_value=result):
            actual = performer._patch_mismatched_audio(
                group, source_video, source_audio,
            )

        self.assertEqual(actual, result)

    def test_unscaled_audio_patch_routes_through_shared_executor(self):
        performer = self._make_performer()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        mapping = [(0, 0), (60000, 60000)]
        group = _GroupContext(
            file_ids={base_video.path: 1, source_audio.path: 2},
            base_video=base_video,
            base_duration_ms=60000,
        )
        group.pair_matches[source_audio.path] = _PairMatchResult(
            matching=SegmentsMappingResult(
                mapping=mapping,
                lhs_all_frames={},
                rhs_all_frames={},
                relation=MappingRelation.GLOBAL_LINEAR,
                lhs_fps=24.0,
                rhs_fps=24.0,
            ),
            source_duration=60000,
        )
        request = Mock(output_path="/tmp/patched.mka", source_audio=source_audio)
        result = AudioPatchResult(
            AudioStreamRef("/tmp/patched.mka", 0, 0, "pol"),
            500,
            59500,
            "unscaled-normalization",
        )

        with patch.object(
                 performer,
                 "_plan_audio",
                 return_value=audio_strategy.AudioPlan(
                     _AudioStrategy.UNSCALED, mapping, unscaled_shift_ms=500,
                 ),
             ), \
             patch.object(performer, "_create_audio_patch_request", return_value=request) as create, \
             patch.object(performer.audio_patcher, "execute", return_value=result) as execute:
            actual = performer._patch_mismatched_audio(
                group, source_video, source_audio,
            )

        self.assertEqual(actual, result)
        self.assertEqual(create.call_args.kwargs["output_end_ms"], 60000)
        self.assertEqual(
            create.call_args.args[6],
            audio_strategy.AudioPlan(_AudioStrategy.UNSCALED, mapping, unscaled_shift_ms=500),
        )
        execute.assert_called_once_with(request)

    def test_audio_patch_request_carries_the_plan(self):
        performer = self._make_performer()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        plan = audio_strategy.AudioPlan(
            _AudioStrategy.SUBSEGMENT,
            [(0, 0), (30000, 30100), (60000, 60900)],
        )

        with patch.object(
                 performer,
                 "_video_to_audio_timeline",
                 return_value=VideoToAudioTimeline(0, 0),
             ), \
             patch.object(performer, "_mapping_to_video_ms", return_value=0):
            request = performer._create_audio_patch_request(
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

        self.assertEqual(request.strategy, _AudioStrategy.SUBSEGMENT)
        self.assertEqual(request.segment_count, audio_strategy.SUBSEGMENT_COUNT)
        self.assertEqual(request.unscaled_shift_ms, 0)
        self.assertEqual(request.mapping, tuple(plan.mapping))
        self.assertEqual(request.target_interval, TimelineInterval(0, 60000))

    def test_subsegment_audio_patch_preserves_selected_stream_identity(self):
        performer = self._make_performer()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, 0, None)
        base_audio = AudioStreamRef("/tmp/base.mkv", 3, 3, "eng")
        source_video = VideoStreamRef("/tmp/source.mkv", 0, 0, None)
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        mapping = [(0, 0), (60000, 60000)]
        group = _GroupContext(
            file_ids={base_video.path: 1, source_audio.path: 2},
            base_video=base_video,
            base_audio=base_audio,
            base_duration_ms=60000,
            base_audio_end_ms=60000,
        )
        group.pair_matches[source_audio.path] = _PairMatchResult(
            matching=SegmentsMappingResult(
                mapping=mapping,
                lhs_all_frames={},
                rhs_all_frames={},
                relation=MappingRelation.GENERIC,
                lhs_fps=24.0,
                rhs_fps=24.0,
            ),
            source_duration=60000,
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
            self.assertEqual(plan.strategy, _AudioStrategy.SUBSEGMENT)
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
                 performer,
                 "_plan_audio",
                 return_value=audio_strategy.AudioPlan(_AudioStrategy.SUBSEGMENT, mapping),
             ), \
             patch.object(performer, "_create_audio_patch_request", side_effect=fake_request), \
             patch.object(performer.audio_patcher, "execute", side_effect=fake_execute):
            patched = performer._patch_mismatched_audio(
                group, source_video, source_audio,
            )

        self.assertEqual(patched.stream.mkvmerge_track_id, 0)
        self.assertEqual(patched.stream.ffprobe_stream_index, 0)
        self.assertEqual(patched.stream.language, "pol")
        self.assertEqual(patched.timeline_start_ms, 125)
        self.assertEqual(patched.duration_ms, 59875)

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
        performer = self._make_performer()

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
                returned_sync_offset = self._execute_performer_global_time_scale(
                    performer, wd, VideoStreamRef("/base.mkv", 0, 0, None),
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
            performer.audio_patcher._global_time_scale_filter(
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
        performer = self._make_performer()

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
                sync = self._execute_performer_global_time_scale(
                    performer, wd, VideoStreamRef("/base.mkv", 0, 0, None),
                    VideoStreamRef("/source.mkv", 0, 0, None),
                    AudioStreamRef("/source.mkv", 1, 1, "pol"), None, output_path, pairs,
                    fill_gaps_from_base=False,
                )

        self.assertEqual(sync, seg1_start,
                         "Without audio start offset, sync offset should be the raw seg1_start")

    def test_patch_audio_fill_gaps_raises_on_start_offset(self):
        """In fill-audio-gaps mode, audio start correction cannot be compensated."""
        performer = self._make_performer()

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
                    self._execute_performer_global_time_scale(
                        performer, wd, VideoStreamRef("/base.mkv", 0, 0, None),
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
