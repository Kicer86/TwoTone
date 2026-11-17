import logging
import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from common import run_ffmpeg
from parameterized import parameterized

from melt.helpers import MeltComponentFactories, _FAKE_PROCESS_OK
from twotone.tools.melt import debug_routines, mapping_validation
from twotone.tools.melt.audio_patcher import AudioPatchResult
from twotone.tools.melt.static_source import StaticSource
from twotone.tools.melt.streams_picker import StreamsPicker
from twotone.tools.melt.melt_common import (
    AttachmentRef,
    AudioStreamRef,
    MeltInputFiles,
    SubtitleStreamRef,
    VideoStreamRef,
)
from twotone.tools.melt.melt_performer import _StreamEntry
from twotone.tools.melt.melt_plan import MeltGroupPlan, MeltTitlePlan
from twotone.tools.melt.pair_matcher import MappingRelation, SegmentsMappingResult
from twotone.tools.utils import process_utils, video_utils


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

    @staticmethod
    def _aligned_files_details(*paths: str, subtitle_format: str | None = None) -> dict:
        details = {}
        for path in paths:
            tracks: dict = {"video": [{"tid": 0, "ffprobe_stream_index": 0, "length": 6000}]}
            if subtitle_format is not None:
                tracks["subtitle"] = [{"tid": 3, "format": subtitle_format}]
            details[path] = {"tracks": tracks}
        return details

    @staticmethod
    def _aligned_match(mapping: list[tuple[int, int]]) -> SegmentsMappingResult:
        return SegmentsMappingResult(
            mapping=mapping,
            lhs_all_frames={},
            rhs_all_frames={},
            relation=MappingRelation.GLOBAL_LINEAR,
            lhs_fps=25.0,
            rhs_fps=25.0,
        )

    @contextmanager
    def _stream_preparation_probes(self, performer):
        with patch.object(performer, "_base_output_end_ms", return_value=None), \
             patch.object(performer, "_base_audio_end_ms", return_value=None), \
             patch.object(performer.timeline, "source_stream_start_offset_ms", return_value=0), \
             patch.object(performer.timeline, "audio_stream_info", return_value={}), \
             patch.object(performer.timeline, "audio_content_start_ms", return_value=0), \
             patch.object(performer.timeline, "track_sync_offset_ms", return_value=None), \
             patch.object(debug_routines, "log_coverage"):
            yield

    def test_prepare_stream_entries_refuses_a_source_before_aligning_any_track(self):
        # A refused source fails the group before another source's audio is
        # patched, so no minutes of audio work precede the refusal.
        performer = self._make_performer()
        base, first, second = "/tmp/base.mkv", "/tmp/first.mkv", "/tmp/second.mkv"
        matches = {
            first: self._aligned_match([(0, 0), (6000, 6000)]),
            second: self._aligned_match([]),
        }

        with self._stream_preparation_probes(performer), \
             patch.object(performer.aligner, "_pair_match", side_effect=lambda _g, _wd, path: matches[path]), \
             patch.object(performer.aligner, "patch_audio") as patch_audio, \
             self.assertRaisesRegex(RuntimeError, "Cannot align audio: no shared video timeline mapping"):
            performer._prepare_stream_entries(
                [VideoStreamRef(base, 0, 0, None)],
                [AudioStreamRef(first, 1, 1, "pol"), AudioStreamRef(second, 1, 1, "eng")],
                [],
                [],
                {base: 1, first: 2, second: 3},
                self._aligned_files_details(base, first, second),
                alignment_paths=(first, second),
            )

        patch_audio.assert_not_called()

    def test_prepare_stream_entries_matches_and_validates_each_source_once(self):
        performer = self._make_performer()
        base, source = "/tmp/base.mkv", "/tmp/source.mkv"
        matching = self._aligned_match([(0, 0), (6000, 6000)])
        audio_streams = [AudioStreamRef(source, 1, 1, "pol"), AudioStreamRef(source, 2, 2, "eng")]
        subtitle = SubtitleStreamRef(source, 3, 3, "pol")

        def patch_audio(_group, audio_stream):
            return AudioPatchResult(audio_stream, 0, 6000, "unscaled-normalization")

        with self._stream_preparation_probes(performer), \
             patch.object(performer.aligner, "_pair_match", return_value=matching) as pair_match, \
             patch.object(
                 mapping_validation,
                 "validate_mapping",
                 wraps=mapping_validation.validate_mapping,
             ) as validate, \
             patch.object(performer.aligner, "patch_audio", side_effect=patch_audio) as aligned_audio, \
             patch.object(performer.aligner, "align_subtitle", return_value=(subtitle, {})) as aligned_subtitle:
            performer._prepare_stream_entries(
                [VideoStreamRef(base, 0, 0, None)],
                audio_streams,
                [subtitle],
                [],
                {base: 1, source: 2},
                self._aligned_files_details(base, source, subtitle_format="SubRip/SRT"),
                alignment_paths=(source,),
            )

        pair_match.assert_called_once()
        validate.assert_called_once()
        self.assertEqual(validate.call_args.kwargs["consumer"], "audio")
        self.assertEqual(aligned_audio.call_count, 2)
        aligned_subtitle.assert_called_once()

    def test_prepare_stream_entries_refuses_image_subtitle_before_matching(self):
        performer = self._make_performer()
        base, source = "/tmp/base.mkv", "/tmp/source.mkv"

        with self._stream_preparation_probes(performer), \
             patch.object(performer.aligner, "_pair_match", side_effect=AssertionError("matched an unusable source")), \
             patch.object(performer.aligner, "patch_audio") as patch_audio, \
             self.assertRaisesRegex(RuntimeError, "image-based subtitle format"):
            performer._prepare_stream_entries(
                [VideoStreamRef(base, 0, 0, None)],
                [AudioStreamRef(source, 1, 1, "pol")],
                [SubtitleStreamRef(source, 3, 3, "pol")],
                [],
                {base: 1, source: 2},
                self._aligned_files_details(base, source, subtitle_format="HDMV PGS"),
                alignment_paths=(source,),
            )

        patch_audio.assert_not_called()

    def test_group_base_duration_reads_the_selected_video_track(self):
        # The base duration pads and validates the output against the
        # selected base video, so it must not come from another video track
        # of the same file.
        performer = self._make_performer()
        base_path = "/media/base.mkv"
        files_details = {
            base_path: {"tracks": {"video": [
                {"tid": 0, "length": 4000},
                {"tid": 1, "length": 6000},
            ]}},
        }

        with patch.object(performer, "_base_audio_end_ms", return_value=None), \
             patch.object(video_utils, "get_video_duration", side_effect=AssertionError("plan data has the length")):
            group = performer._group_context(
                [VideoStreamRef(base_path, 1, 1, None)],
                [],
                {base_path: 1},
                files_details,
                (),
            )

        self.assertEqual(group.base_duration_ms, 6000)

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

    def test_process_duplicates_orders_tracks_by_type_then_language(self):
        # Video, then audio, then subtitles; each type alphabetical by
        # language with unknown languages last.
        source_path = "/library/Movie.mkv"

        args = self._process_single_source_plan(
            source_path,
            {"video": [(source_path, 0, 0, None)]},
            [
                _StreamEntry("subtitle", 5, source_path, "deu"),
                _StreamEntry("audio", 3, source_path, "pol"),
                _StreamEntry("subtitle", 4, source_path, None),
                _StreamEntry("video", 0, source_path, None),
                _StreamEntry("subtitle", 6, source_path, "eng"),
                _StreamEntry("audio", 2, source_path, "eng"),
                _StreamEntry("audio", 1, source_path, None),
            ],
        )

        self.assertEqual(
            args[args.index("--track-order") + 1],
            "0:0,0:2,0:3,0:1,0:5,0:6,0:4",
        )

    def test_process_duplicates_keeps_unknown_language_tracks_in_input_order(self):
        source_path = "/library/Movie.mkv"

        args = self._process_single_source_plan(
            source_path,
            {"video": [(source_path, 0, 0, None)]},
            [
                _StreamEntry("video", 0, source_path, None),
                _StreamEntry("subtitle", 9, source_path, None),
                _StreamEntry("audio", 2, source_path, None),
                _StreamEntry("subtitle", 7, source_path, None),
                _StreamEntry("audio", 1, source_path, "jpn"),
            ],
        )

        self.assertEqual(
            args[args.index("--track-order") + 1],
            "0:0,0:1,0:2,0:9,0:7",
        )

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

        with patch("twotone.tools.melt.melt_performer.video_track_duration_ms", return_value=6000), \
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

        with patch("twotone.tools.melt.melt_performer.video_track_duration_ms", return_value=6000), \
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

        with patch("twotone.tools.melt.melt_performer.video_track_duration_ms", return_value=6000), \
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
             patch("twotone.tools.melt.melt_performer.video_track_duration_ms", return_value=6000), \
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
