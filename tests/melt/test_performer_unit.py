
import logging
import os
import tempfile
import unittest

from parameterized import parameterized
from unittest.mock import Mock, patch

from twotone.tools.utils import files_utils, generic_utils, process_utils, video_utils
from twotone.tools.melt.melt import MeltPerformer
from twotone.tools.melt.melt_common import AudioStreamRef, VideoStreamRef
from twotone.tools.melt.melt_performer import _AudioStrategy, _PairMatchResult, _StreamEntry
from twotone.tools.melt.pair_matcher import MappingRelation, SegmentsMappingResult
from common import run_ffmpeg
from melt.helpers import _FAKE_PROCESS_OK


class MeltPerformerUnitTest(unittest.TestCase):
    """Unit tests for MeltPerformer internal methods."""

    def _make_performer(self) -> MeltPerformer:
        output = files_utils.Workspace.temporary()
        self.addCleanup(output.close)
        return MeltPerformer(
            logger=logging.getLogger("test.MeltPerformer"),
            interruption=generic_utils.InterruptibleProcess(),
            workspace=files_utils.Workspace.temporary(),
            output_dir=output.root,
        )

    def _process_single_source_plan(
        self,
        source_path: str,
        selected_streams: dict[str, list[tuple[str, int, str | None]]],
        prepared_entries: list[_StreamEntry],
    ) -> list[str]:
        performer = self._make_performer()
        plan = [{
            "title": "Title",
            "groups": [{
                "output_name": "Movie",
                "files": [source_path],
                "streams": selected_streams,
            }],
        }]
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

    def test_process_duplicates_rejects_output_aliasing_input(self):
        performer = self._make_performer()
        title_dir = os.path.join(performer.output_dir, "Title")
        os.makedirs(title_dir)
        input_path = os.path.join(title_dir, "Movie.mkv")
        with open(input_path, "wb") as input_file:
            input_file.write(b"source")
        input_alias = os.path.join(performer.output_dir, "movie-source.mkv")
        os.symlink(input_path, input_alias)
        plan = [{
            "title": "Title",
            "groups": [{
                "output_name": "Movie",
                "files": [input_alias],
                "streams": {"video": [(input_alias, 0, None)]},
            }],
        }]

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
            {"video": [(source_path, 0, None)]},
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
                "video": [(source_path, 0, None)],
                "audio": [(source_path, 2, "eng")],
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
            {"audio": [(source_path, 1, "pol")]},
            [_StreamEntry("audio", 1, source_path, "pol", 125)],
        )

        self.assertEqual(args[args.index("--language") + 1], "1:pol")
        self.assertEqual(args[args.index("--sync") + 1], "1:125")

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
        plan = [{
            "title": "Title",
            "groups": [{
                "output_name": "Movie",
                "files": [input_a, input_b],
                "streams": {
                    "video": [(input_a, 0, None)],
                    "audio": [(input_b, 1, "eng")],
                },
            }],
        }]
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

    def test_strict_audio_mapping_collapses_ambiguous_boundary_matches(self):
        mapping = [
            (0, 510),
            (500, 510),
            (24041, 24532),
            (63250, 64540),
            (63250, 65050),
        ]

        result = MeltPerformer._strict_audio_mapping(mapping)

        self.assertEqual(
            result,
            [
                (500, 510),
                (24041, 24532),
                (63250, 64540),
            ],
        )

    def test_unscaled_timeline_shift_accepts_whole_frame_offsets(self):
        # 5 dropped leading frames at 24 fps: deltas sit on the whole-frame grid.
        mapping = [(lhs, lhs - 208) for lhs in (500, 10000, 30000, 60000)]

        shift = MeltPerformer._unscaled_timeline_shift_ms(mapping, 24.0)

        self.assertEqual(shift, 208)

    def test_unscaled_timeline_shift_rejects_off_grid_medians_as_noise(self):
        # Frame-rate-drift pairs are quantized to the coarser 23 fps grid; a
        # +21 ms median is phase noise, not a genuine 1-frame (42 ms) shift.
        mapping = [(lhs, lhs - 21) for lhs in (500, 10000, 30000, 60000)]

        shift = MeltPerformer._unscaled_timeline_shift_ms(mapping, 24.0)

        self.assertEqual(shift, 0)

    def test_unscaled_timeline_shift_applies_stream_bias_correction(self):
        # A container start offset present in one side's frame times only
        # (mkv rebases to zero, mp4 keeps the offset) must cancel out instead
        # of being read as a genuine 12-frame shift.
        mapping = [(lhs, lhs + 495) for lhs in (0, 10000, 30000, 60000)]

        shift = MeltPerformer._unscaled_timeline_shift_ms(mapping, 24.0, stream_bias_ms=500)

        self.assertEqual(shift, 0)

    def test_mapping_stream_bias_measures_rebased_frame_times(self):
        fake_info = {"streams": [{"codec_type": "video", "start_time": "0.500"}]}

        with patch.object(video_utils, "get_video_full_info", return_value=fake_info):
            rebased = MeltPerformer._mapping_stream_bias_ms("/tmp/a.mkv", {0: {}, 41: {}})
            preserved = MeltPerformer._mapping_stream_bias_ms("/tmp/a.mp4", {500: {}, 541: {}})

        self.assertEqual(rebased, 500)
        self.assertEqual(preserved, 0)

    def test_collapse_linear_mapping_sharpens_slope_with_metadata_tempo(self):
        # The matched endpoint carries a ~1 s low-entropy-zone error (observed
        # slope 1.0043 vs the real 1.0204); the effective-fps tempo agrees
        # with the pairs within tolerance, so it replaces the noisy slope and
        # the mapping collapses to one clean global time-scale.  The tempo is
        # derived from stream durations (mp4 duration, matroska DURATION tag)
        # and start offsets.
        mapping = [
            (0, 0),
            (63292, 63562),
        ]
        lhs_frames = {
            round(index * 63292 / 1497): {"path": f"lhs_{index}.jpg"}
            for index in range(1498)
        }
        rhs_frames = {
            round(index * 64030 / 1506): {"path": f"rhs_{index}.jpg"}
            for index in range(1507)
        }

        def fake_full_info(path, **_kwargs):
            if path == "/tmp/base.mp4":
                return {
                    "streams": [
                        {
                            "codec_type": "video",
                            "start_time": "0.500000",
                            "duration": "62.332000",
                            "nb_frames": "1496",
                        }
                    ]
                }
            return {
                "streams": [
                    {
                        "codec_type": "video",
                        "start_time": "0.510000",
                        "tags": {"DURATION": "00:01:04.581000000"},
                    }
                ]
            }

        with patch.object(video_utils, "get_video_full_info", side_effect=fake_full_info):
            tempo_ratio = MeltPerformer._effective_tempo_ratio(
                "/tmp/base.mp4",
                "/tmp/source.mkv",
                lhs_frames,
                rhs_frames,
            )

        result = MeltPerformer._try_collapse_linear_mapping(
            mapping,
            lhs_frames,
            tempo_ratio,
        )

        self.assertEqual(result, [(0, 0), (63292, 64583)])

    def test_collapse_linear_mapping_skips_partial_coverage(self):
        # Metadata agrees with the pairs, but the matched span covers well
        # under 80% of the base — projecting one global time-scale from a
        # small window is guesswork.
        mapping = [
            (5708, 4536),
            (46208, 44438),
        ]
        lhs_frames = {
            round(index * 63292 / 1497): {"path": f"lhs_{index}.jpg"}
            for index in range(1498)
        }

        result = MeltPerformer._try_collapse_linear_mapping(
            mapping,
            lhs_frames,
            tempo_ratio=0.9715,
        )

        self.assertIsNone(result)

    def test_frame_rate_only_drift_mapping_detected(self):
        # A 23<->24 fps resample preserving playback time: the effective fps
        # differ (tempo far from 1) and match the nominal ratio, while the
        # matched spans agree in wall-clock duration.  Such pairs must go the
        # UNSCALED route — _choose_audio_strategy excludes them before any
        # linear-mapping recovery runs.
        mapping = [
            (739, 0),
            (16522, 16063),
            (26217, 25730),
            (53609, 53147),
            (62739, 62272),
            (62826, 62814),
        ]
        lhs_frames = {
            0: {"path": "lhs_0.jpg"},
            62826: {"path": "lhs_62826.jpg"},
        }
        rhs_frames = {
            0: {"path": "rhs_0.jpg"},
            62814: {"path": "rhs_62814.jpg"},
        }

        def fake_full_info(path, **_kwargs):
            if path == "/tmp/base.mkv":
                return {
                    "streams": [
                        {
                            "codec_type": "video",
                            "start_time": "0.000000",
                            "duration": "62.826000",
                            "nb_frames": "1446",
                        }
                    ]
                }
            return {
                "streams": [
                    {
                        "codec_type": "video",
                        "start_time": "0.500000",
                        "duration": "62.332000",
                        "nb_frames": "1496",
                    }
                ]
            }

        with patch.object(video_utils, "get_video_full_info", side_effect=fake_full_info):
            result = MeltPerformer._is_frame_rate_only_drift_mapping(
                "/tmp/base.mkv",
                "/tmp/source.mp4",
                mapping,
                lhs_frames,
                rhs_frames,
                lhs_nominal_fps=23.0,
                rhs_nominal_fps=24.0,
            )

        self.assertTrue(result)

    def test_linear_recovery_rejects_tmnt_piecewise_pairs(self):
        # A 23.976 fps base against a 30 fps resampled transfer with ad-break
        # cuts.  The pairs are piecewise (interior residuals of ~1.4 s against
        # the endpoint line), so no line may be recovered — and the fps tempo
        # 0.7992, contradicting the pairs' ~0.9665 span slope, must never be
        # applied: an fps-derived mapping would stretch the audio by 25%.
        mapping = [
            (1084, 0),
            (4755, 3633),
            (5130, 4000),
            (5589, 4467),
            (17851, 16800),
            (1250918, 1206600),
            (1253337, 1208967),
            (1256590, 1212200),
            (1315148, 1270067),
            (1335794, 1290000),
        ]
        lhs_frames = {
            0: {"path": "lhs_0.jpg"},
            1369994: {"path": "lhs_1369994.jpg"},
        }

        result = MeltPerformer._try_collapse_linear_mapping(
            mapping,
            lhs_frames,
            tempo_ratio=0.7992,
        )

        self.assertIsNone(result)

    def test_collapse_linear_mapping_keeps_pairs_when_metadata_contradicts(self):
        # A clean (cut-free) variant of the TMNT class: pairs perfectly linear
        # at 0.96651, while the effective-fps tempo says 0.7992 (resampled
        # transfer — the fps ratio is unrelated to the time mapping).  The
        # contradicting metadata must not rebuild the mapping.
        pairs = [
            (1084 + i * 300000, round(i * 300000 * 0.96651))
            for i in range(5)
        ]
        lhs_frames = {
            0: {"path": "lhs_0.jpg"},
            1369994: {"path": "lhs_1369994.jpg"},
        }

        result = MeltPerformer._try_collapse_linear_mapping(
            pairs,
            lhs_frames,
            tempo_ratio=0.7992,
        )

        self.assertIsNone(result)

    # ---- _patch_audio_constant_offset ----

    def _collect_ffmpeg_calls(self, performer, segment_pairs, base_duration_ms,
                               source_sample_rate=48000, source_channels=2, source_sample_fmt="s16",
                               source_channel_layout=None,
                               fill_gaps_from_base=True):
        """Run patch_audio_constant_offset with mocked externals and return captured ffmpeg calls."""
        calls = []

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_raise_on_error(result):
            pass

        right_points = [p[1] for p in segment_pairs]
        source_segment_dur = max(right_points) - min(right_points)

        def fake_get_duration(path, **_kwargs):
            if "source_trimmed" in path or "source_scaled" in path or "out." in path:
                return source_segment_dur
            return base_duration_ms

        stream_info = {"codec_type": "audio", "index": 2, "channels": source_channels,
                       "sample_rate": str(source_sample_rate), "sample_fmt": source_sample_fmt}
        if source_channel_layout is not None:
            stream_info["channel_layout"] = source_channel_layout
        fake_full_info = {"streams": [
            {"codec_type": "video", "index": 0},
            stream_info,
            {**stream_info, "index": 3},
        ]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process', side_effect=fake_start_process), \
                 patch.object(process_utils, 'raise_on_error', side_effect=fake_raise_on_error), \
                 patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                 patch.object(video_utils, 'get_video_full_info', return_value=fake_full_info):
                performer.patch_audio_constant_offset(
                    wd,
                    VideoStreamRef("/base.mkv", 0, None),
                    AudioStreamRef("/source.mkv", 2, "pol"),
                    AudioStreamRef("/base.mkv", 3, "eng"),
                    output_path,
                    segment_pairs,
                    fill_gaps_from_base=fill_gaps_from_base,
                )

        return calls

    def test_patch_audio_constant_offset_copy_when_ratio_near_one(self):
        """When source and target durations match, no asetrate filter should be applied."""
        performer = self._make_performer()
        # seg1: 1000..5000 (4s), seg2: 500..4500 (4s) — ratio = 1.0
        pairs = [(1000, 500), (5000, 4500)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=6000)

        # Should NOT contain any asetrate/aresample filter call
        filter_calls = [c for c in calls if any("asetrate" in str(a) for a in c[1])]
        self.assertEqual(filter_calls, [], "asetrate should not be used when ratio ≈ 1.0")

    def test_patch_audio_constant_offset_asetrate_when_ratio_differs(self):
        """When durations differ, asetrate+aresample should be applied."""
        performer = self._make_performer()
        # seg1: 0..4000 (4s), seg2: 0..4100 (4.1s) — ratio ≈ 1.025
        pairs = [(0, 0), (4000, 4100)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=4000, source_sample_rate=48000)

        filter_calls = [c for c in calls if any("asetrate" in str(a) for a in c[1])]
        self.assertEqual(len(filter_calls), 1, "asetrate should be used once")

        filter_arg = next(a for a in filter_calls[0][1] if "asetrate" in str(a))
        # adjusted_rate = 48000 * 4100 / 4000 = 49200
        self.assertIn("asetrate=49200", filter_arg)
        self.assertIn("aresample=48000", filter_arg)

    def test_patch_audio_constant_offset_head_and_tail_extraction(self):
        """Head and tail segments should be extracted from base audio."""
        performer = self._make_performer()
        # Matching region: 2000..8000 in base, so head=[0..2s] and tail=[8s..10s]
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=10000)

        ffmpeg_args_strs = [" ".join(str(a) for a in c[1]) for c in calls if c[0] == "ffmpeg"]

        head_calls = [s for s in ffmpeg_args_strs if "head" in s and "-to" in s]
        self.assertGreaterEqual(len(head_calls), 1, "Head segment should be extracted")
        self.assertIn("2.0", head_calls[0])

        tail_calls = [s for s in ffmpeg_args_strs if "tail" in s and "-ss" in s]
        self.assertGreaterEqual(len(tail_calls), 1, "Tail segment should be extracted")
        self.assertIn("8.0", tail_calls[0])

    def test_patch_audio_constant_offset_no_head_when_at_start(self):
        """When matching starts at 0, no head should be extracted."""
        performer = self._make_performer()
        pairs = [(0, 500), (4000, 4500)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=6000)

        head_calls = [c for c in calls if any("head" in str(a) for a in c[1])]
        self.assertEqual(head_calls, [], "No head should be extracted when seg1 starts at 0")

    def test_patch_audio_constant_offset_no_tail_when_at_end(self):
        """When matching ends at base duration, no tail should be extracted."""
        performer = self._make_performer()
        pairs = [(1000, 500), (6000, 5500)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=6000)

        tail_calls = [c for c in calls if any("tail" in str(a) for a in c[1])]
        self.assertEqual(tail_calls, [], "No tail should be extracted when seg1 ends at base duration")

    def test_patch_audio_head_tail_normalized_to_source_params(self):
        """Head and tail must be re-encoded with source audio parameters so FLAC concat works."""
        performer = self._make_performer()
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(
            performer, pairs, base_duration_ms=10000,
            source_sample_rate=44100, source_channels=6, source_sample_fmt="s32",
        )

        ffmpeg_calls = [c[1] for c in calls if c[0] == "ffmpeg"]
        head_calls = [a for a in ffmpeg_calls if any("head" in str(x) for x in a)]
        tail_calls = [a for a in ffmpeg_calls if any("tail" in str(x) for x in a)]

        self.assertGreaterEqual(len(head_calls), 1, "Head segment should be extracted")
        self.assertGreaterEqual(len(tail_calls), 1, "Tail segment should be extracted")

        for label, call_args in [("head", head_calls[0]), ("tail", tail_calls[0])]:
            self.assertIn("-ac", call_args, f"{label} must have -ac")
            self.assertIn("-ar", call_args, f"{label} must have -ar")
            self.assertIn("-sample_fmt", call_args, f"{label} must have -sample_fmt")
            self.assertEqual(call_args[call_args.index("-ac") + 1], "6", f"{label} channels")
            self.assertEqual(call_args[call_args.index("-ar") + 1], "44100", f"{label} sample rate")
            self.assertEqual(call_args[call_args.index("-sample_fmt") + 1], "s32", f"{label} sample fmt")

    def test_global_audio_patch_uses_selected_source_and_base_streams(self):
        performer = self._make_performer()

        calls = self._collect_ffmpeg_calls(
            performer,
            [(2000, 1000), (8000, 7000)],
            base_duration_ms=10000,
        )

        ffmpeg_calls = [args for tool, args in calls if tool == "ffmpeg"]
        source_decode = next(args for args in ffmpeg_calls if "/source.mkv" in args)
        self.assertEqual(source_decode[source_decode.index("-map") + 1], "0:2")

        base_gap_calls = [args for args in ffmpeg_calls if "/base.mkv" in args]
        self.assertEqual(len(base_gap_calls), 2)
        self.assertTrue(all(args[args.index("-map") + 1] == "0:3" for args in base_gap_calls))

    def test_flac_concat_silently_degrades_when_params_differ(self):
        """Concatenating FLAC files with different params silently downgrades to first file's params.

        When the concat demuxer encounters a parameter change mid-stream, ffmpeg
        reconfigures the filter graph and resamples/downmixes to match the first
        segment.  In the melt audio pipeline head/tail come from the base video
        and the middle segment from the source video — if their params differ,
        the source audio (which is the whole point of melt) gets silently
        degraded.  The normalization fix prevents this by re-encoding head/tail
        to match source params *before* concatenation.
        """
        with tempfile.TemporaryDirectory() as td:
            stereo_path = os.path.join(td, "stereo_44100.flac")
            surround_path = os.path.join(td, "surround_48000.flac")
            concat_list = os.path.join(td, "concat.txt")
            merged_path = os.path.join(td, "merged.flac")

            # head-like: stereo 44100 Hz (base video params)
            run_ffmpeg(["-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=1",
                        "-c:a", "flac", stereo_path], expected_path=stereo_path)
            # source segment: 5.1 surround 48000 Hz (higher quality)
            run_ffmpeg(["-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=5.1:d=1",
                        "-c:a", "flac", surround_path], expected_path=surround_path)

            with open(concat_list, "w") as f:
                f.write(f"file '{stereo_path}'\nfile '{surround_path}'\n")

            run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                        "-c:a", "flac", merged_path], expected_path=merged_path)

            # ffmpeg silently downgrades: output uses the FIRST file's params
            data = video_utils.get_video_data(merged_path)
            self.assertEqual(data["audio"][0]["channels"], 2,
                             "5.1 was silently downmixed to stereo — this is the bug normalization prevents")
            self.assertEqual(data["audio"][0]["sample_rate"], 44100,
                             "48kHz was silently resampled to 44.1kHz")

    def test_flac_concat_preserves_quality_with_normalized_params(self):
        """When all FLAC segments share parameters, concat preserves full quality."""
        with tempfile.TemporaryDirectory() as td:
            a_path = os.path.join(td, "a.flac")
            b_path = os.path.join(td, "b.flac")
            concat_list = os.path.join(td, "concat.txt")
            merged_path = os.path.join(td, "merged.flac")

            # Both normalized to source params: 5.1 surround 48000 Hz
            run_ffmpeg(["-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=5.1:d=1",
                        "-c:a", "flac", a_path], expected_path=a_path)
            run_ffmpeg(["-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=5.1:d=1",
                        "-c:a", "flac", b_path], expected_path=b_path)

            with open(concat_list, "w") as f:
                f.write(f"file '{a_path}'\nfile '{b_path}'\n")

            run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                        "-c:a", "flac", merged_path], expected_path=merged_path)

            data = video_utils.get_video_data(merged_path)
            self.assertEqual(data["audio"][0]["channels"], 6,
                             "5.1 surround must be preserved")
            self.assertEqual(data["audio"][0]["sample_rate"], 48000,
                             "48kHz sample rate must be preserved")

    # ---- _fmt_time ----

    @parameterized.expand([
        ("zero",           0.0,       "0.0s"),
        ("seconds",        2.4,       "2.4s"),
        ("one_minute",    65.3,       "1:05.3"),
        ("ten_minutes",  625.0,      "10:25.0"),
        ("one_hour",    3661.5,       "1:01:01.5"),
        ("large",       6698.0,       "1:51:38.0"),
        ("negative",      -5.0,      "-5.0s"),
    ])
    def test_fmt_time(self, _name, seconds, expected):
        self.assertEqual(MeltPerformer._fmt_time(seconds), expected)

    # ---- _render_overlap_diagram ----

    def test_render_overlap_diagram_both_gaps(self):
        """Diagram with gaps: lhs starts at 2s, rhs starts at 0s → lhs indented, rhs at col 0."""
        performer = self._make_performer()
        # rhs_start_gap_s=0 means rhs starts at beginning of shared content
        # lhs_start_gap_s=2 means lhs starts 2s before shared content → lhs at col 0
        summary = {
            "lhs_start_gap_s": 2.0,
            "rhs_start_gap_s": 0.0,
            "lhs_end_gap_s": 0.0,
            "rhs_end_gap_s": 3.0,
            "full_coverage": False,
        }
        lines = performer._render_overlap_diagram(1, 2, 100_000, 101_000, summary)
        self.assertGreaterEqual(len(lines), 2, "Should have at least 2 bar lines")
        # lhs has start_gap=2 → shared content starts 2s into lhs → lhs starts first
        self.assertTrue(lines[0].startswith("|"), "#1 bar starts at col 0")
        # rhs starts at shared content → rhs is indented
        self.assertTrue(lines[1].startswith(" "), "#2 bar should be indented")
        self.assertIn("#1", lines[0])
        self.assertIn("#2", lines[1])

    def test_render_overlap_diagram_rhs_starts_first(self):
        """When rhs has the initial gap → rhs starts later in the diagram, lhs at col 0."""
        performer = self._make_performer()
        # lhs_start_gap=0 means shared content starts at beginning of lhs
        # rhs_start_gap=3 means shared content starts 3s into rhs → rhs starts earlier
        summary = {
            "lhs_start_gap_s": 0.0,
            "rhs_start_gap_s": 3.0,
            "lhs_end_gap_s": 3.0,
            "rhs_end_gap_s": 0.0,
            "full_coverage": False,
        }
        lines = performer._render_overlap_diagram(1, 2, 100_000, 100_000, summary)
        self.assertGreaterEqual(len(lines), 2)
        # rhs_s = lhs_sg - rhs_sg = 0 - 3 = -3 → rhs starts before lhs
        # So rhs is at col 0, lhs is indented
        self.assertTrue(lines[0].startswith(" "), "#1 bar should be indented")
        self.assertTrue(lines[1].startswith("|"), "#2 bar starts at col 0")

    def test_render_overlap_diagram_no_gaps_returns_empty(self):
        """When gaps are tiny (< threshold), no diagram should be produced."""
        performer = self._make_performer()
        summary = {
            "lhs_start_gap_s": 0.0,
            "rhs_start_gap_s": 0.0,
            "lhs_end_gap_s": 0.0,
            "rhs_end_gap_s": 0.0,
            "full_coverage": False,
        }
        lines = performer._render_overlap_diagram(1, 2, 100_000, 100_000, summary)
        # Bars should be at the same columns (no offset)
        if lines:
            lhs_start = len(lines[0]) - len(lines[0].lstrip())
            rhs_start = len(lines[1]) - len(lines[1].lstrip())
            self.assertEqual(lhs_start, rhs_start, "Both bars should start at same column")

    def test_render_overlap_diagram_has_timestamps(self):
        """Diagram should contain time labels."""
        performer = self._make_performer()
        summary = {
            "lhs_start_gap_s": 0.0,
            "rhs_start_gap_s": 5.0,
            "lhs_end_gap_s": 0.0,
            "rhs_end_gap_s": 0.0,
            "full_coverage": False,
        }
        lines = performer._render_overlap_diagram(1, 2, 120_000, 115_000, summary)
        all_text = "\n".join(lines)
        # Should have at least "0.0s" somewhere in the timestamp rows
        self.assertIn("0.0s", all_text, "Should contain the start timestamp")

    def test_render_overlap_diagram_speed_ratio_projected(self):
        """When rhs plays faster (PAL), rhs should be projected onto lhs timeline.

        With lhs=100s, rhs=96s, lhs_sg=2, rhs_sg=0, lhs_eg=1, rhs_eg=0:
        shared_lhs=97, shared_rhs=96, speed=97/96≈1.0104.
        rhs projected: start=2.0, end=99.0 (=100-1). End gap on diagram ≈1s, not 4s.
        """
        performer = self._make_performer()
        summary = {
            "lhs_start_gap_s": 2.0,
            "rhs_start_gap_s": 0.0,
            "lhs_end_gap_s": 1.0,
            "rhs_end_gap_s": 0.0,
            "full_coverage": False,
        }
        lines = performer._render_overlap_diagram(1, 2, 100_000, 96_000, summary)
        self.assertGreaterEqual(len(lines), 2)
        # lhs starts first (lhs_sg=2 means has content before shared region)
        self.assertTrue(lines[0].startswith("|"), "#1 bar at col 0")
        self.assertTrue(lines[1].startswith(" "), "#2 bar indented")
        # Both should end close (1s gap gets _MIN_GAP=6 columns, but NOT a proportional 4s gap)
        lhs_end_col = len(lines[0].rstrip()) - 1
        rhs_end_col = len(lines[1].rstrip()) - 1
        self.assertAlmostEqual(lhs_end_col, rhs_end_col, delta=8,
                               msg="Speed-adjusted rhs should end close to lhs (small gap)")

    def test_patch_audio_constant_offset_reencodes_even_when_ratio_near_one(self):
        """Even a 1:1 trim is decoded so AAC priming and source starts are normalized."""
        performer = self._make_performer()
        # seg1: 0..6000 (full base), seg2: 500..6500 — same duration, no head/tail
        pairs = [(0, 500), (6000, 6500)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=6000)

        ffmpeg_calls = [c[1] for c in calls if c[0] == "ffmpeg"]
        trim_call = next(a for a in ffmpeg_calls if "source_trimmed" in str(a))
        self.assertNotIn("copy", trim_call)
        self.assertIn("-filter:a", trim_call)
        trim_filter = trim_call[trim_call.index("-filter:a") + 1]
        self.assertIn("atrim=start=0.500000:end=6.500000", trim_filter)
        self.assertIn("asetpts=PTS-STARTPTS", trim_filter)

        concat_calls = [a for a in ffmpeg_calls if "concat" in str(a)]
        self.assertEqual(len(concat_calls), 1)
        self.assertIn("aac", concat_calls[0])

    @parameterized.expand([
        ("absorbed", "0.500000", False),
        ("exposed", "0.479000", True),
    ])
    def test_aac_trim_window_is_anchored_to_content_start(
        self,
        _name,
        stream_start,
        priming_exposed,
    ):
        performer = self._make_performer()
        calls = []
        source_info = {
            "streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.000000"},
                {
                    "codec_type": "audio",
                    "index": 1,
                    "codec_name": "aac",
                    "initial_padding": 1024,
                    "sample_rate": "48000",
                    "start_time": stream_start,
                },
            ]
        }

        def fake_start_process(tool, args, **_kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        with patch.object(video_utils, "get_video_full_info", return_value=source_info), \
             patch.object(performer, "_aac_priming_exposed", return_value=priming_exposed), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error", lambda result: None), \
             patch.object(os.path, "exists", return_value=False):
            performer._decode_source_audio_to_flac(
                "/tmp/source.mkv",
                "/tmp/output.flac",
                trim_start_ms=500,
                trim_end_ms=63250,
                audio_stream_index=1,
            )

        decode_args = calls[1][1]
        trim_filter = decode_args[decode_args.index("-filter:a") + 1]
        self.assertEqual(
            trim_filter,
            "atrim=start=0.021333:end=62.771333,asetpts=PTS-STARTPTS",
        )

    def test_full_selected_audio_decode_is_rebased_and_normalized(self):
        """A whole selected stream uses one zero-based, normalized FLAC timeline."""
        performer = self._make_performer()
        calls = []
        source_info = {
            "streams": [
                {"codec_type": "audio", "index": 2, "codec_name": "flac", "sample_rate": "44100"},
            ],
        }

        def fake_start_process(tool, args, **_kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        with patch.object(video_utils, "get_video_full_info", return_value=source_info), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error", lambda result: None):
            performer._decode_audio_stream_to_flac(
                AudioStreamRef("/tmp/source.mkv", 2, "pol"),
                "/tmp/output.flac",
                normalize_to=(1, 48000, "s32"),
            )

        decode_args = calls[-1][1]
        self.assertEqual(decode_args[decode_args.index("-map") + 1], "0:2")
        self.assertEqual(decode_args[decode_args.index("-filter:a") + 1], "asetpts=PTS-STARTPTS")
        self.assertEqual(decode_args[decode_args.index("-ac") + 1], "1")
        self.assertEqual(decode_args[decode_args.index("-ar") + 1], "48000")

    def test_patch_audio_constant_offset_concat_single_pass(self):
        """When head/tail needed, concat should encode to AAC in a single pass (no intermediate FLAC)."""
        performer = self._make_performer()
        # seg1_start=2000 → has head
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=10000)

        ffmpeg_args_strs = [" ".join(str(a) for a in c[1]) for c in calls if c[0] == "ffmpeg"]
        concat_calls = [s for s in ffmpeg_args_strs if "concat" in s]
        self.assertEqual(len(concat_calls), 1, "Should have exactly one concat call")
        self.assertIn("aac", concat_calls[0], "Concat should encode directly to AAC")
        # No separate FLAC→AAC pass
        flac_to_aac = [s for s in ffmpeg_args_strs if "merged.flac" in s and "aac" in s and "concat" not in s]
        self.assertEqual(flac_to_aac, [], "Should not have a separate FLAC→AAC re-encoding step")

    # ---- channel layout normalization ----

    def test_nonstandard_channel_layout_triggers_aformat(self):
        """5.1(side) is not a native AAC layout — aformat filter must normalize it to avoid PCE."""
        performer = self._make_performer()
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(
            performer, pairs, base_duration_ms=10000,
            source_channels=6, source_channel_layout="5.1(side)",
        )

        # Find the concat→AAC call
        concat_calls = [c for c in calls if c[0] == "ffmpeg" and "concat" in str(c[1])]
        self.assertEqual(len(concat_calls), 1)
        args = concat_calls[0][1]
        self.assertIn("-af", args, "aformat filter should be present for non-standard layout")
        af_idx = args.index("-af")
        self.assertIn("aformat=channel_layouts=", args[af_idx + 1])
        self.assertIn("5.1", args[af_idx + 1], "Standard 5.1 must be in allowed layouts")

    def test_standard_channel_layout_skips_aformat(self):
        """Standard 5.1 layout should NOT trigger the aformat normalization filter."""
        performer = self._make_performer()
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(
            performer, pairs, base_duration_ms=10000,
            source_channels=6, source_channel_layout="5.1",
        )

        concat_calls = [c for c in calls if c[0] == "ffmpeg" and "concat" in str(c[1])]
        self.assertEqual(len(concat_calls), 1)
        args = concat_calls[0][1]
        aformat_args = [a for a in args if isinstance(a, str) and "aformat" in a]
        self.assertEqual(aformat_args, [], "aformat should not be used for standard 5.1")

    def test_stereo_layout_skips_aformat(self):
        """Stereo is a standard AAC layout — no aformat needed."""
        performer = self._make_performer()
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(
            performer, pairs, base_duration_ms=10000,
            source_channels=2, source_channel_layout="stereo",
        )

        concat_calls = [c for c in calls if c[0] == "ffmpeg" and "concat" in str(c[1])]
        self.assertEqual(len(concat_calls), 1)
        args = concat_calls[0][1]
        aformat_args = [a for a in args if isinstance(a, str) and "aformat" in a]
        self.assertEqual(aformat_args, [], "aformat should not be used for stereo")

    def test_unknown_channel_layout_skips_aformat(self):
        """When ffprobe doesn't report channel_layout, aformat should not be added."""
        performer = self._make_performer()
        pairs = [(2000, 1000), (8000, 7000)]
        # No source_channel_layout → defaults to None
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=10000)

        concat_calls = [c for c in calls if c[0] == "ffmpeg" and "concat" in str(c[1])]
        self.assertEqual(len(concat_calls), 1)
        args = concat_calls[0][1]
        aformat_args = [a for a in args if isinstance(a, str) and "aformat" in a]
        self.assertEqual(aformat_args, [], "aformat should not be used when layout is unknown")

    # ---- silence mode (fill_gaps_from_base=False) ----

    def test_silence_mode_skips_head_and_tail(self):
        """With fill_gaps_from_base=False, no head/tail extraction or generation should occur."""
        performer = self._make_performer()
        # Matching region: 2000..8000 in base (10s total) → would have head and tail
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=10000, fill_gaps_from_base=False)

        ffmpeg_args_strs = [" ".join(str(a) for a in c[1]) for c in calls if c[0] == "ffmpeg"]

        # No head/tail extraction or silence generation
        head_tail_calls = [s for s in ffmpeg_args_strs if "head" in s or "tail" in s or "anullsrc" in s]
        self.assertEqual(head_tail_calls, [], "Silence mode should not produce head/tail segments")

        # No base video extraction
        base_extract_calls = [s for s in ffmpeg_args_strs if "/base.mkv" in s]
        self.assertEqual(base_extract_calls, [], "Silence mode should not read from base video")

    def test_silence_mode_reencodes_when_no_scaling(self):
        """With fill_gaps_from_base=False and ratio ≈ 1.0, audio is still decoded for stable timing."""
        performer = self._make_performer()
        pairs = [(0, 500), (4000, 4500)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=6000, fill_gaps_from_base=False)

        ffmpeg_calls = [c[1] for c in calls if c[0] == "ffmpeg"]
        trim_call = next(a for a in ffmpeg_calls if "source_trimmed" in str(a))
        self.assertNotIn("copy", trim_call)
        self.assertIn("-filter:a", trim_call)
        concat_calls = [a for a in ffmpeg_calls if "concat" in str(a)]
        self.assertEqual(len(concat_calls), 1)

    def test_patched_audio_start_preserves_audio_stream_start(self):
        performer = self._make_performer()
        fake_full_info = {
            "format": {"start_time": "0.000000"},
            "streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.000000"},
                {"codec_type": "audio", "index": 1, "start_time": "0.468000"},
            ]
        }

        with patch.object(video_utils, "get_video_full_info", return_value=fake_full_info):
            self.assertEqual(
                performer._patched_audio_start_ms(
                    AudioStreamRef("/tmp/source.mov", 1, "pol"),
                    seg1_start=510,
                    seg2_start=0,
                    base_per_source=1.0,
                ),
                978,
            )
            self.assertEqual(
                performer._patched_audio_start_ms(
                    AudioStreamRef("/tmp/source.mov", 1, "pol"),
                    seg1_start=510,
                    seg2_start=490,
                    base_per_source=1.0,
                ),
                510,
            )

    def test_patched_audio_start_ignores_tail_deficit_when_audio_starts_before_video(self):
        performer = self._make_performer()
        fake_full_info = {
            "format": {"start_time": "0.471000"},
            "streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.492000"},
                {"codec_type": "audio", "index": 1, "start_time": "0.471000"},
            ]
        }

        with patch.object(video_utils, "get_video_full_info", return_value=fake_full_info):
            self.assertEqual(
                performer._patched_audio_start_ms(
                    AudioStreamRef("/tmp/source.mov", 1, "pol"),
                    seg1_start=0,
                    seg2_start=21,
                    base_per_source=1.015,
                ),
                499,
            )

    def test_patched_audio_start_uses_positive_aac_stream_start(self):
        performer = self._make_performer()
        fake_full_info = {
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

        with patch.object(video_utils, "get_video_full_info", return_value=fake_full_info):
            self.assertEqual(
                performer._patched_audio_start_ms(
                    AudioStreamRef("/tmp/source.mkv", 1, "pol"),
                    seg1_start=22,
                    seg2_start=3,
                    base_per_source=1.0,
                ),
                519,
            )

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

    def test_mkvmerge_input_start_offset_preserves_matroska_stream_start(self):
        performer = self._make_performer()
        with patch.object(
            video_utils,
            "get_video_full_info",
            return_value={"streams": [{"codec_type": "video", "index": 0, "start_time": "0.510000"}]},
        ):
            offset = performer._mkvmerge_input_start_offset_ms("/tmp/base.mkv", "video", 0)

        self.assertEqual(offset, 510)

    def test_mkvmerge_input_start_offset_resets_non_matroska_stream_start(self):
        performer = self._make_performer()
        with patch.object(video_utils, "get_video_full_info") as probe:
            offset = performer._mkvmerge_input_start_offset_ms("/tmp/base.mov", "video", 0)

        self.assertEqual(offset, 0)
        probe.assert_not_called()

    def test_source_stream_end_uses_relative_duration_or_matroska_timeline_tag(self):
        performer = self._make_performer()

        def fake_full_info(path, **_kwargs):
            if path.endswith(".mp4"):
                return {
                    "streams": [
                        {
                            "codec_type": "video",
                            "index": 0,
                            "start_time": "0.500000",
                            "duration": "62.332000",
                        }
                    ]
                }
            return {
                "streams": [
                    {
                        "codec_type": "video",
                        "index": 0,
                        "start_time": "0.500000",
                        "tags": {"DURATION": "00:01:02.832000000"},
                    }
                ]
            }

        with patch.object(video_utils, "get_video_full_info", side_effect=fake_full_info):
            self.assertEqual(performer._source_stream_end_offset_ms("/tmp/source.mp4", "video", 0), 62832)
            self.assertEqual(performer._source_stream_end_offset_ms("/tmp/source.mkv", "video", 0), 62832)

    def test_track_sync_offset_compensates_for_mkvmerge_input_start(self):
        performer = self._make_performer()

        def fake_full_info(path, **_kwargs):
            if path.endswith(".mka"):
                return {"streams": [{"codec_type": "audio", "index": 0, "start_time": "0.471000"}]}
            return {"streams": [{"codec_type": "audio", "index": 1, "start_time": "0.471000"}]}

        with patch.object(video_utils, "get_video_full_info", side_effect=fake_full_info):
            mov_offset = performer._track_sync_offset_ms("/tmp/source.mov", "audio", 1, 471)
            mka_offset = performer._track_sync_offset_ms("/tmp/normalized.mka", "audio", 0, 471)

        self.assertEqual(mov_offset, 471)
        self.assertIsNone(mka_offset)

    def test_aac_from_non_matroska_needs_mkvmerge_normalization(self):
        performer = self._make_performer()
        fake_full_info = {
            "streams": [
                {"codec_type": "video", "index": 0, "codec_name": "h264"},
                {"codec_type": "audio", "index": 1, "codec_name": "aac"},
            ]
        }

        with patch.object(video_utils, "get_video_full_info", return_value=fake_full_info):
            self.assertTrue(performer._audio_needs_mkvmerge_normalization(
                AudioStreamRef("/tmp/source.mov", 1, "eng")
            ))
            performer._media_info_cache.clear()
            performer._stream_info_cache.clear()
            self.assertFalse(performer._audio_needs_mkvmerge_normalization(
                AudioStreamRef("/tmp/source.mkv", 1, "eng")
            ))

    def test_matroska_aac_with_priming_needs_mkvmerge_normalization(self):
        """Matroska AAC carrying CodecDelay priming must go through the FLAC flow:
        a raw remux places it build-dependently (~21 ms off on exposing builds)."""
        performer = self._make_performer()

        def fake_full_info(padded):
            return {
                "streams": [
                    {"codec_type": "video", "index": 0, "codec_name": "h264"},
                    {
                        "codec_type": "audio",
                        "index": 1,
                        "codec_name": "aac",
                        "initial_padding": 1024 if padded else 0,
                    },
                ]
            }

        with patch.object(video_utils, "get_video_full_info", return_value=fake_full_info(True)):
            self.assertTrue(performer._audio_needs_mkvmerge_normalization(
                AudioStreamRef("/tmp/source.mkv", 1, "eng")
            ))
        performer._media_info_cache.clear()
        performer._stream_info_cache.clear()
        with patch.object(video_utils, "get_video_full_info", return_value=fake_full_info(False)):
            self.assertFalse(performer._audio_needs_mkvmerge_normalization(
                AudioStreamRef("/tmp/source.mkv", 1, "eng")
            ))

    def test_prepare_normalized_unscaled_audio_decodes_via_flac_then_encodes_aac_once(self):
        performer = self._make_performer()
        calls = []
        source_full_info = {
            "streams": [
                {"codec_type": "video", "index": 0, "codec_name": "h264"},
                {"codec_type": "audio", "index": 1, "codec_name": "aac", "start_time": "0.471000"},
            ]
        }

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_full_info(path, logger=None):
            return source_full_info

        with patch.object(video_utils, "get_video_full_info", side_effect=fake_full_info), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error", lambda r: None):
            prepared = performer._prepare_normalized_unscaled_audio(
                AudioStreamRef("/tmp/source.mov", 1, "pol")
            )
            cached = performer._prepare_normalized_unscaled_audio(
                AudioStreamRef("/tmp/source.mov", 1, "pol")
            )

        self.assertEqual(prepared.stream_index, 0)
        self.assertEqual(prepared.language, "pol")
        self.assertEqual(cached, prepared)

        # The flow runs exactly twice: decode the source to a priming-free FLAC,
        # then encode that FLAC to AAC a single time.  No AAC→AAC re-encode that
        # could re-apply encoder-delay priming.  The second call is fully cached.
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(tool == "ffmpeg" for tool, _ in calls))

        decode_args = calls[0][1]
        self.assertEqual(decode_args[decode_args.index("-map") + 1], "0:1")
        self.assertEqual(decode_args[decode_args.index("-c:a") + 1], "flac")

        encode_args = calls[1][1]
        self.assertEqual(encode_args[encode_args.index("-c:a") + 1], "aac")
        self.assertEqual(encode_args[-1], prepared.path)
        # Source AAC must never be encoded straight to AAC.
        self.assertFalse(any(
            "aac" in args[args.index("-c:a") + 1:] and "/tmp/source.mov" in args
            for _tool, args in calls if "-c:a" in args
        ))

    def test_length_matching_audio_uses_direct_passthrough_when_safe(self):
        performer = self._make_performer()
        base_video = "/tmp/base.mkv"
        source_audio = "/tmp/source.mkv"

        with patch.object(performer, "_video_track_duration", return_value=6000), \
             patch.object(performer, "_base_output_end_ms", return_value=6000), \
             patch.object(performer, "_base_audio_end_ms", return_value=6000), \
             patch.object(performer, "_source_stream_start_offset_ms", return_value=0), \
             patch.object(performer, "_source_stream_end_offset_ms", return_value=5900), \
             patch.object(performer, "_audio_content_start_ms", return_value=0), \
             patch.object(performer, "_stream_info", return_value={}), \
             patch.object(performer, "_track_sync_offset_ms", return_value=None), \
             patch.object(performer, "_audio_needs_mkvmerge_normalization", return_value=False), \
             patch.object(performer, "_prepare_normalized_unscaled_audio", side_effect=AssertionError):
            prepared = performer._prepare_stream_entries(
                video_streams=[(base_video, 0, None)],
                audio_streams=[(source_audio, 1, "eng")],
                subtitle_streams=[],
                required_input_files={base_video, source_audio},
                attachments=[],
                file_ids={base_video: 1, source_audio: 2},
                files_details={},
            )

        self.assertIn(_StreamEntry("audio", 1, source_audio, "eng", None), prepared.entries)
        self.assertEqual(prepared.input_files, {base_video, source_audio})

    def test_track_sync_offset_compensates_when_normalized_audio_start_is_lost(self):
        performer = self._make_performer()
        normalized_full_info = {
            "streams": [
                {"codec_type": "audio", "index": 0, "codec_name": "aac", "start_time": "0.000000"},
            ]
        }

        with patch.object(video_utils, "get_video_full_info", return_value=normalized_full_info):
            offset = performer._track_sync_offset_ms("/tmp/normalized.mka", "audio", 0, 471)

        self.assertEqual(offset, 471)

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

        performer._pair_match_cache[(base_video, audio_a)] = make_match_result(base_duration)
        performer._pair_match_cache[(base_video, audio_b)] = make_match_result(base_duration)

        patch_outputs = []

        def fake_patch_audio_constant_offset(
            _wd,
            _base_video,
            source_audio,
            _base_audio,
            output_path,
            _mapping,
            **_kwargs,
        ):
            self.assertEqual(source_audio.stream_index, 1)
            patch_outputs.append(output_path)
            return 0

        file_ids = {base_video: 1, audio_a: 2, audio_b: 3}
        with patch.object(performer, "_log_coverage", lambda *args, **kwargs: None), \
             patch.object(performer, "patch_audio_constant_offset", side_effect=fake_patch_audio_constant_offset):
            first_stream, first_start = performer._patch_mismatched_audio(
                VideoStreamRef(base_video, 0, None), AudioStreamRef(audio_a, 1, "pol"),
                None, base_duration, None, file_ids,
            )
            second_stream, second_start = performer._patch_mismatched_audio(
                VideoStreamRef(base_video, 0, None), AudioStreamRef(audio_b, 1, "pol"),
                None, base_duration, None, file_ids,
            )

        self.assertEqual(first_stream.stream_index, 0)
        self.assertEqual(second_stream.stream_index, 0)
        self.assertEqual(first_stream.language, "pol")
        self.assertEqual(second_stream.language, "pol")
        self.assertEqual(first_start, 0)
        self.assertEqual(second_start, 0)
        self.assertEqual(patch_outputs, [first_stream.path, second_stream.path])
        self.assertNotEqual(first_stream.path, second_stream.path)

    def test_subsegment_audio_patch_preserves_selected_stream_identity(self):
        performer = self._make_performer()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, None)
        base_audio = AudioStreamRef("/tmp/base.mkv", 3, "eng")
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, "pol")
        mapping = [(0, 0), (60000, 60000)]
        performer._pair_match_cache[(base_video.path, source_audio.path)] = _PairMatchResult(
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

        def fake_patch_audio_segment(
            _wd,
            received_base_video,
            received_source_audio,
            received_base_audio,
            _output_path,
            _mapping,
            _segment_count,
            _lhs_frames,
            _rhs_frames,
            **_kwargs,
        ):
            self.assertEqual(received_base_video, base_video)
            self.assertEqual(received_source_audio, source_audio)
            self.assertEqual(received_base_audio, base_audio)
            return 125

        with patch.object(performer, "_choose_audio_strategy", return_value=_AudioStrategy.SUBSEGMENT), \
             patch.object(performer, "_prepare_audio_mapping", return_value=mapping), \
             patch.object(performer, "_patch_audio_segment", side_effect=fake_patch_audio_segment):
            patched = performer._patch_mismatched_audio(
                base_video,
                source_audio,
                base_audio,
                60000,
                60000,
                {base_video.path: 1, source_audio.path: 2},
            )

        self.assertEqual(patched.stream.stream_index, 0)
        self.assertEqual(patched.stream.language, "pol")
        self.assertEqual(patched.timeline_start_ms, 125)

    def test_subsegment_patcher_decodes_selected_later_audio_stream(self):
        performer = self._make_performer()
        base_video = VideoStreamRef("/tmp/base.mkv", 0, None)
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, "pol")
        extracted_streams = []

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.object(
                 performer,
                 "_extract_selected_audio_to_flac",
                 side_effect=lambda stream, *_args, **_kwargs: extracted_streams.append(stream),
             ), \
             patch.object(performer, "_get_audio_params", return_value=(2, 48000, "s16")) as get_params, \
             patch.object(performer, "_get_audio_channel_layout", return_value="stereo") as get_layout, \
             patch.object(performer, "_concat_and_encode"), \
             patch.object(process_utils, "start_process", return_value=_FAKE_PROCESS_OK), \
             patch.object(process_utils, "raise_on_error"):
            performer._patch_audio_segment(
                tmpdir,
                base_video,
                source_audio,
                None,
                os.path.join(tmpdir, "out.mka"),
                [(0, 0), (60000, 60000)],
                1,
                {},
                {},
                fill_gaps_from_base=False,
            )

        self.assertEqual(extracted_streams, [source_audio])
        get_params.assert_called_once_with(source_audio)
        get_layout.assert_called_once_with(source_audio)

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
                return target_dur
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
                returned_sync_offset = performer.patch_audio_constant_offset(
                    wd, VideoStreamRef("/base.mkv", 0, None),
                    AudioStreamRef("/source.avi", 1, "pol"), None, output_path, pairs,
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
        # adjusted_rate = 48000 × fps_ratio ≈ 46033.69
        expected_rate = 48000 * source_dur / target_dur
        self.assertIn(f"asetrate={expected_rate:.6f}", filter_arg)
        self.assertIn("aresample=48000", filter_arg)

        # --- Trim timestamps must match mapping endpoints sample-accurately ---
        ffmpeg_args = [c[1] for c in calls if c[0] == "ffmpeg"]
        trim_call = next(a for a in ffmpeg_args if "source_trimmed" in str(a))
        self.assertNotIn("-ss", trim_call)
        self.assertNotIn("-to", trim_call)
        self.assertLess(trim_call.index("-i"), trim_call.index("-filter:a"))
        trim_filter = trim_call[trim_call.index("-filter:a") + 1]
        self.assertIn(
            f"atrim=start={seg2_start / 1000:.6f}:end={seg2_end / 1000:.6f}",
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
            if "source_trimmed" in path or "source_scaled" in path or "out." in path:
                return source_dur
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
                sync = performer.patch_audio_constant_offset(
                    wd, VideoStreamRef("/base.mkv", 0, None),
                    AudioStreamRef("/source.mkv", 1, "pol"), None, output_path, pairs,
                    fill_gaps_from_base=False,
                )

        self.assertEqual(sync, seg1_start,
                         "Without audio start offset, sync offset should be the raw seg1_start")

    def test_patch_audio_keeps_priming_aware_source_start(self):
        # A source carrying CodecDelay priming (mkv) patched onto a base that does
        # not (mp4) must be placed by its priming-aware content start with no extra
        # per-container correction.  On builds that expose AAC priming the offset
        # carries the encoder-delay frame (matching the priming-exposed decode); on
        # absorbing builds it does not.  The sync offset is therefore build-specific
        # by design — it must not be "compensated" back, which would shift the track
        # one priming frame on exposing builds (e.g. Ubuntu's ffmpeg).
        pairs = [(22, 3), (62313, 62305)]
        source_dur = 62302
        target_dur = 62291

        def fake_get_duration(path, **_kwargs):
            if "source_trimmed" in path:
                return source_dur
            if "source_scaled" in path or "out." in path:
                return target_dur
            return 63000

        def fake_full_info(path, **_kwargs):
            if path == "/base.mp4":
                return {"streams": [
                    {"codec_type": "video", "index": 0, "start_time": "0.500000"},
                    {"codec_type": "audio", "index": 1, "codec_name": "aac", "channels": 1,
                     "sample_rate": "48000", "sample_fmt": "fltp",
                     "start_time": "0.478000", "initial_padding": 0},
                ]}
            return {"streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.478000"},
                {"codec_type": "audio", "index": 1, "codec_name": "aac", "channels": 1,
                 "sample_rate": "48000", "sample_fmt": "fltp",
                 "start_time": "0.500000", "initial_padding": 1024},
            ]}

        expected_by_exposure = {False: 519, True: 540}
        for priming_exposed, expected in expected_by_exposure.items():
            performer = self._make_performer()
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "out.mka")
                wd = os.path.join(tmpdir, "work")

                with patch.object(process_utils, 'start_process',
                                  side_effect=lambda t, a, **kw: _FAKE_PROCESS_OK), \
                     patch.object(process_utils, 'raise_on_error', lambda r: None), \
                     patch.object(performer, "_aac_priming_exposed", return_value=priming_exposed), \
                     patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                     patch.object(video_utils, 'get_video_full_info', side_effect=fake_full_info):
                    sync = performer.patch_audio_constant_offset(
                        wd, VideoStreamRef("/base.mp4", 0, None),
                        AudioStreamRef("/source.mkv", 1, "pol"), None, output_path, pairs,
                        fill_gaps_from_base=False,
                    )

            self.assertEqual(sync, expected, f"priming_exposed={priming_exposed}")

    def test_patch_audio_keeps_sync_when_base_and_source_have_aac_padding(self):
        performer = self._make_performer()
        pairs = [(0, 0), (62291, 62305)]
        source_dur = 62305
        target_dur = 62291

        def fake_get_duration(path, **_kwargs):
            if "source_trimmed" in path:
                return source_dur
            if "source_scaled" in path or "out." in path:
                return target_dur
            return 63000

        def fake_full_info(_path, **_kwargs):
            return {"streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.478000"},
                {"codec_type": "audio", "index": 1, "codec_name": "aac", "channels": 1,
                 "sample_rate": "48000", "sample_fmt": "fltp",
                 "start_time": "0.500000", "initial_padding": 1024},
            ]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process',
                              side_effect=lambda t, a, **kw: _FAKE_PROCESS_OK), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                 patch.object(video_utils, 'get_video_full_info', side_effect=fake_full_info):
                sync = performer.patch_audio_constant_offset(
                    wd, VideoStreamRef("/base.mkv", 0, None),
                    AudioStreamRef("/source.mkv", 1, "pol"), None, output_path, pairs,
                    fill_gaps_from_base=False,
                )

        self.assertEqual(sync, 500)

    def test_patch_audio_adds_missing_positive_source_timeline_start(self):
        performer = self._make_performer()
        source_dur = 60000
        pairs = [(0, 0), (source_dur, source_dur)]

        def fake_get_duration(path, **_kwargs):
            if "source_trimmed" in path or "source_scaled" in path or "out." in path:
                return source_dur
            return source_dur

        fake_full_info = {
            "format": {"start_time": "0.500000"},
            "streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.500000"},
                {"codec_type": "audio", "index": 1, "channels": 2, "sample_rate": "48000",
                 "sample_fmt": "s16", "start_time": "0.500000"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process',
                              side_effect=lambda t, a, **kw: _FAKE_PROCESS_OK), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                 patch.object(video_utils, 'get_video_full_info', return_value=fake_full_info):
                sync = performer.patch_audio_constant_offset(
                    wd, VideoStreamRef("/base.mkv", 0, None),
                    AudioStreamRef("/source.mkv", 1, "pol"), None, output_path, pairs,
                    fill_gaps_from_base=False,
                )

        self.assertEqual(sync, 500)

    def test_patch_audio_does_not_double_positive_source_timeline_start(self):
        performer = self._make_performer()
        source_dur = 60000
        pairs = [(500, 0), (60500, source_dur)]

        def fake_get_duration(path, **_kwargs):
            if "source_trimmed" in path or "source_scaled" in path or "out." in path:
                return source_dur
            return 60500

        fake_full_info = {
            "format": {"start_time": "0.500000"},
            "streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.500000"},
                {"codec_type": "audio", "index": 1, "channels": 2, "sample_rate": "48000",
                 "sample_fmt": "s16", "start_time": "0.500000"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process',
                              side_effect=lambda t, a, **kw: _FAKE_PROCESS_OK), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                 patch.object(video_utils, 'get_video_full_info', return_value=fake_full_info):
                sync = performer.patch_audio_constant_offset(
                    wd, VideoStreamRef("/base.mkv", 0, None),
                    AudioStreamRef("/source.mkv", 1, "pol"), None, output_path, pairs,
                    fill_gaps_from_base=False,
                )

        self.assertEqual(sync, 500)

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
                    performer.patch_audio_constant_offset(
                        wd, VideoStreamRef("/base.mkv", 0, None),
                        AudioStreamRef("/source.avi", 1, "pol"),
                        AudioStreamRef("/base.mkv", 1, "eng"), output_path, pairs,
                        fill_gaps_from_base=True,
                    )
                self.assertIn("fill-audio-gaps", str(ctx.exception))
                expected_correction = round(
                    start_gap_ms * ((seg1_end - seg1_start) / source_dur)
                )
                self.assertIn(str(expected_correction), str(ctx.exception))

    def test_validate_audio_duration_raises_on_excessive_deviation(self):
        """_validate_audio_duration should raise when deviation exceeds 5%."""
        performer = self._make_performer()
        with self.assertRaises(RuntimeError) as ctx:
            performer._validate_audio_duration(9000, 10000, "test audio")
        self.assertIn("10.0%", str(ctx.exception))
        self.assertIn("test audio", str(ctx.exception))

    def test_validate_audio_duration_passes_within_tolerance(self):
        """_validate_audio_duration should not raise for small deviations."""
        performer = self._make_performer()
        # 2% deviation — within 5% threshold
        performer._validate_audio_duration(9800, 10000, "test audio")
