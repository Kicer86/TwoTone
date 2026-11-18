import os
import tempfile
import unittest
from array import array
from unittest.mock import patch

from common import run_ffmpeg
from parameterized import parameterized

from melt.helpers import MeltComponentFactories, _FAKE_PROCESS_OK
from twotone.tools.melt.audio_patcher import (
    AudioPatcher,
    AudioPatchRequest,
    AudioSourceWindow,
    AudioStrategy,
    TimelineInterval,
    VideoToAudioTimeline,
    _AudioPart,
    segment_range,
)
from twotone.tools.melt.melt_common import AudioStreamRef
from twotone.tools.utils import process_utils, video_utils


class AudioPatcherUnitTest(MeltComponentFactories, unittest.TestCase):
    """Unit tests for the AudioPatcher executor and its timeline value objects."""

    @staticmethod
    def _execute_global_time_scale(
        patcher: AudioPatcher,
        working_dir,
        source_audio,
        base_audio,
        output_path,
        segment_pairs,
        base_duration_ms,
        *,
        fill_gaps_from_base=True,
    ):
        segment = segment_range(segment_pairs)
        request = AudioPatchRequest(
            working_dir=working_dir,
            output_path=output_path,
            source_audio=source_audio,
            base_audio=base_audio,
            mapping=tuple(segment_pairs),
            target_interval=TimelineInterval(segment.lhs_start, segment.lhs_end),
            output_interval=TimelineInterval(0, base_duration_ms),
            source_timeline=VideoToAudioTimeline(0, 0),
            base_timeline=(VideoToAudioTimeline(0, 0) if base_audio is not None else None),
            fill_gaps_from_base=fill_gaps_from_base,
            lhs_frames={},
            rhs_frames={},
            output_mapping_to_video_ms=0,
            strategy=AudioStrategy.GLOBAL_TIME_SCALE,
        )
        return patcher.execute(request).timeline_start_ms

    def _collect_ffmpeg_calls(self, patcher, segment_pairs, base_duration_ms,
                               source_sample_rate=48000, source_channels=2, source_sample_fmt="s16",
                               source_channel_layout=None,
                               fill_gaps_from_base=True):
        """Run global-time-scale patching with mocked externals and capture ffmpeg calls."""
        calls = []

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_raise_on_error(result):
            pass

        left_points = [p[0] for p in segment_pairs]
        right_points = [p[1] for p in segment_pairs]
        target_segment_dur = max(left_points) - min(left_points)
        source_segment_dur = max(right_points) - min(right_points)

        def fake_get_duration(path, **_kwargs):
            if "source_trimmed" in path:
                return source_segment_dur
            if "source_scaled" in path:
                return target_segment_dur
            if "head" in path:
                return min(left_points)
            if "tail" in path:
                return base_duration_ms - max(left_points)
            if "out." in path:
                return base_duration_ms if fill_gaps_from_base else target_segment_dur
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
                self._execute_global_time_scale(
                    patcher,
                    wd,
                    AudioStreamRef("/source.mkv", 2, 2, "pol"),
                    AudioStreamRef("/base.mkv", 3, 3, "eng"),
                    output_path,
                    segment_pairs,
                    base_duration_ms,
                    fill_gaps_from_base=fill_gaps_from_base,
                )

        return calls

    def test_patch_audio_constant_offset_copy_when_ratio_near_one(self):
        """When source and target durations match, no asetrate filter should be applied."""
        patcher = self._make_audio_patcher()
        # seg1: 1000..5000 (4s), seg2: 500..4500 (4s) — ratio = 1.0
        pairs = [(1000, 500), (5000, 4500)]
        calls = self._collect_ffmpeg_calls(patcher, pairs, base_duration_ms=6000)

        # Should NOT contain any asetrate/aresample filter call
        filter_calls = [c for c in calls if any("asetrate" in str(a) for a in c[1])]
        self.assertEqual(filter_calls, [], "asetrate should not be used when ratio ≈ 1.0")

    def test_patch_audio_constant_offset_asetrate_when_ratio_differs(self):
        """When durations differ, asetrate+aresample should be applied."""
        patcher = self._make_audio_patcher()
        # seg1: 0..4000 (4s), seg2: 0..4100 (4.1s) — ratio ≈ 1.025
        pairs = [(0, 0), (4000, 4100)]
        calls = self._collect_ffmpeg_calls(patcher, pairs, base_duration_ms=4000, source_sample_rate=48000)

        filter_calls = [c for c in calls if any("asetrate" in str(a) for a in c[1])]
        self.assertEqual(len(filter_calls), 1, "asetrate should be used once")

        filter_arg = next(a for a in filter_calls[0][1] if "asetrate" in str(a))
        # adjusted_rate = 48000 * 4100 / 4000 = 49200
        self.assertEqual(filter_arg, "asetrate=49200,aresample=48000")

    def test_patch_audio_constant_offset_head_and_tail_extraction(self):
        """Head and tail segments should be extracted from base audio."""
        patcher = self._make_audio_patcher()
        # Matching region: 2000..8000 in base, so head=[0..2s] and tail=[8s..10s]
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(patcher, pairs, base_duration_ms=10000)

        ffmpeg_args_strs = [" ".join(str(a) for a in c[1]) for c in calls if c[0] == "ffmpeg"]

        head_calls = [s for s in ffmpeg_args_strs if "head" in s and "atrim=" in s]
        self.assertGreaterEqual(len(head_calls), 1, "Head segment should be extracted")
        self.assertIn("end=2.000000", head_calls[0])

        tail_calls = [s for s in ffmpeg_args_strs if "tail" in s and "atrim=" in s]
        self.assertGreaterEqual(len(tail_calls), 1, "Tail segment should be extracted")
        self.assertIn("start=8.000000", tail_calls[0])

    def test_patch_audio_constant_offset_no_head_when_at_start(self):
        """When matching starts at 0, no head should be extracted."""
        patcher = self._make_audio_patcher()
        pairs = [(0, 500), (4000, 4500)]
        calls = self._collect_ffmpeg_calls(patcher, pairs, base_duration_ms=6000)

        head_calls = [c for c in calls if any("head" in str(a) for a in c[1])]
        self.assertEqual(head_calls, [], "No head should be extracted when seg1 starts at 0")

    def test_patch_audio_constant_offset_no_tail_when_at_end(self):
        """When matching ends at base duration, no tail should be extracted."""
        patcher = self._make_audio_patcher()
        pairs = [(1000, 500), (6000, 5500)]
        calls = self._collect_ffmpeg_calls(patcher, pairs, base_duration_ms=6000)

        tail_calls = [c for c in calls if any("tail" in str(a) for a in c[1])]
        self.assertEqual(tail_calls, [], "No tail should be extracted when seg1 ends at base duration")

    def test_patch_audio_head_tail_normalized_to_source_params(self):
        """Head and tail must be re-encoded with source audio parameters so FLAC concat works."""
        patcher = self._make_audio_patcher()
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(
            patcher, pairs, base_duration_ms=10000,
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
        patcher = self._make_audio_patcher()

        calls = self._collect_ffmpeg_calls(
            patcher,
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

    def test_patch_audio_constant_offset_reencodes_even_when_ratio_near_one(self):
        """Even a 1:1 trim is decoded so AAC priming and source starts are normalized."""
        patcher = self._make_audio_patcher()
        # seg1: 0..6000 (full base), seg2: 500..6500 — same duration, no head/tail
        pairs = [(0, 500), (6000, 6500)]
        calls = self._collect_ffmpeg_calls(patcher, pairs, base_duration_ms=6000)

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

    def test_patch_audio_constant_offset_concat_single_pass(self):
        """When head/tail needed, concat should encode to AAC in a single pass (no intermediate FLAC)."""
        patcher = self._make_audio_patcher()
        # seg1_start=2000 → has head
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(patcher, pairs, base_duration_ms=10000)

        ffmpeg_args_strs = [" ".join(str(a) for a in c[1]) for c in calls if c[0] == "ffmpeg"]
        concat_calls = [s for s in ffmpeg_args_strs if "concat" in s]
        self.assertEqual(len(concat_calls), 1, "Should have exactly one concat call")
        self.assertIn("aac", concat_calls[0], "Concat should encode directly to AAC")
        # No separate FLAC→AAC pass
        flac_to_aac = [s for s in ffmpeg_args_strs if "merged.flac" in s and "aac" in s and "concat" not in s]
        self.assertEqual(flac_to_aac, [], "Should not have a separate FLAC→AAC re-encoding step")

    def test_nonstandard_channel_layout_triggers_aformat(self):
        """5.1(side) is not a native AAC layout — aformat filter must normalize it to avoid PCE."""
        patcher = self._make_audio_patcher()
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(
            patcher, pairs, base_duration_ms=10000,
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
        patcher = self._make_audio_patcher()
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(
            patcher, pairs, base_duration_ms=10000,
            source_channels=6, source_channel_layout="5.1",
        )

        concat_calls = [c for c in calls if c[0] == "ffmpeg" and "concat" in str(c[1])]
        self.assertEqual(len(concat_calls), 1)
        args = concat_calls[0][1]
        aformat_args = [a for a in args if isinstance(a, str) and "aformat" in a]
        self.assertEqual(aformat_args, [], "aformat should not be used for standard 5.1")

    def test_stereo_layout_skips_aformat(self):
        """Stereo is a standard AAC layout — no aformat needed."""
        patcher = self._make_audio_patcher()
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(
            patcher, pairs, base_duration_ms=10000,
            source_channels=2, source_channel_layout="stereo",
        )

        concat_calls = [c for c in calls if c[0] == "ffmpeg" and "concat" in str(c[1])]
        self.assertEqual(len(concat_calls), 1)
        args = concat_calls[0][1]
        aformat_args = [a for a in args if isinstance(a, str) and "aformat" in a]
        self.assertEqual(aformat_args, [], "aformat should not be used for stereo")

    def test_unknown_channel_layout_skips_aformat(self):
        """When ffprobe doesn't report channel_layout, aformat should not be added."""
        patcher = self._make_audio_patcher()
        pairs = [(2000, 1000), (8000, 7000)]
        # No source_channel_layout → defaults to None
        calls = self._collect_ffmpeg_calls(patcher, pairs, base_duration_ms=10000)

        concat_calls = [c for c in calls if c[0] == "ffmpeg" and "concat" in str(c[1])]
        self.assertEqual(len(concat_calls), 1)
        args = concat_calls[0][1]
        aformat_args = [a for a in args if isinstance(a, str) and "aformat" in a]
        self.assertEqual(aformat_args, [], "aformat should not be used when layout is unknown")

    def test_silence_mode_skips_head_and_tail(self):
        """With fill_gaps_from_base=False, no head/tail extraction or generation should occur."""
        patcher = self._make_audio_patcher()
        # Matching region: 2000..8000 in base (10s total) → would have head and tail
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(patcher, pairs, base_duration_ms=10000, fill_gaps_from_base=False)

        ffmpeg_args_strs = [" ".join(str(a) for a in c[1]) for c in calls if c[0] == "ffmpeg"]

        # No head/tail extraction or silence generation
        head_tail_calls = [s for s in ffmpeg_args_strs if "head" in s or "tail" in s or "anullsrc" in s]
        self.assertEqual(head_tail_calls, [], "Silence mode should not produce head/tail segments")

        # No base video extraction
        base_extract_calls = [s for s in ffmpeg_args_strs if "/base.mkv" in s]
        self.assertEqual(base_extract_calls, [], "Silence mode should not read from base video")

    def test_silence_mode_reencodes_when_no_scaling(self):
        """With fill_gaps_from_base=False and ratio ≈ 1.0, audio is still decoded for stable timing."""
        patcher = self._make_audio_patcher()
        pairs = [(0, 500), (4000, 4500)]
        calls = self._collect_ffmpeg_calls(patcher, pairs, base_duration_ms=6000, fill_gaps_from_base=False)

        ffmpeg_calls = [c[1] for c in calls if c[0] == "ffmpeg"]
        trim_call = next(a for a in ffmpeg_calls if "source_trimmed" in str(a))
        self.assertNotIn("copy", trim_call)
        self.assertIn("-filter:a", trim_call)
        concat_calls = [a for a in ffmpeg_calls if "concat" in str(a)]
        self.assertEqual(len(concat_calls), 1)

    def test_prepare_normalized_unscaled_audio_decodes_via_flac_then_encodes_aac_once(self):
        patcher = self._make_audio_patcher()
        patcher.timeline._input_file_ids = {"/tmp/source.mov": 2}
        calls = []
        progress_options = []
        source_full_info = {
            "streams": [
                {"codec_type": "video", "index": 0, "codec_name": "h264"},
                {"codec_type": "audio", "index": 1, "codec_name": "aac", "start_time": "0.471000"},
            ]
        }

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args)))
            progress_options.append(kwargs)
            return _FAKE_PROCESS_OK

        def fake_full_info(path, logger=None):
            return source_full_info

        with patch.object(video_utils, "get_video_full_info", side_effect=fake_full_info), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error", lambda r: None):
            prepared = patcher.prepare_normalized_unscaled_audio(
                AudioStreamRef("/tmp/source.mov", 1, 1, "pol")
            )
            cached = patcher.prepare_normalized_unscaled_audio(
                AudioStreamRef("/tmp/source.mov", 1, 1, "pol")
            )

        self.assertEqual(prepared.mkvmerge_track_id, 0)
        self.assertEqual(prepared.ffprobe_stream_index, 0)
        self.assertEqual(prepared.language, "pol")
        self.assertEqual(cached, prepared)

        # The flow runs exactly twice: decode the source to a priming-free FLAC,
        # then encode that FLAC to AAC a single time.  No AAC→AAC re-encode that
        # could re-apply encoder-delay priming.  The second call is fully cached.
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(tool == "ffmpeg" for tool, _ in calls))
        self.assertTrue(all(options.get("show_progress") for options in progress_options))
        self.assertEqual(
            [options.get("progress_description") for options in progress_options],
            [
                "Decoding file #2, audio track #1: AAC -> FLAC",
                "Encoding normalized audio from file #2, audio track #1: FLAC -> AAC",
            ],
        )

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

    def test_unscaled_executor_preserves_delayed_source_audio_as_a_virtual_prefix(self):
        """Missing leading samples shift placement instead of breaking duration checks."""
        performer = self._make_performer()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((0, 0), (62792, 62792)),
            target_interval=TimelineInterval(0, 62792),
            output_interval=TimelineInterval(0, 62792),
            source_timeline=VideoToAudioTimeline(0, 478, 62792),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            strategy=AudioStrategy.UNSCALED,
        )
        decoded: list[dict[str, object]] = []

        def fake_decode(_stream, _output_path, **kwargs):
            decoded.append(kwargs)

        def fake_duration(path, **_kwargs):
            if "source_unscaled" in path or path.endswith("out.mka"):
                return 62314
            raise AssertionError(f"Unexpected duration probe for {path}")

        with patch.object(performer.timeline, "decode_audio_stream_to_flac", side_effect=fake_decode), \
             patch.object(performer.audio_patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(performer.audio_patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(performer.audio_patcher, "_concat_and_encode"), \
             patch.object(video_utils, "get_video_duration", side_effect=fake_duration):
            result = performer.audio_patcher.execute(request)

        self.assertEqual(decoded, [{
            "trim_start_ms": 478,
            "trim_end_ms": 62792,
            "normalize_to": (2, 48000, "s16"),
            "logger": performer.logger,
        }])
        self.assertEqual(result.timeline_start_ms, 478)
        self.assertEqual(result.duration_ms, 62314)
        self.assertEqual(result.transformation, "unscaled-normalization")

    def test_unscaled_executor_does_not_turn_a_shared_stream_offset_into_a_gap(self):
        """Audio and video at the same positive PTS still begin together."""
        performer = self._make_performer()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((0, 0), (60000, 60000)),
            target_interval=TimelineInterval(0, 60000),
            output_interval=TimelineInterval(0, 60000),
            source_timeline=VideoToAudioTimeline(500, 500),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            output_mapping_to_video_ms=500,
            strategy=AudioStrategy.UNSCALED,
        )
        decoded: list[dict[str, object]] = []

        def fake_decode(_stream, _output_path, **kwargs):
            decoded.append(kwargs)

        with patch.object(performer.timeline, "decode_audio_stream_to_flac", side_effect=fake_decode), \
             patch.object(performer.audio_patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(performer.audio_patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(performer.audio_patcher, "_concat_and_encode"), \
             patch.object(video_utils, "get_video_duration", return_value=60000):
            result = performer.audio_patcher.execute(request)

        self.assertEqual(decoded, [{
            "trim_start_ms": 500,
            "trim_end_ms": 60500,
            "normalize_to": (2, 48000, "s16"),
            "logger": performer.logger,
        }])
        self.assertEqual(result.timeline_start_ms, 500)

    def test_unscaled_executor_preserves_container_phase_between_mapping_origins(self):
        """A source AAC phase is carried as a small virtual prefix, not dropped."""
        patcher = self._make_audio_patcher()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((22, 0), (60022, 60000)),
            target_interval=TimelineInterval(22, 60022),
            output_interval=TimelineInterval(0, 60022),
            source_timeline=VideoToAudioTimeline(500, 500),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            output_mapping_to_video_ms=478,
            strategy=AudioStrategy.UNSCALED,
        )
        decoded: list[dict[str, object]] = []

        def fake_decode(_stream, _output_path, **kwargs):
            decoded.append(kwargs)

        with patch.object(patcher.timeline, "decode_audio_stream_to_flac", side_effect=fake_decode), \
             patch.object(patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(patcher, "_concat_and_encode"), \
             patch.object(video_utils, "get_video_duration", return_value=60000):
            result = patcher.execute(request)

        self.assertEqual(result.timeline_start_ms, 500)

    def test_global_executor_places_audio_on_base_video_stream_timeline(self):
        """A rebased mapping must be translated back to the muxed base-video PTS."""
        patcher = self._make_audio_patcher()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((0, 0), (60000, 60000)),
            target_interval=TimelineInterval(0, 60000),
            output_interval=TimelineInterval(0, 60000),
            source_timeline=VideoToAudioTimeline(0, 0),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            output_mapping_to_video_ms=510,
            strategy=AudioStrategy.GLOBAL_TIME_SCALE,
        )

        with patch.object(patcher.timeline, "decode_audio_stream_to_flac"), \
             patch.object(patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(patcher, "_concat_and_encode"), \
             patch.object(video_utils, "get_video_duration", return_value=60000):
            result = patcher.execute(request)

        self.assertEqual(result.timeline_start_ms, 510)
        self.assertEqual(result.duration_ms, 60000)
        self.assertEqual(result.transformation, "global-time-scale")

    def test_global_time_scale_uses_a_precise_rational_resample_ratio(self):
        """A fractional virtual rate remains precise without lossy tail trimming."""
        source_duration_ms = 4852640
        target_duration_ms = 5056084
        sample_rate = 48000

        filter_arg = AudioPatcher._global_time_scale_filter(
            source_duration_ms,
            target_duration_ms,
            sample_rate,
        )

        self.assertEqual(
            filter_arg,
            "asetrate=184141,"
            "aresample=191861,"
            "asetrate=48000",
        )
        self.assertNotIn("atempo", filter_arg)
        self.assertNotIn("apad", filter_arg)
        self.assertNotIn("atrim", filter_arg)

        source_samples = source_duration_ms * sample_rate // 1000
        scaled_samples = round(source_samples * 191861 / 184141)
        target_samples = target_duration_ms * sample_rate // 1000
        self.assertEqual(scaled_samples, target_samples)

    def test_global_time_scale_does_not_collapse_long_near_simple_ratio(self):
        """A tiny offset from 11/10 must not accumulate past the sample budget."""
        source_duration_ms = 7200000
        target_duration_ms = 6545460
        sample_rate = 48000

        filter_arg = AudioPatcher._global_time_scale_filter(
            source_duration_ms,
            target_duration_ms,
            sample_rate,
        )

        self.assertEqual(
            filter_arg,
            "asetrate=120000,aresample=109091,asetrate=48000",
        )
        source_samples = source_duration_ms * sample_rate // 1000
        scaled_samples = round(source_samples * 109091 / 120000)
        target_samples = target_duration_ms * sample_rate // 1000
        self.assertEqual(scaled_samples, target_samples)

    def test_global_time_scale_refines_long_near_six_sevenths_ratio(self):
        """A film-length ratio near 6/7 must not collapse to that simple fraction."""
        source_duration_ms = 12178055
        target_duration_ms = 14207737
        sample_rate = 48000

        filter_arg = AudioPatcher._global_time_scale_filter(
            source_duration_ms,
            target_duration_ms,
            sample_rate,
        )

        self.assertEqual(
            filter_arg,
            "asetrate=329135,aresample=383991,asetrate=48000",
        )
        rates = [int(item.split("=", maxsplit=1)[1]) for item in filter_arg.split(",")]
        self.assertTrue(all(0 < rate <= AudioPatcher._MAX_RATIONAL_SCALE_RATE for rate in rates))

        source_samples = source_duration_ms * sample_rate // 1000
        target_samples = target_duration_ms * sample_rate // 1000
        scaled_samples = round(source_samples * rates[1] / rates[0])
        duration_tolerance_ms = AudioPatcher._audio_duration_tolerance_ms(sample_rate)
        self.assertLessEqual(
            abs(scaled_samples - target_samples) * 1000,
            duration_tolerance_ms * sample_rate,
        )

    def test_global_time_scale_keeps_a_direct_rate_that_fits_the_sample_budget(self):
        """Short parts retain a simple integer rate when it is already precise enough."""
        source_duration_ms = 62291
        target_duration_ms = 63562
        sample_rate = 48000

        filter_arg = AudioPatcher._global_time_scale_filter(
            source_duration_ms,
            target_duration_ms,
            sample_rate,
        )

        self.assertEqual(filter_arg, "asetrate=47040,aresample=48000")
        source_samples = source_duration_ms * sample_rate // 1000
        target_samples = target_duration_ms * sample_rate // 1000
        scaled_samples = round(source_samples * sample_rate / 47040)
        duration_tolerance_ms = AudioPatcher._audio_duration_tolerance_ms(sample_rate)
        self.assertLessEqual(
            abs(scaled_samples - target_samples) * 1000,
            duration_tolerance_ms * sample_rate,
        )

    def test_global_time_scale_rejects_an_over_budget_ratio_at_the_rate_cap(self):
        """A bounded rate must fail clearly rather than emit a bad approximation."""
        with patch.object(AudioPatcher, "_MAX_RATIONAL_SCALE_RATE", 192000):
            with self.assertRaisesRegex(RuntimeError, "Cannot represent global audio scaling"):
                AudioPatcher._global_time_scale_filter(12178055, 14207737, 48000)

    def test_global_time_scale_rejects_unsupported_values(self):
        """FFmpeg rate values and requested durations must be positive and bounded."""
        invalid_values = [
            (0, 1000, 48000),
            (1000, 0, 48000),
            (1000, 1000, 0),
            (1000, 1000, AudioPatcher._MAX_RATIONAL_SCALE_RATE + 1),
        ]

        for source_duration_ms, target_duration_ms, sample_rate in invalid_values:
            with self.assertRaises(ValueError):
                AudioPatcher._global_time_scale_filter(
                    source_duration_ms,
                    target_duration_ms,
                    sample_rate,
                )

    def test_global_time_scale_preserves_the_last_source_sample(self):
        """The duration correction must not replace the real source tail with silence."""
        source_duration_ms = 600000
        source_per_base = 4852640 / 5056084
        target_duration_ms = round(source_duration_ms / source_per_base)
        sample_rate = 48000
        last_source_sample = source_duration_ms * sample_rate // 1000 - 1

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.flac")
            scaled_path = os.path.join(tmpdir, "scaled.flac")
            tail_path = os.path.join(tmpdir, "tail.pcm")
            run_ffmpeg([
                "-y",
                "-f", "lavfi",
                "-i", (
                    f"aevalsrc=if(eq(n\\,{last_source_sample})\\,0.9\\,0):"
                    f"s={sample_rate}:d={source_duration_ms / 1000:g}"
                ),
                "-c:a", "flac",
                source_path,
            ], expected_path=source_path)
            patcher = self._make_audio_patcher()
            scaled_part = patcher._scale_audio_part(
                _AudioPart(source_path, source_duration_ms),
                source_per_base,
                scaled_path,
                sample_rate,
                AudioStrategy.GLOBAL_TIME_SCALE,
                label="fractional-rate regression audio",
            )
            run_ffmpeg([
                "-y",
                "-sseof", "-0.020",
                "-i", scaled_path,
                "-map", "0:a:0",
                "-c:a", "pcm_s16le",
                "-f", "s16le",
                tail_path,
            ], expected_path=tail_path)

            samples = array("h")
            with open(tail_path, "rb") as pcm_file:
                samples.frombytes(pcm_file.read())

        self.assertLessEqual(
            abs(scaled_part.duration_ms - target_duration_ms),
            patcher._audio_duration_tolerance_ms(sample_rate),
        )
        self.assertGreaterEqual(len(samples), 900)
        self.assertTrue(any(samples[-sample_rate // 1000:]))

    def test_global_time_scale_refined_near_six_sevenths_preserves_the_last_source_sample(self):
        """The high-precision 6/7-adjacent filter must retain the real audio tail."""
        ratio_source_duration_ms = 12178055
        ratio_target_duration_ms = 14207737
        source_duration_ms = 10000
        target_duration_ms = round(
            source_duration_ms * ratio_target_duration_ms / ratio_source_duration_ms
        )
        sample_rate = 48000
        last_source_sample = source_duration_ms * sample_rate // 1000 - 1
        filter_arg = AudioPatcher._global_time_scale_filter(
            ratio_source_duration_ms,
            ratio_target_duration_ms,
            sample_rate,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.flac")
            scaled_path = os.path.join(tmpdir, "scaled.flac")
            tail_path = os.path.join(tmpdir, "tail.pcm")
            run_ffmpeg([
                "-y",
                "-f", "lavfi",
                "-i", (
                    f"aevalsrc=if(eq(n\\,{last_source_sample})\\,0.9\\,0):"
                    f"s={sample_rate}:d={source_duration_ms / 1000:g}"
                ),
                "-c:a", "flac",
                source_path,
            ], expected_path=source_path)
            run_ffmpeg([
                "-y",
                "-i", source_path,
                "-filter:a", filter_arg,
                "-sample_fmt", "s32", "-c:a", "flac",
                scaled_path,
            ], expected_path=scaled_path)
            run_ffmpeg([
                "-y",
                "-sseof", "-0.020",
                "-i", scaled_path,
                "-map", "0:a:0",
                "-c:a", "pcm_s16le",
                "-f", "s16le",
                tail_path,
            ], expected_path=tail_path)

            samples = array("h")
            with open(tail_path, "rb") as pcm_file:
                samples.frombytes(pcm_file.read())
            actual_duration_ms = video_utils.get_video_duration(scaled_path)

        if actual_duration_ms is not None:
            self.assertLessEqual(
                abs(actual_duration_ms - target_duration_ms),
                AudioPatcher._audio_duration_tolerance_ms(sample_rate),
            )
        else:
            self.fail("Could not determine the scaled regression audio duration.")
        self.assertGreaterEqual(len(samples), 900)
        self.assertTrue(any(samples[-sample_rate // 1000:]))

    @parameterized.expand([
        ("below", 1.000999),
        ("at", 1.001),
        ("above", 1.001001),
        ("below_inverse", 0.999001),
        ("at_inverse", 0.999),
        ("above_inverse", 0.998999),
    ])
    def test_global_executor_scales_material_durations_around_strategy_threshold(
        self,
        _name: str,
        source_per_base: float,
    ):
        """Executor precision, unlike strategy selection, is not a coarse ratio threshold."""
        patcher = self._make_audio_patcher()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        target_duration_ms = 1000000
        source_duration_ms = round(target_duration_ms * source_per_base)
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((0, 0), (target_duration_ms, source_duration_ms)),
            target_interval=TimelineInterval(0, target_duration_ms),
            output_interval=TimelineInterval(0, target_duration_ms),
            source_timeline=VideoToAudioTimeline(0, 0),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            strategy=AudioStrategy.GLOBAL_TIME_SCALE,
        )
        calls: list[tuple[str, list[str]]] = []

        def fake_start_process(tool, args, **_kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_duration(path, **_kwargs):
            if "source_trimmed" in path:
                return source_duration_ms
            if "source_scaled" in path or path.endswith("out.mka"):
                return target_duration_ms
            raise AssertionError(f"Unexpected duration probe for {path}")

        with patch.object(patcher.timeline, "decode_audio_stream_to_flac"), \
             patch.object(patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(patcher, "_concat_and_encode"), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error"), \
             patch.object(video_utils, "get_video_duration", side_effect=fake_duration):
            result = patcher.execute(request)

        filters = [
            args[args.index("-filter:a") + 1]
            for tool, args in calls
            if tool == "ffmpeg" and "-filter:a" in args
        ]
        self.assertEqual(filters, [patcher._global_time_scale_filter(
            source_duration_ms,
            target_duration_ms,
            48000,
        )])
        self.assertEqual(result.timeline_start_ms, 0)
        self.assertEqual(result.duration_ms, target_duration_ms)

    def test_subsegment_executor_scales_a_60_second_near_unity_source_part(self):
        """A physical 60-second part uses atempo when its target differs by 30 ms."""
        patcher = self._make_audio_patcher()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((0, 0), (60000, 60030)),
            target_interval=TimelineInterval(0, 60000),
            output_interval=TimelineInterval(0, 60000),
            source_timeline=VideoToAudioTimeline(0, 0, 60000),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            strategy=AudioStrategy.SUBSEGMENT,
            segment_count=1,
        )
        calls: list[tuple[str, list[str]]] = []

        def fake_start_process(tool, args, **_kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_duration(path, **_kwargs):
            if "source_segment_0_scaled" in path or path.endswith("out.mka"):
                return 59970
            if "source_segment_0" in path:
                return 60000
            raise AssertionError(f"Unexpected duration probe for {path}")

        with patch.object(patcher.timeline, "decode_audio_stream_to_flac"), \
             patch.object(patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(patcher, "_concat_and_encode"), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error"), \
             patch.object(video_utils, "get_video_duration", side_effect=fake_duration):
            result = patcher.execute(request)

        filters = [
            args[args.index("-filter:a") + 1]
            for tool, args in calls
            if tool == "ffmpeg" and "-filter:a" in args
        ]
        self.assertEqual(filters, ["atempo=1.0005"])
        self.assertEqual(result.timeline_start_ms, 0)
        self.assertEqual(result.duration_ms, 59970)

    def test_subsegment_executor_tolerates_a_nonzero_ratio_within_sample_budget(self):
        """A no-op remains valid only when its physical duration fits the budget."""
        patcher = self._make_audio_patcher()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((0, 0), (60000, 60001)),
            target_interval=TimelineInterval(0, 60000),
            output_interval=TimelineInterval(0, 60000),
            source_timeline=VideoToAudioTimeline(0, 0),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            strategy=AudioStrategy.SUBSEGMENT,
            segment_count=1,
        )
        calls: list[tuple[str, list[str]]] = []

        def fake_start_process(tool, args, **_kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_duration(path, **_kwargs):
            if "source_segment_0" in path or path.endswith("out.mka"):
                return 60001
            raise AssertionError(f"Unexpected duration probe for {path}")

        with patch.object(patcher.timeline, "decode_audio_stream_to_flac"), \
             patch.object(patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(patcher, "_concat_and_encode"), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error"), \
             patch.object(video_utils, "get_video_duration", side_effect=fake_duration):
            result = patcher.execute(request)

        filters = [
            args[args.index("-filter:a") + 1]
            for tool, args in calls
            if tool == "ffmpeg" and "-filter:a" in args
        ]
        self.assertEqual(filters, [])
        self.assertEqual(result.timeline_start_ms + result.duration_ms, 60001)

    def test_subsegment_executor_scales_one_frame_local_duration_differences(self):
        """Each near-unity segment is fitted before their final placement is checked."""
        patcher = self._make_audio_patcher()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        target_segment_duration_ms = 60000
        one_24fps_frame_ms = 42
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=(
                (0, 0),
                (target_segment_duration_ms, target_segment_duration_ms + one_24fps_frame_ms),
                (2 * target_segment_duration_ms, 2 * target_segment_duration_ms),
            ),
            target_interval=TimelineInterval(0, 2 * target_segment_duration_ms),
            output_interval=TimelineInterval(0, 2 * target_segment_duration_ms),
            source_timeline=VideoToAudioTimeline(0, 0),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            strategy=AudioStrategy.SUBSEGMENT,
            segment_count=2,
        )
        source_durations = [
            target_segment_duration_ms + one_24fps_frame_ms,
            target_segment_duration_ms - one_24fps_frame_ms,
        ]
        calls: list[tuple[str, list[str]]] = []

        def fake_start_process(tool, args, **_kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_duration(path, **_kwargs):
            if "source_segment_0_scaled" in path or "source_segment_1_scaled" in path:
                return target_segment_duration_ms
            if "source_segment_0" in path:
                return source_durations[0]
            if "source_segment_1" in path:
                return source_durations[1]
            if path.endswith("out.mka"):
                return 2 * target_segment_duration_ms
            raise AssertionError(f"Unexpected duration probe for {path}")

        with patch.object(patcher.timeline, "decode_audio_stream_to_flac"), \
             patch.object(patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(patcher, "_concat_and_encode"), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error"), \
             patch.object(video_utils, "get_video_duration", side_effect=fake_duration):
            result = patcher.execute(request)

        expected_factors = [
            source_durations[0] / target_segment_duration_ms,
            source_durations[1] / target_segment_duration_ms,
        ]
        filters = [
            args[args.index("-filter:a") + 1]
            for tool, args in calls
            if tool == "ffmpeg" and "-filter:a" in args
        ]
        self.assertEqual(filters, [f"atempo={factor:.12g}" for factor in expected_factors])
        self.assertEqual(result.timeline_start_ms, 0)
        self.assertEqual(result.duration_ms, 2 * target_segment_duration_ms)
        self.assertEqual(result.timeline_start_ms + result.duration_ms, 2 * target_segment_duration_ms)

    def test_unscaled_executor_preserves_early_source_end_as_a_virtual_suffix(self):
        """A source tail outside the stream is remembered as output silence."""
        performer = self._make_performer()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((0, 0), (60000, 60000)),
            target_interval=TimelineInterval(0, 60000),
            output_interval=TimelineInterval(0, 60000),
            source_timeline=VideoToAudioTimeline(0, 0, 59000),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            strategy=AudioStrategy.UNSCALED,
        )
        decoded: list[dict[str, object]] = []

        def fake_decode(_stream, _output_path, **kwargs):
            decoded.append(kwargs)

        def fake_duration(path, **_kwargs):
            if "source_unscaled" in path or path.endswith("out.mka"):
                return 59000
            raise AssertionError(f"Unexpected duration probe for {path}")

        with patch.object(performer.timeline, "decode_audio_stream_to_flac", side_effect=fake_decode), \
             patch.object(performer.audio_patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(performer.audio_patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(performer.audio_patcher, "_concat_and_encode"), \
             patch.object(video_utils, "get_video_duration", side_effect=fake_duration):
            result = performer.audio_patcher.execute(request)

        self.assertEqual(decoded, [{
            "trim_start_ms": 0,
            "trim_end_ms": 59000,
            "normalize_to": (2, 48000, "s16"),
            "logger": performer.logger,
        }])
        self.assertEqual(result.timeline_start_ms, 0)
        self.assertEqual(result.duration_ms, 59000)

    def test_decode_audio_window_measures_early_end_as_a_virtual_suffix(self):
        """Decoded samples, not optimistic container metadata, define the source tail."""
        performer = self._make_performer()
        stream = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")

        with patch.object(performer.timeline, "decode_audio_stream_to_flac") as decode, \
             patch.object(video_utils, "get_video_duration", return_value=59991):
            part = performer.audio_patcher._decode_audio_window(
                stream,
                VideoToAudioTimeline(0, 0),
                TimelineInterval(0, 60000),
                "/tmp/source.flac",
                (2, 48000, "s16"),
                label="source audio",
            )

        decode.assert_called_once_with(
            stream,
            "/tmp/source.flac",
            trim_start_ms=0,
            trim_end_ms=60000,
            normalize_to=(2, 48000, "s16"),
            logger=performer.logger,
        )
        self.assertEqual(part.duration_ms, 59991)
        self.assertEqual(part.source_window, AudioSourceWindow(
            TimelineInterval(0, 60000),
            TimelineInterval(0, 59991),
        ))
        self.assertEqual(part.source_window.missing_suffix_ms, 9)

    def test_subsegment_patcher_decodes_selected_later_audio_stream(self):
        patcher = self._make_audio_patcher()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        decoded_streams = []
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((0, 0), (60000, 60000)),
            target_interval=TimelineInterval(0, 60000),
            output_interval=TimelineInterval(0, 60000),
            source_timeline=VideoToAudioTimeline(0, 0),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            strategy=AudioStrategy.SUBSEGMENT,
            segment_count=1,
        )

        def fake_decode(stream, *_args, **_kwargs):
            decoded_streams.append(stream)

        def fake_duration(path, **_kwargs):
            return 60000 if "out.mka" in path or "source_segment" in path else 60000

        with patch.object(patcher.timeline, "decode_audio_stream_to_flac", side_effect=fake_decode), \
             patch.object(patcher, "_get_audio_params", return_value=(2, 48000, "s16")) as get_params, \
             patch.object(patcher, "_get_audio_channel_layout", return_value="stereo") as get_layout, \
             patch.object(patcher, "_concat_and_encode"), \
             patch.object(video_utils, "get_video_duration", side_effect=fake_duration):
            patcher.execute(request)

        self.assertEqual(decoded_streams, [source_audio])
        get_params.assert_called_once_with(source_audio)
        get_layout.assert_called_once_with(source_audio)

    def test_subsegment_patcher_transforms_video_time_before_decoding_audio(self):
        """A delayed stream cuts physical source samples while retaining its virtual prefix."""
        performer = self._make_performer()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((2420, 40), (62420, 60040)),
            target_interval=TimelineInterval(2420, 62420),
            output_interval=TimelineInterval(0, 62420),
            source_timeline=VideoToAudioTimeline(0, 488),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            strategy=AudioStrategy.SUBSEGMENT,
            segment_count=1,
        )
        decoded: list[tuple[AudioStreamRef, dict[str, object]]] = []

        def fake_decode(stream, _output_path, **kwargs):
            decoded.append((stream, kwargs))

        def fake_duration(path, **_kwargs):
            if "source_segment" in path or path.endswith("out.mka"):
                return 59552
            raise AssertionError(f"Unexpected duration probe for {path}")

        with patch.object(performer.timeline, "decode_audio_stream_to_flac", side_effect=fake_decode), \
             patch.object(performer.audio_patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(performer.audio_patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(performer.audio_patcher, "_concat_and_encode"), \
             patch.object(video_utils, "get_video_duration", side_effect=fake_duration):
            result = performer.audio_patcher.execute(request)

        self.assertEqual(decoded, [(
            source_audio,
            {
                "trim_start_ms": 488,
                "trim_end_ms": 60040,
                "normalize_to": (2, 48000, "s16"),
                "logger": performer.logger,
            },
        )])
        self.assertEqual(result.timeline_start_ms, 2868)
        self.assertEqual(result.duration_ms, 59552)
        self.assertEqual(result.transformation, "subsegment-atempo (1 segments)")

    def test_subsegment_patcher_uses_precise_atempo_and_validates_each_segment(self):
        """Long, multi-segment patches retain tempo precision and validate every stage."""
        patcher = self._make_audio_patcher()
        source_audio = AudioStreamRef("/tmp/source.mkv", 2, 2, "pol")
        target_mid, target_end = 3479413, 6958827
        source_mid, source_end = 3336900, 6673800
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=source_audio,
            base_audio=None,
            mapping=((0, 0), (target_mid, source_mid), (target_end, source_end)),
            target_interval=TimelineInterval(0, target_end),
            output_interval=TimelineInterval(0, target_end),
            source_timeline=VideoToAudioTimeline(0, 0),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            strategy=AudioStrategy.SUBSEGMENT,
            segment_count=2,
        )
        calls: list[tuple[str, list[str]]] = []

        def fake_start_process(tool, args, **_kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_duration(path, **_kwargs):
            if "source_segment_0_scaled" in path:
                return target_mid
            if "source_segment_1_scaled" in path:
                return target_end - target_mid
            if "source_segment_0" in path or "source_segment_1" in path:
                return source_mid
            if path.endswith("out.mka"):
                return target_end
            raise AssertionError(f"Unexpected duration probe for {path}")

        with patch.object(patcher.timeline, "decode_audio_stream_to_flac"), \
             patch.object(patcher, "_get_audio_params", return_value=(2, 48000, "s16")), \
             patch.object(patcher, "_get_audio_channel_layout", return_value="stereo"), \
             patch.object(patcher, "_concat_and_encode"), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error"), \
             patch.object(video_utils, "get_video_duration", side_effect=fake_duration), \
             patch.object(patcher, "_validate_audio_duration", wraps=patcher._validate_audio_duration) as validate:
            patcher.execute(request)

        expected_factors = [
            source_mid / target_mid,
            (source_end - source_mid) / (target_end - target_mid),
        ]
        atempo_filters = [
            args[args.index("-filter:a") + 1]
            for tool, args in calls
            if tool == "ffmpeg" and "-filter:a" in args and "atempo=" in args[args.index("-filter:a") + 1]
        ]
        self.assertEqual(atempo_filters, [f"atempo={factor:.12g}" for factor in expected_factors])
        self.assertNotIn(f"atempo={expected_factors[0]:.3f}", atempo_filters)
        labels = [call.args[2] for call in validate.call_args_list]
        self.assertIn("source_segment_0 audio", labels)
        self.assertIn("source_segment_0_scaled audio", labels)
        self.assertIn("source_segment_1 audio", labels)
        self.assertIn("source_segment_1_scaled audio", labels)
        self.assertIn("final patched audio", labels)

    @parameterized.expand([
        ("positive_audio_start", 2420, 6961247, 40, 6673840, 0, 488, 2887),
        ("video_rebased_to_audio_start", 500, 60500, 0, 60000, 500, 500, 500),
        ("aac_priming", 22, 62313, 3, 62305, 0, 521, 540),
        ("negative_audio_start_clamped", 0, 60000, 0, 60000, 0, 0, 0),
    ])
    def test_source_patch_start_uses_explicit_timeline_transform(
        self,
        _name,
        target_start,
        target_end,
        source_start,
        source_end,
        mapping_to_video,
        audio_content_start,
        expected_start,
    ):
        patcher = self._make_audio_patcher()
        request = AudioPatchRequest(
            working_dir="/tmp/work",
            output_path="/tmp/out.mka",
            source_audio=AudioStreamRef("/tmp/source.mkv", 1, 1, "pol"),
            base_audio=None,
            mapping=((target_start, source_start), (target_end, source_end)),
            target_interval=TimelineInterval(target_start, target_end),
            output_interval=TimelineInterval(0, target_end),
            source_timeline=VideoToAudioTimeline(mapping_to_video, audio_content_start),
            base_timeline=None,
            fill_gaps_from_base=False,
            lhs_frames={},
            rhs_frames={},
            strategy=AudioStrategy.SUBSEGMENT,
        )
        first_segment = patcher._build_audio_subsegments(request.mapping, 1)[0]
        source_start_on_stream = source_start + mapping_to_video
        source_window = AudioSourceWindow(
            TimelineInterval(source_start_on_stream, source_end + mapping_to_video),
            TimelineInterval(max(source_start_on_stream, audio_content_start), source_end + mapping_to_video),
        )
        first_part = _AudioPart("/tmp/source.flac", source_window.available.duration_ms, source_window)

        self.assertEqual(patcher._source_patch_start_ms(first_segment, first_part), expected_start)

    def test_audio_timeline_value_objects_preserve_explicit_origins(self):
        interval = TimelineInterval(500, 1000)
        timeline = VideoToAudioTimeline(mapping_to_video_ms=500, audio_content_start_ms=488)

        self.assertEqual(interval.duration_ms, 500)
        self.assertEqual(timeline.video_to_stream_ms(0), 500)
        self.assertEqual(timeline.video_to_rebased_audio_ms(0), 12)
        self.assertEqual(timeline.rebased_audio_to_video_ms(12), 0)
        with self.assertRaises(ValueError):
            TimelineInterval(1000, 500)

    def test_validate_audio_duration_raises_on_excessive_deviation(self):
        """Duration validation has a fixed sample/AAC precision budget, not a percentage."""
        patcher = self._make_audio_patcher()
        with self.assertRaises(RuntimeError) as ctx:
            patcher._validate_audio_duration(10004, 10000, "test audio", sample_rate=48000)
        self.assertIn("difference: 4 ms", str(ctx.exception))
        self.assertIn("max allowed: 3 ms", str(ctx.exception))
        self.assertIn("test audio", str(ctx.exception))

    def test_validate_audio_duration_passes_within_tolerance(self):
        """The allowed error follows sample rounding plus one AAC frame when encoded."""
        patcher = self._make_audio_patcher()
        patcher._validate_audio_duration(10003, 10000, "flac", sample_rate=48000)
        patcher._validate_audio_duration(10025, 10000, "aac", sample_rate=48000, encoded=True)

    def test_validate_audio_duration_allows_one_atempo_tail_block(self):
        """atempo may hold one bounded processing block when the input ends."""
        patcher = self._make_audio_patcher()
        with self.assertRaises(RuntimeError):
            patcher._validate_audio_duration(10024, 10000, "atempo", sample_rate=44100)

        patcher._validate_audio_duration(
            10024,
            10000,
            "atempo",
            sample_rate=44100,
            filter_tail_samples=patcher._ATEMPO_TAIL_SAMPLES,
        )

        with self.assertRaises(RuntimeError):
            patcher._validate_audio_duration(
                10028,
                10000,
                "atempo",
                sample_rate=44100,
                filter_tail_samples=patcher._ATEMPO_TAIL_SAMPLES,
            )


if __name__ == "__main__":
    unittest.main()
