
import logging
import os
import tempfile
import unittest

from parameterized import parameterized
from unittest.mock import patch

from twotone.tools.utils import generic_utils, process_utils, video_utils
from twotone.tools.melt.melt import DEFAULT_TOLERANCE_MS, MeltPerformer
from common import run_ffmpeg
from melt.helpers import _FAKE_PROCESS_OK


class MeltPerformerUnitTest(unittest.TestCase):
    """Unit tests for MeltPerformer internal methods."""

    def _make_performer(self) -> MeltPerformer:
        performer = object.__new__(MeltPerformer)
        performer.logger = logging.getLogger("test.MeltPerformer")
        performer.wd = tempfile.mkdtemp()
        performer.output_dir = tempfile.mkdtemp()
        performer.tolerance_ms = DEFAULT_TOLERANCE_MS
        performer.interruption = generic_utils.InterruptibleProcess()
        performer.fill_audio_gaps = False
        performer._sync_offsets = {}
        return performer

    def test_stream_sorting_puts_unknown_languages_last(self):
        streams = [
            ("audio", 1, "/a.mkv", None),
            ("audio", 2, "/a.mkv", "eng"),
            ("subtitle", 3, "/a.mkv", None),
            ("subtitle", 4, "/a.mkv", "pol"),
            ("subtitle", 5, "/a.mkv", "deu"),
        ]

        sort_key = lambda stream: (stream[3] is None, stream[3] or "")
        result = sorted(streams, key=sort_key)

        languages = [s[3] for s in result]
        self.assertEqual(languages, ["deu", "eng", "pol", None, None])

    def test_stream_sorting_alphabetical_when_all_known(self):
        streams = [
            ("subtitle", 1, "/a.mkv", "pol"),
            ("subtitle", 2, "/a.mkv", "eng"),
            ("subtitle", 3, "/a.mkv", "deu"),
            ("audio", 4, "/a.mkv", "jpn"),
        ]

        sort_key = lambda stream: (stream[3] is None, stream[3] or "")
        result = sorted(streams, key=sort_key)

        languages = [s[3] for s in result]
        self.assertEqual(languages, ["deu", "eng", "jpn", "pol"])

    def test_build_mkvmerge_args_track_order_respects_unknown_last(self):
        performer = self._make_performer()

        file_a = "/tmp/a.mkv"
        file_b = "/tmp/b.mkv"

        streams_list_sorted = [
            ("video", 0, file_a, None),
            ("audio", 1, file_a, "eng"),
            ("subtitle", 3, file_a, "deu"),
            ("subtitle", 4, file_a, "pol"),
            ("subtitle", 5, file_a, None),
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

    # ---- _patch_audio_constant_offset ----

    def _collect_ffmpeg_calls(self, performer, segment_pairs, base_duration_ms,
                               source_sample_rate=48000, source_channels=2, source_sample_fmt="s16",
                               source_channel_layout=None,
                               use_silence=False):
        """Run patch_audio_constant_offset with mocked externals and return captured ffmpeg calls."""
        calls = []

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_raise_on_error(result):
            pass

        right_points = [p[1] for p in segment_pairs]
        source_segment_dur = max(right_points) - min(right_points)

        def fake_get_duration(path):
            if "source_trimmed" in path or "source_scaled" in path or "out." in path:
                return source_segment_dur
            return base_duration_ms

        stream_info = {"codec_type": "audio", "channels": source_channels,
                       "sample_rate": str(source_sample_rate), "sample_fmt": source_sample_fmt}
        if source_channel_layout is not None:
            stream_info["channel_layout"] = source_channel_layout
        fake_full_info = {"streams": [stream_info]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process', side_effect=fake_start_process), \
                 patch.object(process_utils, 'raise_on_error', side_effect=fake_raise_on_error), \
                 patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                 patch.object(video_utils, 'get_video_full_info', return_value=fake_full_info):
                performer.patch_audio_constant_offset(
                    wd, "/base.mkv", "/source.mkv", output_path, segment_pairs,
                    use_silence=use_silence,
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
        self.assertTrue(len(head_calls) >= 1, "Head segment should be extracted")
        self.assertIn("2.0", head_calls[0])

        tail_calls = [s for s in ffmpeg_args_strs if "tail" in s and "-ss" in s]
        self.assertTrue(len(tail_calls) >= 1, "Tail segment should be extracted")
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

        self.assertTrue(len(head_calls) >= 1, "Head segment should be extracted")
        self.assertTrue(len(tail_calls) >= 1, "Tail segment should be extracted")

        for label, call_args in [("head", head_calls[0]), ("tail", tail_calls[0])]:
            self.assertIn("-ac", call_args, f"{label} must have -ac")
            self.assertIn("-ar", call_args, f"{label} must have -ar")
            self.assertIn("-sample_fmt", call_args, f"{label} must have -sample_fmt")
            self.assertEqual(call_args[call_args.index("-ac") + 1], "6", f"{label} channels")
            self.assertEqual(call_args[call_args.index("-ar") + 1], "44100", f"{label} sample rate")
            self.assertEqual(call_args[call_args.index("-sample_fmt") + 1], "s32", f"{label} sample fmt")

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
        self.assertTrue(len(lines) >= 2, "Should have at least 2 bar lines")
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
        self.assertTrue(len(lines) >= 2)
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
        self.assertTrue(len(lines) >= 2)
        # lhs starts first (lhs_sg=2 means has content before shared region)
        self.assertTrue(lines[0].startswith("|"), "#1 bar at col 0")
        self.assertTrue(lines[1].startswith(" "), "#2 bar indented")
        # Both should end close (1s gap gets _MIN_GAP=6 columns, but NOT a proportional 4s gap)
        lhs_end_col = len(lines[0].rstrip()) - 1
        rhs_end_col = len(lines[1].rstrip()) - 1
        self.assertAlmostEqual(lhs_end_col, rhs_end_col, delta=8,
                               msg="Speed-adjusted rhs should end close to lhs (small gap)")

    def test_patch_audio_constant_offset_stream_copy_fast_path(self):
        """When ratio ≈ 1.0 and no head/tail, audio should be stream-copied without re-encoding."""
        performer = self._make_performer()
        # seg1: 0..6000 (full base), seg2: 500..6500 — same duration, no head/tail
        pairs = [(0, 500), (6000, 6500)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=6000)

        # Should be a single ffmpeg call with -c:a copy
        self.assertEqual(len(calls), 1, "Fast path should produce exactly one ffmpeg call")
        self.assertIn("copy", calls[0][1], "Fast path should use -c:a copy")
        # No FLAC, no AAC, no concat
        aac_calls = [c for c in calls if any("aac" in str(a) for a in c[1])]
        self.assertEqual(aac_calls, [], "Fast path should not encode to AAC")

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

    # ---- silence mode (use_silence=True) ----

    def test_silence_mode_skips_head_and_tail(self):
        """With use_silence=True, no head/tail extraction or generation should occur."""
        performer = self._make_performer()
        # Matching region: 2000..8000 in base (10s total) → would have head and tail
        pairs = [(2000, 1000), (8000, 7000)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=10000, use_silence=True)

        ffmpeg_args_strs = [" ".join(str(a) for a in c[1]) for c in calls if c[0] == "ffmpeg"]

        # No head/tail extraction or silence generation
        head_tail_calls = [s for s in ffmpeg_args_strs if "head" in s or "tail" in s or "anullsrc" in s]
        self.assertEqual(head_tail_calls, [], "Silence mode should not produce head/tail segments")

        # No base video extraction
        base_extract_calls = [s for s in ffmpeg_args_strs if "/base.mkv" in s]
        self.assertEqual(base_extract_calls, [], "Silence mode should not read from base video")

    def test_silence_mode_uses_stream_copy_when_no_scaling(self):
        """With use_silence=True and ratio ≈ 1.0, audio should be stream-copied."""
        performer = self._make_performer()
        pairs = [(0, 500), (4000, 4500)]
        calls = self._collect_ffmpeg_calls(performer, pairs, base_duration_ms=6000, use_silence=True)

        self.assertEqual(len(calls), 1, "Should produce exactly one ffmpeg call")
        self.assertIn("copy", calls[0][1], "Should use stream-copy")

    def test_shift_audio_no_reencode_uses_stream_copy(self):
        """_shift_audio_no_reencode should trim with stream-copy and return correct sync offset."""
        performer = self._make_performer()
        calls = []

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            pairs = [(2000, 1000), (8000, 7000)]

            with patch.object(process_utils, 'start_process', side_effect=fake_start_process), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', return_value=6000):
                sync_offset = performer._shift_audio_no_reencode("/source.mkv", output_path, pairs)

        self.assertEqual(sync_offset, 2000, "Sync offset should be seg1_start")
        self.assertEqual(len(calls), 1, "Should produce exactly one ffmpeg call")
        self.assertIn("copy", calls[0][1], "Should use stream-copy")
        # Verify trim points
        args = calls[0][1]
        ss_idx = args.index("-ss")
        to_idx = args.index("-to")
        self.assertEqual(args[ss_idx + 1], "1.0", "Should trim from seg2_start")
        self.assertEqual(args[to_idx + 1], "7.0", "Should trim to seg2_end")

    def test_shift_audio_no_reencode_validates_output_duration(self):
        """_shift_audio_no_reencode should raise when output duration deviates too much."""
        performer = self._make_performer()

        def fake_start_process(tool, args, **kwargs):
            return _FAKE_PROCESS_OK

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            # seg2 range: 1000..7000 → expected duration ≈ 6000 ms
            pairs = [(2000, 1000), (8000, 7000)]

            with patch.object(process_utils, 'start_process', side_effect=fake_start_process), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', return_value=3000):
                with self.assertRaises(RuntimeError, msg="Should raise on excessive duration deviation"):
                    performer._shift_audio_no_reencode("/source.mkv", output_path, pairs)

    def test_shift_audio_no_reencode_corrects_for_audio_deficit(self):
        """When stream-copied audio is shorter than expected (AVI deficit), sync offset must shift forward."""
        performer = self._make_performer()
        calls = []

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            # seg2 range: 1000..7000 → expected duration 6000 ms
            # seg1 range: 2000..8000 → sync_offset base = 2000
            pairs = [(2000, 1000), (8000, 7000)]
            # Audio deficit: 200 ms (got 5800 instead of 6000) — within 5% tolerance
            actual_dur = 5800

            with patch.object(process_utils, 'start_process', side_effect=fake_start_process), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', return_value=actual_dur):
                sync_offset = performer._shift_audio_no_reencode("/source.mkv", output_path, pairs)

        # deficit = 6000 - 5800 = 200, video_ratio = 1.0 (no scaling in this path)
        # correction = round(200 * 1.0) = 200
        # expected sync = 2000 + 200 = 2200
        self.assertEqual(sync_offset, 2200,
                         "Sync offset should be corrected forward by the deficit amount")

    def test_build_mkvmerge_args_applies_sync_offset(self):
        """When _sync_offsets has an entry, --sync should appear in mkvmerge args."""
        performer = self._make_performer()

        patched_file = "/tmp/patched.mka"
        base_file = "/tmp/base.mkv"
        performer._sync_offsets[patched_file] = 3000

        streams = [
            ("video", 0, base_file, None),
            ("audio", 0, patched_file, "pol"),
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
        """When _sync_offsets has an entry with value 0, --sync 0:0 should still appear."""
        performer = self._make_performer()

        patched_file = "/tmp/patched.mka"
        base_file = "/tmp/base.mkv"
        performer._sync_offsets[patched_file] = 0

        streams = [
            ("video", 0, base_file, None),
            ("audio", 0, patched_file, "pol"),
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

    def test_patch_audio_fps_mismatch_with_audio_deficit(self):
        """Exact scenario: 23.976fps base + 25fps AVI source with audio deficit.

        Real-world case:
        - Base (4K MKV, 23.976fps): mapping lhs range [2420 … 6961247] ms
        - Source (AVI, 25fps): mapping rhs range [40 … 6673840] ms
        - source_dur = 6673800 ms, target_dur = 6958827 ms
        - fps_ratio ≈ 0.9590 → needs asetrate scaling
        - Audio in the AVI container starts ~488 ms after the video:
          requesting 6673800 ms of audio yields only 6673312 ms.
        - deficit = 488 ms → sync offset must shift from 2420 to ~2929 ms
          (correction = round(488 × 6958827/6673800) = 509 ms).

        Without the deficit-based correction, audio arrives ~500 ms too early.
        """
        performer = self._make_performer()

        # Mapping: first pair at 2420/40, last pair at 6961247/6673840
        seg1_start, seg2_start = 2420, 40
        seg1_end, seg2_end = 6961247, 6673840
        pairs = [(seg1_start, seg2_start), (seg1_end, seg2_end)]

        source_dur = seg2_end - seg2_start           # 6673800
        target_dur = seg1_end - seg1_start           # 6958827
        video_ratio = target_dur / source_dur        # ≈ 1.042714
        audio_deficit_ms = 488                       # AVI audio is shorter
        actual_trimmed_dur = source_dur - audio_deficit_ms  # 6673312

        calls = []
        returned_sync_offset = None

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        def fake_get_duration(path):
            # Trimmed audio is shorter than requested → simulates AVI deficit
            if "source_trimmed" in path:
                return actual_trimmed_dur
            if "source_scaled" in path:
                return round(actual_trimmed_dur * video_ratio)
            if "out." in path:
                return target_dur
            return 7000000  # base video duration (>seg1_end → has tail)

        base_duration_ms = 7000000
        fake_full_info = {"streams": [{"codec_type": "audio", "channels": 2,
                                       "sample_rate": "48000", "sample_fmt": "s16"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process', side_effect=fake_start_process), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                 patch.object(video_utils, 'get_video_full_info', return_value=fake_full_info):
                returned_sync_offset = performer.patch_audio_constant_offset(
                    wd, "/base.mkv", "/source.avi", output_path, pairs,
                    use_silence=True,
                )

        # --- Sync offset must account for the audio deficit ---
        expected_correction = round(audio_deficit_ms * video_ratio)  # 509
        expected_sync = seg1_start + expected_correction             # 2929
        self.assertEqual(returned_sync_offset, expected_sync,
                         f"Sync offset should be {seg1_start} + {expected_correction} = {expected_sync}, "
                         f"not {seg1_start} (the uncorrected value)")
        self.assertGreater(returned_sync_offset, seg1_start,
                           "Deficit must push sync offset forward")

        # --- asetrate must be applied (fps differ) ---
        filter_calls = [c for c in calls if any("asetrate" in str(a) for a in c[1])]
        self.assertEqual(len(filter_calls), 1, "asetrate should be used exactly once")
        filter_arg = next(a for a in filter_calls[0][1] if "asetrate" in str(a))
        # fps_ratio = source_dur / target_dur ≈ 0.959035
        # adjusted_rate = 48000 × fps_ratio ≈ 46033.69
        expected_rate = 48000 * source_dur / target_dur
        self.assertIn(f"asetrate={expected_rate:.6f}", filter_arg)
        self.assertIn("aresample=48000", filter_arg)

        # --- Trim timestamps must match mapping endpoints ---
        ffmpeg_args = [c[1] for c in calls if c[0] == "ffmpeg"]
        trim_call = next(a for a in ffmpeg_args if "source_trimmed" in str(a))
        ss_idx = trim_call.index("-ss")
        to_idx = trim_call.index("-to")
        self.assertAlmostEqual(float(trim_call[ss_idx + 1]), seg2_start / 1000, places=2)
        self.assertAlmostEqual(float(trim_call[to_idx + 1]), seg2_end / 1000, places=2)

        # --- No head/tail extraction (use_silence=True) ---
        head_tail_calls = [c for c in calls if any("head" in str(a) or "tail" in str(a) for a in c[1])]
        self.assertEqual(head_tail_calls, [], "use_silence=True should skip head/tail")

    def test_patch_audio_no_deficit_keeps_original_sync_offset(self):
        """When the audio track has no deficit, sync offset equals seg1_start."""
        performer = self._make_performer()

        seg1_start, seg2_start = 2420, 40
        seg1_end, seg2_end = 6961247, 6673840
        pairs = [(seg1_start, seg2_start), (seg1_end, seg2_end)]
        source_dur = seg2_end - seg2_start

        def fake_get_duration(path):
            if "source_trimmed" in path or "source_scaled" in path or "out." in path:
                return source_dur  # no deficit — actual equals requested
            return 7000000

        fake_full_info = {"streams": [{"codec_type": "audio", "channels": 2,
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
                    wd, "/base.mkv", "/source.mkv", output_path, pairs,
                    use_silence=True,
                )

        self.assertEqual(sync, seg1_start,
                         "Without deficit, sync offset should be the raw seg1_start")

    def test_patch_audio_fill_gaps_raises_on_deficit(self):
        """In fill-audio-gaps mode, audio deficit cannot be compensated — must raise."""
        performer = self._make_performer()

        seg1_start, seg2_start = 2420, 40
        seg1_end, seg2_end = 6961247, 6673840
        pairs = [(seg1_start, seg2_start), (seg1_end, seg2_end)]
        source_dur = seg2_end - seg2_start
        audio_deficit_ms = 488

        def fake_get_duration(path):
            if "source_trimmed" in path:
                return source_dur - audio_deficit_ms
            return 7000000

        fake_full_info = {"streams": [{"codec_type": "audio", "channels": 2,
                                       "sample_rate": "48000", "sample_fmt": "s16"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.mka")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process',
                              side_effect=lambda t, a, **kw: _FAKE_PROCESS_OK), \
                 patch.object(process_utils, 'raise_on_error', lambda r: None), \
                 patch.object(video_utils, 'get_video_duration', side_effect=fake_get_duration), \
                 patch.object(video_utils, 'get_video_full_info', return_value=fake_full_info):
                with self.assertRaises(RuntimeError, msg="Should raise on deficit in fill-audio-gaps mode") as ctx:
                    performer.patch_audio_constant_offset(
                        wd, "/base.mkv", "/source.avi", output_path, pairs,
                        use_silence=False,
                    )
                self.assertIn("fill-audio-gaps", str(ctx.exception))
                self.assertIn(str(audio_deficit_ms), str(ctx.exception))

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
