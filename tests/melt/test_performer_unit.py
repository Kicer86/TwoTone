import logging
import os
import tempfile
import unittest

from unittest.mock import patch

from twotone.tools.utils import generic_utils, process_utils, video_utils
from twotone.tools.melt.melt import DEFAULT_TOLERANCE_MS, MeltPerformer


class MeltPerformerUnitTest(unittest.TestCase):
    """Unit tests for MeltPerformer internal methods."""

    def _make_performer(self) -> MeltPerformer:
        performer = object.__new__(MeltPerformer)
        performer.logger = logging.getLogger("test.MeltPerformer")
        performer.wd = tempfile.mkdtemp()
        performer.output_dir = tempfile.mkdtemp()
        performer.tolerance_ms = DEFAULT_TOLERANCE_MS
        performer.interruption = generic_utils.InterruptibleProcess()
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

    def _collect_ffmpeg_calls(self, performer, segment_pairs, base_duration_ms, source_sample_rate=48000):
        """Run _patch_audio_constant_offset with mocked externals and return captured ffmpeg calls."""
        calls = []

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args)))
            result = type('ProcessResult', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
            return result

        def fake_raise_on_error(result):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.m4a")
            wd = os.path.join(tmpdir, "work")

            with patch.object(process_utils, 'start_process', side_effect=fake_start_process), \
                 patch.object(process_utils, 'raise_on_error', side_effect=fake_raise_on_error), \
                 patch.object(video_utils, 'get_video_duration', return_value=base_duration_ms), \
                 patch.object(video_utils, 'get_video_data', return_value={"audio": [{"sample_rate": source_sample_rate}]}):
                performer.patch_audio_constant_offset(
                    wd, "/base.mkv", "/source.mkv", output_path, segment_pairs,
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


if __name__ == '__main__':
    unittest.main()
