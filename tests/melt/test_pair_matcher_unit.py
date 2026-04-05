
import logging
import unittest

from unittest.mock import patch

from twotone.tools.utils import generic_utils, video_utils
from twotone.tools.melt.melt import PairMatcher


class PairMatcherUnitTest(unittest.TestCase):
    """Unit tests for PairMatcher internal methods.

    These tests mock _is_rich and bypass the full constructor to test
    algorithmic behaviour without needing video files. They target:
    - _extrapolate_through_low_entropy: 5% noise tolerance
    - snap_to_edges: snap threshold
    - find_boundary (via _look_for_boundaries): look_ahead robustness
    """

    def _make_pair_matcher(self, lhs_fps: float = 25.0, rhs_fps: float = 25.0) -> PairMatcher:
        """Create a PairMatcher with minimal init — no video files needed."""
        pm = object.__new__(PairMatcher)
        pm.logger = logging.getLogger("test.PairMatcher")
        pm.lhs_fps = lhs_fps
        pm.rhs_fps = rhs_fps
        pm.lhs_path = "/fake/lhs.mp4"
        pm.rhs_path = "/fake/rhs.mp4"
        pm.lhs_label = "#1"
        pm.rhs_label = "#2"
        pm.interruption = generic_utils.InterruptibleProcess()
        return pm

    @staticmethod
    def _make_frames(timestamps: list[int], prefix: str = "fake") -> dict[int, dict]:
        """Build a synthetic FramesInfo dict with dummy paths."""
        return {ts: {"path": f"/{prefix}/{ts}.png", "frame_id": i} for i, ts in enumerate(timestamps)}

    # ---- _extrapolate_through_low_entropy ----

    def test_extrapolate_pure_low_entropy_lhs_extends(self):
        """When all frames between boundary and edge are low-entropy, extrapolation happens."""
        pm = self._make_pair_matcher()

        # 100 frames at 40ms intervals from 0..3960ms
        lhs_keys = list(range(0, 4000, 40))
        rhs_keys = list(range(0, 4000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        boundary = (2000, 2000)
        reference = (3000, 3000)

        with patch.object(PairMatcher, '_is_rich', return_value=False):
            result = pm.extrapolate_through_low_entropy(
                lhs, rhs, boundary, reference, ratio=1.0,
                direction=-1, entered_low_entropy=True,
            )

        # Should extrapolate to LHS edge (0ms) and snap RHS to nearest frame
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)

    def test_extrapolate_not_entered_low_entropy_noop(self):
        """When entered_low_entropy=False, boundary is returned unchanged."""
        pm = self._make_pair_matcher()
        lhs = self._make_frames([0, 1000, 2000])
        rhs = self._make_frames([0, 1000, 2000])
        boundary = (1000, 1000)

        result = pm.extrapolate_through_low_entropy(
            lhs, rhs, boundary, (2000, 2000), ratio=1.0,
            direction=-1, entered_low_entropy=False,
        )

        self.assertEqual(result, boundary)

    def test_extrapolate_all_high_entropy_gap_refuses(self):
        """When all gap frames are high-entropy, extrapolation is refused."""
        pm = self._make_pair_matcher()
        lhs = self._make_frames(list(range(0, 4000, 40)))
        rhs = self._make_frames(list(range(0, 4000, 40)))
        boundary = (2000, 2000)

        with patch.object(PairMatcher, '_is_rich', return_value=True):
            result = pm.extrapolate_through_low_entropy(
                lhs, rhs, boundary, (3000, 3000), ratio=1.0,
                direction=-1, entered_low_entropy=True,
            )

        self.assertEqual(result, boundary)

    def test_extrapolate_noise_below_5pct_accepts(self):
        """When < 5% of gap frames are high-entropy, extrapolation proceeds."""
        pm = self._make_pair_matcher()

        # 200 frames from 0..7960ms at 40ms intervals
        lhs_keys = list(range(0, 8000, 40))
        rhs_keys = list(range(0, 8000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        boundary = (4000, 4000)
        # Gap: frames 0..3960 (100 frames), make 3 of them (3%) "rich"
        noisy_timestamps = {120, 2000, 3600}  # 3 out of 100 = 3%

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts in noisy_timestamps

        with patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich):
            result = pm.extrapolate_through_low_entropy(
                lhs, rhs, boundary, (6000, 6000), ratio=1.0,
                direction=-1, entered_low_entropy=True,
            )

        # Should extrapolate — 3% noise is below 5% threshold
        self.assertEqual(result[0], 0)

    def test_extrapolate_noise_above_5pct_refuses(self):
        """When > 5% of gap frames are high-entropy, extrapolation is refused."""
        pm = self._make_pair_matcher()

        lhs_keys = list(range(0, 8000, 40))
        rhs_keys = list(range(0, 8000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        boundary = (4000, 4000)
        # Gap: frames 0..3960 (100 frames), make 10 of them (10%) "rich"
        noisy_timestamps = set(range(0, 400, 40))  # 10 out of 100 = 10%

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts in noisy_timestamps

        with patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich):
            result = pm.extrapolate_through_low_entropy(
                lhs, rhs, boundary, (6000, 6000), ratio=1.0,
                direction=-1, entered_low_entropy=True,
            )

        # Should refuse — 10% noise exceeds 5% threshold
        self.assertEqual(result, boundary)

    def test_extrapolate_end_direction(self):
        """Extrapolation works in the forward (end) direction too."""
        pm = self._make_pair_matcher()

        lhs_keys = list(range(0, 8000, 40))
        rhs_keys = list(range(0, 8000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        boundary = (4000, 4000)
        reference = (2000, 2000)

        with patch.object(PairMatcher, '_is_rich', return_value=False):
            result = pm.extrapolate_through_low_entropy(
                lhs, rhs, boundary, reference, ratio=1.0,
                direction=1, entered_low_entropy=True,
            )

        # Should extrapolate to LHS end edge
        self.assertEqual(result[0], lhs_keys[-1])

    def test_extrapolate_rhs_high_entropy_refuses(self):
        """When LHS is low-entropy but RHS gap is high-entropy, extrapolation is refused."""
        pm = self._make_pair_matcher()

        lhs_keys = list(range(0, 8000, 40))
        rhs_keys = list(range(0, 8000, 40))
        lhs = self._make_frames(lhs_keys, prefix="lhs")
        rhs = self._make_frames(rhs_keys, prefix="rhs")

        boundary = (4000, 4000)

        def smart_mock(path: str) -> bool:
            if path.startswith("/lhs/"):
                return False   # LHS gap: all low-entropy
            if path.startswith("/rhs/"):
                return True    # RHS gap: all high-entropy
            return False

        with patch.object(PairMatcher, '_is_rich', side_effect=smart_mock):
            result = pm.extrapolate_through_low_entropy(
                lhs, rhs, boundary, (6000, 6000), ratio=1.0,
                direction=-1, entered_low_entropy=True,
            )

        # RHS is all high-entropy → should refuse
        self.assertEqual(result, boundary)

    # ---- snap_to_edges ----

    def test_snap_within_4_frames_snaps(self):
        """Pairs within 4 frames of edge should snap to video boundary."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # threshold = 4 * 1000 / 25 = 160ms

        lhs_frames = self._make_frames([0, 40, 80, 100, 5000, 9900, 9960, 10000])
        rhs_frames = self._make_frames([0, 40, 80, 100, 5000, 9900, 9960, 10000])

        matching_pairs = [(100, 100), (5000, 5000), (9900, 9900)]

        with patch.object(video_utils, 'get_video_duration', return_value=10000):
            result = pm.snap_to_edges(matching_pairs, lhs_frames, rhs_frames, snap_frames=4)

        # 100ms is within 160ms threshold → snap to 0
        self.assertEqual(result[0], (0, 0))
        # 10000 - 9900 = 100ms is within 160ms threshold → snap to duration
        self.assertEqual(result[-1], (10000, 10000))

    def test_snap_beyond_4_frames_no_snap(self):
        """Pairs beyond 4 frames of edge should NOT snap."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # threshold = 160ms

        lhs_frames = self._make_frames([0, 200, 5000, 9700, 10000])
        rhs_frames = self._make_frames([0, 200, 5000, 9700, 10000])

        matching_pairs = [(200, 200), (5000, 5000), (9700, 9700)]

        with patch.object(video_utils, 'get_video_duration', return_value=10000):
            result = pm.snap_to_edges(matching_pairs, lhs_frames, rhs_frames, snap_frames=4)

        # 200ms > 160ms → no snap
        self.assertEqual(result[0], (200, 200))
        # 10000 - 9700 = 300ms > 160ms → no snap
        self.assertEqual(result[-1], (9700, 9700))

    def test_snap_asymmetric_fps(self):
        """With different FPS per side, thresholds apply independently."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=50.0)
        # LHS threshold = 4 * 1000 / 25 = 160ms
        # RHS threshold = 4 * 1000 / 50 = 80ms

        lhs_frames = self._make_frames([0, 100, 5000, 10000])
        rhs_frames = self._make_frames([0, 90, 5000, 10000])

        matching_pairs = [(100, 90), (5000, 5000)]

        with patch.object(video_utils, 'get_video_duration', return_value=10000):
            result = pm.snap_to_edges(matching_pairs, lhs_frames, rhs_frames, snap_frames=4)

        # LHS: 100ms <= 160ms → snaps
        self.assertEqual(result[0][0], 0)
        # RHS: 90ms > 80ms → does NOT snap
        self.assertEqual(result[0][1], 90)

    # ---- try_constant_offset_extrapolation ----

    def test_constant_offset_detected_and_extrapolated(self):
        """When all pairs share a constant frame-number offset, boundaries are extrapolated."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # frame_ms = 40ms, frame offset = 3
        # Offset: all pairs at +3 frames (+120ms)
        matching_pairs = [
            (2120, 2000),
            (4120, 4000),
            (6120, 6000),
            (8120, 8000),
        ]

        lhs_keys = list(range(0, 10040, 40))
        rhs_keys = list(range(0, 10040, 40))
        lhs_frames = self._make_frames(lhs_keys, prefix="lhs")
        rhs_frames = self._make_frames(rhs_keys, prefix="rhs")

        result = pm.try_constant_offset_extrapolation(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(result)
        # k=3: first_lhs_frame=3 → ts=120, first_rhs_frame=0 → ts=0
        self.assertEqual(result[0], (120, 0))
        # last_lhs_frame=min(250,250+3)=250 → ts=10000, last_rhs_frame=247 → ts=9880
        self.assertEqual(result[-1], (10000, 9880))

    def test_constant_offset_negative(self):
        """When offset is negative (rhs ahead of lhs), boundaries are correct."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        matching_pairs = [
            (2000, 2120),
            (4000, 4120),
            (6000, 6120),
        ]

        lhs_keys = list(range(0, 10040, 40))
        rhs_keys = list(range(0, 10040, 40))
        lhs_frames = self._make_frames(lhs_keys, prefix="lhs")
        rhs_frames = self._make_frames(rhs_keys, prefix="rhs")

        result = pm.try_constant_offset_extrapolation(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(result)
        # k=-3: first_lhs_frame=0 → ts=0, first_rhs_frame=3 → ts=120
        self.assertEqual(result[0], (0, 120))
        # last_lhs_frame=min(250,250-3)=247 → ts=9880, last_rhs_frame=250 → ts=10000
        self.assertEqual(result[-1], (9880, 10000))

    def test_constant_offset_zero(self):
        """When offset is zero (identical timing), boundaries are at edges."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        matching_pairs = [
            (2000, 2000),
            (4000, 4000),
            (6000, 6000),
            (8000, 8000),
        ]

        lhs_keys = list(range(0, 10040, 40))
        rhs_keys = list(range(0, 10040, 40))
        lhs_frames = self._make_frames(lhs_keys, prefix="lhs")
        rhs_frames = self._make_frames(rhs_keys, prefix="rhs")

        result = pm.try_constant_offset_extrapolation(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(result)
        self.assertEqual(result[0], (0, 0))
        self.assertEqual(result[-1], (10000, 10000))

    def test_constant_offset_rejected_high_std(self):
        """When frame-number offsets vary too much (std > 1), returns None."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # frame offsets: 3, 5, 1 → std ≈ 1.63 > 1.0
        matching_pairs = [
            (2120, 2000),
            (4200, 4000),
            (6040, 6000),
        ]

        lhs_keys = list(range(0, 10040, 40))
        rhs_keys = list(range(0, 10040, 40))
        lhs_frames = self._make_frames(lhs_keys, prefix="lhs")
        rhs_frames = self._make_frames(rhs_keys, prefix="rhs")

        result = pm.try_constant_offset_extrapolation(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNone(result)

    def test_constant_offset_rejected_growing_offsets(self):
        """When frame-number offsets grow (content plays at different rate), returns None."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # frame offsets: 2, 4, 6, 8 → std ≈ 2.24 > 1.0
        matching_pairs = [
            (2080, 2000),
            (4160, 4000),
            (6240, 6000),
            (8320, 8000),
        ]

        lhs_keys = list(range(0, 10040, 40))
        rhs_keys = list(range(0, 10040, 40))
        lhs_frames = self._make_frames(lhs_keys, prefix="lhs")
        rhs_frames = self._make_frames(rhs_keys, prefix="rhs")

        result = pm.try_constant_offset_extrapolation(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNone(result)

    def test_constant_offset_too_few_pairs(self):
        """With fewer than 3 pairs, constant-offset detection is skipped."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        matching_pairs = [(2120, 2000), (4120, 4000)]

        lhs_keys = list(range(0, 10040, 40))
        rhs_keys = list(range(0, 10040, 40))
        lhs_frames = self._make_frames(lhs_keys, prefix="lhs")
        rhs_frames = self._make_frames(rhs_keys, prefix="rhs")

        result = pm.try_constant_offset_extrapolation(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNone(result)

    def test_constant_offset_slight_jitter_within_tolerance(self):
        """Small jitter (< 1 frame) in frame-number offsets should still be accepted."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # frame offsets: 3, 3, 4, 3 → std ≈ 0.43 < 1.0
        matching_pairs = [
            (2120, 2000),
            (4120, 4000),
            (6160, 6000),
            (8120, 8000),
        ]

        lhs_keys = list(range(0, 10040, 40))
        rhs_keys = list(range(0, 10040, 40))
        lhs_frames = self._make_frames(lhs_keys, prefix="lhs")
        rhs_frames = self._make_frames(rhs_keys, prefix="rhs")

        result = pm.try_constant_offset_extrapolation(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(result)
        # Median frame offset is 3 → first=(120, 0), last=(10000, 9880)
        self.assertEqual(result[0], (120, 0))
        self.assertEqual(result[-1], (10000, 9880))

    def test_constant_offset_different_fps(self):
        """Same content with different FPS (e.g. PAL speedup) is detected correctly."""
        pm = self._make_pair_matcher(lhs_fps=20.0, rhs_fps=25.0)
        # lhs_frame_ms=50ms, rhs_frame_ms=40ms
        # Frame offset k=3: frame N in LHS = frame (N-3) in RHS
        matching_pairs = [
            (2500, 1880),   # LHS frame 50, RHS frame 47
            (5000, 3880),   # LHS frame 100, RHS frame 97
            (7500, 5880),   # LHS frame 150, RHS frame 147
        ]

        lhs_keys = list(range(0, 10050, 50))
        rhs_keys = list(range(0, 8040, 40))
        lhs_frames = self._make_frames(lhs_keys, prefix="lhs")
        rhs_frames = self._make_frames(rhs_keys, prefix="rhs")

        result = pm.try_constant_offset_extrapolation(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(result)
        # k=3: first_lhs_frame=3 → ts=150, first_rhs_frame=0 → ts=0
        self.assertEqual(result[0], (150, 0))
        # last_lhs_frame=min(200,200+3)=200 → ts=10000, last_rhs_frame=197 → ts=7880
        self.assertEqual(result[-1], (10000, 7880))

    def test_constant_offset_does_not_duplicate_existing_boundary(self):
        """If extrapolated boundary matches an existing pair, no duplicate is added."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        matching_pairs = [
            (120, 0),
            (4120, 4000),
            (8120, 8000),
        ]

        lhs_keys = list(range(0, 10040, 40))
        rhs_keys = list(range(0, 10040, 40))
        lhs_frames = self._make_frames(lhs_keys, prefix="lhs")
        rhs_frames = self._make_frames(rhs_keys, prefix="rhs")

        result = pm.try_constant_offset_extrapolation(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(result)
        # First pair already matches extrapolated boundary → no duplicate
        self.assertEqual(result[0], (120, 0))
        self.assertEqual(len([p for p in result if p == (120, 0)]), 1)

    def test_create_segments_mapping_skips_edge_snap_after_constant_offset(self):
        """Constant-offset extrapolation should bypass timestamp-based edge snapping."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        pm.lhs_all_wd = "/tmp/lhs_all"
        pm.rhs_all_wd = "/tmp/rhs_all"
        pm.lhs_boundary_wd = "/tmp/lhs_boundary"
        pm.rhs_boundary_wd = "/tmp/rhs_boundary"
        pm.lhs_normalized_wd = "/tmp/lhs_norm"
        pm.rhs_normalized_wd = "/tmp/rhs_norm"
        pm.debug_wd = "/tmp/debug"

        class FakePhash:
            def get(self, path):
                return 0

        pm.phash = FakePhash()

        lhs_probed = self._make_frames([0, 40, 80], prefix="lhs_raw")
        rhs_probed = self._make_frames([0, 40, 80], prefix="rhs_raw")
        extrapolated_pairs = [(120, 0), (10000, 9880)]

        def fake_extract(_video_path, target_dir, _ranges, probed_metadata, **_kwargs):
            for ts, info in probed_metadata.items():
                info["path"] = f"{target_dir}/{ts}.png"

        def fake_normalize(frames_info, wd, desc="Normalizing frames", prefix=""):
            return {
                ts: {"path": f"{wd}/{prefix}{ts}.png", "frame_id": info["frame_id"]}
                for ts, info in frames_info.items()
            }

        with patch.object(video_utils, 'detect_scene_changes', side_effect=[[40], [40]]), \
             patch.object(video_utils, 'probe_frame_timestamps', side_effect=[lhs_probed, rhs_probed]), \
             patch.object(video_utils, 'extract_frames_at_ranges', side_effect=fake_extract), \
             patch('twotone.tools.melt.pair_matcher.DebugRoutines') as debug_cls, \
             patch.object(PairMatcher, '_normalize_frames', side_effect=fake_normalize), \
             patch.object(PairMatcher, '_make_pairs', return_value=[(40, 40)]), \
             patch.object(PairMatcher, 'try_constant_offset_extrapolation', return_value=extrapolated_pairs), \
             patch.object(PairMatcher, 'snap_to_edges', side_effect=AssertionError("snap_to_edges should not be called")):

            debug = debug_cls.return_value
            debug.dump_frames.return_value = None
            debug.dump_matches.return_value = None

            mappings, _, _, is_constant = pm.create_segments_mapping()

        self.assertTrue(is_constant)
        self.assertEqual(mappings, extrapolated_pairs)

    # ---- _look_for_boundaries: look_ahead robustness ----

    @staticmethod
    def _mock_phash_always_match():
        """Return a context manager that replaces PhashCache with an always-matching stub."""
        class FakeHash:
            def __sub__(self, other): return 0
            def __abs__(self): return 0
        class FakePhashCache:
            def __init__(self, hash_size=16): pass
            def get(self, path): return FakeHash()
        return patch('twotone.tools.melt.pair_matcher.PhashCache', FakePhashCache)

    def test_boundary_search_survives_transient_dark_frame(self):
        """A single dark frame at a scene cut should not stop boundary search."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        lhs_keys = list(range(0, 10000, 40))
        rhs_keys = list(range(0, 10000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        dark_timestamps = {3000}

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts not in dark_timestamps

        first = (5000, 5000)
        last = (8000, 8000)

        with (
            patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich),
            patch.object(pm, '_edge_content_matches', return_value=True),
            self._mock_phash_always_match(),
        ):
            result_first, _ = pm.look_for_boundaries(
                lhs, rhs, first, last, cutoff=16, extrapolate=False,
            )

        self.assertLess(result_first[0], 3000,
            "Boundary should extend past the transient dark frame at 3000ms")

    def test_boundary_search_stops_at_sustained_dark_zone(self):
        """A sustained dark zone (>1.5s of dark frames) should stop the search."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        lhs_keys = list(range(0, 10000, 40))
        rhs_keys = list(range(0, 10000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        dark_zone = {ts for ts in lhs_keys if ts <= 2500}

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts not in dark_zone

        first = (5000, 5000)
        last = (8000, 8000)

        with (
            patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich),
            patch.object(pm, '_edge_content_matches', return_value=True),
            self._mock_phash_always_match(),
        ):
            result_first, _ = pm.look_for_boundaries(
                lhs, rhs, first, last, cutoff=16, extrapolate=False,
            )

        self.assertGreaterEqual(result_first[0], 2500,
            "Boundary should not extend into a sustained dark zone")

    def test_boundary_search_consecutive_scene_cuts_dont_stop(self):
        """Two brief dark frames close together should not stop the search."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        lhs_keys = list(range(0, 10000, 40))
        rhs_keys = list(range(0, 10000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        dark_timestamps = {3000, 3200}

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts not in dark_timestamps

        first = (5000, 5000)
        last = (8000, 8000)

        with (
            patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich),
            patch.object(pm, '_edge_content_matches', return_value=True),
            self._mock_phash_always_match(),
        ):
            result_first, _ = pm.look_for_boundaries(
                lhs, rhs, first, last, cutoff=16, extrapolate=False,
            )

        self.assertLess(result_first[0], 3000,
            "Boundary should extend past two isolated dark frames")

    def test_boundary_search_jumps_over_mid_movie_dark_zone(self):
        """A dark zone in the middle of matching content should be jumped over.

        This reproduces the Jumanji scenario: content matches before and after
        a sustained dark zone (scene transition), so the walk should continue
        past the dark zone rather than declaring it a boundary.
        """
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        # 500 frames at 40ms intervals: 0..19960ms
        lhs_keys = list(range(0, 20000, 40))
        rhs_keys = list(range(0, 20000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        # Sustained dark zone from 8000ms..10000ms (2s, ~50 frames)
        dark_zone = {ts for ts in lhs_keys if 8000 <= ts <= 10000}

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts not in dark_zone

        first = (14000, 14000)
        last = (18000, 18000)

        with (
            patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich),
            patch.object(pm, '_edge_content_matches', return_value=True),
            self._mock_phash_always_match(),
        ):
            result_first, _ = pm.look_for_boundaries(
                lhs, rhs, first, last, cutoff=16, extrapolate=False,
            )

        # The walk should have jumped over the dark zone at 8000-10000ms
        # and found matches before it, reaching near the video start.
        self.assertLess(result_first[0], 8000,
            "Boundary should jump over mid-movie dark zone and find earlier matches")

    def test_boundary_search_stops_when_content_differs_after_dark_zone(self):
        """When content on the other side of a dark zone does NOT match,
        the walk should stop (dark zone is the real boundary).
        """
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        lhs_keys = list(range(0, 20000, 40))
        rhs_keys = list(range(0, 20000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        # Dark zone from 8000ms..10000ms
        dark_zone = {ts for ts in lhs_keys if 8000 <= ts <= 10000}

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts not in dark_zone

        first = (14000, 14000)
        last = (18000, 18000)

        class SelectivePhashCache:
            """Phash that matches for ts > 10000 but fails for ts <= 8000."""
            def __init__(self, hash_size=16): pass
            def get(self, path):
                ts = int(path.split("/")[-1].replace(".png", ""))
                return type('Hash', (), {
                    'value': ts,
                    '__sub__': lambda s, o: 0 if s.value > 10000 or o.value > 10000 else 99,
                    '__abs__': lambda s: s,
                    '__lt__': lambda s, o: s.value < (o if isinstance(o, (int, float)) else o.value),
                    '__le__': lambda s, o: s.value <= (o if isinstance(o, (int, float)) else o.value),
                    '__gt__': lambda s, o: s.value > (o if isinstance(o, (int, float)) else o.value),
                    '__int__': lambda s: s.value,
                    '__float__': lambda s: float(s.value),
                })()

        with (
            patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich),
            patch.object(pm, '_edge_content_matches', return_value=True),
            patch('twotone.tools.melt.pair_matcher.PhashCache', SelectivePhashCache),
        ):
            result_first, _ = pm.look_for_boundaries(
                lhs, rhs, first, last, cutoff=16, extrapolate=False,
            )

        # Content doesn't match after the dark zone — walk should stop at/near it
        self.assertGreaterEqual(result_first[0], 8000,
            "Boundary should stop at dark zone when content doesn't match on the other side")


if __name__ == '__main__':
    unittest.main()
