
import logging
import unittest

from typing import cast
from unittest.mock import patch

from twotone.tools.utils import files_utils, generic_utils, image_utils, video_utils
from twotone.tools.melt.melt import MappingRelation, PairMatcher
from twotone.tools.melt.pair_matcher import GlobalLinearFit, _BoundaryVerifyContext, _VerifySide
from twotone.tools.melt.phash_cache import PhashCache


class PairMatcherUnitTest(unittest.TestCase):
    """Unit tests for PairMatcher internal methods.

    These tests mock get_video_data to avoid needing real video files,
    but use the real PairMatcher constructor. They target:
    - _extrapolate_through_low_entropy: 5% noise tolerance
    - snap_to_edges: snap threshold
    - find_boundary (via _look_for_boundaries): look_ahead robustness
    """

    def _make_pair_matcher(self, lhs_fps: float = 25.0, rhs_fps: float = 25.0) -> PairMatcher:
        """Create a PairMatcher using the real constructor with mocked externals."""
        fps_map = {"/fake/lhs.mp4": str(lhs_fps), "/fake/rhs.mp4": str(rhs_fps)}

        wd = files_utils.Workspace.temporary()
        self.addCleanup(wd.close)
        with patch.object(video_utils, 'get_video_data',
                          side_effect=lambda p, **_kwargs: {"video": [{"fps": fps_map[p]}]}):
            pm = PairMatcher(
                interruption=generic_utils.InterruptibleProcess(),
                wd=wd.root,
                lhs_path="/fake/lhs.mp4",
                rhs_path="/fake/rhs.mp4",
                logger=logging.getLogger("test.PairMatcher"),
            )
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

    def test_coverage_summary_accepts_at_most_2_frames_per_edge(self):
        # The last frame starts one frame before the stream end, so an accepted
        # 2-frame boundary error appears as a 3-frame timestamp gap at the end.
        mappings = [(80, 40), (9880, 9940)]

        result = PairMatcher.coverage_summary(
            mappings,
            10000,
            10000,
            lhs_fps=25.0,
            rhs_fps=50.0,
        )

        self.assertTrue(result["full_coverage"])

    def test_coverage_summary_rejects_more_than_2_frames_at_an_edge(self):
        mappings = [(81, 40), (9920, 9960)]

        result = PairMatcher.coverage_summary(
            mappings,
            10000,
            10000,
            lhs_fps=25.0,
            rhs_fps=50.0,
        )

        self.assertFalse(result["full_coverage"])

    def test_coverage_summary_rejects_more_than_2_frames_at_the_end(self):
        mappings = [(80, 40), (9840, 9920)]

        result = PairMatcher.coverage_summary(
            mappings,
            10000,
            10000,
            lhs_fps=25.0,
            rhs_fps=50.0,
        )

        self.assertFalse(result["full_coverage"])

    # ---- detect_global_linear ----
    #
    # detect_global_linear only *detects* the frame relation (slope / intercept /
    # constant-offset / precise audio time-scale).  It no longer decides the
    # first/last common pair — the boundaries come from the content/entropy-aware
    # walk (covered by the integration tests), which uses the fit as a predictor
    # and stops at genuine divergence.  These unit tests therefore assert the
    # detection and the precision of the fit, not extrapolated boundaries.

    def test_constant_offset_detected(self):
        """A constant +3 frame offset is detected as slope=1, intercept=-3."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # frame offset lhs-rhs = +3 frames
        matching_pairs = [
            (2120, 2000),
            (4120, 4000),
            (6120, 6000),
            (8120, 8000),
        ]

        lhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="rhs")

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(fit)
        self.assertTrue(fit.is_constant_offset)
        self.assertEqual(fit.slope, 1.0)
        self.assertEqual(fit.intercept, -3.0)   # rhs_frame = lhs_frame - 3
        self.assertAlmostEqual(fit.time_scale, 1.0, places=6)   # same fps, same speed

    def test_constant_offset_logs_technical_details_at_debug_and_summary_at_info(self):
        """Constant-offset logs should keep technical details out of info."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        matching_pairs = [
            (2120, 2000),
            (4120, 4000),
            (6120, 6000),
            (8120, 8000),
        ]

        lhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="rhs")

        with self.assertLogs("test.PairMatcher", level="DEBUG") as captured:
            pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        debug_messages = [
            record.getMessage() for record in captured.records
            if record.levelno == logging.DEBUG
        ]
        info_messages = [
            record.getMessage() for record in captured.records
            if record.levelno == logging.INFO
        ]

        self.assertTrue(any(
            "Constant offset detected: 3 frame(s) (median=3.0, std=0.00)." in message
            for message in debug_messages
        ))
        self.assertFalse(any("median=" in message for message in info_messages))
        self.assertTrue(any(
            "Files #1 (lhs.mp4) and #2 (rhs.mp4) share content with" in message
            for message in info_messages
        ))
        self.assertTrue(any(
            "small constant frame offset of 3 frame(s)" in message
            for message in info_messages
        ))

    def test_constant_offset_negative(self):
        """A negative offset (rhs ahead of lhs) is detected as intercept=+3."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        matching_pairs = [
            (2000, 2120),
            (4000, 4120),
            (6000, 6120),
        ]

        lhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="rhs")

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(fit)
        self.assertTrue(fit.is_constant_offset)
        self.assertEqual(fit.slope, 1.0)
        self.assertEqual(fit.intercept, 3.0)   # rhs_frame = lhs_frame + 3
        self.assertAlmostEqual(fit.time_scale, 1.0, places=6)

    def test_constant_offset_zero(self):
        """Identical timing is detected as slope=1, intercept=0."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        matching_pairs = [
            (2000, 2000),
            (4000, 4000),
            (6000, 6000),
            (8000, 8000),
        ]

        lhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="rhs")

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(fit)
        self.assertTrue(fit.is_constant_offset)
        self.assertEqual(fit.slope, 1.0)
        self.assertEqual(fit.intercept, 0.0)
        self.assertAlmostEqual(fit.time_scale, 1.0, places=6)

    def test_constant_offset_rejected_high_std(self):
        """When frame-number offsets vary too much (std > 1), returns None."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # frame offsets: 3, 5, 1 → std ≈ 1.63 > 1.0; span too small for drift
        matching_pairs = [
            (2120, 2000),
            (4200, 4000),
            (6040, 6000),
        ]

        lhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="rhs")

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNone(fit)

    def test_constant_offset_rejected_growing_offsets(self):
        """Growing offsets over a short span fit neither constant offset nor drift."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # frame offsets: 2, 4, 6, 8 → std ≈ 2.24 > 1.0; span 156 frames < min 250
        matching_pairs = [
            (2080, 2000),
            (4160, 4000),
            (6240, 6000),
            (8320, 8000),
        ]

        lhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="rhs")

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNone(fit)

    def test_linear_frame_drift_detected(self):
        """A 23/24 frame-count conversion (same speed) is detected as a drift."""
        pm = self._make_pair_matcher(lhs_fps=24.0, rhs_fps=23.0)

        # RHS has 23 frames for each 24 LHS frames while preserving playback
        # time: rhs_frame = 23/24 * lhs_frame + 1.
        lhs_frames = self._make_frames(list(range(0, 10001 * 40, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 9590 * 40, 40)), prefix="rhs")

        matching_frame_ids = [1000, 4000, 7000, 9000]
        matching_pairs = [
            (lhs_id * 40, int(round((23 / 24) * lhs_id + 1)) * 40)
            for lhs_id in matching_frame_ids
        ]

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(fit)
        self.assertFalse(fit.is_constant_offset)
        self.assertAlmostEqual(fit.slope, 23 / 24, places=3)
        # 23/24 frames scaled by 24/23 fps → same playback speed, no audio stretch.
        self.assertAlmostEqual(fit.time_scale, 1.0, places=2)

    def test_global_linear_accepts_time_scaled_drift(self):
        """A near-identity frame slope with an FPS speed change yields time_scale != 1."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=26.5)

        lhs_frames = self._make_frames(list(range(0, 10001 * 40, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 10291 * 40, 40)), prefix="rhs")

        matching_frame_ids = [1000, 4000, 7000, 9000]
        matching_pairs = [
            (lhs_id * 40, int(round(1.028 * lhs_id)) * 40)
            for lhs_id in matching_frame_ids
        ]

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(fit)
        self.assertFalse(fit.is_constant_offset)
        self.assertAlmostEqual(fit.slope, 1.028, places=3)
        # slope 1.028 * 25/26.5 ≈ 0.9698 → a real speed change, audio time-scaled.
        self.assertAlmostEqual(fit.time_scale, 1.028 * 25 / 26.5, places=4)
        self.assertGreater(abs(fit.time_scale - 1.0), 0.02)

    def test_linear_frame_drift_rejects_extreme_slope_delta(self):
        """Extreme frame-count conversion is not a near-identity drift → None."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        lhs_frames = self._make_frames(list(range(0, 10001 * 40, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 11001 * 40, 40)), prefix="rhs")

        matching_frame_ids = [1000, 4000, 7000, 9000]
        matching_pairs = [
            (lhs_id * 40, int(round(1.08 * lhs_id + 1)) * 40)
            for lhs_id in matching_frame_ids
        ]

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNone(fit)

    def test_constant_offset_two_nearby_pairs_are_insufficient(self):
        """Two nearby pairs do not cover enough content to prove a constant offset."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        matching_pairs = [(2120, 2000), (4120, 4000)]

        lhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="rhs")

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNone(fit)

    def test_constant_offset_two_widely_separated_pairs_are_sufficient(self):
        """Two distant pairs can prove the constant frame offset seen on Ubuntu CI."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        lhs_frames = self._make_frames(list(range(480, 60441, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 60441, 40)), prefix="rhs")
        matching_pairs = [
            (10480, 10480),
            (50480, 50480),
        ]

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(fit)
        self.assertTrue(fit.is_constant_offset)
        self.assertEqual(fit.slope, 1.0)
        # lhs key list starts at 480ms (frame_id 0) while rhs starts at 0ms
        # (frame_id 0), so at a shared timestamp the lhs frame_id trails the rhs
        # frame_id by 12 (480/40): offset lhs-rhs = -12 → intercept +12.
        self.assertEqual(fit.intercept, 12.0)

    def test_constant_offset_slight_jitter_within_tolerance(self):
        """Small jitter (< 1 frame) in frame-number offsets is still a constant offset."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # frame offsets: 3, 3, 4, 3 → std ≈ 0.43 < 1.0, median 3
        matching_pairs = [
            (2120, 2000),
            (4120, 4000),
            (6160, 6000),
            (8120, 8000),
        ]

        lhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 10040, 40)), prefix="rhs")

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(fit)
        self.assertTrue(fit.is_constant_offset)
        self.assertEqual(fit.slope, 1.0)
        self.assertEqual(fit.intercept, -3.0)   # median offset 3

    def test_constant_offset_different_fps(self):
        """Same frames at different FPS (PAL speedup) → slope=1, time_scale=lhs/rhs fps."""
        pm = self._make_pair_matcher(lhs_fps=20.0, rhs_fps=25.0)
        # lhs_frame_ms=50ms, rhs_frame_ms=40ms; frame offset k=3
        matching_pairs = [
            (2500, 1880),   # LHS frame 50, RHS frame 47
            (5000, 3880),   # LHS frame 100, RHS frame 97
            (7500, 5880),   # LHS frame 150, RHS frame 147
        ]

        lhs_frames = self._make_frames(list(range(0, 10050, 50)), prefix="lhs")
        rhs_frames = self._make_frames(list(range(0, 8040, 40)), prefix="rhs")

        fit = pm.detect_global_linear(matching_pairs, lhs_frames, rhs_frames)

        self.assertIsNotNone(fit)
        self.assertTrue(fit.is_constant_offset)
        self.assertEqual(fit.slope, 1.0)
        self.assertEqual(fit.intercept, -3.0)
        # Same frames played at 20 vs 25 fps → a real 0.8x speed change.
        self.assertAlmostEqual(fit.time_scale, 20.0 / 25.0, places=6)

    def test_create_segments_mapping_verified_extrapolation_for_global_linear(self):
        """Global-linear uses the fit's verified extrapolation, not the refine walk."""
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
        extrapolated_pairs = [(0, 0), (10000, 9880)]
        fit = GlobalLinearFit(slope=1.0, intercept=-3.0, is_constant_offset=True, time_scale=0.96)

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
             patch.object(PairMatcher, 'detect_global_linear', return_value=fit), \
             patch.object(PairMatcher, '_extrapolate_and_verify_global_linear', return_value=extrapolated_pairs) as extrap, \
             patch.object(PairMatcher, '_extract_and_refine_boundaries', side_effect=AssertionError("refine walk must not run for global-linear")), \
             patch.object(PairMatcher, 'snap_to_edges', side_effect=AssertionError("edge snap must not run for global-linear")):

            debug = debug_cls.return_value
            debug.dump_frames.return_value = None
            debug.dump_matches.return_value = None

            result = pm.create_segments_mapping()

        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)
        # Boundaries come from the fit's content-verified extrapolation.
        extrap.assert_called_once()
        self.assertEqual(result.mapping, extrapolated_pairs)

    # ---- _extrapolate_and_verify_global_linear ----

    def _make_pm_with_frames(self, lhs_ts, rhs_ts):
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        pm.lhs_all_frames = self._make_frames(lhs_ts, prefix="lhs")
        pm.rhs_all_frames = self._make_frames(rhs_ts, prefix="rhs")
        return pm

    @staticmethod
    def _patch_verify_ctx():
        """Patch the verify-context builder — the tests patch the consumer
        (_boundary_content_matches), so the real builder's image work is
        neither needed nor possible on the fake frame paths."""
        return patch.object(PairMatcher, '_build_boundary_verify_context', return_value=None)

    def _make_verify_ctx(self, phash_values: dict, cutoff: float,
                         lhs_images: dict, rhs_images: dict) -> _BoundaryVerifyContext:
        """Build a context with pre-seeded comparison images and a fake phash."""
        class FakePhash:
            def get(self, path):
                return phash_values[path]

        def side(images):
            return _VerifySide(
                video_path="/fake", raw_dir="/fake_raw", all_frames={}, normalized={},
                crop_fn=None, comparison_dir="/fake_cmp", comparison_cache=dict(images),
            )

        return _BoundaryVerifyContext(
            lhs=side(lhs_images), rhs=side(rhs_images),
            phash=cast(PhashCache, FakePhash()), cutoff=cutoff,
        )

    def test_verified_extrapolation_extends_to_edges_when_content_matches(self):
        """When the extrapolated boundary frames verify, the boundary reaches the edge."""
        pm = self._make_pm_with_frames(list(range(0, 10040, 40)), list(range(0, 10040, 40)))
        # Zero offset, matches only in the middle of the video.
        matching_pairs = [(4000, 4000), (6000, 6000)]
        fit = GlobalLinearFit(slope=1.0, intercept=0.0, is_constant_offset=True, time_scale=1.0)

        with self._patch_verify_ctx(), \
             patch.object(PairMatcher, '_boundary_content_matches', return_value=True):
            result = pm._extrapolate_and_verify_global_linear(
                fit, matching_pairs, pm.lhs_all_frames, pm.rhs_all_frames)

        self.assertEqual(result[0], (0, 0))
        self.assertEqual(result[-1], (10000, 10000))

    def test_verified_extrapolation_stops_at_outermost_match_when_divergent(self):
        """When the extrapolated boundary frames do NOT verify, the boundary stays put."""
        pm = self._make_pm_with_frames(list(range(0, 10040, 40)), list(range(0, 10040, 40)))
        matching_pairs = [(3000, 3000), (6000, 6000)]
        fit = GlobalLinearFit(slope=1.0, intercept=0.0, is_constant_offset=True, time_scale=1.0)

        with self._patch_verify_ctx(), \
             patch.object(PairMatcher, '_boundary_content_matches', return_value=False):
            result = pm._extrapolate_and_verify_global_linear(
                fit, matching_pairs, pm.lhs_all_frames, pm.rhs_all_frames)

        # No extension — divergent intro/outro is not crossed.
        self.assertEqual(result[0], (3000, 3000))
        self.assertEqual(result[-1], (6000, 6000))

    def test_verified_extrapolation_snaps_equal_length_lead_in_to_edge(self):
        """A small fit residual at the lhs edge snaps the rhs onto its edge → (0,0)."""
        pm = self._make_pm_with_frames(list(range(0, 10040, 40)), list(range(0, 10040, 40)))
        matching_pairs = [(4000, 4120), (6000, 6120)]
        # rhs_frame = lhs_frame + 3 → at lhs edge (0) the line predicts rhs frame 3;
        # within snap tolerance of the rhs edge, so it maps edge-to-edge.
        fit = GlobalLinearFit(slope=1.0, intercept=3.0, is_constant_offset=True, time_scale=1.0)

        with self._patch_verify_ctx(), \
             patch.object(PairMatcher, '_boundary_content_matches', return_value=True):
            result = pm._extrapolate_and_verify_global_linear(
                fit, matching_pairs, pm.lhs_all_frames, pm.rhs_all_frames)

        self.assertEqual(result[0], (0, 0))

    def test_verified_extrapolation_keeps_offset_lead_in_position(self):
        """A large offset (2s vs 6s black) is far past the snap tolerance and is kept."""
        pm = self._make_pm_with_frames(list(range(0, 10040, 40)), list(range(0, 10040, 40)))
        matching_pairs = [(4000, 6000), (6000, 8000)]
        # rhs_frame = lhs_frame + 50 (a 2s offset) → at lhs edge (0) the line
        # predicts rhs frame 50 (2000ms), well past the snap tolerance.
        fit = GlobalLinearFit(slope=1.0, intercept=50.0, is_constant_offset=True, time_scale=1.0)

        with self._patch_verify_ctx(), \
             patch.object(PairMatcher, '_boundary_content_matches', return_value=True):
            result = pm._extrapolate_and_verify_global_linear(
                fit, matching_pairs, pm.lhs_all_frames, pm.rhs_all_frames)

        self.assertEqual(result[0], (0, 2000))

    @staticmethod
    def _patch_entropy(values: dict):
        """Patch image entropy per comparison-image path."""
        return patch.object(image_utils, 'image_entropy', side_effect=lambda path: values[path])

    def test_boundary_content_matches_accepts_both_black(self):
        """Two low-entropy (black) boundary frames verify as a shared lead-in/out."""
        pm = self._make_pm_with_frames([0, 40], [0, 40])
        # No phash values — both-black short-circuits before any hash lookup.
        ctx = self._make_verify_ctx({}, cutoff=16,
                                    lhs_images={0: "/l.png"}, rhs_images={0: "/r.png"})
        with self._patch_entropy({"/l.png": 0.5, "/r.png": 0.5}):
            self.assertTrue(pm._boundary_content_matches(ctx, 0, 0, 0, 0))

    def test_boundary_content_matches_rejects_black_vs_content(self):
        """A decisively black frame vs a rich frame never verifies (one lead-in is black)."""
        pm = self._make_pm_with_frames([0, 40], [0, 40])
        ctx = self._make_verify_ctx({}, cutoff=16,
                                    lhs_images={0: "/l.png"}, rhs_images={0: "/r.png"})
        with self._patch_entropy({"/l.png": 0.5, "/r.png": 5.0}):
            self.assertFalse(pm._boundary_content_matches(ctx, 0, 0, 0, 0))

    def test_boundary_content_matches_flat_vs_rich_falls_through_to_phash(self):
        """A flat-but-lit frame (title card, sky) vs a rich one is not a
        black-vs-content mismatch; the phash comparison decides.

        A framing difference between two transfers can push one side of a
        genuinely shared flat scene just under the richness threshold — a
        hard reject there would collapse the boundary in front of it.
        """
        pm = self._make_pm_with_frames([0, 40], [0, 40])
        ctx = self._make_verify_ctx({"/l.png": 0, "/r.png": 12}, cutoff=16,
                                    lhs_images={0: "/l.png"}, rhs_images={0: "/r.png"})
        with self._patch_entropy({"/l.png": 3.2, "/r.png": 5.0}):
            self.assertTrue(pm._boundary_content_matches(ctx, 0, 0, 0, 0))

    def test_boundary_content_matches_accepts_within_cutoff(self):
        """A distance within the pair-calibrated cutoff verifies as the same frame."""
        pm = self._make_pm_with_frames([0, 40], [0, 40])
        ctx = self._make_verify_ctx({"/l.png": 0, "/r.png": 12}, cutoff=16,
                                    lhs_images={0: "/l.png"}, rhs_images={0: "/r.png"})
        with self._patch_entropy({"/l.png": 5.0, "/r.png": 5.0}):
            self.assertTrue(pm._boundary_content_matches(ctx, 0, 0, 0, 0))

    def test_boundary_content_matches_rejects_beyond_cutoff(self):
        """Divergent content (e.g. grass vs atoms intros) lands far above the
        calibrated cutoff and is rejected, keeping the extrapolation from
        crossing a different intro/outro."""
        pm = self._make_pm_with_frames([0, 40], [0, 40])
        ctx = self._make_verify_ctx({"/l.png": 0, "/r.png": 124}, cutoff=16,
                                    lhs_images={0: "/l.png"}, rhs_images={0: "/r.png"})
        with self._patch_entropy({"/l.png": 5.0, "/r.png": 5.0}):
            self.assertFalse(pm._boundary_content_matches(ctx, 0, 0, 0, 0))

    def test_boundary_content_matches_rejects_missing_comparison_image(self):
        """A frame whose comparison image cannot be produced never verifies."""
        pm = self._make_pm_with_frames([0, 40], [0, 40])
        ctx = self._make_verify_ctx({}, cutoff=16,
                                    lhs_images={0: None}, rhs_images={0: "/r.png"})
        self.assertFalse(pm._boundary_content_matches(ctx, 0, 0, 0, 0))

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

    # ---- find_content_discontinuities ----

    def test_content_discontinuity_flags_commercial_cut(self):
        # TMNT-like pair: steady 0.9864 slope, but 7 s of rhs content missing
        # in one gap (an ad-break cut in the rhs transfer).
        mapping = [
            (lhs, round(lhs * 0.9864) - (7000 if lhs > 300000 else 0))
            for lhs in range(0, 600001, 10000)
        ]

        result = PairMatcher.find_content_discontinuities(mapping)

        self.assertEqual(result, [(300000, 310000, 295920, 298784, 7000)])

    def test_content_discontinuity_tolerates_speed_waviness(self):
        # VHS-style source: scene durations wobble ±3% around the median
        # speed, with no content missing on either side.
        mapping = [(0, 0)]
        lhs = rhs = 0
        for i in range(60):
            lhs += 10000
            rhs += round(10000 * (1.03 if i % 2 else 0.97))
            mapping.append((lhs, rhs))

        result = PairMatcher.find_content_discontinuities(mapping)

        self.assertEqual(result, [])

    def test_content_discontinuity_threshold_scales_with_gap_length(self):
        # The same 2.5 s deviation is drift/matcher noise across a sparse
        # 160 s gap (under the 2% relative threshold) but a genuine hole in
        # a 20 s gap.
        base = [(i * 10000, i * 10000) for i in range(30)]
        last_lhs, last_rhs = base[-1]

        sparse = base + [(last_lhs + 160000, last_rhs + 157500)]
        dense = base + [(last_lhs + 20000, last_rhs + 17500)]

        self.assertEqual(PairMatcher.find_content_discontinuities(sparse), [])
        self.assertEqual(
            PairMatcher.find_content_discontinuities(dense),
            [(last_lhs, last_lhs + 20000, last_rhs, last_rhs + 17500, 2500)],
        )

    # ---- _drop_pairs_breaking_local_linearity ----

    def test_local_linearity_filter_drops_mismatched_gendarme_pair(self):
        # Real pairs #310-#329 from a Gendarme Gets Married melt (24 fps
        # BluRay vs 25 fps AVI transfer, slope 0.96).  The pair
        # (3671813, 3527080) matched two near-identical shots 2.8 s apart,
        # bending the line: gap deficits -2800/+2800 ms around it cancel
        # once it is removed.
        pm = self._make_pair_matcher()
        good = [
            (3608813, 3463840), (3611938, 3466840), (3626188, 3480560),
            (3644438, 3498000), (3648771, 3502160), (3656021, 3509160),
            (3668313, 3520960), (3669605, 3522160), (3670563, 3523080),
            (3678355, 3530560), (3681021, 3533120), (3682980, 3535000),
            (3684105, 3536120), (3688813, 3540600), (3690730, 3542400),
            (3693855, 3545440), (3698313, 3549720), (3704605, 3555760),
            (3708980, 3559960),
        ]
        rogue = (3671813, 3527080)

        result = pm._drop_pairs_breaking_local_linearity(sorted(good + [rogue]))

        self.assertEqual(result, good)
        self.assertEqual(PairMatcher.find_content_discontinuities(result), [])

    def test_local_linearity_filter_keeps_genuine_content_cut(self):
        # A commercial-break cut shifts the offset persistently: its deficit
        # is one-sided, no removal can cancel it, so every pair must survive
        # for find_content_discontinuities to report the hole.
        pm = self._make_pair_matcher()
        mapping = [
            (lhs, round(lhs * 0.9864) - (7000 if lhs > 300000 else 0))
            for lhs in range(0, 600001, 10000)
        ]

        result = pm._drop_pairs_breaking_local_linearity(mapping)

        self.assertEqual(result, mapping)
        self.assertEqual(len(PairMatcher.find_content_discontinuities(result)), 1)

    def test_local_linearity_filter_keeps_adjacent_same_sign_cuts(self):
        # Two cuts in consecutive gaps push the deficit the same way; they do
        # not cancel, so the pair between them is genuine and must survive.
        pm = self._make_pair_matcher()
        mapping = []
        for lhs in range(0, 600001, 10000):
            cut = (7000 if lhs > 300000 else 0) + (7000 if lhs > 310000 else 0)
            mapping.append((lhs, lhs - cut))

        result = pm._drop_pairs_breaking_local_linearity(mapping)

        self.assertEqual(result, mapping)

    def test_local_linearity_filter_tolerates_speed_waviness(self):
        # VHS-style wobble oscillates below the deficit threshold — no gap
        # gets flagged and no pair gets dropped.
        pm = self._make_pair_matcher()
        mapping = [(0, 0)]
        lhs = rhs = 0
        for i in range(60):
            lhs += 10000
            rhs += round(10000 * (1.03 if i % 2 else 0.97))
            mapping.append((lhs, rhs))

        result = pm._drop_pairs_breaking_local_linearity(mapping)

        self.assertEqual(result, mapping)

    def test_local_linearity_filter_drops_multiple_isolated_rogue_pairs(self):
        # Each mismatched pair bends the line locally; removals repeat until
        # every remaining gap follows the median slope again.
        pm = self._make_pair_matcher()
        base = [(i * 10000, i * 10000) for i in range(40)]
        corrupted = list(base)
        corrupted[10] = (100000, 103000)
        corrupted[30] = (300000, 297200)

        result = pm._drop_pairs_breaking_local_linearity(corrupted)

        expected = [p for i, p in enumerate(base) if i not in (10, 30)]
        self.assertEqual(result, expected)

    # ---- detect_global_linear on refined generic pairs ----

    def test_detect_global_linear_certifies_refined_linear_pairs(self):
        # 24 fps base vs a 25 fps frame-for-frame transfer: the refined pairs
        # lie on one line, so the detection certifies the frame relation and
        # the implied time scale — earning the mapping the verified boundary
        # extension in the GENERIC branch.
        pm = self._make_pair_matcher(lhs_fps=24.0, rhs_fps=25.0)
        pm.lhs_all_frames = {i * 500: {"frame_id": i * 12} for i in range(121)}
        pm.rhs_all_frames = {i * 480: {"frame_id": i * 12} for i in range(121)}
        pairs = [(i * 500, i * 480) for i in (0, 30, 60, 90, 120)]

        fit = pm.detect_global_linear(pairs, pm.lhs_all_frames, pm.rhs_all_frames)

        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit.slope, 1.0, places=6)
        self.assertAlmostEqual(fit.intercept, 0.0, places=6)
        self.assertAlmostEqual(fit.time_scale, 24.0 / 25.0, places=6)

    def test_detect_global_linear_rejects_piecewise_refined_pairs(self):
        # 7 s of content missing on the rhs side halfway through: the pairs do
        # not lie on one line, so no boundary predictor may be built and the
        # edges stay where the content search left them.
        pm = self._make_pair_matcher(lhs_fps=24.0, rhs_fps=24.0)
        pm.lhs_all_frames = {i * 500: {"frame_id": i * 12} for i in range(121)}
        pm.rhs_all_frames = {i * 500: {"frame_id": i * 12} for i in range(121)}
        pairs = [
            (i * 500, i * 500 - (7000 if i >= 60 else 0))
            for i in (0, 30, 45, 60, 90, 120)
        ]

        self.assertIsNone(
            pm.detect_global_linear(pairs, pm.lhs_all_frames, pm.rhs_all_frames)
        )


if __name__ == '__main__':
    unittest.main()
