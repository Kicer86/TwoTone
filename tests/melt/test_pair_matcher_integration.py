
import logging

from twotone.tools.utils import generic_utils, video_utils
from twotone.tools.melt.melt import MappingRelation, PairMatcher
from common import add_to_test_dir
from melt.helpers import MeltTestBase


class PairMatcherIntegrationTest(MeltTestBase):

    def test_scene_detection_is_stable_and_content_driven(self):
        """Scene detection must be stable — the whole frame-matching pipeline
        keys off it, so non-deterministic cuts would make the mapping unstable.

        Validates three properties on the real fixtures:
        1. Deterministic: identical scene-change timestamps across repeated runs.
        2. Finds the known synthetic cut: the 3s intro boundary is detected.
        3. Content-driven: bbb_gi3 (grass intro) and bbb_bi3 (black intro) share
           the same bbb body starting at 3s, so their cuts from 3s onward match —
           the detector keys off content, not the differing intro.
        """
        grass_intro, _ = self.edge_fixtures["diff_intro_same"]   # bbb_gi3: 3s grass intro + bbb
        black_intro, _ = self.edge_fixtures["black_intro_same"]  # bbb_bi3: 3s black intro + bbb

        interruption = generic_utils.InterruptibleProcess()
        scenes_grass = video_utils.detect_scene_changes(
            grass_intro, threshold=0.3, interruption=interruption, logger=self.logger,
        )
        scenes_grass_again = video_utils.detect_scene_changes(
            grass_intro, threshold=0.3, interruption=interruption, logger=self.logger,
        )

        # 1. Deterministic across runs.
        self.assertEqual(scenes_grass, scenes_grass_again)

        # 2. The known 3s intro boundary is detected (within ~3 frames).
        self.assertTrue(
            any(abs(t - 3000) <= 120 for t in scenes_grass),
            f"expected a scene cut near 3000ms, got {scenes_grass}",
        )

        # 3. Content-driven & stable: the bbb body cuts match across fixtures that
        #    share it, regardless of the (grass vs black) intro before it.
        scenes_black = video_utils.detect_scene_changes(
            black_intro, threshold=0.3, interruption=interruption, logger=self.logger,
        )
        body_grass = [t for t in scenes_grass if t >= 3000]
        body_black = [t for t in scenes_black if t >= 3000]
        self.assertEqual(body_grass, body_black)
        # A real movie body has several cuts — this is not a degenerate match.
        self.assertGreaterEqual(len(body_grass), 5)

    def test_pair_matcher_precision(self):
        """Multi-scene sample (82.6s) vs VHS degraded version (78.7s, 1.05x speed, crop, blur)."""
        file1 = add_to_test_dir(self.wd.path, self.sample_video_file)
        file2 = add_to_test_dir(self.wd.path, self.sample_vhs_video_file)

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1, file2, logging.getLogger("PM"))
        mappings = pair_matcher.create_segments_mapping().mapping

        # At least 6 pairs across the 82s video
        self.assertGreaterEqual(len(mappings), 6)
        # Edge: first pair snapped to (0, 0)
        self.assertEqual(mappings[0], (0, 0))
        # Edge: last pair near video duration
        self.assertAlmostEqual(mappings[-1][0], 82582, delta=500)
        self.assertAlmostEqual(mappings[-1][1], 78700, delta=500)
        # Monotonicity
        for i in range(1, len(mappings)):
            self.assertGreaterEqual(mappings[i][0], mappings[i-1][0])
            self.assertGreaterEqual(mappings[i][1], mappings[i-1][1])

        # Both edges snapped to video boundaries. Full coverage.
        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1),
            video_utils.get_video_duration(file2),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_black_intro_same_length(self):
        """Both files have the same length black intro — boundary should extend through it."""
        file1_path, file2_path = self.edge_fixtures["black_intro_same"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        result = pair_matcher.create_segments_mapping()
        mappings = result.mapping
        # The fixtures are related by a 1.03x linear speed change, so the
        # detected relation must be GLOBAL_LINEAR (stable across ffmpeg builds).
        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)

        # 3s black intro on both files — boundary extends through black to edge
        # LHS: bbb_bi3 (65.3s), RHS: bi3_deg103 (63.4s)
        self.assertGreaterEqual(len(mappings), 3)
        # Edge: first pair snapped to (0, 0) through black intro
        self.assertEqual(mappings[0], (0, 0))
        # Edge: last pair near video duration
        self.assertAlmostEqual(mappings[-1][0], 65337, delta=500)
        self.assertAlmostEqual(mappings[-1][1], 63433, delta=500)

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_black_intro_different_length(self):
        """Files have different length black intros — algorithm should find content pairs despite offset."""
        file1_path, file2_path = self.edge_fixtures["black_intro_diff"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        result = pair_matcher.create_segments_mapping()
        mappings = result.mapping
        # The fixtures are related by a 1.03x linear speed change, so the
        # detected relation must be GLOBAL_LINEAR (stable across ffmpeg builds).
        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)

        # LHS: bbb_bi2 (64.3s, 2s black intro), RHS: bi6_deg103 (66.5s, 6s black intro)
        # LHS extends to edge through black; RHS starts inside its 6s intro (3-6.5s)
        self.assertGreaterEqual(len(mappings), 3)
        # Edge: LHS snapped to 0; RHS somewhere inside its 6s black intro (3-6.5s)
        self.assertEqual(mappings[0][0], 0)
        self.assertGreater(mappings[0][1], 3000)
        self.assertLess(mappings[0][1], 6500)
        # Edge: both near video duration
        self.assertAlmostEqual(mappings[-1][0], 64103, delta=500)
        self.assertAlmostEqual(mappings[-1][1], 66325, delta=500)

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        # LHS snapped to 0, RHS inside its 6s black intro — RHS start gap < 6s.
        # Both ends reach their video boundary.
        self.assertFalse(coverage["full_coverage"])
        self.assertEqual(coverage["lhs_start_gap_s"], 0.0)
        self.assertGreater(coverage["rhs_start_gap_s"], 3.0)
        self.assertLess(coverage["rhs_start_gap_s"], 6.5)
        self.assertLess(coverage["lhs_end_gap_s"], 0.5)
        self.assertLess(coverage["rhs_end_gap_s"], 0.5)

    def test_pair_matcher_black_outro(self):
        """Both files have black outro — last pair should extend through it to edge."""
        file1_path, file2_path = self.edge_fixtures["black_outro"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        result = pair_matcher.create_segments_mapping()
        mappings = result.mapping
        # The fixtures are related by a 1.03x linear speed change, so the
        # detected relation must be GLOBAL_LINEAR (stable across ffmpeg builds).
        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)

        # LHS: bbb_bo3 (65.3s, 3s black outro), RHS: bo3_deg103 (63.4s, 3s black outro)
        self.assertGreaterEqual(len(mappings), 3)
        # Edge: first pair snapped to (0, 0) — shared content from the start
        self.assertEqual(mappings[0], (0, 0))
        # Edge: last pair snapped to video duration (through black outro)
        self.assertAlmostEqual(mappings[-1][0], 65337, delta=500)
        self.assertAlmostEqual(mappings[-1][1], 63433, delta=500)

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_both_intro_and_outro_black(self):
        """Files have black intro AND outro — both boundaries should be handled."""
        file1_path, file2_path = self.edge_fixtures["both_black"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        result = pair_matcher.create_segments_mapping()
        mappings = result.mapping
        # The fixtures are related by a 1.03x linear speed change, so the
        # detected relation must be GLOBAL_LINEAR (stable across ffmpeg builds).
        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)

        # LHS: bi2_bo2 (66.3s, 2s black intro + 2s black outro)
        # RHS: bi2_bo2_deg103 (64.4s, same + 1.03x speed)
        self.assertGreaterEqual(len(mappings), 3)
        # Edge: first pair snapped to (0, 0) through black intro
        self.assertEqual(mappings[0], (0, 0))
        # Edge: last pair snapped to video duration through black outro
        self.assertAlmostEqual(mappings[-1][0], 66343, delta=500)
        self.assertAlmostEqual(mappings[-1][1], 64415, delta=500)

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        # Both edges snapped through black intro/outro to video boundaries.
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_no_speed_change(self):
        """Same content with no speed change (ratio ~1.0), only quality degradation."""
        file1_path, file2_path = self.edge_fixtures["no_speed"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        result = pair_matcher.create_segments_mapping()
        mappings = result.mapping
        # Same speed, zero offset: a constant-offset global-linear relation.
        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)

        # LHS: bbb (62.3s), RHS: bbb_deg10 (62.3s, same speed, degraded quality)
        self.assertGreaterEqual(len(mappings), 3)
        # Edge: first pair at (0, 0)
        self.assertEqual(mappings[0], (0, 0))
        # Edge: last pair near video duration (~62.3s)
        self.assertAlmostEqual(mappings[-1][0], 62314, delta=500)
        self.assertAlmostEqual(mappings[-1][1], 62313, delta=500)

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_open_matte_vs_widescreen_crop(self):
        """Same content in two transfers of different picture geometry.

        LHS is the full 16:9 (open-matte) master; RHS is a widescreen
        ~2.35:1 crop of it, degraded and speed-changed by 1.03 — the same
        relation as a 4K open-matte release vs an SD widescreen rip of one
        film.  The mapping must still cover both videos edge to edge.

        Regression: the global-linear boundary verification compared raw
        extracted frames, where the geometry difference pushes even the
        exact same picture past any fixed phash gate.  Every gap sample
        failed, the boundaries collapsed to the outermost scene-change
        matches and several minutes of shared content were cut on a real
        4K 16:9 release vs an SD 2.35:1 release of the same film.
        """
        file1_path, file2_path = self.edge_fixtures["open_matte"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        result = pair_matcher.create_segments_mapping()

        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)

        coverage = PairMatcher.coverage_summary(
            result.mapping,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_different_intro_same_length(self):
        """Files have different high-entropy intros of similar length, then shared content."""
        file1_path, file2_path = self.edge_fixtures["diff_intro_same"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        result = pair_matcher.create_segments_mapping()
        mappings = result.mapping
        # The shared bodies are related by a 1.03x linear speed change, so
        # the detected relation must be GLOBAL_LINEAR regardless of the
        # divergent intros/outros around them.
        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)

        # LHS: bbb_gi3 (65.3s, 3s grass intro), RHS: atoms_i3_deg (63.5s, 3s atoms intro)
        self.assertGreaterEqual(len(mappings), 3)
        # First pair NOT at edge — content starts at ~3s (after different intros)
        self.assertAlmostEqual(mappings[0][0], 3023, delta=500)
        self.assertAlmostEqual(mappings[0][1], 3057, delta=500)
        # Last pair snapped to video duration (fine-sweep extends to edge)
        self.assertAlmostEqual(mappings[-1][0], 65302, delta=100)
        self.assertAlmostEqual(mappings[-1][1], 63471, delta=100)

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        # Both files have 3s intros (grass / atoms). Common content starts at ~3s.
        # Start gaps must match the known intro duration within 1s tolerance.
        # End gaps effectively 0 — boundary walk + snap reaches video edge.
        self.assertFalse(coverage["full_coverage"])
        self.assertAlmostEqual(coverage["lhs_start_gap_s"], 3.0, delta=1.0)
        self.assertAlmostEqual(coverage["rhs_start_gap_s"], 3.0, delta=1.0)
        self.assertLess(coverage["lhs_end_gap_s"], 0.1)
        self.assertLess(coverage["rhs_end_gap_s"], 0.1)

    def test_pair_matcher_different_intro_different_length(self):
        """Files have different high-entropy intros of DIFFERENT lengths."""
        file1_path, file2_path = self.edge_fixtures["diff_intro_diff"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        result = pair_matcher.create_segments_mapping()
        mappings = result.mapping
        # The shared bodies are related by a 1.03x linear speed change, so
        # the detected relation must be GLOBAL_LINEAR regardless of the
        # divergent intros/outros around them.
        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)

        # LHS: bbb_gi2 (64.3s, 2s grass intro), RHS: atoms_i5_deg (65.5s, 5s atoms intro)
        self.assertGreaterEqual(len(mappings), 3)
        # First pair NOT at edge — after different intros (2s vs 5s)
        self.assertAlmostEqual(mappings[0][0], 2023, delta=500)
        self.assertAlmostEqual(mappings[0][1], 5057, delta=500)
        # Last pair snapped to video duration (fine-sweep extends to edge)
        self.assertAlmostEqual(mappings[-1][0], 64302, delta=100)
        self.assertAlmostEqual(mappings[-1][1], 65471, delta=100)

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        # LHS has 2s grass intro, RHS has 5s atoms intro.
        # Start gaps must match known intro durations within 1s tolerance.
        # End gaps effectively 0 — boundary walk + snap reaches video edge.
        self.assertFalse(coverage["full_coverage"])
        self.assertAlmostEqual(coverage["lhs_start_gap_s"], 2.0, delta=1.0)
        self.assertAlmostEqual(coverage["rhs_start_gap_s"], 5.0, delta=1.0)
        self.assertLess(coverage["lhs_end_gap_s"], 0.1)
        self.assertLess(coverage["rhs_end_gap_s"], 0.1)

    def test_pair_matcher_different_outro(self):
        """Files share content but have different high-entropy outros."""
        file1_path, file2_path = self.edge_fixtures["diff_outro"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        result = pair_matcher.create_segments_mapping()
        mappings = result.mapping
        # The fixtures are related by a 1.03x linear speed change, so the
        # detected relation must be GLOBAL_LINEAR (stable across ffmpeg builds).
        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)

        # LHS: bbb_wo3 (65.3s, 3s woman outro), RHS: deg103_atoms_o3 (63.5s, 3s atoms outro)
        self.assertGreaterEqual(len(mappings), 3)
        # Edge: first pair snapped to (0, 0) — shared content from the start
        self.assertEqual(mappings[0], (0, 0))
        # Last pair NOT at edge — content ends before the divergent outros
        # (~62s lhs, ~60s rhs); the walk stops at the last shared frame.
        self.assertAlmostEqual(mappings[-1][0], 62103, delta=1000)
        self.assertAlmostEqual(mappings[-1][1], 60325, delta=1000)

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        # Start snapped to edge. Both files have 3s outros (woman / atoms) that
        # differ, so the shared range ends before them — end gaps ≈ the outro.
        self.assertFalse(coverage["full_coverage"])
        self.assertEqual(coverage["lhs_start_gap_s"], 0.0)
        self.assertEqual(coverage["rhs_start_gap_s"], 0.0)
        self.assertAlmostEqual(coverage["lhs_end_gap_s"], 3.0, delta=1.0)
        self.assertAlmostEqual(coverage["rhs_end_gap_s"], 3.0, delta=1.0)

    def test_pair_matcher_different_intro_and_outro(self):
        """Files share content but have BOTH different intros AND different outros."""
        file1_path, file2_path = self.edge_fixtures["diff_both"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        result = pair_matcher.create_segments_mapping()
        mappings = result.mapping
        # The shared bodies are related by a 1.03x linear speed change, so
        # the detected relation must be GLOBAL_LINEAR regardless of the
        # divergent intros/outros around them.
        self.assertEqual(result.relation, MappingRelation.GLOBAL_LINEAR)

        # LHS: gi3_wo3 (57.1s, 3s grass intro + 3s woman outro)
        # RHS: ai3d_swo3 (66.5s, 3s atoms intro + 3s seawaves outro)
        self.assertGreaterEqual(len(mappings), 3)
        # First pair NOT at edge — content starts at ~3s (after different intros)
        self.assertAlmostEqual(mappings[0][0], 3263, delta=500)
        self.assertAlmostEqual(mappings[0][1], 3208, delta=500)

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        # Both files have 3s intros and 3s outros — start/end gaps expected.
        self.assertFalse(coverage["full_coverage"])
        self.assertAlmostEqual(coverage["lhs_start_gap_s"], 3.0, delta=1.0)
        self.assertAlmostEqual(coverage["lhs_end_gap_s"], 3.0, delta=1.0)
        self.assertAlmostEqual(coverage["rhs_start_gap_s"], 3.0, delta=1.0)
        self.assertAlmostEqual(coverage["rhs_end_gap_s"], 3.0, delta=1.0)

    # ---- coverage_summary tests ----

    def test_coverage_summary_full_coverage(self):
        """Pairs touching both edges → full_coverage=True."""
        mappings = [(10, 5), (50000, 47500), (99920, 99920)]
        result = PairMatcher.coverage_summary(
            mappings, 100000, 100000, lhs_fps=25.0, rhs_fps=25.0,
        )
        self.assertTrue(result["full_coverage"])
        self.assertAlmostEqual(result["lhs_start_gap_s"], 0.01, places=3)
        self.assertAlmostEqual(result["rhs_start_gap_s"], 0.005, places=3)
        self.assertAlmostEqual(result["lhs_end_gap_s"], 0.08, places=3)
        self.assertAlmostEqual(result["rhs_end_gap_s"], 0.08, places=3)

    def test_coverage_summary_start_mismatch(self):
        """First pair far from start → full_coverage=False, start gaps reported."""
        mappings = [(3000, 5000), (60000, 58000)]
        result = PairMatcher.coverage_summary(
            mappings, 62000, 60000, lhs_fps=25.0, rhs_fps=25.0,
        )
        self.assertFalse(result["full_coverage"])
        self.assertAlmostEqual(result["lhs_start_gap_s"], 3.0, places=1)
        self.assertAlmostEqual(result["rhs_start_gap_s"], 5.0, places=1)
        # End gaps should be small
        self.assertAlmostEqual(result["lhs_end_gap_s"], 2.0, places=1)
        self.assertAlmostEqual(result["rhs_end_gap_s"], 2.0, places=1)

    def test_coverage_summary_end_mismatch(self):
        """Last pair far from end → full_coverage=False, end gaps reported."""
        mappings = [(20, 15), (55000, 53000)]
        result = PairMatcher.coverage_summary(
            mappings, 62000, 60000, lhs_fps=25.0, rhs_fps=25.0,
        )
        self.assertFalse(result["full_coverage"])
        self.assertAlmostEqual(result["lhs_end_gap_s"], 7.0, places=1)
        self.assertAlmostEqual(result["rhs_end_gap_s"], 7.0, places=1)

    def test_coverage_summary_with_real_no_speed(self):
        """Use actual no_speed fixture — should be full_coverage."""
        file1_path, file2_path = self.edge_fixtures["no_speed"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings = pair_matcher.create_segments_mapping().mapping

        d1 = video_utils.get_video_duration(file1_path)
        d2 = video_utils.get_video_duration(file2_path)
        result = PairMatcher.coverage_summary(
            mappings,
            d1,
            d2,
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        self.assertTrue(result["full_coverage"])

    def test_coverage_summary_with_real_diff_intro(self):
        """Use actual diff_intro_same fixture — should NOT be full_coverage (start gap)."""
        file1_path, file2_path = self.edge_fixtures["diff_intro_same"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings = pair_matcher.create_segments_mapping().mapping

        d1 = video_utils.get_video_duration(file1_path)
        d2 = video_utils.get_video_duration(file2_path)
        result = PairMatcher.coverage_summary(
            mappings,
            d1,
            d2,
            lhs_fps=pair_matcher.lhs_fps,
            rhs_fps=pair_matcher.rhs_fps,
        )
        # Both files have 3s intros. Start gaps must match intro duration.
        self.assertFalse(result["full_coverage"])
        self.assertAlmostEqual(result["lhs_start_gap_s"], 3.0, delta=1.0)
        self.assertAlmostEqual(result["rhs_start_gap_s"], 3.0, delta=1.0)
        self.assertLess(result["lhs_end_gap_s"], 0.1)
        self.assertLess(result["rhs_end_gap_s"], 0.1)
