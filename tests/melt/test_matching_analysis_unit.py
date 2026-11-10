import unittest

from twotone.tools.melt import matching_analysis


class MatchingAnalysisUnitTest(unittest.TestCase):
    def test_content_discontinuity_flags_commercial_cut(self):
        mapping = [
            (lhs, round(lhs * 0.9864) - (7000 if lhs > 300000 else 0))
            for lhs in range(0, 600001, 10000)
        ]

        result = matching_analysis.find_content_discontinuities(mapping)

        self.assertEqual(result, [(300000, 310000, 295920, 298784, 7000)])

    def test_content_discontinuity_tolerates_speed_waviness(self):
        mapping = [(0, 0)]
        lhs = rhs = 0
        for index in range(60):
            lhs += 10000
            rhs += round(10000 * (1.03 if index % 2 else 0.97))
            mapping.append((lhs, rhs))

        self.assertEqual(matching_analysis.find_content_discontinuities(mapping), [])

    def test_content_discontinuity_threshold_scales_with_gap_length(self):
        base = [(index * 10000, index * 10000) for index in range(30)]
        last_lhs, last_rhs = base[-1]
        sparse = base + [(last_lhs + 160000, last_rhs + 157500)]
        dense = base + [(last_lhs + 20000, last_rhs + 17500)]

        self.assertEqual(matching_analysis.find_content_discontinuities(sparse), [])
        self.assertEqual(
            matching_analysis.find_content_discontinuities(dense),
            [(last_lhs, last_lhs + 20000, last_rhs, last_rhs + 17500, 2500)],
        )

    def test_coverage_summary_accepts_at_most_2_frames_per_edge(self):
        # The last frame starts one frame before the stream end, so an accepted
        # 2-frame boundary error appears as a 3-frame timestamp gap at the end.
        mappings = [(80, 40), (9880, 9940)]

        result = matching_analysis.coverage_summary(
            mappings, 10000, 10000, lhs_fps=25.0, rhs_fps=50.0,
        )

        self.assertTrue(result["full_coverage"])

    def test_coverage_summary_rejects_more_than_2_frames_at_an_edge(self):
        mappings = [(81, 40), (9920, 9960)]

        result = matching_analysis.coverage_summary(
            mappings, 10000, 10000, lhs_fps=25.0, rhs_fps=50.0,
        )

        self.assertFalse(result["full_coverage"])

    def test_coverage_summary_rejects_more_than_2_frames_at_the_end(self):
        mappings = [(80, 40), (9840, 9920)]

        result = matching_analysis.coverage_summary(
            mappings, 10000, 10000, lhs_fps=25.0, rhs_fps=50.0,
        )

        self.assertFalse(result["full_coverage"])
