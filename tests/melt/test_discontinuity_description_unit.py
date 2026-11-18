import unittest

from twotone.tools.melt.matching_analysis import describe_discontinuity_regions


class DiscontinuityDescriptionUnitTest(unittest.TestCase):
    def test_describes_shared_region_and_step_facts(self):
        lhs_frames = {
            40: {"frame_id": 10},
            80: {"frame_id": 11},
            120: {"frame_id": 12},
        }
        rhs_frames = {
            40: {"frame_id": 20},
            120: {"frame_id": 22},
        }
        group = [
            (40, 80, 40, 120, -40),
            (80, 120, 120, 120, 40),
        ]

        descriptions = describe_discontinuity_regions(
            [group],
            lhs_frames,
            rhs_frames,
            rhs_fps=25.0,
            compensation_tolerance_frames=2.0,
        )

        self.assertEqual(len(descriptions), 1)
        description = descriptions[0]
        self.assertEqual(description.group, tuple(group))
        self.assertEqual(description.lhs_range, (40, 120))
        self.assertEqual(description.rhs_range, (40, 120))
        self.assertEqual(description.lhs_frame_range, (10, 12))
        self.assertEqual(description.rhs_frame_range, (20, 22))
        self.assertEqual(
            description.step_frame_ranges,
            ((10, 11, 20, 22), (11, 12, 22, 22)),
        )
        self.assertEqual(description.frame_errors, (-1.0, 1.0))
        self.assertTrue(description.compensating)

    def test_preserves_compensation_tolerance_rule(self):
        frames = {
            0: {"frame_id": 0},
            100: {"frame_id": 1},
            200: {"frame_id": 2},
        }
        group = [
            (0, 100, 0, 100, -80),
            (100, 200, 100, 200, 40),
        ]

        description = describe_discontinuity_regions(
            [group],
            frames,
            frames,
            rhs_fps=25.0,
            compensation_tolerance_frames=0.5,
        )[0]

        self.assertEqual(description.frame_errors, (-2.0, 1.0))
        self.assertFalse(description.compensating)


if __name__ == "__main__":
    unittest.main()
