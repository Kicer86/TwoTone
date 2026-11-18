import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from parameterized import parameterized

from twotone.tools.melt import debug_routines
from twotone.tools.melt.debug_routines import (
    DebugRoutines,
    unresolved_content_discontinuities_error,
)


class DebugRoutinesUnitTest(unittest.TestCase):
    def test_summary_logging_skips_calculation_above_debug(self):
        logger = logging.getLogger("test.debug_routines.info_summaries")
        logger.setLevel(logging.INFO)

        with patch.object(debug_routines, "summarize_pairs") as pairs_summary:
            debug_routines.log_pairs_summary(
                logger, MagicMock(), {}, {}, "Pairs: ", [(0, 0)],
            )

        pairs_summary.assert_not_called()

    def test_summary_logging_preserves_debug_prefix_and_verbose_detail(self):
        logger = logging.getLogger("test.debug_routines.debug_summaries")
        logger.setLevel(logging.DEBUG)
        phash = MagicMock()
        pairs = [(0, 0)]
        lhs = {0: {"path": "/lhs.png", "frame_id": 0}}
        rhs = {0: {"path": "/rhs.png", "frame_id": 0}}

        with patch.object(
            debug_routines, "summarize_pairs", return_value="rendered pairs",
        ) as pairs_summary, self.assertLogs(logger, logging.DEBUG) as captured:
            debug_routines.log_pairs_summary(
                logger, phash, lhs, rhs, "Initial candidates:        ", pairs,
                verbose=True,
            )

        self.assertEqual(
            [record.getMessage() for record in captured.records],
            ["Initial candidates:        rendered pairs"],
        )
        pairs_summary.assert_called_once_with(
            phash, pairs, lhs, rhs, verbose=True,
        )

    def test_content_error_uses_same_region_facts_as_debug_details(self):
        lhs_frames = {
            40: {"frame_id": 10},
            80: {"frame_id": 11},
            120: {"frame_id": 12},
        }
        rhs_frames = {
            40: {"frame_id": 9},
            120: {"frame_id": 11},
        }
        groups = [[
            (40, 80, 40, 120, -40),
            (80, 120, 120, 120, 40),
        ]]
        logger = MagicMock()

        error = unresolved_content_discontinuities_error(
            logger,
            groups,
            lhs_frames,
            rhs_frames,
            lhs_label="#2",
            rhs_label="#1",
            rhs_fps=25.0,
            compensation_tolerance_frames=2.0,
            suggested_temporary_offsets=[],
            debug_path=None,
        )

        summary = logger.error.call_args_list[0].args
        self.assertEqual(summary[6:8], (10, 12))
        self.assertEqual(summary[10:12], (9, 11))
        step = logger.debug.call_args_list[0].args
        self.assertEqual(step[2:4], (10, 11))
        self.assertEqual(step[6:8], (9, 11))
        self.assertEqual(step[-2:], (-40, -1.0))
        self.assertIn("1 unresolved region", error)

    def test_dump_content_discontinuities_keeps_every_frame_and_writes_details(self):
        with tempfile.TemporaryDirectory() as root:
            source_dir = os.path.join(root, "source")
            debug_dir = os.path.join(root, "debug")
            os.makedirs(source_dir)
            os.makedirs(debug_dir)

            def frames(side: str, timestamps: list[int]) -> dict[int, dict]:
                result = {}
                for frame_id, timestamp in enumerate(timestamps):
                    path = os.path.join(source_dir, f"{side}_{frame_id}.png")
                    with open(path, "wb"):
                        pass
                    result[timestamp] = {"frame_id": frame_id, "path": path}
                return result

            lhs_frames = frames("lhs", [0, 40, 80, 120, 160, 200, 240])
            rhs_frames = frames("rhs", [0, 40, 80, 120, 160, 200, 240, 280])
            debug = DebugRoutines(debug_dir, lhs_frames, rhs_frames)
            prepared = []
            groups = [
                [
                    (40, 80, 40, 120, -40),
                    (80, 120, 120, 120, 40),
                ],
                [
                    (160, 200, 160, 240, -40),
                    (200, 240, 240, 280, 40),
                ],
            ]

            target_dir = debug.dump_content_discontinuities(
                groups,
                lhs_label="#2",
                rhs_label="#1",
                lhs_fps=25.0,
                rhs_fps=25.0,
                prepare_frames=lambda: prepared.append(True),
            )

            self.assertEqual(prepared, [True])
            self.assertIsNotNone(target_dir)
            assert target_dir is not None
            self.assertEqual(os.path.basename(target_dir), "#0 content discontinuity error")
            with open(os.path.join(target_dir, "details.txt"), encoding="utf-8") as details_file:
                details = details_file.read()
            self.assertIn("compensating adjacent frame-step errors", details)
            self.assertIn(
                "#2: 00:00:00,040-00:00:00,120 (40-120 ms); "
                "decoded frames 1 -> 3",
                details,
            )
            self.assertIn(
                "#1: 00:00:00,040-00:00:00,120 (40-120 ms); "
                "decoded frames 1 -> 3",
                details,
            )
            self.assertIn("Region 1: compensating adjacent frame-step errors", details)
            self.assertIn(
                "#2: 00:00:00,160-00:00:00,240 (160-240 ms); "
                "decoded frames 4 -> 6",
                details,
            )
            self.assertIn(
                "#1: 00:00:00,160-00:00:00,280 (160-280 ms); "
                "decoded frames 4 -> 7",
                details,
            )

            lhs_dump = os.listdir(os.path.join(target_dir, "region_000", "lhs"))
            rhs_dump = os.listdir(os.path.join(target_dir, "region_000", "rhs"))
            self.assertEqual(len(lhs_dump), 3)
            self.assertEqual(len(rhs_dump), 3)
            self.assertTrue(all(os.path.isfile(
                os.path.join(target_dir, "region_000", "lhs", name)
            ) for name in lhs_dump))
            self.assertEqual(
                len(os.listdir(os.path.join(target_dir, "region_001", "lhs"))),
                3,
            )
            self.assertEqual(
                len(os.listdir(os.path.join(target_dir, "region_001", "rhs"))),
                4,
            )

    def test_dump_content_discontinuities_uses_error_compensation_tolerance(self):
        # Steps of -1 and +4 frames leave a 3-frame residue: beyond the
        # 2-frame tolerance the user-facing error applies, so the dump must
        # not describe the region as compensating either.
        with tempfile.TemporaryDirectory() as root:
            source_dir = os.path.join(root, "source")
            debug_dir = os.path.join(root, "debug")
            os.makedirs(source_dir)
            os.makedirs(debug_dir)

            def frames(side: str, timestamps: list[int]) -> dict[int, dict]:
                result = {}
                for frame_id, timestamp in enumerate(timestamps):
                    path = os.path.join(source_dir, f"{side}_{frame_id}.png")
                    with open(path, "wb"):
                        pass
                    result[timestamp] = {"frame_id": frame_id, "path": path}
                return result

            lhs_frames = frames("lhs", [0, 40, 80, 120])
            rhs_frames = frames("rhs", [0, 40, 80, 120, 160, 200, 240])
            debug = DebugRoutines(debug_dir, lhs_frames, rhs_frames)
            groups = [[
                (40, 80, 40, 80, -40),
                (80, 120, 80, 240, 160),
            ]]

            target_dir = debug.dump_content_discontinuities(
                groups,
                lhs_label="#2",
                rhs_label="#1",
                lhs_fps=25.0,
                rhs_fps=25.0,
                prepare_frames=lambda: None,
            )

            assert target_dir is not None
            with open(os.path.join(target_dir, "details.txt"), encoding="utf-8") as details_file:
                details = details_file.read()
            self.assertIn("Region 0: unresolved frame-step error", details)
            self.assertNotIn("Region 0: compensating", details)

    def test_debug_frame_uses_hardlink_when_symlinks_are_unavailable(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "source.png")
            target = os.path.join(root, "target.png")
            with open(source, "wb") as source_file:
                source_file.write(b"frame")

            with patch.object(os, "symlink", side_effect=OSError("not permitted")):
                DebugRoutines._link_file(source, target)

            self.assertTrue(os.path.isfile(target))
            self.assertTrue(os.path.samefile(source, target))

    def test_dump_pairs_skips_frames_whose_image_is_gone(self):
        # Matched frames may point into a matcher working directory that was
        # already removed; without symlinks the copy fallback would fail.
        with tempfile.TemporaryDirectory() as root:
            kept = os.path.join(root, "kept.png")
            with open(kept, "wb") as kept_file:
                kept_file.write(b"frame")
            routines = DebugRoutines(
                root,
                {0: {"frame_id": 0, "path": kept}, 40: {"frame_id": 1, "path": os.path.join(root, "gone.png")}},
                {},
            )

            with patch.object(os, "symlink", side_effect=OSError("not permitted")):
                routines.dump_pairs([(0, 40, 0, 40)])

            dumped = os.listdir(os.path.join(root, "#0 subsegments"))
            self.assertEqual(dumped, ["000000_lhs_b_00000000"])

    def test_dump_timing_discontinuities_keeps_every_frame_and_writes_details(self):
        with tempfile.TemporaryDirectory() as root:
            source_dir = os.path.join(root, "source")
            debug_dir = os.path.join(root, "debug")
            os.makedirs(source_dir)
            os.makedirs(debug_dir)

            def frames(side: str, timestamps: list[int]) -> dict[int, dict]:
                result = {}
                for frame_id, timestamp in enumerate(timestamps):
                    path = os.path.join(source_dir, f"{side}_{frame_id}.png")
                    with open(path, "wb"):
                        pass
                    result[timestamp] = {"frame_id": frame_id, "path": path}
                return result

            lhs_frames = frames("lhs", [0, 40, 80, 120, 160, 200, 240])
            rhs_frames = frames("rhs", [0, 40, 80, 120, 160, 200, 240, 280])
            debug = DebugRoutines(debug_dir, lhs_frames, rhs_frames)
            prepared = []

            target_dir = debug.dump_timing_discontinuities(
                [
                    [(40, 120, 40, 160, -40)],
                    [(200, 240, 240, 280, 40)],
                ],
                lhs_label="#2",
                rhs_label="#1",
                lhs_fps=25.0,
                rhs_fps=25.0,
                prepare_frames=lambda: prepared.append(True),
            )

            self.assertEqual(prepared, [True])
            self.assertIsNotNone(target_dir)
            assert target_dir is not None
            self.assertEqual(os.path.basename(target_dir), "#0 timing discontinuity error")
            with open(os.path.join(target_dir, "details.txt"), encoding="utf-8") as details_file:
                details = details_file.read()
            self.assertIn("Timing discontinuity diagnostics", details)
            self.assertIn("Region 0: unresolved matched-timeline step", details)
            self.assertIn(
                "#2: 00:00:00,040-00:00:00,120 (40-120 ms); "
                "decoded frames 1 -> 3",
                details,
            )
            self.assertEqual(
                len(os.listdir(os.path.join(target_dir, "region_000", "lhs"))),
                3,
            )
            self.assertEqual(
                len(os.listdir(os.path.join(target_dir, "region_000", "rhs"))),
                4,
            )
            self.assertIn("Region 1: unresolved matched-timeline step", details)
            self.assertEqual(
                len(os.listdir(os.path.join(target_dir, "region_001", "lhs"))),
                2,
            )
            self.assertEqual(
                len(os.listdir(os.path.join(target_dir, "region_001", "rhs"))),
                2,
            )

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
        self.assertEqual(debug_routines.format_time(seconds), expected)

    def test_render_overlap_diagram_both_gaps(self):
        """Diagram with gaps: lhs starts at 2s, rhs starts at 0s → lhs indented, rhs at col 0."""
        # rhs_start_gap_s=0 means rhs starts at beginning of shared content
        # lhs_start_gap_s=2 means lhs starts 2s before shared content → lhs at col 0
        summary = {
            "lhs_start_gap_s": 2.0,
            "rhs_start_gap_s": 0.0,
            "lhs_end_gap_s": 0.0,
            "rhs_end_gap_s": 3.0,
            "full_coverage": False,
        }
        lines = debug_routines.render_overlap_diagram(1, 2, 100_000, 101_000, summary)
        self.assertGreaterEqual(len(lines), 2, "Should have at least 2 bar lines")
        # lhs has start_gap=2 → shared content starts 2s into lhs → lhs starts first
        self.assertTrue(lines[0].startswith("|"), "#1 bar starts at col 0")
        # rhs starts at shared content → rhs is indented
        self.assertTrue(lines[1].startswith(" "), "#2 bar should be indented")
        self.assertIn("#1", lines[0])
        self.assertIn("#2", lines[1])

    def test_render_overlap_diagram_rhs_starts_first(self):
        """When rhs has the initial gap → rhs starts later in the diagram, lhs at col 0."""
        # lhs_start_gap=0 means shared content starts at beginning of lhs
        # rhs_start_gap=3 means shared content starts 3s into rhs → rhs starts earlier
        summary = {
            "lhs_start_gap_s": 0.0,
            "rhs_start_gap_s": 3.0,
            "lhs_end_gap_s": 3.0,
            "rhs_end_gap_s": 0.0,
            "full_coverage": False,
        }
        lines = debug_routines.render_overlap_diagram(1, 2, 100_000, 100_000, summary)
        self.assertGreaterEqual(len(lines), 2)
        # rhs_s = lhs_sg - rhs_sg = 0 - 3 = -3 → rhs starts before lhs
        # So rhs is at col 0, lhs is indented
        self.assertTrue(lines[0].startswith(" "), "#1 bar should be indented")
        self.assertTrue(lines[1].startswith("|"), "#2 bar starts at col 0")

    def test_render_overlap_diagram_no_gaps_returns_empty(self):
        """When gaps are tiny (< threshold), no diagram should be produced."""
        summary = {
            "lhs_start_gap_s": 0.0,
            "rhs_start_gap_s": 0.0,
            "lhs_end_gap_s": 0.0,
            "rhs_end_gap_s": 0.0,
            "full_coverage": False,
        }
        lines = debug_routines.render_overlap_diagram(1, 2, 100_000, 100_000, summary)
        # Bars should be at the same columns (no offset)
        if lines:
            lhs_start = len(lines[0]) - len(lines[0].lstrip())
            rhs_start = len(lines[1]) - len(lines[1].lstrip())
            self.assertEqual(lhs_start, rhs_start, "Both bars should start at same column")

    def test_render_overlap_diagram_has_timestamps(self):
        """Diagram should contain time labels."""
        summary = {
            "lhs_start_gap_s": 0.0,
            "rhs_start_gap_s": 5.0,
            "lhs_end_gap_s": 0.0,
            "rhs_end_gap_s": 0.0,
            "full_coverage": False,
        }
        lines = debug_routines.render_overlap_diagram(1, 2, 120_000, 115_000, summary)
        all_text = "\n".join(lines)
        # Should have at least "0.0s" somewhere in the timestamp rows
        self.assertIn("0.0s", all_text, "Should contain the start timestamp")

    def test_render_overlap_diagram_speed_ratio_projected(self):
        """When rhs plays faster (PAL), rhs should be projected onto lhs timeline.

        With lhs=100s, rhs=96s, lhs_sg=2, rhs_sg=0, lhs_eg=1, rhs_eg=0:
        shared_lhs=97, shared_rhs=96, speed=97/96≈1.0104.
        rhs projected: start=2.0, end=99.0 (=100-1). End gap on diagram ≈1s, not 4s.
        """
        summary = {
            "lhs_start_gap_s": 2.0,
            "rhs_start_gap_s": 0.0,
            "lhs_end_gap_s": 1.0,
            "rhs_end_gap_s": 0.0,
            "full_coverage": False,
        }
        lines = debug_routines.render_overlap_diagram(1, 2, 100_000, 96_000, summary)
        self.assertGreaterEqual(len(lines), 2)
        # lhs starts first (lhs_sg=2 means has content before shared region)
        self.assertTrue(lines[0].startswith("|"), "#1 bar at col 0")
        self.assertTrue(lines[1].startswith(" "), "#2 bar indented")
        # Both should end close (1s gap gets _MIN_GAP=6 columns, but NOT a proportional 4s gap)
        lhs_end_col = len(lines[0].rstrip()) - 1
        rhs_end_col = len(lines[1].rstrip()) - 1
        self.assertAlmostEqual(lhs_end_col, rhs_end_col, delta=8,
                               msg="Speed-adjusted rhs should end close to lhs (small gap)")


if __name__ == "__main__":
    unittest.main()
