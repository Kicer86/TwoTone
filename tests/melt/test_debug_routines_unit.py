import os
import tempfile
import unittest
from unittest.mock import patch

from twotone.tools.melt.debug_routines import DebugRoutines


class DebugRoutinesUnitTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
