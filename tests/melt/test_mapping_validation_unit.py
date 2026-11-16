import logging
import os
import unittest

from twotone.tools.melt import mapping_validation, matching_analysis
from twotone.tools.melt.matching_analysis import Discontinuity
from twotone.tools.melt.pair_matcher import MappingRelation, SegmentsMappingResult


def _frames(timestamps: list[int], frame_ids: list[int]) -> dict[int, dict]:
    return {
        timestamp: {"frame_id": frame_id, "path": None}
        for timestamp, frame_id in zip(timestamps, frame_ids)
    }


def _matching(**fields) -> SegmentsMappingResult:
    mapping = fields.pop("mapping", [(0, 0), (60000, 60000)])
    return SegmentsMappingResult(
        mapping=mapping,
        lhs_all_frames=fields.pop("lhs_all_frames", {}),
        rhs_all_frames=fields.pop("rhs_all_frames", {}),
        relation=fields.pop("relation", MappingRelation.GENERIC),
        lhs_fps=25.0,
        rhs_fps=25.0,
        **fields,
    )


GAP = Discontinuity(100000, 200000, 100000, 180000, -20000)
TIMING_STEP = Discontinuity(1600000, 1700000, 1600000, 1703000, -3000)


class MappingValidationTest(unittest.TestCase):
    def _validate(
        self,
        matching: SegmentsMappingResult,
        consumer: str = "audio",
        *,
        lhs_id: int = 1,
        rhs_id: int = 2,
        max_temporary_frame_offset: float = mapping_validation.DEFAULT_MAX_TEMPORARY_FRAME_OFFSET,
    ) -> None:
        mapping_validation.validate_mapping(
            matching,
            consumer=consumer,
            lhs_id=lhs_id,
            rhs_id=rhs_id,
            max_temporary_frame_offset=max_temporary_frame_offset,
            logger=logging.getLogger("test.mapping_validation"),
        )

    def test_accepts_continuous_mapping(self):
        self._validate(_matching())

    def test_refuses_generic_content_gap(self):
        with self.assertLogs("test.mapping_validation", level="ERROR"), \
             self.assertRaisesRegex(
                 RuntimeError,
                 "holes in the scene sequence.*patching subtitle across",
             ):
            self._validate(
                _matching(generic_content_discontinuities=(GAP,)),
                consumer="subtitle",
            )

    def test_timing_step_is_reported_before_content_gaps(self):
        with self.assertLogs("test.mapping_validation", level="ERROR"), \
             self.assertRaisesRegex(
                 RuntimeError,
                 "Unsupported audio timeline discontinuity.*PTS step",
             ):
            self._validate(_matching(
                timing_discontinuities=(TIMING_STEP,),
                generic_content_discontinuities=(GAP,),
            ))

    def test_temporary_frame_offset_limit_allows_only_compensated_excursion(self):
        lhs_timestamps = [0, 1000, 2000, 3000]
        rhs_timestamps = [0, 1000, 1800, 3000]
        mapping = list(zip(lhs_timestamps, rhs_timestamps))
        lhs_frames = _frames(lhs_timestamps, [0, 25, 50, 75])
        rhs_frames = _frames(rhs_timestamps, [0, 25, 45, 75])
        gaps = matching_analysis.find_global_linear_content_discontinuities(
            mapping, lhs_frames, rhs_frames,
            frame_slope=1.0, lhs_fps=25.0, rhs_fps=25.0,
        )
        self.assertEqual([gap[4] for gap in gaps], [200, -200])

        def matching(content: list[Discontinuity]) -> SegmentsMappingResult:
            return _matching(
                mapping=mapping,
                lhs_all_frames=lhs_frames,
                rhs_all_frames=rhs_frames,
                relation=MappingRelation.GLOBAL_LINEAR,
                content_discontinuities=tuple(content),
            )

        with self.assertLogs("test.mapping_validation", level="WARNING") as captured:
            self._validate(matching(gaps), max_temporary_frame_offset=5.0)
        self.assertTrue(any(
            "Accepting temporary 5-frame alignment offset" in message
            and "00:00:02,000 (2000 ms)" in message
            and "00:00:01,800 (1800 ms)" in message
            for message in captured.output
        ))

        # One step alone does not return to the outer offset.
        with self.assertLogs("test.mapping_validation", level="ERROR"), \
             self.assertRaisesRegex(RuntimeError, "local frame-sequence difference"):
            self._validate(matching(gaps[:1]), max_temporary_frame_offset=5.0)

    def test_default_temporary_frame_offset_limit_rejects_three_frames(self):
        lhs_timestamps = [0, 1000, 2000, 3000]
        rhs_timestamps = [0, 1000, 1880, 3000]
        mapping = list(zip(lhs_timestamps, rhs_timestamps))
        lhs_frames = _frames(lhs_timestamps, [0, 25, 50, 75])
        rhs_frames = _frames(rhs_timestamps, [0, 25, 47, 75])
        gaps = matching_analysis.find_global_linear_content_discontinuities(
            mapping, lhs_frames, rhs_frames,
            frame_slope=1.0, lhs_fps=25.0, rhs_fps=25.0,
        )
        self.assertEqual([gap[4] for gap in gaps], [120, -120])

        with self.assertLogs("test.mapping_validation", level="ERROR"), \
             self.assertRaises(RuntimeError) as raised:
            self._validate(_matching(
                mapping=mapping,
                lhs_all_frames=lhs_frames,
                rhs_all_frames=rhs_frames,
                relation=MappingRelation.GLOBAL_LINEAR,
                content_discontinuities=tuple(gaps),
            ))
        self.assertIn("local frame-sequence difference", str(raised.exception))
        self.assertIn("--max-temporary-frame-offset 3", str(raised.exception))

    def test_compensating_discontinuity_reports_outer_offset_and_inner_shift(self):
        lhs_frames = {
            2473639: {"frame_id": 59307},
            2476726: {"frame_id": 59381},
            2485901: {"frame_id": 59601},
        }
        rhs_frames = {
            2473558: {"frame_id": 59306},
            2476853: {"frame_id": 59385},
            2485821: {"frame_id": 59600},
        }
        gaps = (
            Discontinuity(2473639, 2476726, 2473558, 2476853, -209),
            Discontinuity(2476726, 2485901, 2476853, 2485821, 209),
        )

        with self.assertLogs("test.mapping_validation", level="DEBUG") as captured, \
             self.assertRaises(RuntimeError) as raised:
            self._validate(
                _matching(
                    lhs_all_frames=lhs_frames,
                    rhs_all_frames=rhs_frames,
                    relation=MappingRelation.GLOBAL_LINEAR,
                    content_discontinuities=gaps,
                    content_discontinuity_debug_path="/tmp/debug",
                ),
                lhs_id=2,
                rhs_id=1,
            )

        summary = next(
            message for message in captured.output
            if "Local frame-sequence difference" in message
        )
        self.assertIn(
            "first and last matched frames use the same 1-frame offset "
            "(#2 = #1 + 1)",
            summary,
        )
        self.assertIn("inside this range the match shifts by 5 frames", summary)
        self.assertIn(
            "#2: frames 59307-59601; "
            "00:41:13,639-00:41:25,901 (2473639-2485901 ms)",
            summary,
        )
        self.assertIn(
            "#1: frames 59306-59600; "
            "00:41:13,558-00:41:25,821 (2473558-2485821 ms)",
            summary,
        )
        details_path = os.path.join("/tmp/debug", "details.txt")
        self.assertTrue(any(
            f"Full frame diagnostics: {details_path}" in message
            for message in captured.output
        ))
        self.assertIn("Inspect the reported region", str(raised.exception))
        self.assertIn(details_path, str(raised.exception))
        self.assertIn(
            "--max-temporary-frame-offset 5",
            str(raised.exception),
        )
        self.assertIn("report a PairMatcher bug", str(raised.exception))
        self.assertIn("include /tmp/debug", str(raised.exception))

    def test_permanent_discontinuity_does_not_suggest_temporary_offset_option(self):
        with self.assertLogs("test.mapping_validation", level="DEBUG"), \
             self.assertRaises(RuntimeError) as raised:
            self._validate(_matching(
                lhs_all_frames={0: {"frame_id": 0}, 1000: {"frame_id": 25}},
                rhs_all_frames={0: {"frame_id": 0}, 1200: {"frame_id": 30}},
                relation=MappingRelation.GLOBAL_LINEAR,
                content_discontinuities=(Discontinuity(0, 1000, 0, 1200, -200),),
                content_discontinuity_debug_path="/tmp/debug",
            ))

        message = str(raised.exception)
        self.assertIn("Inspect the reported region", message)
        self.assertIn("report a PairMatcher bug", message)
        self.assertNotIn("--max-temporary-frame-offset", message)

    def test_frame_exact_content_gap_is_reported_before_timing_step(self):
        with self.assertLogs("test.mapping_validation", level="ERROR"), \
             self.assertRaisesRegex(RuntimeError, "local frame-sequence difference"):
            self._validate(_matching(
                lhs_all_frames={0: {"frame_id": 0}, 1000: {"frame_id": 25}},
                rhs_all_frames={0: {"frame_id": 0}, 1200: {"frame_id": 30}},
                relation=MappingRelation.GLOBAL_LINEAR,
                content_discontinuities=(Discontinuity(0, 1000, 0, 1200, -200),),
                timing_discontinuities=(TIMING_STEP,),
            ))

    def test_refuses_empty_mapping(self):
        with self.assertRaisesRegex(RuntimeError, "no shared video timeline mapping"):
            self._validate(_matching(mapping=[]))


if __name__ == "__main__":
    unittest.main()
