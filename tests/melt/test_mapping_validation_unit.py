import logging
import unittest

from twotone.tools.melt import mapping_validation
from twotone.tools.melt.matching_analysis import Discontinuity
from twotone.tools.melt.pair_matcher import MappingRelation, SegmentsMappingResult


def _matching(**fields) -> SegmentsMappingResult:
    mapping = fields.pop("mapping", [(0, 0), (60000, 60000)])
    return SegmentsMappingResult(
        mapping=mapping,
        lhs_all_frames={},
        rhs_all_frames={},
        relation=fields.pop("relation", MappingRelation.GENERIC),
        lhs_fps=25.0,
        rhs_fps=25.0,
        **fields,
    )


GAP = Discontinuity(100000, 200000, 100000, 180000, -20000)
TIMING_STEP = Discontinuity(1600000, 1700000, 1600000, 1703000, -3000)


class MappingValidationTest(unittest.TestCase):
    def _validate(self, matching: SegmentsMappingResult, consumer: str = "audio") -> None:
        mapping_validation.validate_mapping(
            matching,
            consumer=consumer,
            lhs_id=1,
            rhs_id=2,
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

    def test_refuses_empty_mapping(self):
        with self.assertRaisesRegex(RuntimeError, "no shared video timeline mapping"):
            self._validate(_matching(mapping=[]))


if __name__ == "__main__":
    unittest.main()
