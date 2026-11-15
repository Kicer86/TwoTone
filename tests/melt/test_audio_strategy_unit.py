import unittest

from twotone.tools.melt import audio_strategy
from twotone.tools.melt.audio_patcher import AudioStrategy
from twotone.tools.melt.pair_matcher import MappingRelation, SegmentsMappingResult


def _matching(
    mapping: list[tuple[int, int]],
    relation: MappingRelation,
    *,
    lhs_frames: dict | None = None,
    fps: float = 24.0,
) -> SegmentsMappingResult:
    return SegmentsMappingResult(
        mapping=mapping,
        lhs_all_frames=lhs_frames or {},
        rhs_all_frames={},
        relation=relation,
        lhs_fps=fps,
        rhs_fps=fps,
    )


class AudioStrategyPlanTest(unittest.TestCase):
    def test_unscaled_plan_keeps_raw_pairs_and_places_by_their_shift(self):
        # Same-speed global-linear relation: no time-scale, only a placement
        # shift.  The raw pairs (including the duplicate rhs observation) are
        # kept because they only derive that shift.
        mapping = [(500, 292), (10000, 9792), (10040, 9792), (60000, 59792)]

        plan = audio_strategy.plan_audio(
            _matching(mapping, MappingRelation.GLOBAL_LINEAR),
            tempo_ratio=None,
            stream_bias_ms=0,
        )

        self.assertEqual(plan.strategy, AudioStrategy.UNSCALED)
        self.assertEqual(plan.mapping, mapping)
        self.assertEqual(plan.unscaled_shift_ms, 208)
        self.assertEqual(plan.segment_count, 1)
        self.assertFalse(plan.collapsed_linear_mapping)

    def test_unscaled_plan_applies_stream_bias(self):
        mapping = [(lhs, lhs + 495) for lhs in (0, 10000, 30000, 60000)]

        plan = audio_strategy.plan_audio(
            _matching(mapping, MappingRelation.GLOBAL_LINEAR),
            tempo_ratio=None,
            stream_bias_ms=500,
        )

        self.assertEqual(plan.strategy, AudioStrategy.UNSCALED)
        self.assertEqual(plan.unscaled_shift_ms, 0)

    def test_global_time_scale_plan_uses_strict_mapping(self):
        # A real 0.96 speed change.  The duplicate rhs observation collapses
        # to its latest lhs before the trim ranges are derived.
        mapping = [(0, 0), (30000, 28800), (30040, 28800), (60000, 57600)]

        plan = audio_strategy.plan_audio(
            _matching(mapping, MappingRelation.GLOBAL_LINEAR),
            tempo_ratio=None,
            stream_bias_ms=0,
        )

        self.assertEqual(plan.strategy, AudioStrategy.GLOBAL_TIME_SCALE)
        self.assertEqual(plan.mapping, [(0, 0), (30040, 28800), (60000, 57600)])
        self.assertEqual(plan.unscaled_shift_ms, 0)
        self.assertEqual(plan.segment_count, 1)

    def test_subsegment_plan_collapses_tempo_confirmed_linear_mapping(self):
        # The pairs' noisy slope (1.0043) agrees with the 1.0204 metadata
        # tempo within tolerance and covers the whole base, so the mapping
        # becomes one line at the metadata tempo.
        mapping = [(0, 0), (63292, 63562)]

        plan = audio_strategy.plan_audio(
            _matching(
                mapping,
                MappingRelation.GENERIC,
                lhs_frames={0: {}, 63292: {}},
                fps=25.0,
            ),
            tempo_ratio=1.0204,
            stream_bias_ms=0,
        )

        self.assertEqual(plan.strategy, AudioStrategy.SUBSEGMENT)
        self.assertEqual(plan.mapping, [(0, 0), (63292, 64583)])
        self.assertTrue(plan.collapsed_linear_mapping)
        self.assertEqual(plan.segment_count, audio_strategy.SUBSEGMENT_COUNT)

    def test_subsegment_plan_keeps_pairs_without_tempo(self):
        mapping = [(0, 0), (30000, 30100), (30040, 30100), (60000, 60900)]

        plan = audio_strategy.plan_audio(
            _matching(mapping, MappingRelation.GENERIC, lhs_frames={0: {}, 60000: {}}),
            tempo_ratio=None,
            stream_bias_ms=0,
        )

        self.assertEqual(plan.strategy, AudioStrategy.SUBSEGMENT)
        self.assertEqual(plan.mapping, [(0, 0), (30040, 30100), (60000, 60900)])
        self.assertFalse(plan.collapsed_linear_mapping)


if __name__ == "__main__":
    unittest.main()
