import unittest
from unittest.mock import patch

from parameterized import parameterized

from melt.helpers import MeltComponentFactories
from twotone.tools.melt import audio_strategy
from twotone.tools.melt.audio_patcher import AudioStrategy
from twotone.tools.melt.melt_common import FrameSpan
from twotone.tools.melt.pair_matcher import MappingRelation, SegmentsMappingResult
from twotone.tools.utils import video_utils


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


class AudioStrategyDecisionTest(MeltComponentFactories, unittest.TestCase):
    """Strategy selection, tempo-line collapse and unscaled placement shift."""

    def test_unscaled_timeline_shift_accepts_whole_frame_offsets(self):
        # 5 dropped leading frames at 24 fps: deltas sit on the whole-frame grid.
        mapping = [(lhs, lhs - 208) for lhs in (500, 10000, 30000, 60000)]

        shift = audio_strategy.unscaled_timeline_shift_ms(mapping, 24.0)

        self.assertEqual(shift, 208)

    def test_unscaled_timeline_shift_rejects_off_grid_medians_as_noise(self):
        # Frame-rate-drift pairs are quantized to the coarser 23 fps grid; a
        # +21 ms median is phase noise, not a genuine 1-frame (42 ms) shift.
        mapping = [(lhs, lhs - 21) for lhs in (500, 10000, 30000, 60000)]

        shift = audio_strategy.unscaled_timeline_shift_ms(mapping, 24.0)

        self.assertEqual(shift, 0)

    def test_unscaled_timeline_shift_accepts_large_stable_cross_fps_offset(self):
        # Regression: a 29.97 fps base and a 23.976 fps source share the same
        # playback speed, but their independently quantized timestamps put the
        # real ~1.15 s content offset between base-frame grid positions.  The
        # offset stays stable across the whole film and must not be discarded
        # merely because its median is off the base frame grid.
        mapping = [
            (4371, 3212),
            (1_000_000, 998_850),
            (2_700_000, 2_698_851),
            (4_000_000, 3_998_858),
            (5_408_208, 5_407_068),
        ]

        shift = audio_strategy.unscaled_timeline_shift_ms(
            mapping,
            29.97,
            rhs_fps=24000 / 1001,
        )

        self.assertEqual(shift, 1149)

    def test_unscaled_timeline_shift_applies_stream_bias_correction(self):
        # A container start offset present in one side's frame times only
        # (mkv rebases to zero, mp4 keeps the offset) must cancel out instead
        # of being read as a genuine 12-frame shift.
        mapping = [(lhs, lhs + 495) for lhs in (0, 10000, 30000, 60000)]

        shift = audio_strategy.unscaled_timeline_shift_ms(mapping, 24.0, stream_bias_ms=500)

        self.assertEqual(shift, 0)

    def test_collapse_linear_mapping_sharpens_slope_with_metadata_tempo(self):
        aligner = self._make_source_aligner()
        # The matched endpoint carries a ~1 s low-entropy-zone error (observed
        # slope 1.0043 vs the real 1.0204); the effective-fps tempo agrees
        # with the pairs within tolerance, so it replaces the noisy slope and
        # the mapping collapses to one clean global time-scale.  The tempo is
        # derived from stream durations (mp4 duration, matroska DURATION tag)
        # and start offsets.
        mapping = [
            (0, 0),
            (63292, 63562),
        ]
        lhs_frames = {
            round(index * 63292 / 1497): {"path": f"lhs_{index}.jpg"}
            for index in range(1498)
        }
        rhs_frames = {
            round(index * 64030 / 1506): {"path": f"rhs_{index}.jpg"}
            for index in range(1507)
        }

        def fake_full_info(path, **_kwargs):
            if path == "/tmp/base.mp4":
                return {
                    "streams": [
                        {
                            "codec_type": "video",
                            "start_time": "0.500000",
                            "duration": "62.332000",
                            "nb_frames": "1496",
                        }
                    ]
                }
            return {
                "streams": [
                    {
                        "codec_type": "video",
                        "start_time": "0.510000",
                        "tags": {"DURATION": "00:01:04.581000000"},
                    }
                ]
            }

        with patch.object(video_utils, "get_video_full_info", side_effect=fake_full_info) as probe:
            tempo_ratio = aligner._effective_tempo_ratio(
                "/tmp/base.mp4",
                "/tmp/source.mkv",
                FrameSpan.of(lhs_frames),
                FrameSpan.of(rhs_frames),
            )
        self.assertEqual(probe.call_count, 2)

        result = audio_strategy.try_collapse_linear_mapping(
            mapping,
            FrameSpan.of(lhs_frames),
            tempo_ratio,
        )

        self.assertEqual(result, [(0, 0), (63292, 64583)])

    def test_collapse_linear_mapping_skips_partial_coverage(self):
        # Metadata agrees with the pairs, but the matched span covers well
        # under 80% of the base — projecting one global time-scale from a
        # small window is guesswork.
        mapping = [
            (5708, 4536),
            (46208, 44438),
        ]
        lhs_frames = {
            round(index * 63292 / 1497): {"path": f"lhs_{index}.jpg"}
            for index in range(1498)
        }

        result = audio_strategy.try_collapse_linear_mapping(
            mapping,
            FrameSpan.of(lhs_frames),
            tempo_ratio=0.9715,
        )

        self.assertIsNone(result)

    def test_frame_rate_only_drift_mapping_detected(self):
        aligner = self._make_source_aligner()
        # A 23<->24 fps resample preserving playback time: the effective fps
        # differ (tempo far from 1) and match the nominal ratio, while the
        # matched spans agree in wall-clock duration.  Such pairs must go the
        # UNSCALED route — audio_strategy.choose_strategy excludes them before any
        # linear-mapping recovery runs.
        mapping = [
            (739, 0),
            (16522, 16063),
            (26217, 25730),
            (53609, 53147),
            (62739, 62272),
            (62826, 62814),
        ]
        lhs_frames = {
            0: {"path": "lhs_0.jpg"},
            62826: {"path": "lhs_62826.jpg"},
        }
        rhs_frames = {
            0: {"path": "rhs_0.jpg"},
            62814: {"path": "rhs_62814.jpg"},
        }

        def fake_full_info(path, **_kwargs):
            if path == "/tmp/base.mkv":
                return {
                    "streams": [
                        {
                            "codec_type": "video",
                            "start_time": "0.000000",
                            "duration": "62.826000",
                            "nb_frames": "1446",
                        }
                    ]
                }
            return {
                "streams": [
                    {
                        "codec_type": "video",
                        "start_time": "0.500000",
                        "duration": "62.332000",
                        "nb_frames": "1496",
                    }
                ]
            }

        with patch.object(video_utils, "get_video_full_info", side_effect=fake_full_info):
            tempo_ratio = aligner._effective_tempo_ratio(
                "/tmp/base.mkv",
                "/tmp/source.mp4",
                FrameSpan.of(lhs_frames),
                FrameSpan.of(rhs_frames),
            )
        result = audio_strategy.is_frame_rate_only_drift(
            mapping,
            tempo_ratio,
            lhs_nominal_fps=23.0,
            rhs_nominal_fps=24.0,
        )

        self.assertTrue(result)

    def test_generic_smooth_timing_variation_uses_subsegment_audio(self):
        initial_pairs = [
            (i * 100000, i * 100000 - (600 if i == 9 else 0))
            for i in range(20)
        ]
        refined_pairs = sorted({
            *initial_pairs,
            *(
                (
                    timestamp,
                    timestamp - round(
                        (1 - abs(timestamp - 900000) / 100000) * 600 / 40
                    ) * 40,
                )
                for timestamp in range(805000, 1000000, 5000)
            ),
        })
        matching = SegmentsMappingResult(
            mapping=refined_pairs,
            lhs_all_frames={},
            rhs_all_frames={},
            relation=MappingRelation.GENERIC,
            lhs_fps=25.0,
            rhs_fps=25.0,
        )

        # No effective-fps tempo is known, so the mapping cannot count as a
        # pure frame-rate drift.
        strategy = audio_strategy.choose_strategy(matching, tempo_ratio=None)

        self.assertEqual(strategy, AudioStrategy.SUBSEGMENT)

    def test_linear_recovery_rejects_tmnt_piecewise_pairs(self):
        # A 23.976 fps base against a 30 fps resampled transfer with ad-break
        # cuts.  The pairs are piecewise (interior residuals of ~1.4 s against
        # the endpoint line), so no line may be recovered — and the fps tempo
        # 0.7992, contradicting the pairs' ~0.9665 span slope, must never be
        # applied: an fps-derived mapping would stretch the audio by 25%.
        mapping = [
            (1084, 0),
            (4755, 3633),
            (5130, 4000),
            (5589, 4467),
            (17851, 16800),
            (1250918, 1206600),
            (1253337, 1208967),
            (1256590, 1212200),
            (1315148, 1270067),
            (1335794, 1290000),
        ]
        lhs_frames = {
            0: {"path": "lhs_0.jpg"},
            1369994: {"path": "lhs_1369994.jpg"},
        }

        result = audio_strategy.try_collapse_linear_mapping(
            mapping,
            FrameSpan.of(lhs_frames),
            tempo_ratio=0.7992,
        )

        self.assertIsNone(result)

    def test_collapse_linear_mapping_keeps_pairs_when_metadata_contradicts(self):
        # A clean (cut-free) variant of the TMNT class: pairs perfectly linear
        # at 0.96651, while the effective-fps tempo says 0.7992 (resampled
        # transfer — the fps ratio is unrelated to the time mapping).  The
        # contradicting metadata must not rebuild the mapping.
        pairs = [
            (1084 + i * 300000, round(i * 300000 * 0.96651))
            for i in range(5)
        ]
        lhs_frames = {
            0: {"path": "lhs_0.jpg"},
            1369994: {"path": "lhs_1369994.jpg"},
        }

        result = audio_strategy.try_collapse_linear_mapping(
            pairs,
            FrameSpan.of(lhs_frames),
            tempo_ratio=0.7992,
        )

        self.assertIsNone(result)

    @parameterized.expand([
        ("below_positive", 1.000999, False),
        ("at_positive", 1.001, True),
        ("above_positive", 1.001001, True),
        ("below_negative", 0.999001, False),
        ("at_negative", 0.999, True),
        ("above_negative", 0.998999, True),
    ])
    def test_global_speed_classification_has_a_stable_ratio_threshold(
        self,
        _name: str,
        source_per_base: float,
        expected: bool,
    ):
        """The strategy-classification threshold must not depend on float artifacts."""
        self.assertEqual(audio_strategy.needs_fps_scaling(source_per_base), expected)


if __name__ == "__main__":
    unittest.main()
