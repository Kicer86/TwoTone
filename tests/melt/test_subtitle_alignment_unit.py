import os
import tempfile
import unittest
from unittest.mock import patch

import pysubs2
from common import run_ffmpeg, write_srt_subtitle

from melt.helpers import MeltComponentFactories
from twotone.tools.melt.melt_common import FramesInfo, SubtitleStreamRef, VideoStreamRef
from twotone.tools.melt.pair_matcher import MappingRelation, SegmentsMappingResult
from twotone.tools.melt.source_aligner import AlignmentGroup


class SubtitleAlignmentUnitTest(MeltComponentFactories, unittest.TestCase):
    """Unit tests for retiming and patching mismatched subtitle tracks."""

    @staticmethod
    def _subtitle_matching(
        mapping: list[tuple[int, int]],
        relation: MappingRelation,
    ) -> SegmentsMappingResult:
        lhs_frames: FramesInfo = {
            lhs: {"frame_id": index, "path": None}
            for index, (lhs, _rhs) in enumerate(mapping)
        }
        rhs_frames: FramesInfo = {
            rhs: {"frame_id": index, "path": None}
            for index, (_lhs, rhs) in enumerate(mapping)
        }
        return SegmentsMappingResult(
            mapping,
            lhs_frames,
            rhs_frames,
            relation,
            25.0,
            25.0,
        )

    def test_retime_subtitle_content_applies_global_shift_and_scale(self):
        aligner = self._make_source_aligner()
        content = pysubs2.SSAFile()
        content.append(pysubs2.SSAEvent(start=2000, end=4000, text="Line"))
        matching = self._subtitle_matching(
            [(0, 1000), (10000, 9000)],
            MappingRelation.GLOBAL_LINEAR,
        )

        aligner.subtitle_alignment.retime_content(
            content,
            matching,
            source_mapping_to_video_ms=0,
            output_mapping_to_video_ms=500,
            source_description="subtitle track #2 from file #2",
        )

        self.assertEqual((content[0].start, content[0].end), (1750, 4250))

    def test_retime_subtitle_content_uses_piecewise_generic_mapping(self):
        aligner = self._make_source_aligner()
        content = pysubs2.SSAFile()
        content.append(pysubs2.SSAEvent(start=2000, end=7000, text="Line"))
        matching = self._subtitle_matching(
            [(0, 0), (5000, 4000), (10000, 10000)],
            MappingRelation.GENERIC,
        )

        aligner.subtitle_alignment.retime_content(
            content,
            matching,
            source_mapping_to_video_ms=0,
            output_mapping_to_video_ms=0,
            source_description="subtitle track #2 from file #2",
        )

        self.assertEqual((content[0].start, content[0].end), (2500, 7500))

    def test_retime_subtitle_content_rejects_event_outside_shared_content(self):
        aligner = self._make_source_aligner()
        content = pysubs2.SSAFile()
        content.append(pysubs2.SSAEvent(start=500, end=1500, text="Intro"))
        matching = self._subtitle_matching(
            [(0, 1000), (10000, 11000)],
            MappingRelation.GLOBAL_LINEAR,
        )

        with self.assertRaisesRegex(RuntimeError, "extends beyond verified shared content"):
            aligner.subtitle_alignment.retime_content(
                content,
                matching,
                source_mapping_to_video_ms=0,
                output_mapping_to_video_ms=0,
                source_description="subtitle track #2 from file #2",
            )

    def test_align_subtitle_rejects_image_based_track(self):
        aligner = self._make_source_aligner()
        base_path = "/tmp/base.mkv"
        source_path = "/tmp/source.mkv"
        subtitle = SubtitleStreamRef(source_path, 2, 2, "pol")
        details = {
            source_path: {
                "tracks": {
                    "subtitle": [
                        {"tid": 2, "format": "HDMV PGS", "forced": True},
                    ],
                },
            },
        }

        with self.assertRaisesRegex(RuntimeError, "image-based subtitle format"):
            aligner.align_subtitle(
                AlignmentGroup(
                    file_ids={base_path: 1, source_path: 2},
                    base_video=VideoStreamRef(base_path, 0, 0, None),
                    base_duration_ms=10000,
                    files_details=details,
                ),
                subtitle,
            )

    def test_align_subtitle_extracts_text_track_from_mp4(self):
        aligner = self._make_source_aligner()
        with tempfile.TemporaryDirectory() as tmpdir:
            subtitle_path = os.path.join(tmpdir, "source.srt")
            source_path = os.path.join(tmpdir, "source.mp4")
            write_srt_subtitle(subtitle_path, [(1000, 2000, "Line")])
            run_ffmpeg(
                [
                    "-y",
                    "-f", "lavfi",
                    "-i", "color=c=black:s=64x64:r=25:d=5",
                    "-i", subtitle_path,
                    "-map", "0:v:0",
                    "-map", "1:s:0",
                    "-c:v", "mpeg4",
                    "-c:s", "mov_text",
                    "-t", "5",
                    source_path,
                ],
                expected_path=source_path,
            )
            matching = self._subtitle_matching(
                [(0, 0), (5000, 5000)],
                MappingRelation.GLOBAL_LINEAR,
            )
            base_path = "/tmp/base.mkv"
            details = {
                source_path: {
                    "tracks": {
                        "video": [
                            {"tid": 0, "ffprobe_stream_index": 0},
                        ],
                        "subtitle": [
                            {"tid": 1, "format": "mov_text"},
                        ],
                    },
                },
            }

            group = AlignmentGroup(
                file_ids={base_path: 1, source_path: 2},
                base_video=VideoStreamRef(base_path, 0, 0, None),
                base_duration_ms=5000,
                files_details=details,
            )
            group.aligned_sources[source_path] = matching
            with patch.object(aligner, "_mapping_to_video_ms", return_value=0):
                output_stream, _metadata = aligner.align_subtitle(
                    group,
                    SubtitleStreamRef(source_path, 1, 1, "pol"),
                )

            content = pysubs2.load(output_stream.path)
            self.assertEqual(len(content), 1)
            self.assertEqual(content[0].text, "Line")
            self.assertEqual((content[0].start, content[0].end), (1000, 2000))


if __name__ == "__main__":
    unittest.main()
