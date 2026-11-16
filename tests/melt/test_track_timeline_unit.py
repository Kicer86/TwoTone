import os
import unittest
from unittest.mock import patch

from parameterized import parameterized

from melt.helpers import MeltComponentFactories, _FAKE_PROCESS_OK
from twotone.tools.melt.melt_common import AudioStreamRef
from twotone.tools.utils import process_utils, video_utils


class TrackTimelineUnitTest(MeltComponentFactories, unittest.TestCase):
    """Unit tests for TrackTimeline stream offsets and AAC-priming handling."""

    @parameterized.expand([
        ("absorbed", "0.500000", False),
        ("exposed", "0.479000", True),
    ])
    def test_aac_trim_window_is_anchored_to_content_start(
        self,
        _name,
        stream_start,
        priming_exposed,
    ):
        timeline = self._make_track_timeline()
        timeline._input_file_ids = {"/tmp/source.mkv": 1}
        calls = []
        source_info = {
            "streams": [
                {"codec_type": "video", "index": 0, "start_time": "0.000000"},
                {
                    "codec_type": "audio",
                    "index": 1,
                    "codec_name": "aac",
                    "initial_padding": 1024,
                    "sample_rate": "48000",
                    "start_time": stream_start,
                },
            ]
        }

        def fake_start_process(tool, args, **kwargs):
            calls.append((tool, list(args), kwargs))
            return _FAKE_PROCESS_OK

        with patch.object(video_utils, "get_video_full_info", return_value=source_info), \
             patch.object(timeline, "aac_priming_exposed", return_value=priming_exposed), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error", lambda result: None), \
             patch.object(os.path, "exists", return_value=False):
            timeline.decode_audio_stream_to_flac(
                AudioStreamRef("/tmp/source.mkv", 1, 1, None),
                "/tmp/output.flac",
                trim_start_ms=500,
                trim_end_ms=63250,
            )

        decode_args = calls[1][1]
        trim_filter = decode_args[decode_args.index("-filter:a") + 1]
        self.assertEqual(
            trim_filter,
            "asetpts=PTS-STARTPTS,atrim=start=0.021333:end=62.771333,asetpts=PTS-STARTPTS",
        )
        self.assertEqual(
            [call[2].get("progress_description") for call in calls],
            [
                "Extracting AAC bitstream from file #1, audio track #1",
                "Decoding file #1, audio track #1: AAC -> FLAC",
            ],
        )

    def test_non_primed_trim_window_is_anchored_to_content_start(self):
        """MP4/MOV decode rebases positive frame PTS before and after ``atrim``."""
        timeline = self._make_track_timeline()
        calls = []
        source_info = {
            "streams": [{
                "codec_type": "audio",
                "index": 1,
                "codec_name": "aac",
                "sample_rate": "48000",
                "start_time": "0.478000",
            }],
        }

        def fake_start_process(tool, args, **_kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        with patch.object(video_utils, "get_video_full_info", return_value=source_info), \
             patch.object(timeline, "aac_priming_exposed", return_value=False), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error", lambda result: None):
            timeline.decode_audio_stream_to_flac(
                AudioStreamRef("/tmp/source.mp4", 1, 1, None),
                "/tmp/output.flac",
                trim_start_ms=541,
                trim_end_ms=62791,
            )

        decode_args = calls[0][1]
        trim_filter = decode_args[decode_args.index("-filter:a") + 1]
        self.assertEqual(
            trim_filter,
            "asetpts=PTS-STARTPTS,atrim=start=0.063000:end=62.313000,asetpts=PTS-STARTPTS",
        )

    def test_full_selected_audio_decode_is_rebased_and_normalized(self):
        """A whole selected stream uses one zero-based, normalized FLAC timeline."""
        timeline = self._make_track_timeline()
        calls = []
        source_info = {
            "streams": [
                {"codec_type": "audio", "index": 2, "codec_name": "flac", "sample_rate": "44100"},
            ],
        }

        def fake_start_process(tool, args, **_kwargs):
            calls.append((tool, list(args)))
            return _FAKE_PROCESS_OK

        with patch.object(video_utils, "get_video_full_info", return_value=source_info), \
             patch.object(process_utils, "start_process", side_effect=fake_start_process), \
             patch.object(process_utils, "raise_on_error", lambda result: None):
            timeline.decode_audio_stream_to_flac(
                AudioStreamRef("/tmp/source.mkv", 2, 2, "pol"),
                "/tmp/output.flac",
                normalize_to=(1, 48000, "s32"),
            )

        decode_args = calls[-1][1]
        self.assertEqual(decode_args[decode_args.index("-map") + 1], "0:2")
        self.assertEqual(decode_args[decode_args.index("-filter:a") + 1], "asetpts=PTS-STARTPTS")
        self.assertEqual(decode_args[decode_args.index("-ac") + 1], "1")
        self.assertEqual(decode_args[decode_args.index("-ar") + 1], "48000")

    def test_mkvmerge_input_start_offset_preserves_matroska_stream_start(self):
        timeline = self._make_track_timeline()
        with patch.object(
            video_utils,
            "get_video_full_info",
            return_value={"streams": [{"codec_type": "video", "index": 0, "start_time": "0.510000"}]},
        ):
            offset = timeline.mkvmerge_input_start_offset_ms("/tmp/base.mkv", "video", 0)

        self.assertEqual(offset, 510)

    def test_mkvmerge_input_start_offset_resets_non_matroska_stream_start(self):
        timeline = self._make_track_timeline()
        with patch.object(video_utils, "get_video_full_info") as probe:
            offset = timeline.mkvmerge_input_start_offset_ms("/tmp/base.mov", "video", 0)

        self.assertEqual(offset, 0)
        probe.assert_not_called()

    def test_source_stream_end_uses_relative_duration_or_matroska_timeline_tag(self):
        timeline = self._make_track_timeline()

        def fake_full_info(path, **_kwargs):
            if path.endswith(".mp4"):
                return {
                    "streams": [
                        {
                            "codec_type": "video",
                            "index": 0,
                            "start_time": "0.500000",
                            "duration": "62.332000",
                        }
                    ]
                }
            return {
                "streams": [
                    {
                        "codec_type": "video",
                        "index": 0,
                        "start_time": "0.500000",
                        "tags": {"DURATION": "00:01:02.832000000"},
                    }
                ]
            }

        with patch.object(video_utils, "get_video_full_info", side_effect=fake_full_info):
            self.assertEqual(timeline.source_stream_end_offset_ms("/tmp/source.mp4", "video", 0), 62832)
            self.assertEqual(timeline.source_stream_end_offset_ms("/tmp/source.mkv", "video", 0), 62832)

    def test_track_sync_offset_compensates_for_mkvmerge_input_start(self):
        timeline = self._make_track_timeline()

        def fake_full_info(path, **_kwargs):
            if path.endswith(".mka"):
                return {"streams": [{"codec_type": "audio", "index": 0, "start_time": "0.471000"}]}
            return {"streams": [{"codec_type": "audio", "index": 1, "start_time": "0.471000"}]}

        with patch.object(video_utils, "get_video_full_info", side_effect=fake_full_info):
            mov_offset = timeline.track_sync_offset_ms("/tmp/source.mov", "audio", 1, 471)
            mka_offset = timeline.track_sync_offset_ms("/tmp/normalized.mka", "audio", 0, 471)

        self.assertEqual(mov_offset, 471)
        self.assertIsNone(mka_offset)

    def test_aac_from_non_matroska_needs_mkvmerge_normalization(self):
        timeline = self._make_track_timeline()
        fake_full_info = {
            "streams": [
                {"codec_type": "video", "index": 0, "codec_name": "h264"},
                {"codec_type": "audio", "index": 1, "codec_name": "aac"},
            ]
        }

        with patch.object(video_utils, "get_video_full_info", return_value=fake_full_info):
            self.assertIsNotNone(timeline.audio_mkvmerge_normalization_reason(
                AudioStreamRef("/tmp/source.mov", 1, 1, "eng")
            ))
            timeline._media_info_cache.clear()
            timeline._stream_info_cache.clear()
            self.assertIsNone(timeline.audio_mkvmerge_normalization_reason(
                AudioStreamRef("/tmp/source.mkv", 1, 1, "eng")
            ))

    def test_matroska_aac_with_priming_needs_mkvmerge_normalization(self):
        """Matroska AAC carrying CodecDelay priming must go through the FLAC flow:
        a raw remux places it build-dependently (~21 ms off on exposing builds)."""
        timeline = self._make_track_timeline()

        def fake_full_info(padded):
            return {
                "streams": [
                    {"codec_type": "video", "index": 0, "codec_name": "h264"},
                    {
                        "codec_type": "audio",
                        "index": 1,
                        "codec_name": "aac",
                        "initial_padding": 1024 if padded else 0,
                    },
                ]
            }

        with patch.object(video_utils, "get_video_full_info", return_value=fake_full_info(True)):
            padded_stream = AudioStreamRef("/tmp/source.mkv", 1, 1, "eng")
            self.assertIsNotNone(timeline.audio_mkvmerge_normalization_reason(padded_stream))
            self.assertEqual(
                timeline.audio_mkvmerge_normalization_reason(padded_stream),
                "removing AAC encoder delay (1024 priming samples)",
            )
        timeline._media_info_cache.clear()
        timeline._stream_info_cache.clear()
        with patch.object(video_utils, "get_video_full_info", return_value=fake_full_info(False)):
            unpadded_stream = AudioStreamRef("/tmp/source.mkv", 1, 1, "eng")
            self.assertIsNone(timeline.audio_mkvmerge_normalization_reason(unpadded_stream))

    def test_track_sync_offset_compensates_when_normalized_audio_start_is_lost(self):
        timeline = self._make_track_timeline()
        normalized_full_info = {
            "streams": [
                {"codec_type": "audio", "index": 0, "codec_name": "aac", "start_time": "0.000000"},
            ]
        }

        with patch.object(video_utils, "get_video_full_info", return_value=normalized_full_info):
            offset = timeline.track_sync_offset_ms("/tmp/normalized.mka", "audio", 0, 471)

        self.assertEqual(offset, 471)


if __name__ == "__main__":
    unittest.main()
