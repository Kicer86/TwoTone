
import os
import unittest
from parameterized import parameterized
from unittest.mock import patch

from twotone.tools.utils import process_utils
from twotone.tools.utils import subtitles_utils, video_utils
from common import TwoToneTestCase, get_video, remove_key, write_subtitle, generate_subtitles


class UtilsTests(TwoToneTestCase):
    def _test_content(self, ext: str, content: str, valid: bool):
        subtitle_path = os.path.join(self.wd.path, f"subtitle.{ext}")

        if ext == "sub":
            generate_subtitles(subtitle_path, length=2, unit="seconds", fps=25, interval_ms=1000, event_ms=500)
        else:
            write_subtitle(subtitle_path, [content])

        if valid:
            self.assertTrue(subtitles_utils.is_subtitle(subtitle_path))
        else:
            self.assertFalse(subtitles_utils.is_subtitle(subtitle_path))


    def assertDictSubset(self, expected: dict, actual: dict, context: str) -> None:
        for key, value in expected.items():
            self.assertIn(key, actual, f"{context}: missing key '{key}'")
            self.assertEqual(value, actual[key], f"{context}: expected {key}={value!r}, got {actual[key]!r}")


    def assertStreamsSubset(self, expected_streams: dict, actual_streams: dict) -> None:
        actual_streams = dict(actual_streams)

        for stream_type, expected_stream_list in expected_streams.items():
            self.assertIn(stream_type, actual_streams, f"missing stream type '{stream_type}'")
            actual_stream_list = actual_streams[stream_type]
            self.assertEqual(
                len(expected_stream_list),
                len(actual_stream_list),
                f"stream type '{stream_type}': expected {len(expected_stream_list)} streams, got {len(actual_stream_list)}",
            )

            for idx, expected_stream in enumerate(expected_stream_list):
                self.assertDictSubset(expected_stream, actual_stream_list[idx], f"{stream_type}[{idx}]")


    subtitle_samples = [
        (
            "SubRip (SRT)",
            "srt",
            "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n",
            True,
        ),
        (
            "MicroDVD",
            "sub",
            "",
            True,
        ),
        (
            "WebVTT",
            "vtt",
            "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello world\n",
            True,
        ),
        (
            "Plain text file",
            "txt",
            "This is just a plain text file, not subtitles.",
            False,
        ),
    ]

    @parameterized.expand(subtitle_samples)
    def test_subtitle_detection(self, name, ext, content, valid):
        self._test_content(ext, content, valid)

    def test_idx_language_detection(self):
        subtitle_path = os.path.join(self.wd.path, "movie.idx")
        write_subtitle(
            subtitle_path,
            [
                "# VobSub index file, v7",
                "size: 720x576",
                "palette: 000000,ffffff",
                "id: zho, index: 0",
            ],
        )

        subtitle = subtitles_utils.build_subtitle_from_path(subtitle_path, language=None)
        self.assertEqual(subtitle.language, "zho")

    @parameterized.expand([
        ("negative_start", "-0.021000", -21),
        ("zero_start", "0.000000", 0),
        ("positive_start", "0.510000", 0),
    ])
    def test_showinfo_timestamp_correction_uses_only_negative_container_start(
        self,
        _name: str,
        ffprobe_output: str,
        expected_correction_ms: int,
    ):
        result = process_utils.ProcessResult(0, ffprobe_output, "")
        with patch.object(process_utils, "start_process", return_value=result):
            correction = video_utils._showinfo_timestamp_correction_ms("input.mkv", self.logger)

        self.assertEqual(expected_correction_ms, correction)

    def test_probe_frame_timestamps_corrects_negative_container_start(self):
        stderr_lines = [
            "[Parsed_showinfo_0] n:   0 pts:     21 pts_time:0.021\n",
            "[Parsed_showinfo_0] n:   1 pts:     62 pts_time:0.062\n",
        ]
        proc = unittest.mock.Mock(returncode=0)

        def fake_start_ffmpeg_streaming(args, interruption=None, on_line=None, logger=None):
            for line in stderr_lines:
                on_line(line)
            return proc, stderr_lines

        with patch.object(video_utils, "get_video_frames_count", return_value=2), \
             patch.object(video_utils, "_showinfo_timestamp_correction_ms", return_value=-21), \
             patch.object(video_utils, "_start_ffmpeg_streaming", side_effect=fake_start_ffmpeg_streaming):
            mapping = video_utils.probe_frame_timestamps("input.mkv", logger=self.logger)

        self.assertEqual(
            {
                0: {"frame_id": 0, "path": None},
                41: {"frame_id": 1, "path": None},
            },
            mapping,
        )


    test_videos = [
        # case: merge all audio tracks
        (
            "MP4 - camera",
            # input
            "DSC_8073.MP4",
            # expected output
            {
                'video':
                [{
                    'fps': '30000/1001', 'length': 3403, 'width': 3840, 'height': 2160, 'bitrate': None, 'codec': 'hevc', 'tid': 0,
                    'default': True, 'forced': False
                }],
                'audio':
                [{
                    'language': 'eng', 'channels': 2, 'sample_rate': 48000, 'tid': 1,
                    'default': True, 'forced': False
                }]
            }
        ),
        (
            "MP4 - camera2",
            # input
            "moon.mp4",
            # expected output
            {
                'video':
                [{
                    'fps': '29999/500', 'length': 1000, 'width': 2160, 'height': 3840, 'bitrate': None, 'codec': 'hevc','tid': 0,
                    'default': True, 'forced': False
                }],
                'audio':
                [{
                    'language': 'eng', 'channels': 2, 'sample_rate': 48000, 'tid': 1,
                    'default': True, 'forced': False
                }]
            }
        ),
        (
            "MOV - no audio",
            # input
            "Blue_Sky_and_Clouds_Timelapse_0892__Videvo.mov",
            # expected output
            {
                'video':
                [{
                    'fps': '25/1', 'length': 15600, 'width': 1920, 'height': 1080, 'bitrate': None, 'codec': 'hevc', 'tid': 0,
                    'default': True, 'forced': False
                }]
            }
        ),
    ]

    @parameterized.expand(test_videos)
    def test_video_info(self, name, input, expected_streams):
        input_file_name = get_video(input)
        file_info = video_utils.get_video_data(input_file_name)

        self.assertStreamsSubset(expected_streams, file_info)

    test_videos_mkv = [
        # case: merge all audio tracks
        (
            "MP4 - camera",
            # input
            "DSC_8073.MP4",
            # expected output
            {
                "attachments": [],
                "tracks":
                {
                    'video':
                    [{
                        'default': False, 'enabled': True, 'forced': False, 'fps': '0', 'language': None, 'length': None, 'width': 3840, 'height': 2160, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                    }],
                    'audio':
                    [{
                        'codec': 'AAC', 'default': False, 'enabled': True, 'forced': False, 'language': 'eng', 'length': None, 'channels': 2, 'sample_rate': 48000, 'tid': 1, 'uid': None,
                    }]
                }
            }
        ),
        (
            "MP4 - camera2",
            # input
            "moon.mp4",
            # expected output
            {
                "attachments": [],
                "tracks":
                {
                    'video':
                    [{
                        'default': False, 'enabled': True, 'forced': False, 'fps': '0', 'language': None, 'length': None, 'width': 2160, 'height': 3840, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                    }],
                    'audio':
                    [{
                        'codec': 'AAC', 'default': False, 'enabled': True, 'forced': False,'language': 'eng', 'length': None, 'channels': 2, 'sample_rate': 48000, 'tid': 1, 'uid': None,
                    }]
                }
            }
        ),
        (
            "MOV - no audio",
            # input
            "Blue_Sky_and_Clouds_Timelapse_0892__Videvo.mov",
            # expected output
            {
                "attachments": [],
                "tracks":
                {
                    'video':
                    [{
                        'default': False, 'enabled': True, 'forced': False, 'fps': '0', 'language': None, 'length': None, 'width': 1920, 'height': 1080, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                    }]
                }
            }
        ),
        (
            "MP4 - image attachment",
            # input
            "Chess alert 🚨 This is why you need to protect your king 🚨 [Iwj5vgXMeVE].mp4",
            # expected output
            {
                "attachments":
                [
                    {
                        'tid': 0, 'uid': 15146541822372754365, 'content_type': 'image/png', 'file_name': 'cover.png'
                    }
                ],
                "tracks":
                {
                    'video':
                    [{
                        'default': False, 'enabled': True, 'forced': False, 'fps': '0', 'language': None, 'length': None, 'width': 1080, 'height': 1112, 'bitrate': None, 'codec': 'VP9', 'tid': 0, 'uid': None,
                    }],
                    'audio':
                    [{
                        'codec': 'AAC', 'default': False, 'enabled': True, 'forced': False, 'language': None, 'length': None, 'channels': 2, 'sample_rate': 44100, 'tid': 1, 'uid': None,
                    }]
                }
            }
        ),
    ]

    @parameterized.expand(test_videos_mkv)
    def test_video_mkvinfo(self, name, input, expected_streams):
        self.maxDiff = None
        input_file_name = get_video(input)
        file_info = video_utils.get_video_data_mkvmerge(input_file_name)

        file_info = remove_key(file_info, "uid")
        expected_streams = remove_key(expected_streams, "uid")

        self.assertEqual(expected_streams, file_info)

    test_videos_mkv = [
        # case: merge all audio tracks
        (
            "MP4 - camera",
            # input
            "DSC_8073.MP4",
            # expected output
            {
                "attachments": [],
                "tracks":
                {
                    'video':
                    [{
                        'fps': '30000/1001', 'language': None, 'length': 3403, 'width': 3840, 'height': 2160, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                        'default': True, 'forced': False, 'enabled': True,
                    }],
                    'audio':
                    [{
                        'codec': 'AAC', 'language': 'eng', 'length': None, 'channels': 2, 'sample_rate': 48000, 'tid': 1, 'uid': None,
                        'default': True, 'forced': False, 'enabled': True,
                    }]
                }
            }
        ),
        (
            "MP4 - camera2",
            # input
            "moon.mp4",
            # expected output
            {
                "attachments": [],
                "tracks":
                {
                    'video':
                    [{
                        'fps': '29999/500', 'language': None, 'length': 1000, 'width': 2160, 'height': 3840, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                        'default': True, 'forced': False, 'enabled': True,
                    }],
                    'audio':
                    [{
                        'codec': 'AAC', 'language': 'eng', 'length': None, 'channels': 2, 'sample_rate': 48000, 'tid': 1, 'uid': None,
                        'default': True, 'forced': False, 'enabled': True,
                    }]
                }
            }
        ),
        (
            "MOV - no audio",
            # input
            "Blue_Sky_and_Clouds_Timelapse_0892__Videvo.mov",
            # expected output
            {
                "attachments": [],
                "tracks":
                {
                    'video':
                    [{
                        'fps': '25/1', 'language': None, 'length': 15600, 'width': 1920, 'height': 1080, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                        'default': True, 'forced': False, 'enabled': True,
                    }]
                }
            }
        ),
        (
            "MP4 - image attachment",
            # input
            "Chess alert 🚨 This is why you need to protect your king 🚨 [Iwj5vgXMeVE].mp4",
            # expected output
            {
                "attachments":
                [
                    {
                        'tid': 0, 'uid': 7991968932496793124, 'content_type': 'image/png', 'file_name': 'cover.png'
                    }
                ],
                "tracks":
                {
                    'video':
                    [{
                        'fps': '60/1', 'language': None, 'length': 11583, 'width': 1080, 'height': 1112, 'bitrate': None, 'codec': 'VP9', 'tid': 0, 'uid': None,
                        'default': True, 'forced': False, 'enabled': True,
                    }],
                    'audio':
                    [{
                        'codec': 'AAC', 'language': 'und', 'length': None, 'channels': 2, 'sample_rate': 44100, 'tid': 1, 'uid': None,
                        'default': True, 'forced': False, 'enabled': True,
                    }]
                }
            }
        ),
    ]

    @parameterized.expand(test_videos_mkv)
    def test_video_mkvinfo_enriched(self, name, input, expected_streams):
        self.maxDiff = None
        input_file_name = get_video(input)
        file_info = video_utils.get_video_data_mkvmerge(input_file_name, enrich = True)

        file_info = remove_key(file_info, "uid")
        expected_streams = remove_key(expected_streams, "uid")

        self.assertEqual(expected_streams, file_info)


if __name__ == '__main__':
    unittest.main()
