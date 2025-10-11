
import os
import unittest
from parameterized import parameterized

from twotone.tools.utils import subtitles_utils, video_utils
from common import TwoToneTestCase, get_video, remove_key, write_subtitle


class UtilsTests(TwoToneTestCase):
    def _test_content(self, content: str, valid: bool):
        subtitle_path = os.path.join(self.wd.path, "subtitle.txt")

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


    def test_subtitle_detection(self):
        self._test_content("12:34:56:test", True)
        self._test_content("{1}{2}test", True)
        self._test_content("12:34:56:test\n21:01:45:test2", True)
        self._test_content("12:34:5:test", False)
        self._test_content("12:test", False)
        self._test_content("{12}:test", False)
        self._test_content("{a}{b}:test", False)


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
                        'fps': '0', 'length': None, 'width': 3840, 'height': 2160, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                    }],
                    'audio':
                    [{
                        'language': 'eng', 'channels': 2, 'sample_rate': 48000, 'tid': 1, 'uid': None,
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
                        'fps': '0', 'length': None, 'width': 2160, 'height': 3840, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                    }],
                    'audio':
                    [{
                        'language': 'eng', 'channels': 2, 'sample_rate': 48000, 'tid': 1, 'uid': None,
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
                        'fps': '0', 'length': None, 'width': 1920, 'height': 1080, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
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
                        'fps': '0', 'length': None, 'width': 1080, 'height': 1112, 'bitrate': None, 'codec': 'VP9', 'tid': 0, 'uid': None,
                    }],
                    'audio':
                    [{
                        'language': 'und', 'channels': 2, 'sample_rate': 44100, 'tid': 1, 'uid': None,
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
                        'fps': '30000/1001', 'length': 3403, 'width': 3840, 'height': 2160, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                        'default': True, 'forced': False,
                    }],
                    'audio':
                    [{
                        'codec': 'aac', 'language': 'eng', 'channels': 2, 'sample_rate': 48000, 'tid': 1, 'uid': None,
                        'default': True, 'forced': False,
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
                        'fps': '29999/500', 'length': 1000, 'width': 2160, 'height': 3840, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                        'default': True, 'forced': False,
                    }],
                    'audio':
                    [{
                        'codec': 'aac', 'language': 'eng', 'channels': 2, 'sample_rate': 48000, 'tid': 1, 'uid': None,
                        'default': True, 'forced': False,
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
                        'fps': '25/1', 'length': 15600, 'width': 1920, 'height': 1080, 'bitrate': None, 'codec': 'HEVC/H.265/MPEG-H', 'tid': 0, 'uid': None,
                        'default': True, 'forced': False,
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
                        'fps': '60/1', 'length': 11583, 'width': 1080, 'height': 1112, 'bitrate': None, 'codec': 'VP9', 'tid': 0, 'uid': None,
                        'default': True, 'forced': False,
                    }],
                    'audio':
                    [{
                        'codec': 'aac', 'language': 'und', 'channels': 2, 'sample_rate': 44100, 'tid': 1, 'uid': None,
                        'default': True, 'forced': False,
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
