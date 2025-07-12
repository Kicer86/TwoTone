
import os
import unittest
from parameterized import parameterized

from twotone.tools.utils import subtitles_utils, video_utils
from common import WorkingDirectoryForTest, TwoToneTestCase, get_video, write_subtitle


class UtilsTests(TwoToneTestCase):
    def _test_content(self, content: str, valid: bool):
        subtitle_path = os.path.join(self.wd.path, "subtitle.txt")

        write_subtitle(subtitle_path, [content])

        if valid:
            self.assertTrue(subtitles_utils.is_subtitle(subtitle_path))
        else:
            self.assertFalse(subtitles_utils.is_subtitle(subtitle_path))


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
                    'fps': '30000/1001',
                    'length': 3403,
                    'width': 3840,
                    'height': 2160,
                    'bitrate': None,
                    'codec': 'hevc',
                    'attached_pic': False,
                    'tid': 0
                }],
                'audio':
                [{
                    'language': 'eng',
                    'channels': 2,
                    'sample_rate': '48000',
                    'tid': 1
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
                    'fps': '29999/500',
                    'length': 1000,
                    'width': 2160,
                    'height': 3840,
                    'bitrate': None,
                    'codec': 'hevc',
                    'attached_pic': False,
                    'tid': 0
                }],
                'audio':
                [{
                    'language': 'eng',
                    'channels': 2,
                    'sample_rate': '48000',
                    'tid': 1
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
                    'fps': '25/1',
                    'length': 15600,
                    'width': 1920,
                    'height': 1080,
                    'bitrate': None,
                    'codec': 'hevc',
                    'attached_pic': False,
                    'tid': 0
                }]
            }
        ),
    ]

    @parameterized.expand(test_videos)
    def test_video_info(self, name, input, expected_streams):
        input_file_name = get_video(input)
        file_info = video_utils.get_video_data(input_file_name)

        self.assertEqual(expected_streams, file_info)


if __name__ == '__main__':
    unittest.main()
