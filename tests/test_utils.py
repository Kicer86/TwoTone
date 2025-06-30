
import os
import unittest

from twotone.tools.utils import subtitles_utils
from common import WorkingDirectoryForTest, TwoToneTestCase, write_subtitle


class UtilsTests(TwoToneTestCase):
    def _test_content(self, content: str, valid: bool):
        with WorkingDirectoryForTest() as wd:
            subtitle_path = os.path.join(wd.path, "subtitle.txt")

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


if __name__ == '__main__':
    unittest.main()
