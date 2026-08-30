import unittest

import pysubs2

from twotone.tools.utils import subtitles_utils


class StripMicrodvdHeaderTest(unittest.TestCase):
    def test_strips_one_frame_fps_declaration(self):
        subtitles = pysubs2.SSAFile.from_string(
            "{1}{1}60\n{0}{30}caption\n",
            format_="microdvd",
            fps=60,
        )

        subtitles_utils._strip_microdvd_header(subtitles, fps=60)

        self.assertEqual(["caption"], [event.text for event in subtitles])


if __name__ == "__main__":
    unittest.main()
