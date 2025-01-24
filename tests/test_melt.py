
import logging
import unittest

import twotone.tools.utils as utils
from common import WorkingDirectoryForTest, run_twotone


class MeltingTest(unittest.TestCase):

    def setUp(self):
        logging.getLogger().setLevel(logging.ERROR)

    def test_simple_duplicate_detection(self):
        pass


if __name__ == '__main__':
    unittest.main()
