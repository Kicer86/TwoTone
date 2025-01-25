
import logging
import unittest
from overrides import override
from typing import Dict, List

import twotone.tools.utils as utils
from twotone.tools.melt import Melter, DuplicatesSource
from common import WorkingDirectoryForTest, add_test_media



class Duplicates(DuplicatesSource):
    def __init__(self, interruption: utils.InterruptibleProcess):
        super().__init__(interruption)
        self.duplicates = {}

    def setDuplicates(self, duplicates: Dict):
        self.duplicates = duplicates

    @override
    def collect_duplicates(self) -> Dict[str, List[str]]:
        return self.duplicates


class MeltingTest(unittest.TestCase):

    def setUp(self):
        logging.getLogger().setLevel(logging.ERROR)

    def test_simple_duplicate_detection(self):
        with WorkingDirectoryForTest() as td:
            file1 = add_test_media("Grass - 66810.mp4", td.path, suffixes = ["v1"])
            file2 = add_test_media("Grass - 66810.mp4", td.path, suffixes = ["v2"])
            files = [*file1, *file2]

            interruption = utils.InterruptibleProcess()
            duplicates = Duplicates(interruption)
            duplicates.setDuplicates({"Grass": files})

            melter = Melter(interruption, duplicates)
            melter.melt()



if __name__ == '__main__':
    unittest.main()
