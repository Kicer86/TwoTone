
import logging
import os
import shutil
import unittest
from typing import List

from twotone.tools.utils2.files import split_path
from common import WorkingDirectoryForTest, add_test_media, list_files, run_twotone


class ConcatenateTests(unittest.TestCase):

    def _create_media(self, wd: str, base_file: str, partnames: List[str]):
        media_file_components = split_path(base_file)

        def build_part(part: str) -> str:
            return os.path.join(wd, media_file_components[1]) + part + "." + media_file_components[2]

        for partname in partnames:
            target = build_part(partname)
            shutil.copy2(base_file, target)


    def _setup_valid_media(self, wd: str):
        media_files = add_test_media("Frog.*mp4", wd)

        wdX = [os.path.join(wd, str(i)) for i in range(7)]

        for wdx in wdX:
            os.makedirs(wdx)

        self.assertEqual(len(media_files), 1)
        media_file = media_files[0]

        self._create_media(wdX[0], media_file, [" CD1", " CD2"])
        self._create_media(wdX[1], media_file, ["-CD1", "-CD2", "-CD3", "-CD4"])
        self._create_media(wdX[2], media_file, [".CD1", ".CD2", ".CD3", ".CD4", ".CD5", ".CD6", ".CD7"])
        self._create_media(wdX[3], media_file, [" cd1", " cd2", " cd3", " cd4", " cd5", " cd6", " cd7", " cd8", " cd9", " cd10", " cd11", " cd12", ])
        self._create_media(wdX[4], media_file, [" Cd01", " Cd02", " Cd03", " Cd04", " Cd05", " Cd06", " Cd07", " Cd08", " Cd09", " Cd010", " Cd11", " Cd12"])

        shutil.copy2(media_file, os.path.join(wdX[5], "cd1.mp4"))
        shutil.copy2(media_file, os.path.join(wdX[5], "cd2.mp4"))

        shutil.copy2(media_file, os.path.join(wdX[6], "''v'' cd1.mp4"))
        shutil.copy2(media_file, os.path.join(wdX[6], "''v'' cd2.mp4"))


    def _setup_invalid_media(self, wd: str):
        media_files = add_test_media("Frog.*mp4", wd)

        wdX = [os.path.join(wd, str(i)) for i in range(4)]

        for wdx in wdX:
            os.makedirs(wdx)

        self.assertEqual(len(media_files), 1)
        media_file = media_files[0]

        self._create_media(wdX[0], media_file, [" CD2"])
        self._create_media(wdX[1], media_file, ["-CD1"])
        self._create_media(wdX[2], media_file, [".CD1", ".CD2", ".CD3", ".CD5", ".CD6", ".CD7"])
        shutil.copy2(media_file, os.path.join(wdX[3], "cd1.mp4"))

        return wdX


    def test_dry_run_is_respected(self):
        with WorkingDirectoryForTest() as td:
            self._setup_valid_media(td.path)

            files_before = list_files(td.path)
            run_twotone("concatenate", [td.path])

            files_after = list_files(td.path)
            self.assertEqual(files_after, files_before)


    def test_concatenation(self):
        with WorkingDirectoryForTest() as td:
            self._setup_valid_media(td.path)

            run_twotone("concatenate", [td.path], ["-r"])

            files_after = list_files(td.path)
            tdl = len(td.path) + 1

            short_paths = [path[tdl:] for path in files_after]
            short_paths.sort()
            self.assertEqual(short_paths, ['0/Frog - 113403.mp4', '1/Frog - 113403.mp4', '2/Frog - 113403.mp4', '3/Frog - 113403.mp4', '4/Frog - 113403.mp4', '5/5.mp4', "6/''v''.mp4", 'Frog - 113403.mp4'])


    def test_invalid_scenarios(self):
         with WorkingDirectoryForTest() as td:
            cases = self._setup_invalid_media(td.path)
            files_before = list_files(td.path)
            for case in cases:
                run_twotone("concatenate", [case], ["-r"])

            files_after = list_files(td.path)
            self.assertEqual(files_after, files_before)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.CRITICAL)
    unittest.main()
