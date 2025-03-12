
import logging
import os
import unittest

import twotone.tools.utils as utils
from common import WorkingDirectoryForTest, add_test_media, hashes, run_twotone
from unittest.mock import patch

from twotone.tools.utils2 import files


class SimpleSubtitlesMerge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._start_process = utils.start_process
        logging.getLogger().setLevel(logging.CRITICAL)


    def test_no_changes_when_mkvmerge_exits_with_error(self):

        def start_process(cmd, args):
            _, exec_name, _ = files.split_path(cmd)

            if exec_name == "mkvmerge":
                return utils.ProcessResult(1, b"", b"")
            else:
                return self._start_process.__func__(cmd, args)

        with patch("twotone.tools.utils.start_process") as mock_start_process, WorkingDirectoryForTest() as td:
            mock_start_process.side_effect = start_process
            add_test_media("Blue_Sky_and_Clouds_Timelapse.*(?:mov|srt)", td.path)

            hashes_before = hashes(td.path)
            self.assertEqual(len(hashes_before), 2)
            try:
                run_twotone("merge", [td.path], ["--no-dry-run"])
            except RuntimeError:
                pass

            hashes_after = hashes(td.path)

            self.assertEqual(hashes_before, hashes_after)
            self.assertEqual(mock_start_process.call_count, 3)


    def test_no_changes_when_ffprobe_exits_with_error(self):

        def start_process(cmd, args):
            _, exec_name, _ = files.split_path(cmd)

            if exec_name == "ffprobe":
                return utils.ProcessResult(1, b"", b"")
            else:
                return self._start_process.__func__(cmd, args)

        with patch("twotone.tools.utils.start_process") as mock_start_process, WorkingDirectoryForTest() as td:
            mock_start_process.side_effect = start_process
            add_test_media("Blue_Sky_and_Clouds_Timelapse.*(?:mov|srt)", td.path)

            hashes_before = hashes(td.path)
            self.assertEqual(len(hashes_before), 2)
            try:
                run_twotone("merge", [td.path], ["--no-dry-run"])
            except RuntimeError:
                pass

            hashes_after = hashes(td.path)

            self.assertEqual(hashes_before, hashes_after)
            self.assertEqual(mock_start_process.call_count, 1)

    def test_no_changes_when_ffmpeg_exits_with_error(self):

        def start_process(cmd, args):
            _, exec_name, _ = files.split_path(cmd)

            if exec_name == "ffmpeg":
                return utils.ProcessResult(1, b"", b"")
            else:
                return self._start_process.__func__(cmd, args)

        with patch("twotone.tools.utils.start_process") as mock_start_process, WorkingDirectoryForTest() as td:
            mock_start_process.side_effect = start_process
            add_test_media("Blue_Sky_and_Clouds_Timelapse.*(?:mov|srt)", td.path)

            hashes_before = hashes(td.path)
            self.assertEqual(len(hashes_before), 2)
            try:
                run_twotone("merge", [td.path], ["--no-dry-run"])
            except RuntimeError:
                pass

            hashes_after = hashes(td.path)

            self.assertEqual(hashes_before, hashes_after)
            self.assertEqual(mock_start_process.call_count, 2)


if __name__ == '__main__':
    unittest.main()
