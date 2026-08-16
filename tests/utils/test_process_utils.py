import subprocess
import unittest

from unittest.mock import Mock, call, patch

from twotone.tools.utils import process_utils


class StartProcessTest(unittest.TestCase):
    def test_ffprobe_progress_drains_output_while_process_is_running(self):
        process = Mock(returncode=0)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["ffprobe"], 0.1),
            ('{"streams": []}', ""),
        ]

        with patch.object(process_utils.subprocess, "Popen", return_value=process), \
             patch.object(process_utils, "tqdm"):
            result = process_utils.start_process("ffprobe", [], show_progress=True)

        self.assertEqual(result, process_utils.ProcessResult(0, '{"streams": []}', ""))
        self.assertEqual(process.communicate.call_args_list, [call(timeout=0.1), call(timeout=0.1)])


if __name__ == "__main__":
    unittest.main()
