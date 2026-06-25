import io
import os
import sys
import unittest

from contextlib import redirect_stdout
from unittest.mock import patch

from twotone import twotone
from twotone.tools.utils import process_utils


class RuntimeVersionTest(unittest.TestCase):
    def test_version_option_prints_runtime_report_without_requiring_a_tool(self):
        output = io.StringIO()

        with patch.object(twotone, "_runtime_version_report", return_value="runtime info"), \
             redirect_stdout(output):
            twotone.execute(["--version"])

        self.assertEqual(output.getvalue(), "runtime info\n")

    def test_runtime_version_report_identifies_launcher_source_and_revision(self):
        git_result = process_utils.ProcessResult(0, "v1.3.0-2-g1234567-dirty\n", "")
        launcher = "/home/user/.local/bin/twotone"

        with patch("importlib.metadata.version", return_value="1.3.0"), \
             patch.object(twotone.shutil, "which", return_value="/usr/bin/git"), \
             patch.object(twotone.process_utils, "start_process", return_value=git_result) as start_process, \
             patch.object(sys, "argv", [launcher]):
            report = twotone._runtime_version_report()

        source_dir = os.path.dirname(os.path.abspath(twotone.__file__))
        expected_launcher = os.path.abspath(os.path.expanduser(launcher))
        self.assertIn("TwoTone 1.3.0", report)
        self.assertIn(f"Launcher: {expected_launcher}", report)
        self.assertIn(f"Source: {source_dir}", report)
        self.assertIn(f"Python: {sys.executable}", report)
        self.assertIn("Git: v1.3.0-2-g1234567-dirty", report)
        start_process.assert_called_once_with(
            "git",
            ["describe", "--always", "--dirty", "--long"],
            cwd=source_dir,
        )

    def test_runtime_version_report_does_not_require_git(self):
        with patch("importlib.metadata.version", return_value="1.3.0"), \
             patch.object(twotone.shutil, "which", return_value=None), \
             patch.object(twotone.process_utils, "start_process") as start_process:
            report = twotone._runtime_version_report()

        self.assertNotIn("Git:", report)
        start_process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
