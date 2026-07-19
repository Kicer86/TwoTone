import io
import os
import sys
import unittest

from contextlib import redirect_stdout
from dataclasses import dataclass
from unittest.mock import patch

from twotone import twotone
from twotone.tools.tool import Tool
from twotone.tools.utils import process_utils


@dataclass
class _TestPlan:
    paths: set[str]

    def is_empty(self) -> bool:
        return False

    def render(self, _logger) -> None:
        return None

    def input_files(self) -> set[str]:
        return self.paths


class _TestTool(Tool):
    def __init__(self, plan: _TestPlan) -> None:
        self.plan = plan
        self.performed = False

    def setup_parser(self, _parser) -> None:
        return None

    def analyze(self, _args, logger, workspace) -> _TestPlan:
        return self.plan

    def perform(self, _args, logger, workspace, plan) -> None:
        self.performed = True


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

    def test_executor_validates_plan_inputs_before_perform(self):
        with self.subTest("global validator receives only the analyzed plan inputs"):
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = os.path.join(temp_dir, "input.mkv")
                with open(input_path, "wb") as file:
                    file.write(b"media")
                tool = _TestTool(_TestPlan({input_path}))
                report = twotone.input_validation.ValidationReport((), 1, 0)
                work_dir = os.path.join(temp_dir, "work")
                cache_dir = os.path.join(temp_dir, "cache")

                with patch.dict(twotone.TOOLS, {"test": (tool, "test tool")}, clear=True), \
                     patch.object(twotone.process_utils, "ensure_tools_exist"), \
                     patch.object(twotone.input_validation, "InputValidator") as validator:
                    validator.return_value.validate.return_value = report
                    twotone.execute([
                        "-r", "--working-dir", work_dir,
                        "--validation-cache-dir", cache_dir,
                        "test",
                    ])

                validator.assert_called_once()
                validator.return_value.validate.assert_called_once_with({input_path})
                self.assertTrue(tool.performed)


if __name__ == "__main__":
    unittest.main()
