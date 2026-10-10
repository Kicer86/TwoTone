import argparse
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from twotone import twotone
from twotone.completion import build_bash_completion


class BashCompletionTest(unittest.TestCase):
    version = "1.4.0+test-revision"
    parser: ClassVar[argparse.ArgumentParser]
    script: ClassVar[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = twotone._create_parser()
        cls.script = build_bash_completion(cls.parser, cls.version)

    @staticmethod
    def _bash_path(path: Path) -> str:
        return path.as_posix()

    def test_generated_script_contains_runtime_version(self):
        self.assertEqual(
            self.script.splitlines()[0],
            f"# twotone-completion-version: {self.version}",
        )

    def complete(self, words: list[str], current_word: int) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            script_path = Path(directory) / "twotone-completion.bash"
            script_path.write_text(self.script, encoding="utf-8")
            quoted_words = " ".join(shlex.quote(word) for word in words)
            command = (
                'source "$1"; '
                f"COMP_WORDS=({quoted_words}); "
                f"COMP_CWORD={current_word}; "
                '_twotone_complete; printf "%s\\n" "${COMPREPLY[@]}"'
            )
            result = subprocess.run(
                ["bash", "-c", command, "bash", self._bash_path(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        return result.stdout.splitlines()

    def test_generated_script_has_valid_bash_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            script_path = Path(directory) / "twotone-completion.bash"
            script_path.write_text(self.script, encoding="utf-8")
            result = subprocess.run(
                ["bash", "-n", self._bash_path(script_path)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_completes_top_level_commands(self):
        self.assertEqual(self.complete(["twotone", "m"], 1), ["melt", "merge"])

    def test_completes_options_for_selected_tool(self):
        self.assertEqual(
            self.complete(["twotone", "merge", "--l"], 2),
            ["--language", "--languages-priority"],
        )

    def test_completes_option_choices(self):
        self.assertEqual(
            self.complete(["twotone", "--validate-inputs", "f"], 2),
            ["fast", "full"],
        )

    def test_completes_nested_utility_command(self):
        self.assertEqual(
            self.complete(["twotone", "utilities", "s"], 2),
            ["scenes"],
        )

    def test_loaded_completion_reloads_after_file_is_refreshed(self):
        old_parser = twotone._create_parser()
        old_script = build_bash_completion(old_parser, "old-version")
        old_parser.add_argument("--new-option", action="store_true")
        new_script = build_bash_completion(old_parser, "new-version")

        with tempfile.TemporaryDirectory() as directory:
            installed_path = Path(directory) / "twotone"
            updated_path = Path(directory) / "twotone.updated"
            installed_path.write_text(old_script, encoding="utf-8")
            updated_path.write_text(new_script, encoding="utf-8")
            command = (
                'source "$1"; cp "$2" "$1"; '
                'COMP_WORDS=(twotone --new); COMP_CWORD=1; '
                '_twotone_complete; printf "%s\\n" "${COMPREPLY[@]}"'
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    command,
                    "bash",
                    self._bash_path(installed_path),
                    self._bash_path(updated_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.stdout.splitlines(), ["--new-option"])


class CompletionVersionTest(unittest.TestCase):
    def test_completion_version_combines_package_and_git_versions(self):
        with patch.object(
            twotone,
            "_runtime_version_details",
            return_value=("1.4.0", "v1.4.0-2-g1234567"),
        ):
            version = twotone._completion_version()

        self.assertEqual(version, "1.4.0+v1.4.0-2-g1234567")

    def test_completion_version_uses_package_version_without_git(self):
        with patch.object(
            twotone,
            "_runtime_version_details",
            return_value=("1.4.0", None),
        ):
            version = twotone._completion_version()

        self.assertEqual(version, "1.4.0")


class CompletionRefreshTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = twotone._create_parser()
        self.data_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.data_directory.cleanup)
        self.environment = patch.dict(
            "os.environ",
            {"XDG_DATA_HOME": self.data_directory.name},
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.version = patch.object(
            twotone,
            "_completion_version",
            return_value="test-version",
        )
        self.version.start()
        self.addCleanup(self.version.stop)

    def completion_path(self) -> Path:
        return Path(self.data_directory.name) / "bash-completion" / "completions" / "twotone"

    def test_refresh_does_nothing_when_completion_is_not_installed(self):
        with patch.object(twotone, "_write_completion") as write_completion:
            twotone._refresh_completion_if_installed(self.parser)

        write_completion.assert_not_called()

    def test_refresh_does_not_rewrite_current_completion(self):
        twotone._install_completion(self.parser)

        with patch.object(twotone, "_write_completion") as write_completion:
            twotone._refresh_completion_if_installed(self.parser)

        write_completion.assert_not_called()

    def test_refresh_atomically_replaces_stale_completion(self):
        completion_path = self.completion_path()
        completion_path.parent.mkdir(parents=True)
        completion_path.write_text(
            build_bash_completion(self.parser, "old-version"),
            encoding="utf-8",
        )

        twotone._refresh_completion_if_installed(self.parser)

        self.assertEqual(
            completion_path.read_text(encoding="utf-8"),
            build_bash_completion(self.parser, "test-version"),
        )
        self.assertEqual(list(completion_path.parent.iterdir()), [completion_path])

    def test_execute_refreshes_completion_before_parsing(self):
        with patch.object(twotone, "_refresh_completion_if_installed") as refresh, \
             self.assertRaises(SystemExit):
            twotone.execute(["--help"])

        refresh.assert_called_once()

    def test_install_does_not_refresh_before_overwriting_completion(self):
        with patch.object(twotone, "_refresh_completion_if_installed") as refresh:
            twotone.execute(["--install-completion"])

        refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
