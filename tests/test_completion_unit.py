import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from twotone import twotone
from twotone.completion import build_bash_completion


class BashCompletionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = build_bash_completion(twotone._create_parser())

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
                ["bash", "-c", command, "bash", str(script_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        return result.stdout.splitlines()

    def test_generated_script_has_valid_bash_syntax(self):
        result = subprocess.run(
            ["bash", "-n"],
            input=self.script,
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


if __name__ == "__main__":
    unittest.main()
