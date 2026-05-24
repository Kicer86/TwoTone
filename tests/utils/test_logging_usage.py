import ast
import unittest
from pathlib import Path


FORBIDDEN_LOGGING_CALLS = {
    "critical",
    "debug",
    "error",
    "exception",
    "info",
    "log",
    "warn",
    "warning",
}


class LoggingUsageTests(unittest.TestCase):
    def test_production_code_does_not_use_root_logging_calls(self):
        project_root = Path(__file__).resolve().parents[2]
        production_root = project_root / "twotone"

        offenders: list[str] = []
        for path in sorted(production_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                value = func.value
                if not isinstance(value, ast.Name) or value.id != "logging":
                    continue
                if func.attr not in FORBIDDEN_LOGGING_CALLS:
                    continue

                rel_path = path.relative_to(project_root)
                offenders.append(f"{rel_path}:{node.lineno} logging.{func.attr}()")

        self.assertEqual([], offenders)

    def test_production_code_uses_twotone_logger_hierarchy(self):
        project_root = Path(__file__).resolve().parents[2]
        production_root = project_root / "twotone"

        offenders: list[str] = []
        for path in sorted(production_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                value = func.value
                if not isinstance(value, ast.Name) or value.id != "logging":
                    continue
                if func.attr != "getLogger":
                    continue

                rel_path = path.relative_to(project_root)
                if not node.args:
                    offenders.append(f"{rel_path}:{node.lineno} logging.getLogger()")
                    continue

                logger_name = node.args[0]
                if isinstance(logger_name, ast.Constant) and isinstance(logger_name.value, str):
                    if not logger_name.value.startswith("TwoTone"):
                        offenders.append(f"{rel_path}:{node.lineno} logging.getLogger({logger_name.value!r})")
                    continue

                offenders.append(f"{rel_path}:{node.lineno} logging.getLogger(<dynamic>)")

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
