import unittest

from twotone.tools.tool import EmptyPlan, Tool
from twotone.tools.utils import process_utils, requirements_utils


def _uses_ffmpeg_positional():
    process_utils.start_process("ffmpeg", ["-i", "input.mp4"])


def _uses_mkvmerge_keyword():
    tool_name = "mkvmerge"
    process_utils.start_process(process=tool_name, args=[])


def _helper_uses_ffprobe():
    process_utils.start_process("ffprobe", ["-i", "input.mp4"])


def _entry_calls_helper():
    _helper_uses_ffprobe()


def _explicit_required_tools():
    return None


def _dep_helper_uses_mkvextract():
    process_utils.start_process("mkvextract", ["tracks", "input.mkv", "0:out.srt"])


def _entry_with_dep_attr():
    return None


class DummyRequirementsTool(Tool):
    def setup_parser(self, parser) -> None:
        return None

    def analyze(self, args, logger, working_dir):
        self._entrypoint()
        return EmptyPlan()

    def perform(self, args, logger, working_dir, plan):
        self._perform_helper()
        return None

    def _entrypoint(self):
        self._first()
        _entry_calls_helper()

    def _first(self):
        self._second()

    def _second(self):
        process_utils.start_process("ffmpeg", ["-i", "input.mp4"])

    def _perform_helper(self):
        process_utils.start_process("mkvinfo", ["input.mkv"])


setattr(
    _explicit_required_tools,
    requirements_utils._REQUIRED_TOOLS_ATTR,
    {"exiftool"},
)
setattr(
    _entry_with_dep_attr,
    requirements_utils._REQUIRED_DEPS_ATTR,
    {_dep_helper_uses_mkvextract},
)
setattr(
    DummyRequirementsTool._second,
    requirements_utils._REQUIRED_DEPS_ATTR,
    {_dep_helper_uses_mkvextract},
)


class RequirementsUtilsTests(unittest.TestCase):
    def test_collects_tools_from_start_process_positional(self):
        tools = requirements_utils.collect_required_tools(_uses_ffmpeg_positional)
        self.assertEqual(tools, {"ffmpeg"})

    def test_collects_tools_from_start_process_keyword(self):
        tools = requirements_utils.collect_required_tools(_uses_mkvmerge_keyword)
        self.assertEqual(tools, {"mkvmerge"})

    def test_collects_tools_from_helper_call(self):
        tools = requirements_utils.collect_required_tools(_entry_calls_helper)
        self.assertEqual(tools, {"ffprobe"})

    def test_collects_tools_from_explicit_attr(self):
        tools = requirements_utils.collect_required_tools(_explicit_required_tools)
        self.assertEqual(tools, {"exiftool"})

    def test_collects_tools_from_dependency_attr(self):
        tools = requirements_utils.collect_required_tools(_entry_with_dep_attr)
        self.assertEqual(tools, {"mkvextract"})

    def test_collects_tools_from_tool_required_tools(self):
        tools = DummyRequirementsTool().required_tools()
        self.assertEqual(tools, {"ffmpeg", "ffprobe", "mkvextract", "mkvinfo"})


if __name__ == "__main__":
    unittest.main()
