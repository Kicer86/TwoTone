
import logging
import unittest

from twotone.tools.melt.melt_plan import MeltPlan


class MeltPlanFormatTrackLineTest(unittest.TestCase):
    """Unit tests for MeltPlan._format_track_line."""

    def test_format_track_line_audio_with_language(self):
        stream = {"tid": 1, "language": "eng", "channels": 6, "sample_rate": 48000}
        result = MeltPlan._format_track_line("audio", stream, used=True)
        self.assertIn("English", result)
        self.assertIn("used", result)

    def test_format_track_line_audio_unknown_language(self):
        stream = {"tid": 1, "channels": 2, "sample_rate": 44100}
        result = MeltPlan._format_track_line("audio", stream, used=True)
        self.assertIn("unknown", result)

    def test_format_track_line_audio_with_override_language(self):
        """When override_lang is provided, it should be shown instead of the stream's raw language."""
        stream = {"tid": 1, "channels": 6, "sample_rate": 48000}
        result = MeltPlan._format_track_line("audio", stream, used=True, override_lang="pol")
        self.assertIn("Polish", result)
        self.assertNotIn("unknown", result)

    def test_format_track_line_override_replaces_original(self):
        """Override language should replace the original one."""
        stream = {"tid": 1, "language": "eng", "channels": 6, "sample_rate": 48000}
        result = MeltPlan._format_track_line("audio", stream, used=True, override_lang="pol")
        self.assertIn("Polish", result)
        self.assertNotIn("English", result)


class MeltPlanCollectSelectedTest(unittest.TestCase):
    """Unit tests for MeltPlan._collect_selected."""

    def test_collect_selected_returns_language_overrides(self):
        group = {
            "streams": {
                "video": [("/a.mkv", 0, None)],
                "audio": [("/a.mkv", 1, "eng"), ("/b.avi", 1, "pol")],
                "subtitle": [("/a.mkv", 2, "eng")],
            },
            "attachments": [],
        }
        selected, _, lang_overrides = MeltPlan._collect_selected(group)

        # Basic selected IDs
        self.assertEqual(selected["audio"]["/a.mkv"], {1})
        self.assertEqual(selected["audio"]["/b.avi"], {1})

        # Language overrides
        self.assertEqual(lang_overrides["audio"][("/a.mkv", 1)], "eng")
        self.assertEqual(lang_overrides["audio"][("/b.avi", 1)], "pol")

    def test_collect_selected_none_language_not_in_overrides(self):
        group = {
            "streams": {
                "video": [("/a.mkv", 0, None)],
                "audio": [],
                "subtitle": [],
            },
            "attachments": [],
        }
        _, _, lang_overrides = MeltPlan._collect_selected(group)
        self.assertEqual(lang_overrides["video"], {})


class MeltPlanRenderOverrideTest(unittest.TestCase):
    """Test that render() shows overridden languages for used streams."""

    def test_render_shows_overridden_audio_language(self):
        """File #2 audio with --audio-lang pl should show 'Polish' in rendered output."""
        group = {
            "files": ["/a.mkv", "/b.avi"],
            "output_name": "output",
            "streams": {
                "video": [("/a.mkv", 0, None)],
                "audio": [("/a.mkv", 1, "eng"), ("/b.avi", 1, "pol")],
                "subtitle": [],
            },
            "attachments": [],
            "files_details": {
                "/a.mkv": {
                    "tracks": {
                        "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                        "audio": [{"tid": 1, "language": "eng", "channels": 6, "sample_rate": 48000}],
                        "subtitle": [],
                    },
                    "attachments": [],
                },
                "/b.avi": {
                    "tracks": {
                        "video": [{"tid": 0, "width": 720, "height": 384, "fps": "25/1"}],
                        "audio": [{"tid": 1, "channels": 6, "sample_rate": 48000}],
                        "subtitle": [],
                    },
                    "attachments": [],
                },
            },
        }
        plan = MeltPlan(
            items=[{"title": "Test Movie", "groups": [group]}],
            output_dir="/output",
        )
        logger = logging.getLogger("test.render")
        messages: list[str] = []
        logger.handlers.clear()

        class Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        logger.addHandler(Capture())
        logger.setLevel(logging.DEBUG)

        plan.render(logger)

        # Find line for File #2 audio
        file2_audio_lines = [
            m for m in messages
            if "#1 (" in m and "audio" not in m.split("#1 (")[0].rsplit("\n", 1)[-1].lower()
        ]
        # More targeted: find the audio line under File #2
        all_lines = "\n".join(messages)
        self.assertIn("Polish", all_lines,
                       f"Expected 'Polish' in rendered plan output but got:\n{all_lines}")
