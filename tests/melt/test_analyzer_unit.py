import os
import unittest

from parameterized import parameterized
from unittest.mock import patch

from common import TwoToneTestCase
from twotone.tools.melt.melt import MeltAnalyzer, StaticSource
from twotone.tools.melt.melt_analyzer import AlignmentRequirement, UnsupportedMeltInputError
from twotone.tools.melt.melt_common import AudioStreamRef, MeltInputFiles, SubtitleStreamRef, VideoStreamRef
from twotone.tools.utils import generic_utils, video_utils


class MeltAnalyzerTest(TwoToneTestCase):
    def setUp(self):
        super().setUp()
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        self.analyzer = MeltAnalyzer(
            self.logger,
            duplicates,
            self.workspace,
            allow_video_timeline_mismatch=False,
        )


    @staticmethod
    def _mkvmerge_info(**overrides):
        info = {
            "tracks": [],
            "attachments": [],
            "chapters": [],
            "global_tags": [],
            "track_tags": [],
        }
        info.update(overrides)
        return info

    @parameterized.expand([
        (
            "buttons_track",
            {"tracks": [{"id": 3, "type": "buttons", "properties": {}}]},
            "track type 'buttons'",
        ),
        (
            "unknown_track",
            {"tracks": [{"id": 4, "type": "control", "properties": {}}]},
            "track type 'control'",
        ),
        (
            "font_attachment",
            {
                "attachments": [{
                    "id": 5,
                    "content_type": "application/x-truetype-font",
                    "file_name": "font.ttf",
                    "properties": {},
                }],
            },
            "attachment 'font.ttf'",
        ),
        (
            "unknown_attachment",
            {
                "attachments": [{
                    "id": 6,
                    "content_type": "application/octet-stream",
                    "file_name": "payload.bin",
                    "properties": {},
                }],
            },
            "attachment 'payload.bin'",
        ),
    ])
    def test_probe_inputs_rejects_unsupported_elements(
        self,
        _name,
        raw_overrides,
        expected_description,
    ):
        # Exercise the Windows path that previously broke regex-based assertions,
        # even when this test runs on another platform.
        input_path = r"C:\Users\runneradmin\AppData\Local\Temp\input.mkv"
        raw_info = self._mkvmerge_info(**raw_overrides)

        with patch.object(video_utils, "get_video_full_info_mkvmerge", return_value=raw_info), \
             patch.object(video_utils, "get_video_data_mkvmerge") as parse_info:
            with self.assertRaises(UnsupportedMeltInputError) as raised:
                self.analyzer._probe_inputs([input_path], {input_path: 1})

        message = str(raised.exception)
        self.assertTrue(message.startswith("File #1"))
        self.assertIn(expected_description, message)
        self.assertIn("not supported yet", message)
        self.assertIn("File #1", message)
        self.assertNotIn("Melt input", message)
        parse_info.assert_not_called()

    def test_probe_inputs_rejects_multiple_thumbnails_across_group(self):
        first_path = os.path.join(self.wd.path, "first.mkv")
        second_path = os.path.join(self.wd.path, "second.mkv")
        first_info = self._mkvmerge_info(attachments=[{
            "id": 0,
            "content_type": "image/jpeg",
            "file_name": "first.jpg",
            "properties": {},
        }])
        second_info = self._mkvmerge_info(attachments=[{
            "id": 0,
            "content_type": "image/png",
            "file_name": "second.png",
            "properties": {},
        }])

        with patch.object(
            video_utils,
            "get_video_full_info_mkvmerge",
            side_effect=[first_info, second_info],
        ), patch.object(video_utils, "get_video_data_mkvmerge") as parse_info:
            with self.assertRaisesRegex(
                UnsupportedMeltInputError,
                "2 thumbnails.*not supported yet",
            ):
                self.analyzer._probe_inputs([first_path, second_path], {first_path: 1, second_path: 2})

        parse_info.assert_not_called()

    def test_probe_inputs_accepts_one_thumbnail_and_standard_tags(self):
        input_path = os.path.join(self.wd.path, "input.mkv")
        raw_info = self._mkvmerge_info(
            tracks=[
                {"id": 0, "type": "video", "properties": {}},
                {"id": 1, "type": "audio", "properties": {}},
                {"id": 2, "type": "subtitles", "properties": {}},
            ],
            attachments=[{
                "id": 0,
                "content_type": "image/jpeg",
                "file_name": "cover.jpg",
                "properties": {},
            }],
            global_tags=[{"num_entries": 1}],
            track_tags=[{"num_entries": 2, "track_id": 0}],
        )
        parsed_info = {
            "attachments": [{
                "tid": 0,
                "content_type": "image/jpeg",
                "file_name": "cover.jpg",
            }],
            "tracks": {
                "video": [{"tid": 0}],
                "audio": [{"tid": 1}],
                "subtitle": [{"tid": 2}],
            },
        }

        with patch.object(
            video_utils,
            "get_video_full_info_mkvmerge",
            return_value=raw_info,
        ) as raw_probe, patch.object(
            video_utils,
            "get_video_data_mkvmerge",
            return_value=parsed_info,
        ) as parse_info:
            details, attachments, tracks = self.analyzer._probe_inputs([input_path], {input_path: 1})

        self.assertEqual(
            {input_path: {**parsed_info, "chapters": []}},
            details,
        )
        self.assertEqual({input_path: parsed_info["attachments"]}, attachments)
        self.assertEqual({input_path: parsed_info["tracks"]}, tracks)
        raw_probe.assert_called_once_with(input_path, logger=self.logger)
        parse_info.assert_called_once_with(
            input_path,
            enrich=True,
            logger=self.logger,
            _mkvmerge_info=raw_info,
        )

    def test_pick_chapter_source_prefers_base_video(self):
        base_path = os.path.join(self.wd.path, "base.mkv")
        other_path = os.path.join(self.wd.path, "other.mkv")
        details = {
            base_path: {"chapters": [{"num_entries": 2}]},
            other_path: {"chapters": [{"num_entries": 3}]},
        }
        tracks = {
            base_path: {"video": [{"tid": 0, "length": 1000}]},
            other_path: {"video": [{"tid": 0, "length": 2000}]},
        }

        source = self.analyzer._pick_chapter_source(
            details,
            tracks,
            [VideoStreamRef(base_path, 0, 0, None)],
            {base_path: 1, other_path: 2},
        )

        self.assertEqual(source, base_path)

    def test_pick_chapter_source_accepts_same_length_non_base_video(self):
        base_path = os.path.join(self.wd.path, "base.mkv")
        other_path = os.path.join(self.wd.path, "other.mkv")
        details = {
            base_path: {"chapters": []},
            other_path: {"chapters": [{"num_entries": 2}]},
        }
        tracks = {
            base_path: {"video": [{"tid": 0, "length": 1000}]},
            other_path: {"video": [{"tid": 0, "length": 1000}]},
        }

        source = self.analyzer._pick_chapter_source(
            details,
            tracks,
            [VideoStreamRef(base_path, 0, 0, None)],
            {base_path: 1, other_path: 2},
        )

        self.assertEqual(source, other_path)

    def test_pick_chapter_source_skips_different_length_non_base_video(self):
        base_path = os.path.join(self.wd.path, "base.mkv")
        other_path = os.path.join(self.wd.path, "other.mkv")
        details = {
            base_path: {"chapters": []},
            other_path: {"chapters": [{"num_entries": 2}]},
        }
        tracks = {
            base_path: {"video": [{"tid": 0, "length": 1000}]},
            other_path: {"video": [{"tid": 0, "length": 1001}]},
        }

        source = self.analyzer._pick_chapter_source(
            details,
            tracks,
            [VideoStreamRef(base_path, 0, 0, None)],
            {base_path: 1, other_path: 2},
        )

        self.assertIsNone(source)

    def test_analyze_duplicates_skips_only_group_with_unsupported_input(self):
        unsupported_path = os.path.join(self.wd.path, "unsupported.mkv")
        valid_path = os.path.join(self.wd.path, "valid.mkv")
        base_plan = [{
            "title": "Title",
            "groups": [
                {"files": [unsupported_path], "output_name": "unsupported"},
                {"files": [valid_path], "output_name": "valid"},
            ],
        }]
        valid_details = {"marker": "analyzed"}
        unsupported_issue = "Buttons are not supported yet"

        with self.assertLogs(self.logger, level="INFO") as logged, \
             patch.object(
                 self.analyzer,
                 "_prepare_duplicates_set",
                 return_value=base_plan,
             ), patch.object(
                 self.analyzer,
                 "_analyze_group",
                 side_effect=[
                     UnsupportedMeltInputError(unsupported_issue),
                     (valid_details, None, {}),
                 ],
             ):
            plan = self.analyzer.analyze_duplicates({})

        first_input = next(index for index, message in enumerate(logged.output) if f"#1: {unsupported_path}" in message)
        issue = next(index for index, message in enumerate(logged.output) if unsupported_issue in message)
        self.assertLess(first_input, issue)
        self.assertNotIn("Title Title: Buttons are not supported yet", logged.output[issue])

        self.assertEqual(
            [{
                "title": "Title",
                "groups": [{
                    "files": [valid_path],
                    "output_name": "valid",
                    **valid_details,
                }],
                "skipped_groups": [{
                    "files": [unsupported_path],
                    "output_name": "unsupported",
                    "issue": unsupported_issue,
                }],
            }],
            plan,
        )

    def test_analyze_duplicates_propagates_unrelated_runtime_error(self):
        input_path = os.path.join(self.wd.path, "input.mkv")
        base_plan = [{
            "title": "Title",
            "groups": [{"files": [input_path], "output_name": "input"}],
        }]

        with patch.object(
            self.analyzer,
            "_prepare_duplicates_set",
            return_value=base_plan,
        ), patch.object(
            self.analyzer,
            "_analyze_group",
            side_effect=RuntimeError("ffprobe failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "ffprobe failed"):
                self.analyzer.analyze_duplicates({})

    def test_analyze_duplicates_displays_paths_relative_to_input(self):
        input_dir = os.path.join(self.wd.path, "input")
        input_path = os.path.join(input_dir, "nested", "input.mkv")
        self.analyzer.input_paths = (input_dir,)
        base_plan = [{
            "title": "Title",
            "groups": [{"files": [input_path], "output_name": "input"}],
        }]

        with self.assertLogs(self.logger, level="INFO") as logged, \
             patch.object(self.analyzer, "_prepare_duplicates_set", return_value=base_plan), \
             patch.object(self.analyzer, "_analyze_group", return_value=({}, None, {})):
            self.analyzer.analyze_duplicates({})

        relative_path = os.path.join("nested", "input.mkv")
        self.assertTrue(any(f"#1: {relative_path}" in message for message in logged.output))

    @parameterized.expand([
        ("content_mismatch_without_flag", "Video content mismatch", False),
        ("content_mismatch_with_flag", "Video content mismatch", True),
        ("length_mismatch_without_flag", "Video length mismatch", False),
        ("length_mismatch_with_flag", "Video length mismatch", True),
    ])
    def test_analyze_group_rejects_subtitles_when_source_requires_timeline_alignment(
        self,
        _name,
        requirement_issue,
        allow_video_timeline_mismatch,
    ):
        base_path = os.path.join(self.wd.path, "base.mkv")
        subtitle_path = os.path.join(self.wd.path, "subtitle-source.mkv")
        tracks = {
            base_path: {"video": [{"tid": 0, "length": 6000}], "audio": [], "subtitle": []},
            subtitle_path: {"video": [{"tid": 0, "length": 6000}], "audio": [], "subtitle": []},
        }
        details = {path: {"tracks": value, "attachments": []} for path, value in tracks.items()}
        self.analyzer.allow_video_timeline_mismatch = allow_video_timeline_mismatch

        with patch.object(self.analyzer, "_probe_inputs", return_value=(details, {base_path: [], subtitle_path: []}, tracks)), \
             patch.object(
                 self.analyzer,
                 "_pick_streams",
                 return_value=(
                     [VideoStreamRef(base_path, 0, 0, None)],
                     [AudioStreamRef(base_path, 1, 1, "eng")],
                     [SubtitleStreamRef(subtitle_path, 2, 2, "pol")],
                 ),
             ), patch.object(
                 self.analyzer,
                 "_find_alignment_requirements",
                 return_value=[AlignmentRequirement(subtitle_path, requirement_issue)],
             ):
            plan, issue, _ = self.analyzer._analyze_group(
                [base_path, subtitle_path], {base_path: 1, subtitle_path: 2}, "Title",
            )

        self.assertIsNone(plan)
        self.assertEqual(
            issue,
            "Subtitle streams from #2 require video timeline alignment, which is not supported yet.",
        )


class MeltInputFilesTest(unittest.TestCase):
    def test_assigns_stable_one_based_ids_and_formats_paths_from_inputs(self):
        files = MeltInputFiles(
            ["/media/first.mkv", "/media/input/nested/second.mkv", "/net/library/third.mkv"],
            input_paths=["/media/first.mkv", "/media/input"],
        )

        self.assertEqual(files.id_for("/media/first.mkv"), 1)
        self.assertEqual(files.id_for("/media/input/nested/second.mkv"), 2)
        self.assertEqual(files.reference("/media/input/nested/second.mkv"), "file #2")
        self.assertEqual(files.display_path("/media/first.mkv"), "/media/first.mkv")
        self.assertEqual(
            files.display_path("/media/input/nested/second.mkv"),
            os.path.join("nested", "second.mkv"),
        )
        self.assertEqual(files.display_path("/net/library/third.mkv"), "/net/library/third.mkv")
        self.assertEqual(files.ids, {
            "/media/first.mkv": 1,
            "/media/input/nested/second.mkv": 2,
            "/net/library/third.mkv": 3,
        })


if __name__ == "__main__":
    unittest.main()
