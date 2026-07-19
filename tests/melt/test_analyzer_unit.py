import os
import unittest

from parameterized import parameterized
from unittest.mock import patch

from common import TwoToneTestCase
from twotone.tools.melt.melt import MeltAnalyzer, StaticSource
from twotone.tools.melt.melt_analyzer import UnsupportedMeltInputError
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
            allow_length_mismatch=False,
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
        (
            "chapters",
            {"chapters": [{"num_entries": 2}]},
            "chapters",
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
                self.analyzer._probe_inputs([input_path])

        message = str(raised.exception)
        self.assertIn(expected_description, message)
        self.assertIn("not supported yet", message)
        self.assertIn(input_path, message)
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
                self.analyzer._probe_inputs([first_path, second_path])

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
            details, attachments, tracks = self.analyzer._probe_inputs([input_path])

        self.assertEqual({input_path: parsed_info}, details)
        self.assertEqual({input_path: parsed_info["attachments"]}, attachments)
        self.assertEqual({input_path: parsed_info["tracks"]}, tracks)
        raw_probe.assert_called_once_with(input_path, logger=self.logger)
        parse_info.assert_called_once_with(
            input_path,
            enrich=True,
            logger=self.logger,
            _mkvmerge_info=raw_info,
        )

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

        with patch.object(
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


if __name__ == "__main__":
    unittest.main()
