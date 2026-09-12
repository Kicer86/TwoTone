import logging
import os
import tempfile
import unittest

from unittest.mock import patch

from twotone.tools.utils import files_utils, generic_utils, media_analysis, process_utils


class MediaAnalysisSessionTest(unittest.TestCase):
    def setUp(self):
        self.workspace = files_utils.Workspace.temporary()
        self.addCleanup(self.workspace.close)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = os.path.join(self.temp_dir.name, "input.mkv")
        with open(self.path, "wb") as file:
            file.write(b"media")

        self.session = media_analysis.MediaAnalysisSession(
            self.workspace,
            generic_utils.InterruptibleProcess(),
            logging.getLogger("MediaAnalysisSessionTest"),
            validate_all_streams=True,
        )

    def test_probe_reuses_result_for_the_same_unchanged_file(self):
        success = process_utils.ProcessResult(
            0,
            '{"streams": [{"codec_type": "video", "codec_name": "h264"}]}',
            "",
        )

        with patch.object(process_utils, "start_process", return_value=success) as start_process:
            first = self.session.probe(self.path)
            second = self.session.probe(self.path)

        self.assertIs(first, second)
        self.assertTrue(first.has_video)
        start_process.assert_called_once()


if __name__ == "__main__":
    unittest.main()
