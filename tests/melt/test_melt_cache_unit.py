import logging
import os
import tempfile
import unittest
from unittest.mock import mock_open, patch

from twotone.tools.melt.melt_cache import MeltCache


class MeltCacheUnitTest(unittest.TestCase):
    def test_code_hash_includes_all_media_analysis_implementations(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MeltCache(cache_dir, logging.getLogger("test.MeltCache"))
            source = mock_open(read_data=b"source")

            with patch("builtins.open", source), \
                 patch.object(os.path, "isfile", return_value=True):
                cache._code_hash()

        hashed_paths = {os.path.basename(call.args[0]) for call in source.call_args_list}
        self.assertEqual(
            hashed_paths,
            {"media_analysis.py", "pair_matcher.py", "video_utils.py"},
        )


if __name__ == "__main__":
    unittest.main()
