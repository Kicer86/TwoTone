import hashlib
import logging
import os
import tempfile
import unittest
from unittest.mock import mock_open, patch

from twotone.tools.melt.melt_cache import MeltCache


class MeltCacheUnitTest(unittest.TestCase):
    def test_cache_key_does_not_reuse_legacy_timestamp_data(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            video_path = os.path.join(cache_dir, "input.mkv")
            with open(video_path, "wb") as file:
                file.write(b"video")

            stat = os.stat(video_path)
            legacy_identity = (
                f"{os.path.realpath(video_path)}:{stat.st_size}:{stat.st_mtime_ns}"
            )
            legacy_key = hashlib.sha256(legacy_identity.encode()).hexdigest()[:16]

            cache = MeltCache(cache_dir, logging.getLogger("test.MeltCache"))

            self.assertNotEqual(legacy_key, cache._cache_key(video_path))

    def test_cache_key_does_not_reuse_old_timestamp_data(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            video_path = os.path.join(cache_dir, "input.avi")
            with open(video_path, "wb") as file:
                file.write(b"video")

            stat = os.stat(video_path)
            cache = MeltCache(cache_dir, logging.getLogger("test.MeltCache"))
            for version in (2, 3):
                with self.subTest(version=version):
                    old_identity = (
                        f"{version}:{os.path.realpath(video_path)}:{stat.st_size}:{stat.st_mtime_ns}"
                    )
                    old_key = hashlib.sha256(old_identity.encode()).hexdigest()[:16]
                    self.assertNotEqual(old_key, cache._cache_key(video_path))

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
