
import os
import shutil
import tempfile
import unittest

from twotone.tools.utils import files_utils


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_workspace_is_path_like(self):
        workspace = files_utils.Workspace(self.root)

        self.assertEqual(os.fspath(workspace), self.root)
        self.assertEqual(os.path.join(workspace, "x"), os.path.join(self.root, "x"))

    def test_unique_names_do_not_repeat_within_a_run(self):
        workspace = files_utils.Workspace(self.root)

        first = workspace.unique_file("clip", "mkv")
        second = workspace.unique_file("clip", "mkv")

        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith(".mkv"))

    def test_unique_names_skip_leftovers_from_previous_runs(self):
        leftover = os.path.join(self.root, "clip-0.mkv")
        with open(leftover, "w"):
            pass

        workspace = files_utils.Workspace(self.root)
        path = workspace.unique_file("clip", "mkv")

        self.assertNotEqual(path, leftover)

    def test_unique_dir_is_created(self):
        workspace = files_utils.Workspace(self.root)

        path = workspace.unique_dir("frames")

        self.assertTrue(os.path.isdir(path))

    def test_scoped_dir_is_removed_on_exit(self):
        workspace = files_utils.Workspace(self.root)

        with workspace.scoped_dir("matching") as path:
            self.assertTrue(os.path.isdir(path))
            with open(os.path.join(path, "data"), "w"):
                pass

        self.assertFalse(os.path.exists(path))

    def test_scoped_dir_survives_in_keep_mode(self):
        workspace = files_utils.Workspace(self.root, keep=True)

        with workspace.scoped_dir("matching") as path:
            pass

        self.assertTrue(os.path.isdir(path))

    def test_text_file_holds_content_and_is_removed_on_exit(self):
        workspace = files_utils.Workspace(self.root)

        with workspace.text_file("hello", "txt") as path:
            with open(path) as text_file:
                self.assertEqual(text_file.read(), "hello")

        self.assertFalse(os.path.exists(path))

    def test_text_file_survives_in_keep_mode(self):
        workspace = files_utils.Workspace(self.root, keep=True)

        with workspace.text_file("hello") as path:
            pass

        self.assertTrue(os.path.exists(path))

    def test_subdir_shares_uniqueness_with_parent(self):
        workspace = files_utils.Workspace(self.root)

        child = workspace.subdir("melt")
        child_file = child.unique_file("clip")
        parent_file = workspace.unique_file("clip")

        self.assertTrue(child_file.startswith(os.path.join(self.root, "melt") + os.sep))
        self.assertNotEqual(os.path.basename(child_file), os.path.basename(parent_file))

    def test_remove_created_removes_only_own_entries(self):
        foreign = os.path.join(self.root, "user_data")
        os.makedirs(foreign)

        workspace = files_utils.Workspace(self.root)
        own_dir = workspace.unique_dir("frames")
        own_file = workspace.unique_file("clip", "mkv")
        with open(own_file, "w"):
            pass

        workspace.remove_created()

        self.assertFalse(os.path.exists(own_dir))
        self.assertFalse(os.path.exists(own_file))
        self.assertTrue(os.path.isdir(foreign))

    def test_remove_created_is_a_no_op_in_keep_mode(self):
        workspace = files_utils.Workspace(self.root, keep=True)
        own_dir = workspace.unique_dir("frames")

        workspace.remove_created()

        self.assertTrue(os.path.isdir(own_dir))


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.user_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.user_dir, ignore_errors=True)
        self.workspace = files_utils.Workspace(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.workspace.root, ignore_errors=True)
        self.target = os.path.join(self.user_dir, "movie.mkv")

    def test_staging_file_lives_visible_next_to_target(self):
        with self.workspace.staging_for(self.target) as staging:
            directory, name = os.path.split(staging.path)

            self.assertEqual(directory, self.user_dir)
            self.assertFalse(name.startswith("."))
            self.assertIn("twotone-tmp", name)
            self.assertTrue(name.endswith(".mkv"))

    def test_commit_replaces_target_atomically(self):
        with open(self.target, "w") as target:
            target.write("old")

        with self.workspace.staging_for(self.target) as staging:
            with open(staging.path, "w") as staged:
                staged.write("new")
            staging.commit()

        with open(self.target) as target:
            self.assertEqual(target.read(), "new")
        self.assertFalse(os.path.exists(staging.path))

    def test_uncommitted_staging_is_removed_on_exit(self):
        with self.workspace.staging_for(self.target) as staging:
            with open(staging.path, "w"):
                pass

        self.assertFalse(os.path.exists(staging.path))

    def test_uncommitted_staging_is_removed_on_error(self):
        with self.assertRaises(RuntimeError):
            with self.workspace.staging_for(self.target) as staging:
                with open(staging.path, "w"):
                    pass
                raise RuntimeError("processing failed")

        self.assertFalse(os.path.exists(staging.path))

    def test_staging_survives_when_process_dies_before_context_exit(self):
        # Simulates a hard kill: nothing cleans the file up, and by design
        # it must remain visible in the target directory.
        cm = self.workspace.staging_for(self.target)
        staging = cm.__enter__()
        with open(staging.path, "w"):
            pass

        self.assertTrue(os.path.exists(staging.path))

    def test_staging_dir_shares_filesystem_with_target_and_is_removed(self):
        with self.workspace.staging_dir_for(self.user_dir) as scratch:
            self.assertEqual(os.path.dirname(scratch), self.user_dir)
            inner = os.path.join(scratch, "frame.jpg")
            with open(inner, "w"):
                pass
            os.rename(inner, os.path.join(self.user_dir, "frame.jpg"))

        self.assertFalse(os.path.exists(scratch))
        self.assertTrue(os.path.exists(os.path.join(self.user_dir, "frame.jpg")))


if __name__ == "__main__":
    unittest.main()
