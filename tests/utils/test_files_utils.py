
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


if __name__ == "__main__":
    unittest.main()
