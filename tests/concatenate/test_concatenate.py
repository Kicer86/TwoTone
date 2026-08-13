
import os
import shutil
import unittest
from pathlib import Path

from twotone.tools.utils.files_utils import split_path
from twotone.tools.utils import process_utils, video_utils
from common import TwoToneTestCase, add_test_media, list_files, run_twotone


class ConcatenateTests(TwoToneTestCase):
    def _create_media(self, wd: str, base_file: str, partnames: list[str]):
        media_file_components = split_path(base_file)

        def build_part(part: str) -> str:
            return os.path.join(wd, media_file_components[1]) + part + "." + media_file_components[2]

        for partname in partnames:
            target = build_part(partname)
            shutil.copy2(base_file, target)


    def _setup_valid_media(self, wd: str):
        media_files = add_test_media("Frog.*mp4", wd)

        wdX = [os.path.join(wd, str(i)) for i in range(7)]

        for wdx in wdX:
            os.makedirs(wdx)

        self.assertEqual(len(media_files), 1)
        media_file = media_files[0]

        self._create_media(wdX[0], media_file, [" CD1", " CD2"])
        self._create_media(wdX[1], media_file, ["-CD1", "-CD2", "-CD3", "-CD4"])
        self._create_media(wdX[2], media_file, [".CD1", ".CD2", ".CD3", ".CD4", ".CD5", ".CD6", ".CD7"])
        self._create_media(wdX[3], media_file, [" cd1", " cd2", " cd3", " cd4", " cd5", " cd6", " cd7", " cd8", " cd9", " cd10", " cd11", " cd12", ])
        self._create_media(wdX[4], media_file, [" Cd01", " Cd02", " Cd03", " Cd04", " Cd05", " Cd06", " Cd07", " Cd08", " Cd09", " Cd010", " Cd11", " Cd12"])

        shutil.copy2(media_file, os.path.join(wdX[5], "cd1.mp4"))
        shutil.copy2(media_file, os.path.join(wdX[5], "cd2.mp4"))

        shutil.copy2(media_file, os.path.join(wdX[6], "''v'' cd1.mp4"))
        shutil.copy2(media_file, os.path.join(wdX[6], "''v'' cd2.mp4"))


    def _setup_invalid_media(self, wd: str):
        media_files = add_test_media("Frog.*mp4", wd)

        wdX = [os.path.join(wd, str(i)) for i in range(4)]

        for wdx in wdX:
            os.makedirs(wdx)

        self.assertEqual(len(media_files), 1)
        media_file = media_files[0]

        self._create_media(wdX[0], media_file, [" CD2"])
        self._create_media(wdX[1], media_file, ["-CD1"])
        self._create_media(wdX[2], media_file, [".CD1", ".CD2", ".CD3", ".CD5", ".CD6", ".CD7"])
        shutil.copy2(media_file, os.path.join(wdX[3], "cd1.mp4"))

        return wdX


    def test_dry_run_is_respected(self):
        self._setup_valid_media(self.wd.path)

        files_before = list_files(self.wd.path)
        run_twotone("concatenate", [self.wd.path])

        files_after = list_files(self.wd.path)
        self.assertEqual(files_after, files_before)


    def test_concatenation(self):
        self._setup_valid_media(self.wd.path)

        run_twotone("concatenate", [self.wd.path], ["-r"])

        files_after = list_files(self.wd.path)
        tdl = len(self.wd.path) + 1

        short_paths = [path[tdl:] for path in files_after]
        short_paths.sort()
        short_paths = [Path(path).as_posix() for path in short_paths]
        self.assertEqual(short_paths, ['0/Frog - 113403.mp4', '1/Frog - 113403.mp4', '2/Frog - 113403.mp4', '3/Frog - 113403.mp4', '4/Frog - 113403.mp4', '5/5.mp4', "6/''v''.mp4", 'Frog - 113403.mp4'])


    def test_explicit_relative_files(self):
        media_file = add_test_media("Frog.*mp4", self.wd.path)[0]
        os.makedirs(os.path.join(self.wd.path, "first"))
        os.makedirs(os.path.join(self.wd.path, "second"))
        first_file = os.path.join(self.wd.path, "first", "part.mkv")
        second_file = os.path.join(self.wd.path, "second", "part.mkv")
        shutil.copy2(media_file, first_file)
        shutil.copy2(media_file, second_file)

        previous_cwd = os.getcwd()
        os.chdir(self.wd.path)
        self.addCleanup(os.chdir, previous_cwd)
        run_twotone("concatenate", ["first/part.mkv", "second/part.mkv", "--output", "combined.mkv"], ["-r"])

        self.assertTrue(os.path.exists(os.path.join(self.wd.path, "combined.mkv")))
        self.assertFalse(os.path.exists(first_file))
        self.assertFalse(os.path.exists(second_file))


    def test_mkv_concatenation_preserves_chapters_and_attachments(self):
        media_file = add_test_media("Frog.*mp4", self.wd.path)[0]
        first_part = os.path.join(self.wd.path, "first.mkv")
        second_part = os.path.join(self.wd.path, "second.mkv")
        output = os.path.join(self.wd.path, "combined.mkv")
        first_attachment = os.path.join(self.wd.path, "first-font.txt")
        second_attachment = os.path.join(self.wd.path, "second-font.txt")
        with open(first_attachment, "w") as file:
            file.write("attachment")
        with open(second_attachment, "w") as file:
            file.write("attachment")

        self._create_mkv_part(media_file, first_part, "First part", first_attachment)
        self._create_mkv_part(media_file, second_part, "Second part", second_attachment)

        run_twotone("concatenate", [first_part, second_part, "--output", output], ["-r"])

        output_info = video_utils.get_video_full_info_mkvmerge(output, logger=self.logger)
        self.assertEqual(sum(chapter["num_entries"] for chapter in output_info["chapters"]), 2)
        self.assertEqual(len(output_info["attachments"]), 2)

        chapters_path = os.path.join(self.wd.path, "chapters.txt")
        status = process_utils.start_process("mkvextract", [output, "chapters", "--simple", chapters_path], logger=self.logger)
        self.assertEqual(status.returncode, 0, status.stderr)
        with open(chapters_path) as file:
            chapters = file.read()
        self.assertIn("CHAPTER01NAME=First part", chapters)
        self.assertIn("CHAPTER02NAME=Second part", chapters)

    def _create_mkv_part(self, media_file: str, output: str, chapter_name: str, attachment: str | None = None):
        chapters_path = f"{output}.chapters.txt"
        with open(chapters_path, "w") as file:
            file.write(f"CHAPTER01=00:00:00.000\nCHAPTER01NAME={chapter_name}\n")

        args = ["-o", output, "--chapters", chapters_path]
        if attachment:
            args.extend(["--attach-file", attachment])
        args.append(media_file)
        status = process_utils.start_process("mkvmerge", args, logger=self.logger)
        self.assertIn(status.returncode, (0, 1), status.stderr)


    def test_invalid_scenarios(self):
        cases = self._setup_invalid_media(self.wd.path)
        files_before = list_files(self.wd.path)
        for case in cases:
            run_twotone("concatenate", [case], ["-r"])

        files_after = list_files(self.wd.path)
        self.assertEqual(files_after, files_before)


if __name__ == '__main__':
    unittest.main()
