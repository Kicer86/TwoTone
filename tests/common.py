
import hashlib
import inspect
import logging
import os
import re
import shutil
import tempfile
import unittest

from contextlib import contextmanager
from pathlib import Path
from platformdirs import user_cache_dir
from typing import Dict, List, Union
from unittest.mock import patch

from twotone.tools.utils import video_utils

import twotone.twotone

from twotone.tools.utils import files_utils, generic_utils, process_utils, subtitles_utils


current_path = os.path.dirname(os.path.abspath(__file__))
generic_utils.DISABLE_PROGRESSBARS = True


class WorkingDirectoryForTest:
    def __init__(self, class_name: str | None = None, test_name: str | None = None):
        self.directory = None
        self.class_name = class_name
        self.test_name = test_name

    @property
    def path(self):
        return self.directory

    def __enter__(self):
        cname = self.class_name
        tname = self.test_name
        if cname is None or tname is None:
            stack_level = inspect.stack()[1]
            frame = stack_level.frame
            if cname is None:
                if 'self' in frame.f_locals:
                    cname = frame.f_locals['self'].__class__.__name__
                elif 'cls' in frame.f_locals:
                    cname = frame.f_locals['cls'].__name__
                else:
                    cname = ""
            if tname is None:
                tname = stack_level.function

        self.directory = os.path.join(tempfile.gettempdir(), "twotone_tests", cname, tname)
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)

        os.makedirs(self.directory, exist_ok=True)
        return self

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.directory)


class TwoToneTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        logging.getLogger().setLevel(logging.CRITICAL)
        cls.logger = logging.getLogger(cls.__name__)

    def setUp(self):
        super().setUp()
        self.logger = self.__class__.logger
        self.wd = WorkingDirectoryForTest(self.__class__.__name__, self._testMethodName)
        self.wd.__enter__()

    def tearDown(self):
        self.wd.__exit__(None, None, None)
        super().tearDown()


class FileCache:
    def __init__(self, app_name: str):
        self.base_dir = Path(user_cache_dir(app_name))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_or_generate(self, name: str, version: str, extension: str, generator_fn) -> Path:
        """
        name: logical name (e.g. 'embedding-A')
        version: version string (e.g. 'v1.0')
        extension: file extension without dot (e.g. 'json', 'npy')
        generator_fn(out_path: Path) -> None
        """
        filename = f"{name}_{version}.{extension}"
        out_path = self.base_dir / filename

        if out_path.exists():
            return out_path

        # Clean up older versions with the same name and extension
        for file in self.base_dir.glob(f"{name}_*.{extension}"):
            if file != out_path:
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Warning: could not delete {file}: {e}")

        # Run generator
        generator_fn(out_path)

        if not out_path.exists():
            raise RuntimeError(f"Generator did not produce expected file: {out_path}")

        return out_path


def list_files(path: str) -> List[str]:
    results = []

    for root, _, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)

            if os.path.isfile(filepath):
                results.append(filepath)

    return results


def remove_key(obj, key_to_remove):
    if isinstance(obj, dict):
        return {
            k: remove_key(v, key_to_remove)
            for k, v in obj.items()
            if k != key_to_remove
        }
    elif isinstance(obj, list):
        return [remove_key(item, key_to_remove) for item in obj]
    else:
        return obj


def add_test_media(filter: str, test_case_path: str, suffixes: List[str] | None = None, copy: bool = False) -> List[str]:
    suffixes = suffixes or [""]
    filter_regex = re.compile(filter)
    output_files = []

    suffixes = suffixes or [None]

    for media in ["subtitles", "subtitles_txt", "videos"]:
        for root, _, files in os.walk(os.path.join(current_path, media)):
            for file in files:
                if filter_regex.fullmatch(file):
                    for suffix in suffixes:
                        suffix = "-" + suffix if suffix else ""
                        file_path = Path(os.path.join(root, file))
                        dst_file_name = file_path.stem + suffix + file_path.suffix

                        src = os.path.join(current_path, media, file_path)
                        dst = add_to_test_dir(test_case_path, src, copy, dst_file_name)

                        output_files.append(dst)

    return output_files


def add_to_test_dir(test_case_path: str, file_path: str, copy: bool = False, dst_file_name: Union[str, None] = None) -> str:
    basename = os.path.basename(file_path) if dst_file_name is None else dst_file_name
    dst = os.path.join(test_case_path, basename)

    if copy:
        shutil.copy2(file_path, dst)
    else:
        os.symlink(file_path, dst)

    return dst


def get_audio(name: str) -> str:
    return os.path.join(current_path, "audio", name)


def get_video(name: str) -> str:
    return os.path.join(current_path, "videos", name)


def get_image(name: str) -> str:
    return os.path.join(current_path, "images", name)


def get_subtitle(name: str) -> str:
    return os.path.join(current_path, "subtitles", name)


def build_test_video(
    output_path: str,
    wd: str,
    video_name: str | None,
    *,
    audio_name: Union[
        str,
        bool,
        tuple[Union[str, bool], str],
        List[Union[str, bool, tuple[Union[str, bool], str]]],
        None,
    ] = None,
    subtitle: Union[
        str,
        bool,
        tuple[str, str],
        List[Union[str, bool, tuple[str, str]]],
        None,
    ] = None,
    thumbnail_name: Union[str, None] = None,
    duration: int | None = None,
    width: int = 640,
    height: int = 480,
) -> str:
    def _parse_media(media, get_func, build_func):
        if media is None:
            return []
        if isinstance(media, (str, bool)):
            path = get_func(media)
            return [build_func(path)]
        if isinstance(media, tuple):
            path = get_func(media[0])
            lang = media[1]
            return [build_func(path, lang)]
        if isinstance(media, list):
            result = [subitem for item2 in media for subitem in _parse_media(item2, get_func, build_func)]
            return result
        return []

    with tempfile.TemporaryDirectory(dir=wd) as tmp_dir:
        class TempFileManager:
            def __init__(self, tmp_dir):
                self.tmp_dir = tmp_dir
                self.counter = 0
            def next(self, prefix, ext):
                self.counter += 1
                return os.path.join(self.tmp_dir, f"{prefix}_{self.counter}.{ext}")

        temp_files = TempFileManager(tmp_dir)

        if video_name is None:
            if duration is None:
                raise ValueError("duration must be provided when video_name is None")
            generated_video = temp_files.next("generated_video", "mp4")
            process_utils.start_process(
                "ffmpeg",
                [
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=black:s={width}x{height}:d={duration}",
                    "-pix_fmt",
                    "yuv420p",
                    generated_video,
                ],
            )
            original_video_path = generated_video
        else:
            original_video_path = get_video(video_name)

        stripped_video_path = temp_files.next("stripped_video", "mkv")
        process_utils.start_process(
            "ffmpeg",
            [
                "-i", original_video_path,
                "-map", "0:v:0",
                "-c:v", "copy",
                "-an",
                "-sn",
                "-dn",
                stripped_video_path,
            ],
        )
        video_path = stripped_video_path
        thumbnail_path = None if thumbnail_name is None else get_image(thumbnail_name)

        def subtitle_builder(path, lang=None):
            return subtitles_utils.build_subtitle_from_path(path, lang) if lang else subtitles_utils.build_subtitle_from_path(path)

        def subtitle_getter(sub):
            if isinstance(sub, bool):
                if sub:
                    video_length = video_utils.get_video_duration(video_path)
                    subtitle_path = temp_files.next("temporary_subtitle", "srt")
                    generate_subrip_subtitles(subtitle_path, length=video_length)
                    return subtitle_path
                else:
                    return None
            else:
                return get_subtitle(sub)

        def audio_builder(path, lang=None):
            return subtitles_utils.build_audio_from_path(path, lang) if lang else subtitles_utils.build_audio_from_path(path)

        def audio_getter(aud):
            if isinstance(aud, bool):
                if aud:
                    video_length = video_utils.get_video_duration(video_path)
                    audio_path = temp_files.next("temporary_audio", "wav")
                    # Generate simple sine wave audio using ffmpeg
                    duration_sec = max(1, int(video_length / 1000))
                    process_utils.start_process(
                        "ffmpeg",
                        [
                            "-y",
                            "-f", "lavfi",
                            "-i", f"sine=frequency=1000:sample_rate=44100:duration={duration_sec}",
                            audio_path,
                        ],
                    )
                    return audio_path
                else:
                    return None
            else:
                return get_audio(aud)

        subtitle_objs = None
        if subtitle is not None:
            subtitle_objs = _parse_media(subtitle, subtitle_getter, subtitle_builder)
            if isinstance(subtitle_objs, list):
                subtitle_objs = [s for s in subtitle_objs if s]
                if not subtitle_objs:
                    subtitle_objs = None

        audio_objs = None
        if audio_name is not None:
            audio_objs = _parse_media(audio_name, audio_getter, audio_builder)
            if isinstance(audio_objs, list):
                audio_objs = [a for a in audio_objs if a]
                if not audio_objs:
                    audio_objs = None

        video_utils.generate_mkv(
            output_path,
            video_path,
            subtitle_objs,
            audio_objs,
            thumbnail_path,
        )

        return output_path


def assert_video_info(testcase: unittest.TestCase, path: str,
                      expected_video_tracks: int = 1,
                      expected_subtitles: int | None = None):
    testcase.assertTrue(path.endswith(".mkv"))
    tracks = video_utils.get_video_data(path)
    testcase.assertEqual(len(tracks["video"]), expected_video_tracks)
    if expected_subtitles is not None:
        testcase.assertEqual(len(tracks["subtitle"]), expected_subtitles)
    return tracks


def hashes(path: str) -> Dict[str, str]:
    results = {}

    files = list_files(path)

    for filepath in files:
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)

            results[filepath] = file_hash.hexdigest()

    return results


def generate_microdvd_subtitles(path: str, length: int, fps: float = 60):
    with open(path, "w") as sf:
        b = 0
        e = 0
        for t in range(length):
            b = int(round(t * fps, 0))
            e = int(round(b + fps/2, 0))
            sf.write(f"{{{b}}}{{{e}}}{t}\n")

        # microdvd requires at least 3 entries for ffmpeg to consume it
        if length < 3:
            # add some empty entries to satisfy ffmpeg
            sf.write(f"{{{e}}}{{{e + 1}}}\n")
            sf.write(f"{{{e + 1}}}{{{e + 2}}}\n")


def generate_subrip_subtitles(path: str, length: int):
    content = []

    for i, t in enumerate(range(0, length, 1000)):
        b = generic_utils.ms_to_time(t)
        e = generic_utils.ms_to_time(t + 100)

        content.append(str(i + 1))
        content.append(f"{b} --> {e}")
        content.append(str(i))
        content.append("\n")

    write_subtitle(path, content)


def write_subtitle(path: str, lines: list[str], *, encoding: str = "utf-8") -> str:
    with open(path, "w", encoding=encoding) as f:
        for line in lines:
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")
    return path


def extract_subtitles(video_path: str, out_path: str):
    process_utils.start_process("ffmpeg", ["-i", video_path, "-map", "0:s:0", out_path])


def run_twotone(tool: str, tool_options = [], global_options = None):
    if global_options is None:
        global_options = []

    for opt in global_options:
        if opt in ("-w", "--working-dir") or opt.startswith("--working-dir=") or opt.startswith("-w="):
            raise ValueError("Tests must not override working directory")

    wd = generic_utils.get_twotone_working_dir()
    os.makedirs(wd, exist_ok=True)

    global_options.extend(["--quiet", "--working-dir", wd])

    twotone.twotone.execute([*global_options, tool, *tool_options])


@contextmanager
def simulate_process_failure(target_exec: str):
    original = process_utils.start_process

    def wrapper(cmd, args):
        _, exec_name, _ = files_utils.split_path(cmd)
        if exec_name == target_exec:
            return process_utils.ProcessResult(1, b"", b"")
        return original(cmd, args)

    with patch("twotone.tools.utils.process_utils.start_process", side_effect=wrapper) as p:
        yield p
