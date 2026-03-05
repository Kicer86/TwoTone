
import logging
import tempfile
import unittest
import os
import platform
import argparse

from functools import partial
from itertools import permutations
from parameterized import parameterized
from pathlib import Path
from typing import Iterator

from twotone.tools.utils import generic_utils, process_utils, video_utils
from twotone.tools.melt.melt import DEFAULT_TOLERANCE_MS, MeltAnalyzer, MeltPerformer, MeltTool, PairMatcher, StaticSource, StreamsPicker
from twotone.tools.utils.files_utils import ScopedDirectory
from twotone.tools.melt.phash_cache import PhashCache
from unittest.mock import patch
from common import (
    TwoToneTestCase,
    FileCache,
    add_test_media,
    add_to_test_dir,
    build_test_video,
    get_audio,
    get_video,
    hashes,
    run_ffmpeg,
)


def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, list):
        return sorted((normalize(item) for item in obj), key=lambda x: repr(x))
    elif isinstance(obj, tuple):
        return tuple(normalize(item) for item in obj)
    else:
        return obj


def all_key_orders(d: dict) -> Iterator[dict]:
    """
    Yield dictionaries with all possible key orderings (same keys and values).
    """
    keys = list(d.keys())
    for perm in permutations(keys):
        yield {k: d[k] for k in perm}


def analyze_duplicates_helper(
    logger: logging.Logger,
    duplicates_source: StaticSource,
    working_dir: str,
    allow_length_mismatch: bool = False,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
):
    os.makedirs(working_dir, exist_ok=True)
    duplicates_raw = duplicates_source.collect_duplicates()
    duplicates = {title: list(files) for title, files in duplicates_raw.items()}
    analyzer = MeltAnalyzer(
        logger,
        duplicates_source,
        working_dir,
        allow_length_mismatch,
        tolerance_ms,
    )
    return analyzer.analyze_duplicates(duplicates)


def process_duplicates_helper(
    logger: logging.Logger,
    interruption: generic_utils.InterruptibleProcess,
    working_dir: str,
    output_dir: str,
    plan,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
):
    performer = MeltPerformer(
        logger,
        interruption,
        working_dir,
        output_dir,
        tolerance_ms,
    )
    performer.process_duplicates(plan)


def _build_path_to_id_map(input: dict) -> dict[str, int]:
    return {path: idx for idx, path in enumerate(input.keys())}


class MeltingTest(TwoToneTestCase):

    def setUp(self):
        super().setUp()

        def gen_sample(out_path: Path):
            videos = ["Atoms - 8579.mp4",
                      "Blue_Sky_and_Clouds_Timelapse_0892__Videvo.mov",
                      "Frog - 113403.mp4", "sea-waves-crashing-on-beach-shore-4793288.mp4",
                      "Woman - 58142.mp4"]
            audios = ["806912__kevp888__250510_122217_fr_large_crowd_in_palais_garnier.wav",
                      "807385__josefpres__piano-loops-066-efect-4-octave-long-loop-120-bpm.wav",
                      "807184__logicmoon__mirrors.wav",
                      "807419__kvgarlic__light-spring-rain-and-kids-and-birds-may-13-2025-vtwo.wav",
                      "806912__kevp888__250510_122217_fr_large_crowd_in_palais_garnier.wav"]

            #unify fps and add audio path
            output_dir = os.path.join(self.wd.path, "gen_sample")

            with ScopedDirectory(output_dir) as sd:
                output_files = []
                for video, audio in zip(videos, audios):
                    video_input_path = get_video(video)
                    audio_input_path = get_audio(audio)
                    output_path = os.path.join(output_dir, video + ".mp4")
                    run_ffmpeg(
                        [
                            "-i", video_input_path,
                            "-i", audio_input_path,
                            "-r", "25",
                            "-vf", "fps=25",
                            "-c:v", "libx264",
                            "-preset", "veryfast",
                            "-crf", "18",
                            "-pix_fmt", "yuv420p",
                            "-shortest",
                            "-map", "0:v:0",
                            "-map", "1:a:0",
                            "-c:a", "aac",
                            "-ar", "44100",
                            output_path,
                        ],
                        expected_path=output_path,
                    )
                    output_files.append(output_path)

                # concatenate
                files_list_path = os.path.join(output_dir, "filelist.txt")
                with open(files_list_path, "w", encoding="utf-8") as f:
                    for path in output_files:
                        # Escape single quotes if needed
                        safe_path = path.replace("'", "'\\''")
                        f.write(f"file '{safe_path}'\n")

                run_ffmpeg(
                    [
                        "-f", "concat",
                        "-safe", "0",
                        "-i", files_list_path,
                        "-c", "copy",
                        str(out_path),
                    ],
                    expected_path=str(out_path),
                )

        def gen_vhs(out_path: Path, input: str):
            """
                Process input file and worse its quality
            """
            duration = video_utils.get_video_duration(input) / 1000                                     # duration of original video

            vf = ",".join([
                "fps=26.5",                                                                             # use non standard fps
                "setpts=PTS/1.05",                                                                      # speed it up by 5%
                "boxblur=enable='between(t,5,10)':lr=2",                                                # add some blur
                f"crop=w=iw*0.9:h=ih*0.9:x='(iw-iw*0.9)*t/{duration}':y='(ih-ih*0.9)*t/{duration}'",    # add a crop (90% H and W) which moves from top left corner to bottom right
                f"scale={960}:{720}"                                                                    # scale to 4:3
            ])

            af = "atempo=1.05"

            args = [
                "-i", input,
                "-filter_complex", vf,
                "-filter:a", af,
                "-c:v", "libx264",
                "-crf", "40",                                                                           # use low quality
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(out_path)
            ]

            run_ffmpeg(args, expected_path=str(out_path))

        file_cache = FileCache("TwoToneTests")

        self.sample_video_file = str(file_cache.get_or_generate("melter_tests_sample", "2", "mp4", gen_sample))
        self.sample_vhs_video_file = str(file_cache.get_or_generate("melter_tests_vhs", "2", "mp4", partial(gen_vhs, input = self.sample_video_file)))

        self._setup_edge_fixtures(file_cache)

    def _setup_edge_fixtures(self, file_cache: FileCache):
        """Pre-generate and cache all edge-case PairMatcher test fixtures.

        Builds a layered dependency tree so shared intermediates (like
        the degraded big_buck_bunny) are generated once and reused.
        """
        V = "1"  # bump to invalidate all edge fixtures

        bbb = get_video("big_buck_bunny_720p_10mb.mp4")
        grass = get_video("Grass - 66810.mp4")
        atoms = get_video("Atoms - 8579.mp4")
        woman = get_video("Woman - 58142.mp4")
        seawaves = get_video("sea-waves-crashing-on-beach-shore-4793288.mp4")

        # --- Level 1: single-operation transforms on bbb ---

        def gen_deg103(out: Path):
            self._degrade_video(bbb, str(out), speed=1.03)

        def gen_deg10(out: Path):
            self._degrade_video(bbb, str(out), speed=1.0)

        def gen_black_intro_3s(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._prepend_black(bbb, str(out), 3.0, tmp_dir=td)

        def gen_black_intro_2s(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._prepend_black(bbb, str(out), 2.0, tmp_dir=td)

        def gen_black_outro_3s(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._append_black(bbb, str(out), 3.0, tmp_dir=td)

        def gen_grass_intro_3s(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._prepend_video(grass, bbb, str(out), intro_seconds=3.0, tmp_dir=td)

        def gen_grass_intro_2s(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._prepend_video(grass, bbb, str(out), intro_seconds=2.0, tmp_dir=td)

        def gen_woman_outro_3s(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._append_video(bbb, woman, str(out), outro_seconds=3.0, tmp_dir=td)

        bbb_deg103 = str(file_cache.get_or_generate("pm_bbb_deg103", V, "mp4", gen_deg103))
        bbb_deg10 = str(file_cache.get_or_generate("pm_bbb_deg10", V, "mp4", gen_deg10))
        bbb_bi3 = str(file_cache.get_or_generate("pm_bbb_bi3", V, "mp4", gen_black_intro_3s))
        bbb_bi2 = str(file_cache.get_or_generate("pm_bbb_bi2", V, "mp4", gen_black_intro_2s))
        bbb_bo3 = str(file_cache.get_or_generate("pm_bbb_bo3", V, "mp4", gen_black_outro_3s))
        bbb_gi3 = str(file_cache.get_or_generate("pm_bbb_gi3", V, "mp4", gen_grass_intro_3s))
        bbb_gi2 = str(file_cache.get_or_generate("pm_bbb_gi2", V, "mp4", gen_grass_intro_2s))
        bbb_wo3 = str(file_cache.get_or_generate("pm_bbb_wo3", V, "mp4", gen_woman_outro_3s))

        # --- Level 2: transforms depending on level 1 ---

        def gen_bi3_deg103(out: Path):
            self._degrade_video(bbb_bi3, str(out), speed=1.03)

        def gen_bi6_from_deg103(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._prepend_black(bbb_deg103, str(out), 6.0, tmp_dir=td)

        def gen_bo3_deg103(out: Path):
            self._degrade_video(bbb_bo3, str(out), speed=1.03)

        def gen_bi2_bo2(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._append_black(bbb_bi2, str(out), 2.0, tmp_dir=td)

        def gen_atoms_i3_deg(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._prepend_video(atoms, bbb_deg103, str(out), intro_seconds=3.0, tmp_dir=td)

        def gen_atoms_i5_deg(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._prepend_video(atoms, bbb_deg103, str(out), intro_seconds=5.0, tmp_dir=td)

        def gen_deg103_atoms_o3(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._append_video(bbb_deg103, atoms, str(out), outro_seconds=3.0, tmp_dir=td)

        bi3_deg103 = str(file_cache.get_or_generate("pm_bi3_deg103", V, "mp4", gen_bi3_deg103))
        bi6_deg103 = str(file_cache.get_or_generate("pm_bi6_deg103", V, "mp4", gen_bi6_from_deg103))
        bo3_deg103 = str(file_cache.get_or_generate("pm_bo3_deg103", V, "mp4", gen_bo3_deg103))
        bi2_bo2 = str(file_cache.get_or_generate("pm_bi2_bo2", V, "mp4", gen_bi2_bo2))
        atoms_i3_deg = str(file_cache.get_or_generate("pm_atoms_i3_deg", V, "mp4", gen_atoms_i3_deg))
        atoms_i5_deg = str(file_cache.get_or_generate("pm_atoms_i5_deg", V, "mp4", gen_atoms_i5_deg))
        deg103_atoms_o3 = str(file_cache.get_or_generate("pm_deg103_atoms_o3", V, "mp4", gen_deg103_atoms_o3))

        # --- Level 3: complex composites ---

        def gen_bi2_bo2_deg103(out: Path):
            self._degrade_video(bi2_bo2, str(out), speed=1.03)

        def gen_gi3_wo3(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._append_video(bbb_gi3, woman, str(out), outro_seconds=3.0, tmp_dir=td)

        def gen_ai3d_swo3(out: Path):
            with tempfile.TemporaryDirectory() as td:
                self._append_video(atoms_i3_deg, seawaves, str(out), outro_seconds=3.0, tmp_dir=td)

        bi2_bo2_deg103 = str(file_cache.get_or_generate("pm_bi2_bo2_deg103", V, "mp4", gen_bi2_bo2_deg103))
        gi3_wo3 = str(file_cache.get_or_generate("pm_gi3_wo3", V, "mp4", gen_gi3_wo3))
        ai3d_swo3 = str(file_cache.get_or_generate("pm_ai3d_swo3", V, "mp4", gen_ai3d_swo3))

        # Expose as fixture pairs: (file_a, file_b) per test scenario
        self.bbb = bbb
        self.edge_fixtures = {
            "black_intro_same":  (bbb_bi3, bi3_deg103),
            "black_intro_diff":  (bbb_bi2, bi6_deg103),
            "black_outro":       (bbb_bo3, bo3_deg103),
            "both_black":        (bi2_bo2, bi2_bo2_deg103),
            "no_speed":          (bbb, bbb_deg10),
            "diff_intro_same":   (bbb_gi3, atoms_i3_deg),
            "diff_intro_diff":   (bbb_gi2, atoms_i5_deg),
            "diff_outro":        (bbb_wo3, deg103_atoms_o3),
            "diff_both":         (gi3_wo3, ai3d_swo3),
        }

    # ---- Private helpers for edge-case fixture generation ----

    def _prepend_black(self, input_path: str, output_path: str, black_seconds: float, tmp_dir: str | None = None) -> str:
        """Prepend *black_seconds* of black video+silence before *input_path*."""
        td = tmp_dir or self.wd.path
        # Get input properties
        data = video_utils.get_video_data(input_path)
        width = int(data["video"][0]["width"])
        height = int(data["video"][0]["height"])
        fps = data["video"][0]["fps"]
        fps_float = generic_utils.fps_str_to_float(fps)
        has_audio = len(data.get("audio", [])) > 0

        black_path = os.path.join(td, "black_intro.mp4")
        args = [
            "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps_float}:d={black_seconds}",
        ]
        if has_audio:
            args += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={black_seconds}"]
            args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", black_path]
        else:
            args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", black_path]
        run_ffmpeg(args, expected_path=black_path)

        # Reencode input to match codec/fps for concat
        reencoded_path = os.path.join(td, "reencoded_input.mp4")
        args = ["-y", "-i", input_path, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-r", str(fps_float), "-vf", f"fps={fps_float},scale={width}:{height}"]
        if has_audio:
            args += ["-c:a", "aac", "-ar", "44100"]
        else:
            args += ["-an"]
        args.append(reencoded_path)
        run_ffmpeg(args, expected_path=reencoded_path)

        # Concat
        filelist = os.path.join(td, "concat_list.txt")
        with open(filelist, "w") as f:
            f.write(f"file '{black_path}'\nfile '{reencoded_path}'\n")

        run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", filelist, "-c", "copy", output_path],
                         expected_path=output_path)
        return output_path

    def _append_black(self, input_path: str, output_path: str, black_seconds: float, tmp_dir: str | None = None) -> str:
        """Append *black_seconds* of black video+silence after *input_path*."""
        td = tmp_dir or self.wd.path
        data = video_utils.get_video_data(input_path)
        width = int(data["video"][0]["width"])
        height = int(data["video"][0]["height"])
        fps = data["video"][0]["fps"]
        fps_float = generic_utils.fps_str_to_float(fps)
        has_audio = len(data.get("audio", [])) > 0

        black_path = os.path.join(td, "black_outro.mp4")
        args = [
            "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps_float}:d={black_seconds}",
        ]
        if has_audio:
            args += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={black_seconds}"]
            args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", black_path]
        else:
            args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", black_path]
        run_ffmpeg(args, expected_path=black_path)

        reencoded_path = os.path.join(td, "reencoded_input_outro.mp4")
        args = ["-y", "-i", input_path, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-r", str(fps_float), "-vf", f"fps={fps_float},scale={width}:{height}"]
        if has_audio:
            args += ["-c:a", "aac", "-ar", "44100"]
        else:
            args += ["-an"]
        args.append(reencoded_path)
        run_ffmpeg(args, expected_path=reencoded_path)

        filelist = os.path.join(td, "concat_list_outro.txt")
        with open(filelist, "w") as f:
            f.write(f"file '{reencoded_path}'\nfile '{black_path}'\n")

        run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", filelist, "-c", "copy", output_path],
                         expected_path=output_path)
        return output_path

    def _degrade_video(self, input_path: str, output_path: str, speed: float = 1.0) -> str:
        """Create a degraded copy: lower quality, optional speed change."""
        duration = video_utils.get_video_duration(input_path) / 1000

        vf_parts = ["fps=26.5", f"scale=640:480", "boxblur=lr=1"]
        if speed != 1.0:
            vf_parts.insert(1, f"setpts=PTS/{speed}")

        af = f"atempo={speed}" if speed != 1.0 else "aresample=44100"

        args = [
            "-y", "-i", input_path,
            "-vf", ",".join(vf_parts),
            "-af", af,
            "-c:v", "libx264", "-crf", "35", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            output_path,
        ]
        run_ffmpeg(args, expected_path=output_path)
        return output_path

    def _reencode_for_concat(self, input_path: str, output_path: str, width: int, height: int, fps: float) -> str:
        """Reencode a video to specific resolution/fps so it can be concatenated with concat demuxer."""
        args = [
            "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-vf", f"fps={fps},scale={width}:{height}",
            "-c:a", "aac", "-ar", "44100",
            "-shortest",
            output_path,
        ]
        run_ffmpeg(args, expected_path=output_path)
        return output_path

    def _concat_videos(self, parts: list[str], output_path: str, tmp_dir: str | None = None) -> str:
        """Concatenate video files using ffmpeg concat demuxer."""
        td = tmp_dir or self.wd.path
        filelist = os.path.join(td, f"concat_{os.path.basename(output_path)}.txt")
        with open(filelist, "w") as f:
            for part in parts:
                f.write(f"file '{part}'\n")
        run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", filelist, "-c", "copy", output_path],
                         expected_path=output_path)
        return output_path

    def _prepend_video(self, intro_source: str, content_path: str, output_path: str, intro_seconds: float | None = None, tmp_dir: str | None = None) -> str:
        """Prepend content from *intro_source* before *content_path*.

        If *intro_seconds* is given, the intro is trimmed to that duration.
        Both parts are reencoded to matching format before concat.
        """
        td = tmp_dir or self.wd.path
        data = video_utils.get_video_data(content_path)
        width = int(data["video"][0]["width"])
        height = int(data["video"][0]["height"])
        fps = generic_utils.fps_str_to_float(data["video"][0]["fps"])

        # Reencode intro
        intro_reenc = os.path.join(td, f"intro_reenc_{os.path.basename(output_path)}")
        trim_args = ["-t", str(intro_seconds)] if intro_seconds else []
        args = ["-y", "-i", intro_source] + trim_args + [
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-vf", f"fps={fps},scale={width}:{height}",
            "-c:a", "aac", "-ar", "44100", "-shortest",
            intro_reenc,
        ]
        run_ffmpeg(args, expected_path=intro_reenc)

        # Reencode content
        content_reenc = os.path.join(td, f"content_reenc_{os.path.basename(output_path)}")
        self._reencode_for_concat(content_path, content_reenc, width, height, fps)

        return self._concat_videos([intro_reenc, content_reenc], output_path, tmp_dir=td)

    def _append_video(self, content_path: str, outro_source: str, output_path: str, outro_seconds: float | None = None, tmp_dir: str | None = None) -> str:
        """Append content from *outro_source* after *content_path*."""
        td = tmp_dir or self.wd.path
        data = video_utils.get_video_data(content_path)
        width = int(data["video"][0]["width"])
        height = int(data["video"][0]["height"])
        fps = generic_utils.fps_str_to_float(data["video"][0]["fps"])

        content_reenc = os.path.join(td, f"content_reenc_outro_{os.path.basename(output_path)}")
        self._reencode_for_concat(content_path, content_reenc, width, height, fps)

        outro_reenc = os.path.join(td, f"outro_reenc_{os.path.basename(output_path)}")
        trim_args = ["-t", str(outro_seconds)] if outro_seconds else []
        args = ["-y", "-i", outro_source] + trim_args + [
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-vf", f"fps={fps},scale={width}:{height}",
            "-c:a", "aac", "-ar", "44100", "-shortest",
            outro_reenc,
        ]
        run_ffmpeg(args, expected_path=outro_reenc)

        return self._concat_videos([content_reenc, outro_reenc], output_path, tmp_dir=td)

    # ---- Test methods ----

    def test_simple_duplicate_detection(self):
        file1 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = ["v1"])[0]
        file2 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = ["v2"])[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Grass", file1)
        duplicates.add_entry("Grass", file2)

        input_file_hashes = hashes(self.wd.path)
        self.assertEqual(len(input_file_hashes), 2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # expect output to be equal to the first of files
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        # check if file was not altered
        self.assertEqual(list(output_file_hash.values())[0], input_file_hashes[file1])


    def test_static_source_production_audio_language_metadata(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        path = "/tmp/fake.mkv" if platform.system() != "Windows" else "c:\tmp\fake.mkv"
        duplicates.add_entry("Some title", path)
        duplicates.add_metadata(path, "audio_prod_lang", "eng")

        self.assertEqual(
            "eng",
            duplicates.get_metadata_for(path)["audio_prod_lang"],
        )
        self.assertIsNone(
            duplicates.get_metadata_for("/not/exists").get("audio_prod_lang")
        )


    def test_dry_run_is_being_respected(self):
        file1 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = ["v1"])[0]
        file2 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = ["v2"])[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Grass", file1)
        duplicates.add_entry("Grass", file2)

        input_file_hashes = hashes(self.wd.path)
        self.assertEqual(len(input_file_hashes), 2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        # Dry run: only prepare plan, do not execute
        _ = analyze_duplicates_helper(logger, duplicates, self.wd.path)

        # expect output to be empty
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 0)

    def test_inputs_are_kept_by_default(self):
        file1 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes=["r1"])[0]
        file2 = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes=["r2"])[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Grass", file1)
        duplicates.add_entry("Grass", file2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        self.assertTrue(os.path.exists(file1))
        self.assertTrue(os.path.exists(file2))
        self.assertEqual(len(hashes(output_dir)), 1)


    def test_skip_on_length_mismatch(self):
        file1 = add_test_media("DSC_8073.MP4", self.wd.path)[0]
        file2 = add_test_media("moon.mp4", self.wd.path)[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)
        duplicates.add_metadata(file1, "audio_lang", "eng")
        duplicates.add_metadata(file2, "audio_lang", "de")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 0)

    def test_allow_length_mismatch(self):
        file1 = add_to_test_dir(self.wd.path, str(self.sample_video_file))
        file2 = add_to_test_dir(self.wd.path, str(self.sample_vhs_video_file))

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)
        duplicates.add_metadata(file1, "audio_lang", "eng")
        duplicates.add_metadata(file2, "audio_lang", "de")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(
            logger,
            duplicates,
            self.wd.path,
            allow_length_mismatch=True,
        )
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash.keys())[0]
        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["audio"]), 2)

    def test_streams_picker_prefers_higher_sample_rate_audio(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file1 = os.path.join(self.wd.path, "audio_48k.mkv")
        file2 = os.path.join(self.wd.path, "audio_24k.mkv")

        files_details = {
            file1: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 6, "sample_rate": 48000}],
                "subtitle": [],
            },
            file2: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 6, "sample_rate": 24000}],
                "subtitle": [],
            },
        }
        ids = {file1: 1, file2: 2}

        _, audio_streams, _ = sp.pick_streams(files_details, ids)

        self.assertEqual(audio_streams[0][0], file1)

    def test_melt_tool_parses_force_all_streams_as_per_input_flag(self):
        parser = argparse.ArgumentParser()
        MeltTool().setup_parser(parser)

        args = parser.parse_args([
            "-o", "/tmp/out",
            "-t", "Example",
            "-i", "/tmp/a.mkv",
            "--force-all-streams",
            "-i", "/tmp/b.mkv",
        ])

        self.assertTrue(args.input_entries[0]["force_all_streams"])
        self.assertNotIn("force_all_streams", args.input_entries[1])

    def test_streams_picker_keeps_forced_streams_including_unknown_language(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file_forced = os.path.join(self.wd.path, "forced.mkv")
        file_other = os.path.join(self.wd.path, "other.mkv")

        duplicates.add_metadata(file_forced, "force_all_streams", True)

        files_details = {
            file_forced: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [
                    {"tid": 1, "language": None, "channels": 2, "sample_rate": 24000},
                    {"tid": 2, "language": "eng", "channels": 2, "sample_rate": 24000},
                ],
                "subtitle": [
                    {"tid": 3, "language": None},
                ],
            },
            file_other: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [
                    {"tid": 5, "language": "pol", "channels": 2, "sample_rate": 48000},
                    {"tid": 6, "language": "eng", "channels": 2, "sample_rate": 96000},
                ],
                "subtitle": [
                    {"tid": 8, "language": "deu"},
                ],
            },
        }
        ids = {file_forced: 1, file_other: 2}

        _, audio_streams, subtitle_streams = sp.pick_streams(files_details, ids)

        self.assertEqual(audio_streams, [
            (file_forced, 1, None),
            (file_forced, 2, "eng"),
            (file_other, 5, "pol"),
        ])
        self.assertEqual(subtitle_streams, [
            (file_forced, 3, None),
            (file_other, 8, "deu"),
        ])

    def test_streams_picker_raises_on_unknown_language_without_force_flag(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file1 = os.path.join(self.wd.path, "unknown_audio_1.mkv")
        file2 = os.path.join(self.wd.path, "unknown_audio_2.mkv")

        files_details = {
            file1: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [{"tid": 1, "language": None, "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
            file2: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "24000/1001"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
        }
        ids = {file1: 1, file2: 2}

        with self.assertRaises(RuntimeError):
            sp.pick_streams(files_details, ids)

    def test_force_all_streams_does_not_affect_video_selection(self):
        """Force flag only applies to audio/subtitle — video uses normal preference."""
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file_forced = os.path.join(self.wd.path, "forced_lo.mkv")
        file_other = os.path.join(self.wd.path, "other_hi.mkv")

        duplicates.add_metadata(file_forced, "force_all_streams", True)

        files_details = {
            file_forced: {
                "video": [{"tid": 0, "width": 640, "height": 480, "fps": "25"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
            file_other: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
        }
        ids = {file_forced: 1, file_other: 2}

        video_streams, _, _ = sp.pick_streams(files_details, ids)

        # Higher resolution from non-forced file should be preferred
        self.assertEqual(video_streams[0][0], file_other)

    def test_force_all_streams_treats_und_as_unknown(self):
        """'und' language is normalized to None, then treated as undefined for forced inputs."""
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file_forced = os.path.join(self.wd.path, "forced_und.mkv")

        duplicates.add_metadata(file_forced, "force_all_streams", True)

        files_details = {
            file_forced: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [{"tid": 1, "language": "und", "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
        }
        ids = {file_forced: 1}

        _, audio_streams, _ = sp.pick_streams(files_details, ids)

        # 'und' → None in output (normalized through undefined bucket)
        self.assertEqual(len(audio_streams), 1)
        self.assertIsNone(audio_streams[0][2])

    def test_force_all_streams_parser_requires_preceding_input(self):
        """--force-all-streams before any -i should fail."""
        parser = argparse.ArgumentParser()
        MeltTool().setup_parser(parser)

        with self.assertRaises(SystemExit):
            parser.parse_args(["--force-all-streams", "-i", "/tmp/a.mkv", "-o", "/out", "-t", "X"])

    def test_force_all_streams_both_inputs_forced_same_language(self):
        """Two forced inputs with the same language keep all streams from both."""
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file_a = os.path.join(self.wd.path, "forced_a.mkv")
        file_b = os.path.join(self.wd.path, "forced_b.mkv")

        duplicates.add_metadata(file_a, "force_all_streams", True)
        duplicates.add_metadata(file_b, "force_all_streams", True)

        files_details = {
            file_a: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [{"tid": 1, "language": "eng", "channels": 2, "sample_rate": 48000}],
                "subtitle": [],
            },
            file_b: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [{"tid": 2, "language": "eng", "channels": 2, "sample_rate": 96000}],
                "subtitle": [],
            },
        }
        ids = {file_a: 1, file_b: 2}

        _, audio_streams, _ = sp.pick_streams(files_details, ids)

        # Both forced — both eng streams kept
        self.assertEqual(len(audio_streams), 2)
        paths = {s[0] for s in audio_streams}
        self.assertEqual(paths, {file_a, file_b})

    def test_force_all_streams_covers_all_languages_non_forced_skipped(self):
        """When forced input covers all unique keys, non-forced contributes nothing."""
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file_forced = os.path.join(self.wd.path, "forced_full.mkv")
        file_other = os.path.join(self.wd.path, "other.mkv")

        duplicates.add_metadata(file_forced, "force_all_streams", True)

        files_details = {
            file_forced: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [
                    {"tid": 1, "language": "eng", "channels": 6, "sample_rate": 48000},
                    {"tid": 2, "language": "pol", "channels": 6, "sample_rate": 48000},
                ],
                "subtitle": [{"tid": 3, "language": "eng"}],
            },
            file_other: {
                "video": [{"tid": 0, "width": 1920, "height": 1080, "fps": "25"}],
                "audio": [
                    {"tid": 4, "language": "eng", "channels": 6, "sample_rate": 96000},
                    {"tid": 5, "language": "pol", "channels": 6, "sample_rate": 96000},
                ],
                "subtitle": [{"tid": 6, "language": "eng"}],
            },
        }
        ids = {file_forced: 1, file_other: 2}

        _, audio_streams, subtitle_streams = sp.pick_streams(files_details, ids)

        # All from forced, nothing from other (same language+channels = same key)
        forced_audio = [s for s in audio_streams if s[0] == file_forced]
        other_audio = [s for s in audio_streams if s[0] == file_other]
        self.assertEqual(len(forced_audio), 2)
        self.assertEqual(len(other_audio), 0)

        forced_subs = [s for s in subtitle_streams if s[0] == file_forced]
        other_subs = [s for s in subtitle_streams if s[0] == file_other]
        self.assertEqual(len(forced_subs), 1)
        self.assertEqual(len(other_subs), 0)

    def test_streams_picker_prefers_higher_resolution_video(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        sp = StreamsPicker(self.logger.getChild("StreamsPicker"), duplicates, self.wd.path)

        file1 = os.path.join(self.wd.path, "video_800.mkv")
        file2 = os.path.join(self.wd.path, "video_796.mkv")

        files_details = {
            file1: {
                "video": [{"tid": 0, "width": 1920, "height": 800, "fps": "24000/1001"}],
                "audio": [],
                "subtitle": [],
            },
            file2: {
                "video": [{"tid": 0, "width": 1920, "height": 796, "fps": "24000/1001"}],
                "audio": [],
                "subtitle": [],
            },
        }
        ids = {file1: 1, file2: 2}

        video_streams, _, _ = sp.pick_streams(files_details, ids)

        self.assertEqual(video_streams[0][0], file1)


    def test_mismatch_unused_file_ignored(self):
        file1 = build_test_video(
            os.path.join(self.wd.path, "rich.mkv"),
            self.wd.path,
            None,
            duration=1,
            width=1280,
            height=720,
            audio_name=(True, "eng"),
        )
        file2 = build_test_video(
            os.path.join(self.wd.path, "unused.mkv"),
            self.wd.path,
            None,
            duration=2,
            width=640,
            height=480,
        )

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)
        duplicates.add_metadata(file1, "audio_lang", "eng")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash.keys())[0]
        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["audio"]), 1)


    def test_mismatch_unused_third_input(self):
        file1 = build_test_video(
            os.path.join(self.wd.path, "a.mkv"),
            self.wd.path,
            None,
            duration=1,
            width=1280,
            height=720,
            audio_name=(True, "eng"),
        )
        file2 = build_test_video(
            os.path.join(self.wd.path, "b.mkv"),
            self.wd.path,
            None,
            duration=1,
            width=640,
            height=480,
            audio_name=(True, "de"),
        )
        file3 = build_test_video(
            os.path.join(self.wd.path, "c.mkv"),
            self.wd.path,
            None,
            duration=2,
            width=320,
            height=240,
        )

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)
        duplicates.add_entry("Video", file3)
        duplicates.add_metadata(file1, "audio_lang", "eng")
        duplicates.add_metadata(file2, "audio_lang", "de")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash.keys())[0]
        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["audio"]), 2)
        self.assertEqual({a["language"] for a in output_file_data["tracks"]["audio"]}, {"eng", "deu"})


    def test_same_multiscene_video_duplicate_detection(self):
        file1 = add_to_test_dir(self.wd.path, self.sample_video_file)
        file2 = add_to_test_dir(self.wd.path, self.sample_vhs_video_file)

        files = [file1, file2]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("video", file1)
        duplicates.add_entry("video", file2)
        duplicates.add_metadata(file1, "audio_lang", "eng")
        duplicates.add_metadata(file2, "audio_lang", "pol")

        input_file_hashes = hashes(self.wd.path)
        self.assertEqual(len(input_file_hashes), 2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(
            logger,
            duplicates,
            self.wd.path,
            allow_length_mismatch=True,
        )
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data(output_file)
        self.assertEqual(len(output_file_data["video"]), 1)
        self.assertEqual(output_file_data["video"][0]["height"], 1080)
        self.assertEqual(output_file_data["video"][0]["width"], 1920)

        self.assertEqual(len(output_file_data["audio"]), 2)
        self.assertEqual(output_file_data["audio"][0]["language"], "eng")
        self.assertEqual(output_file_data["audio"][1]["language"], "pol")


    def test_series_duplication(self):
        series1_dir = os.path.join(self.wd.path, "series1")
        series2_dir = os.path.join(self.wd.path, "series2")

        os.makedirs(series1_dir)
        os.makedirs(series2_dir)

        for episode in range(5):
            add_test_media("Grass - 66810.mp4", series1_dir, suffixes = [f"suf-S1E{episode}"])[0]
            add_test_media("Grass - 66810.mp4", series2_dir, suffixes = [f"S1E{episode}"])[0]

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Grass", series1_dir)
        duplicates.add_entry("Grass", series2_dir)
        duplicates.add_metadata(series1_dir, "audio_lang", "nor")
        duplicates.add_metadata(series2_dir, "audio_lang", "ger")

        input_file_hashes = hashes(self.wd.path)
        self.assertEqual(len(input_file_hashes), 10)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate output
        output_file_hash = hashes(output_dir)
        output_files = sorted(list(output_file_hash))

        for i, output_file in enumerate(output_files):
            output_file_name = os.path.basename(output_file)
            self.assertEqual(output_file_name, f"Grass - 66810-suf-S1E{i}.mkv")

            output_file_data = video_utils.get_video_data(output_file)
            self.assertEqual(len(output_file_data["video"]), 1)
            self.assertEqual(output_file_data["video"][0]["height"], 2160)
            self.assertEqual(output_file_data["video"][0]["width"], 3840)

            self.assertEqual(len(output_file_data["audio"]), 2)
            self.assertEqual(output_file_data["audio"][0]["language"], "deu")
            self.assertEqual(output_file_data["audio"][1]["language"], "nor")


    def test_pair_matcher_precision(self):
        """Multi-scene sample (82.6s) vs VHS degraded version (78.7s, 1.05x speed, crop, blur)."""
        file1 = add_to_test_dir(self.wd.path, self.sample_video_file)
        file2 = add_to_test_dir(self.wd.path, self.sample_vhs_video_file)

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1, file2, logging.getLogger("PM"))
        mappings, _, _ = pair_matcher.create_segments_mapping()

        self.assertEqual(mappings, [
            (0, 0),          # ← edge (snapped from 23ms/75ms — within 3 frames)
            (8663, 8264),
            (12503, 11925),
            (15863, 15132),
            (20663, 19698),
            (25463, 24264),
            (29783, 28377),
            (44463, 42340),
            (44943, 42830),
            (46863, 44642),
            (48783, 46491),
            (54543, 52038),
            (57903, 55208),
            (61263, 58377),
            (65583, 62491),
            (71823, 68453),
            (75663, 72075),
            (82582, 78700),  # ← edge (snapped to video duration)
        ])

        # Both edges snapped to video boundaries. Full coverage.
        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1),
            video_utils.get_video_duration(file2),
        )
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_black_intro_same_length(self):
        """Both files have the same length black intro — boundary should extend through it."""
        file1_path, file2_path = self.edge_fixtures["black_intro_same"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        # 3s black intro on both files — boundary extends through black to edge
        # LHS: bbb_bi3 (65.3s), RHS: bi3_deg103 (63.4s)
        self.assertEqual(mappings, [
            (0, 0),          # ← edge (snapped from 23ms/38ms — within 4 frames)
            (3223, 3132),
            (19063, 18528),
            (24743, 24038),
            (26583, 25849),
            (28743, 27925),
            (45023, 43736),
            (55023, 53434),
            (65337, 63433),  # ← edge (snapped to video duration)
        ])

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
        )
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_black_intro_different_length(self):
        """Files have different length black intros — algorithm should find content pairs despite offset."""
        file1_path, file2_path = self.edge_fixtures["black_intro_diff"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        # LHS: bbb_bi2 (64.3s, 2s black intro), RHS: bi6_deg103 (66.5s, 6s black intro)
        # LHS extends to edge through black; RHS starts at 4098ms (inside 6s intro)
        self.assertEqual(mappings, [
            (0, 4098),       # ← LHS snapped to edge; RHS 4.1s in (inside the 6s black intro)
            (2223, 6249),
            (10383, 14174),
            (18063, 21608),
            (25583, 28891),
            (27743, 31042),
            (44063, 46853),
            (54023, 56514),
            (64103, 66325),  # ← LHS ~edge; RHS ~edge
        ])

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
        )
        # LHS snapped to 0, RHS inside its 6s black intro — RHS start gap < 6s.
        # Both ends reach their video boundary.
        self.assertFalse(coverage["full_coverage"])
        self.assertEqual(coverage["lhs_start_gap_s"], 0.0)
        self.assertGreater(coverage["rhs_start_gap_s"], 3.0)
        self.assertLess(coverage["rhs_start_gap_s"], 6.5)
        self.assertLess(coverage["lhs_end_gap_s"], 0.5)
        self.assertLess(coverage["rhs_end_gap_s"], 0.5)

    def test_pair_matcher_black_outro(self):
        """Both files have black outro — last pair should extend through it to edge."""
        file1_path, file2_path = self.edge_fixtures["black_outro"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        # LHS: bbb_bo3 (65.3s, 3s black outro), RHS: bo3_deg103 (63.4s, 3s black outro)
        self.assertEqual(mappings, [
            (0, 0),          # ← edge (snapped from 23ms/38ms — within 3 frames)
            (16063, 15623),
            (23623, 22943),
            (25743, 25019),
            (27823, 27019),
            (42103, 40868),
            (52063, 50566),
            (62143, 60377),
            (65337, 63433),  # ← edge (snapped to video duration, through black outro)
        ])

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
        )
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_both_intro_and_outro_black(self):
        """Files have black intro AND outro — both boundaries should be handled."""
        file1_path, file2_path = self.edge_fixtures["both_black"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        # LHS: bi2_bo2 (66.3s, 2s black intro + 2s black outro)
        # RHS: bi2_bo2_deg103 (64.4s, same + 1.03x speed)
        # Resolution change improved matching — now finds 12 pairs spanning nearly full video
        self.assertEqual(mappings, [
            (0, 0),          # ← edge (snapped through black intro)
            (2263, 2226),
            (10423, 10113),
            (18103, 17623),
            (23823, 23094),
            (27783, 26981),
            (29863, 28981),
            (44063, 42792),
            (45743, 44415),
            (54063, 52528),
            (64143, 62302),
            (66343, 64415),  # ← edge (snapped to video duration, through black outro)
        ])

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
        )
        # With improved resolution, algorithm now achieves full coverage.
        # Both edges snapped through black intro/outro to video boundaries.
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_no_speed_change(self):
        """Same content with no speed change (ratio ~1.0), only quality degradation."""
        file1_path, file2_path = self.edge_fixtures["no_speed"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        # LHS: bbb (62.3s), RHS: bbb_deg10 (62.3s, same speed, degraded quality)
        self.assertEqual(mappings, [
            (0, 0),          # ← edge (exact first frame)
            (8360, 8377),
            (12960, 12943),
            (16040, 16038),
            (23560, 23547),
            (25720, 25736),
            (27800, 27811),
            (42000, 42000),
            (52000, 52000),
            (56240, 56226),
            (62314, 62313),  # ← edge (snapped to video duration)
        ])

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
        )
        self.assertTrue(coverage["full_coverage"])

    def test_pair_matcher_different_intro_same_length(self):
        """Files have different high-entropy intros of similar length, then shared content."""
        file1_path, file2_path = self.edge_fixtures["diff_intro_same"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        # LHS: bbb_gi3 (65.3s, 3s grass intro), RHS: atoms_i3_deg (63.5s, 3s atoms intro)
        self.assertEqual(mappings, [
            (3263, 3283),    # ← NOT at edge — content starts at ~3s (after different intros)
            (7583, 7509),
            (11423, 11245),
            (16023, 15660),
            (19063, 18604),
            (24703, 24113),
            (26583, 25887),
            (28743, 28038),
            (30823, 30038),
            (46703, 45472),
            (55103, 53585),
            (65302, 63471),  # ← edge (snapped to video duration)
        ])

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
        )
        # Both files have 3s intros (grass / atoms). Common content starts at ~3s.
        # Start gaps must match the known intro duration within 1s tolerance.
        # End gaps must be negligible — last pair reaches near video boundary.
        self.assertFalse(coverage["full_coverage"])
        self.assertAlmostEqual(coverage["lhs_start_gap_s"], 3.0, delta=1.0)
        self.assertAlmostEqual(coverage["rhs_start_gap_s"], 3.0, delta=1.0)
        self.assertLess(coverage["lhs_end_gap_s"], 0.5)
        self.assertLess(coverage["rhs_end_gap_s"], 0.5)

    def test_pair_matcher_different_intro_different_length(self):
        """Files have different high-entropy intros of DIFFERENT lengths."""
        file1_path, file2_path = self.edge_fixtures["diff_intro_diff"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        # LHS: bbb_gi2 (64.3s, 2s grass intro), RHS: atoms_i5_deg (65.5s, 5s atoms intro)
        self.assertEqual(mappings, [
            (2263, 5283),    # ← NOT at edge — first pair after different intros (2s vs 5s)
            (6583, 9509),
            (10423, 13245),
            (15023, 17660),
            (18063, 20604),
            (23703, 26113),
            (25583, 27887),
            (27743, 30038),
            (29823, 32038),
            (45703, 47472),
            (54103, 55585),
            (64302, 65471),  # ← edge (snapped to video duration)
        ])

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
        )
        # LHS has 2s grass intro, RHS has 5s atoms intro.
        # Start gaps must match known intro durations within 1s tolerance.
        # End gaps must be negligible — last pair reaches near video boundary.
        self.assertFalse(coverage["full_coverage"])
        self.assertAlmostEqual(coverage["lhs_start_gap_s"], 2.0, delta=1.0)
        self.assertAlmostEqual(coverage["rhs_start_gap_s"], 5.0, delta=1.0)
        self.assertLess(coverage["lhs_end_gap_s"], 0.5)
        self.assertLess(coverage["rhs_end_gap_s"], 0.5)

    def test_pair_matcher_different_outro(self):
        """Files share content but have different high-entropy outros."""
        file1_path, file2_path = self.edge_fixtures["diff_outro"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        # LHS: bbb_wo3 (65.3s, 3s woman outro), RHS: deg103_atoms_o3 (63.5s, 3s atoms outro)
        self.assertEqual(mappings, [
            (0, 0),          # ← edge (snapped from 23ms/61ms — within 3 frames)
            (8383, 8174),
            (16063, 15608),
            (23583, 22891),
            (25743, 25042),
            (42063, 40853),
            (43703, 42476),
            (52023, 50514),
            (62103, 60325),  # ← NOT at edge — content ends at 62s, different outros follow
        ])

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
        )
        # Start snapped to edge. Both files have 3s outros (woman / atoms).
        # End gaps must match the known outro duration within 1s tolerance.
        self.assertFalse(coverage["full_coverage"])
        self.assertEqual(coverage["lhs_start_gap_s"], 0.0)
        self.assertEqual(coverage["rhs_start_gap_s"], 0.0)
        self.assertAlmostEqual(coverage["lhs_end_gap_s"], 3.0, delta=1.0)
        self.assertAlmostEqual(coverage["rhs_end_gap_s"], 3.0, delta=1.0)

    def test_pair_matcher_different_intro_and_outro(self):
        """Files share content but have BOTH different intros AND different outros."""
        file1_path, file2_path = self.edge_fixtures["diff_both"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        # LHS: gi3_wo3 (57.1s, 3s grass intro + 3s woman outro)
        # RHS: ai3d_swo3 (66.5s, 3s atoms intro + 3s seawaves outro)
        # Resolution change improved matching — now finds 5 pairs (vs 3 before)
        self.assertEqual(mappings, [
            (3263, 3208),    # ← NOT at edge — content starts at ~3s (after different intros)
            (19103, 18604),
            (26623, 25887),
            (46743, 45472),
            (53463, 52000),  # ← NOT at edge — 4s from end of shared content
        ])

        coverage = PairMatcher.coverage_summary(
            mappings,
            video_utils.get_video_duration(file1_path),
            video_utils.get_video_duration(file2_path),
        )
        # Improved from 3 to 5 pairs, but still not full coverage.
        # Both files have 3s intros and 3s outros — start/end gaps expected.
        self.assertFalse(coverage["full_coverage"])
        self.assertAlmostEqual(coverage["lhs_start_gap_s"], 3.0, delta=1.0)
        self.assertGreater(coverage["lhs_end_gap_s"], 3.0)
        self.assertAlmostEqual(coverage["rhs_start_gap_s"], 3.0, delta=1.0)
        self.assertGreater(coverage["rhs_end_gap_s"], 10.0)

    # ---- coverage_summary tests ----

    def test_coverage_summary_full_coverage(self):
        """Pairs touching both edges → full_coverage=True."""
        mappings = [(10, 5), (50000, 47500), (99910, 99920)]
        result = PairMatcher.coverage_summary(mappings, 100000, 100000)
        self.assertTrue(result["full_coverage"])
        self.assertAlmostEqual(result["lhs_start_gap_s"], 0.01, places=3)
        self.assertAlmostEqual(result["rhs_start_gap_s"], 0.005, places=3)
        self.assertAlmostEqual(result["lhs_end_gap_s"], 0.09, places=3)
        self.assertAlmostEqual(result["rhs_end_gap_s"], 0.08, places=3)

    def test_coverage_summary_start_mismatch(self):
        """First pair far from start → full_coverage=False, start gaps reported."""
        mappings = [(3000, 5000), (60000, 58000)]
        result = PairMatcher.coverage_summary(mappings, 62000, 60000)
        self.assertFalse(result["full_coverage"])
        self.assertAlmostEqual(result["lhs_start_gap_s"], 3.0, places=1)
        self.assertAlmostEqual(result["rhs_start_gap_s"], 5.0, places=1)
        # End gaps should be small
        self.assertAlmostEqual(result["lhs_end_gap_s"], 2.0, places=1)
        self.assertAlmostEqual(result["rhs_end_gap_s"], 2.0, places=1)

    def test_coverage_summary_end_mismatch(self):
        """Last pair far from end → full_coverage=False, end gaps reported."""
        mappings = [(20, 15), (55000, 53000)]
        result = PairMatcher.coverage_summary(mappings, 62000, 60000)
        self.assertFalse(result["full_coverage"])
        self.assertAlmostEqual(result["lhs_end_gap_s"], 7.0, places=1)
        self.assertAlmostEqual(result["rhs_end_gap_s"], 7.0, places=1)

    def test_coverage_summary_with_real_no_speed(self):
        """Use actual no_speed fixture — should be full_coverage."""
        file1_path, file2_path = self.edge_fixtures["no_speed"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        d1 = video_utils.get_video_duration(file1_path)
        d2 = video_utils.get_video_duration(file2_path)
        result = PairMatcher.coverage_summary(mappings, d1, d2)
        self.assertTrue(result["full_coverage"])

    def test_coverage_summary_with_real_diff_intro(self):
        """Use actual diff_intro_same fixture — should NOT be full_coverage (start gap)."""
        file1_path, file2_path = self.edge_fixtures["diff_intro_same"]

        interruption = generic_utils.InterruptibleProcess()
        pair_matcher = PairMatcher(interruption, self.wd.path, file1_path, file2_path, self.logger)
        mappings, _, _ = pair_matcher.create_segments_mapping()

        d1 = video_utils.get_video_duration(file1_path)
        d2 = video_utils.get_video_duration(file2_path)
        result = PairMatcher.coverage_summary(mappings, d1, d2)
        # Both files have 3s intros. Start gaps must match intro duration.
        self.assertFalse(result["full_coverage"])
        self.assertAlmostEqual(result["lhs_start_gap_s"], 3.0, delta=1.0)
        self.assertAlmostEqual(result["rhs_start_gap_s"], 3.0, delta=1.0)
        self.assertLess(result["lhs_end_gap_s"], 0.5)
        self.assertLess(result["rhs_end_gap_s"], 0.5)


    def test_languages_ordering(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        langs = ["pol", "en", "ger", "ja", "nor"]

        for i in range(5):
            file = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = [f"v{i}"])[0]
            duplicates.add_entry("Grass", file)
            duplicates.add_metadata(file, "audio_lang", langs[i])

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate alphabetical order
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash)[0]
        output_file_data = video_utils.get_video_data(output_file)
        self.assertEqual(output_file_data["audio"][0]["language"], "deu")
        self.assertEqual(output_file_data["audio"][1]["language"], "eng")
        self.assertEqual(output_file_data["audio"][2]["language"], "jpn")
        self.assertEqual(output_file_data["audio"][3]["language"], "nor")
        self.assertEqual(output_file_data["audio"][4]["language"], "pol")

    def test_unknown_language_streams_sorted_last(self):
        """Streams with unknown language (from --force-all-streams) should appear after all known-language streams."""
        video1 = build_test_video(os.path.join(self.wd.path, "o1.mkv"), self.wd.path, "sea-waves-crashing-on-beach-shore-4793288.mp4", subtitle = True)
        video2 = build_test_video(os.path.join(self.wd.path, "o2.mkv"), self.wd.path, "sea-waves-crashing-on-beach-shore-4793288.mp4", subtitle = True)

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Sea Waves", video1)
        duplicates.add_entry("Sea Waves", video2)
        duplicates.add_metadata(video1, "audio_lang", "eng")
        duplicates.add_metadata(video2, "audio_lang", "pol")
        duplicates.add_metadata(video1, "subtitle_lang", "pol")
        # video2 subtitle: unknown language, kept via force_all_streams
        duplicates.add_metadata(video2, "force_all_streams", True)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data(output_file)
        subtitles = output_file_data["subtitle"]
        self.assertEqual(len(subtitles), 2)
        # Known language (pol) should come first, unknown last
        self.assertEqual(subtitles[0]["language"], "pol")
        self.assertIsNone(subtitles[1]["language"])

    def test_default_language(self):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        langs = ["pol", "en", "ger", "ja", "nor"]

        for i in range(5):
            file = add_test_media("Grass - 66810.mp4", self.wd.path, suffixes = [f"v{i}"])[0]
            duplicates.add_entry("Grass", file)
            duplicates.add_metadata(file, "audio_lang", langs[i])
            duplicates.add_metadata(file, "audio_prod_lang", "ja")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate alphabetical order
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)

        output_file = list(output_file_hash)[0]
        output_file_data = video_utils.get_video_data(output_file)
        self.assertEqual(output_file_data["audio"][0]["language"], "deu")
        self.assertEqual(output_file_data["audio"][1]["language"], "eng")
        self.assertEqual(output_file_data["audio"][2]["language"], "jpn")
        self.assertEqual(output_file_data["audio"][2]["default"], True)
        self.assertEqual(output_file_data["audio"][3]["language"], "nor")
        self.assertEqual(output_file_data["audio"][4]["language"], "pol")


    def test_subtitle_streams(self):
        video1 = build_test_video(os.path.join(self.wd.path, "o1.mkv"), self.wd.path, "sea-waves-crashing-on-beach-shore-4793288.mp4", subtitle = True)
        video2 = build_test_video(os.path.join(self.wd.path, "o2.mkv"), self.wd.path, "sea-waves-crashing-on-beach-shore-4793288.mp4", subtitle = True)

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Sea Waves", video1)
        duplicates.add_entry("Sea Waves", video2)
        duplicates.add_metadata(video1, "audio_lang", "eng")
        duplicates.add_metadata(video2, "audio_lang", "eng")
        duplicates.add_metadata(video1, "subtitle_lang", "jpn")
        duplicates.add_metadata(video2, "subtitle_lang", "br")

        input_file_hashes = hashes(self.wd.path)
        self.assertEqual(len(input_file_hashes), 2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = logging.getLogger("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data(output_file)
        self.assertEqual(len(output_file_data["video"]), 1)
        self.assertEqual(output_file_data["video"][0]["height"], 1080)
        self.assertEqual(output_file_data["video"][0]["width"], 1920)

        self.assertEqual(len(output_file_data["subtitle"]), 2)
        languages = { output_file_data["subtitle"][0]["language"],
                      output_file_data["subtitle"][1]["language"] }
        self.assertEqual(languages, {"jpn", "bre"})


    def test_additional_attachements(self):
        video1 = build_test_video(os.path.join(self.wd.path, "o1.mkv"), self.wd.path, "fog-over-mountainside-13008647.mp4", subtitle = True, thumbnail_name = "parrot.jpeg")
        video2 = build_test_video(os.path.join(self.wd.path, "o2.mkv"), self.wd.path, "fog-over-mountainside-13008647.mp4", subtitle = True)

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Fog", video1)
        duplicates.add_entry("Fog", video2)
        duplicates.add_metadata(video1, "audio_lang", "eng")
        duplicates.add_metadata(video2, "audio_lang", "eng")
        duplicates.add_metadata(video1, "subtitle_lang", "pol")
        duplicates.add_metadata(video2, "subtitle_lang", "eng")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = logging.getLogger("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["video"]), 1)
        self.assertEqual(len(output_file_data["attachments"]), 1)


    def test_attachement_in_file_with_useless_streams(self):
        # video #1 comes with all interesting data. the only thing video #2 can offer is an attachment.
        video1 = build_test_video(os.path.join(self.wd.path, "o1.mkv"), self.wd.path, "fog-over-mountainside-13008647.mp4", subtitle = True)
        video2 = build_test_video(os.path.join(self.wd.path, "o2.mkv"), self.wd.path, "fog-over-mountainside-13008647.mp4", thumbnail_name = "parrot.jpeg")

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_metadata(video1, "subtitle_lang", "eng")
        duplicates.add_metadata(video1, "audio_lang", "eng")
        duplicates.add_metadata(video2, "audio_lang", "eng")
        duplicates.add_entry("Fog", video1)
        duplicates.add_entry("Fog", video2)

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = logging.getLogger("Melter")
        plan = analyze_duplicates_helper(logger, duplicates, self.wd.path)
        process_duplicates_helper(logger, interruption, self.wd.path, output_dir, plan)

        # validate output
        output_file_hash = hashes(output_dir)
        self.assertEqual(len(output_file_hash), 1)
        output_file = list(output_file_hash)[0]

        output_file_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_file_data["tracks"]["video"]), 1)
        self.assertEqual(len(output_file_data["attachments"]), 1)


    sample_streams = [
        # case: merge all audio tracks
        (
            "mix audios",
            # input
            {
                "fileA": {
                    "video": [{"height": "1024", "width": "1024", "fps": "24", "tid": 0}],
                    "audio": [{"language": "jp", "channels": "2", "sample_rate": "32000", "tid": 2},
                              {"language": "de", "channels": "2", "sample_rate": "32000", "tid": 4}]
                },
                "fileB": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30", "tid": 6}],
                    "audio": [{"language": "br", "channels": "2", "sample_rate": "32000", "tid": 8},
                              {"language": "nl", "channels": "2", "sample_rate": "32000", "tid": 10}]
                }
            },
            # expected output
            (
                [("fileB", 6, None)],
                [("fileA", 2, "jp"), ("fileA", 4, "de"), ("fileB", 8, "br"), ("fileB", 10, "nl")],
                []
            )
        ),
        # case: pick one file whenever possible

        (
            "prefer one file",
            # input (fileB is a superset of fileA, so prefer it)
            {
                "fileA": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30", "tid": 1}],
                    "audio": [{"language": "cz", "channels": "2", "sample_rate": "32000", "tid": 2}],
                    "subtitle": [{"language": "pl", "tid": 3}]
                },
                "fileB": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30", "tid": 1}],
                    "audio": [{"language": "cz", "channels": "2", "sample_rate": "32000", "tid": 2}],
                    "subtitle": [{"language": "pl", "tid": 4}, {"language": "br", "tid": 3}]
                }
            },
            # expected output
            # Explanation: fileB is a superset of fileA, so no need to pick any streams from fileA
            (
                [("fileB", 1, None)],
                [("fileB", 2, "cz")],
                [("fileB", 4, "pl"), ("fileB", 3, "br")]
            )
        ),

        (
            "same but different",
            # input
            {
                "fileA": {
                    "video": [{"height": "1024", "width": "1024", "fps": "24", "tid": 1}],
                    "audio": [{"language": "jp", "channels": "2", "sample_rate": "32000", "tid": 4},
                              {"language": "jp", "channels": "2", "sample_rate": "32000", "tid": 6}],
                    "subtitle": [{"language": "de", "tid": 15}, {"language": "de", "tid": 8}]
                },
                "fileB": {
                    "video": [{"height": "1024", "width": "1024", "fps": "30", "tid": 2}],
                    "audio": [{"language": "jp", "channels": "2", "sample_rate": "32000", "tid": 1},
                              {"language": "jp", "channels": "6", "sample_rate": "32000", "tid": 0}],
                    "subtitle": [{"language": "pl", "tid": 15}, {"language": "pl", "tid": 17}]
                }
            },
            # expected output
            # Explanation:
            # There are two identical (basing on parameters) audio inputs in file A.
            # Consider them different (why would there be two identical audio stracks?) and include both in output.
            #
            # Include 6 channel audio track from file B (as best one) but ignore 2 channel one (assume it's a duplicate of tracks from file A).
            #
            # Same logic goes for subtitles. Include both (most likely different) subtitle tracks from file A and
            # both subtitle tracks from file B
            (
                [("fileB", 2, None)],
                [("fileA", 4, "jp"), ("fileA", 6, "jp"), ("fileB", 0, "jp")],
                [("fileA", 15, "de"), ("fileA", 8, "de"), ("fileB", 15, "pl"), ("fileB", 17, "pl")]
            )
        ),
    ]

    @parameterized.expand(sample_streams)
    def test_streams_pick_decision(self, name, input, expected_streams):
        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        streams_picker = StreamsPicker(self.logger.getChild("Melter"), duplicates, self.wd.path)

        ids = _build_path_to_id_map(input)

        # Test all possible combinations of order of input files. Output should be stable
        for video_info in all_key_orders(input):
            picked_streams = streams_picker.pick_streams(video_info, ids)
            picked_streams_normalized = normalize(picked_streams)
            expected_streams_normalized = normalize(expected_streams)

            self.assertEqual(picked_streams_normalized, expected_streams_normalized)



class MeltPerformerUnitTest(unittest.TestCase):
    """Unit tests for MeltPerformer internal methods."""

    def _make_performer(self) -> MeltPerformer:
        performer = object.__new__(MeltPerformer)
        performer.logger = logging.getLogger("test.MeltPerformer")
        performer.wd = tempfile.mkdtemp()
        performer.output_dir = tempfile.mkdtemp()
        performer.tolerance_ms = DEFAULT_TOLERANCE_MS
        performer.interruption = generic_utils.InterruptibleProcess()
        return performer

    def test_stream_sorting_puts_unknown_languages_last(self):
        streams = [
            ("audio", 1, "/a.mkv", None),
            ("audio", 2, "/a.mkv", "eng"),
            ("subtitle", 3, "/a.mkv", None),
            ("subtitle", 4, "/a.mkv", "pol"),
            ("subtitle", 5, "/a.mkv", "deu"),
        ]

        sort_key = lambda stream: (stream[3] is None, stream[3] or "")
        result = sorted(streams, key=sort_key)

        languages = [s[3] for s in result]
        self.assertEqual(languages, ["deu", "eng", "pol", None, None])

    def test_stream_sorting_alphabetical_when_all_known(self):
        streams = [
            ("subtitle", 1, "/a.mkv", "pol"),
            ("subtitle", 2, "/a.mkv", "eng"),
            ("subtitle", 3, "/a.mkv", "deu"),
            ("audio", 4, "/a.mkv", "jpn"),
        ]

        sort_key = lambda stream: (stream[3] is None, stream[3] or "")
        result = sorted(streams, key=sort_key)

        languages = [s[3] for s in result]
        self.assertEqual(languages, ["deu", "eng", "jpn", "pol"])

    def test_build_mkvmerge_args_track_order_respects_unknown_last(self):
        performer = self._make_performer()

        file_a = "/tmp/a.mkv"
        file_b = "/tmp/b.mkv"

        streams_list_sorted = [
            ("video", 0, file_a, None),
            ("audio", 1, file_a, "eng"),
            ("subtitle", 3, file_a, "deu"),
            ("subtitle", 4, file_a, "pol"),
            ("subtitle", 5, file_a, None),
        ]

        args = performer._build_mkvmerge_args(
            "/tmp/out.mkv",
            streams_list_sorted,
            attachments=[],
            preferred_audio=None,
            required_input_files=[file_a],
        )

        # Track order should preserve the sorted order
        track_order_idx = args.index("--track-order")
        track_order = args[track_order_idx + 1]
        self.assertEqual(track_order, "0:0,0:1,0:3,0:4,0:5")
class PairMatcherUnitTest(unittest.TestCase):
    """Unit tests for PairMatcher internal methods.

    These tests mock _is_rich and bypass the full constructor to test
    algorithmic behaviour without needing video files. They target:
    - _extrapolate_through_low_entropy: 5% noise tolerance
    - _snap_to_edges: snap_frames=4 threshold
    - find_boundary (via _look_for_boundaries): look_ahead robustness
    """

    def _make_pair_matcher(self, lhs_fps: float = 25.0, rhs_fps: float = 25.0) -> PairMatcher:
        """Create a PairMatcher with minimal init — no video files needed."""
        pm = object.__new__(PairMatcher)
        pm.logger = logging.getLogger("test.PairMatcher")
        pm.lhs_fps = lhs_fps
        pm.rhs_fps = rhs_fps
        pm.phash = None  # not used in unit tests
        pm.lhs_path = "/fake/lhs.mp4"
        pm.rhs_path = "/fake/rhs.mp4"
        pm.interruption = generic_utils.InterruptibleProcess()
        return pm

    @staticmethod
    def _make_frames(timestamps: list[int], prefix: str = "fake") -> dict[int, dict]:
        """Build a synthetic FramesInfo dict with dummy paths."""
        return {ts: {"path": f"/{prefix}/{ts}.png", "frame_id": i} for i, ts in enumerate(timestamps)}

    # ---- _extrapolate_through_low_entropy ----

    def test_extrapolate_pure_low_entropy_lhs_extends(self):
        """When all frames between boundary and edge are low-entropy, extrapolation happens."""
        pm = self._make_pair_matcher()

        # 100 frames at 40ms intervals from 0..3960ms
        lhs_keys = list(range(0, 4000, 40))
        rhs_keys = list(range(0, 4000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        boundary = (2000, 2000)
        reference = (3000, 3000)

        with patch.object(PairMatcher, '_is_rich', return_value=False):
            result = pm._extrapolate_through_low_entropy(
                lhs, rhs, boundary, reference, ratio=1.0,
                direction=-1, entered_low_entropy=True,
            )

        # Should extrapolate to LHS edge (0ms) and snap RHS to nearest frame
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 0)

    def test_extrapolate_not_entered_low_entropy_noop(self):
        """When entered_low_entropy=False, boundary is returned unchanged."""
        pm = self._make_pair_matcher()
        lhs = self._make_frames([0, 1000, 2000])
        rhs = self._make_frames([0, 1000, 2000])
        boundary = (1000, 1000)

        result = pm._extrapolate_through_low_entropy(
            lhs, rhs, boundary, (2000, 2000), ratio=1.0,
            direction=-1, entered_low_entropy=False,
        )

        self.assertEqual(result, boundary)

    def test_extrapolate_all_high_entropy_gap_refuses(self):
        """When all gap frames are high-entropy, extrapolation is refused."""
        pm = self._make_pair_matcher()
        lhs = self._make_frames(list(range(0, 4000, 40)))
        rhs = self._make_frames(list(range(0, 4000, 40)))
        boundary = (2000, 2000)

        with patch.object(PairMatcher, '_is_rich', return_value=True):
            result = pm._extrapolate_through_low_entropy(
                lhs, rhs, boundary, (3000, 3000), ratio=1.0,
                direction=-1, entered_low_entropy=True,
            )

        self.assertEqual(result, boundary)

    def test_extrapolate_noise_below_5pct_accepts(self):
        """When < 5% of gap frames are high-entropy, extrapolation proceeds."""
        pm = self._make_pair_matcher()

        # 200 frames from 0..7960ms at 40ms intervals
        lhs_keys = list(range(0, 8000, 40))
        rhs_keys = list(range(0, 8000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        boundary = (4000, 4000)
        # Gap: frames 0..3960 (100 frames), make 3 of them (3%) "rich"
        noisy_timestamps = {120, 2000, 3600}  # 3 out of 100 = 3%

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts in noisy_timestamps

        with patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich):
            result = pm._extrapolate_through_low_entropy(
                lhs, rhs, boundary, (6000, 6000), ratio=1.0,
                direction=-1, entered_low_entropy=True,
            )

        # Should extrapolate — 3% noise is below 5% threshold
        self.assertEqual(result[0], 0)

    def test_extrapolate_noise_above_5pct_refuses(self):
        """When > 5% of gap frames are high-entropy, extrapolation is refused."""
        pm = self._make_pair_matcher()

        lhs_keys = list(range(0, 8000, 40))
        rhs_keys = list(range(0, 8000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        boundary = (4000, 4000)
        # Gap: frames 0..3960 (100 frames), make 10 of them (10%) "rich"
        noisy_timestamps = set(range(0, 400, 40))  # 10 out of 100 = 10%

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts in noisy_timestamps

        with patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich):
            result = pm._extrapolate_through_low_entropy(
                lhs, rhs, boundary, (6000, 6000), ratio=1.0,
                direction=-1, entered_low_entropy=True,
            )

        # Should refuse — 10% noise exceeds 5% threshold
        self.assertEqual(result, boundary)

    def test_extrapolate_end_direction(self):
        """Extrapolation works in the forward (end) direction too."""
        pm = self._make_pair_matcher()

        lhs_keys = list(range(0, 8000, 40))
        rhs_keys = list(range(0, 8000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        boundary = (4000, 4000)
        reference = (2000, 2000)

        with patch.object(PairMatcher, '_is_rich', return_value=False):
            result = pm._extrapolate_through_low_entropy(
                lhs, rhs, boundary, reference, ratio=1.0,
                direction=1, entered_low_entropy=True,
            )

        # Should extrapolate to LHS end edge
        self.assertEqual(result[0], lhs_keys[-1])

    def test_extrapolate_rhs_high_entropy_refuses(self):
        """When LHS is low-entropy but RHS gap is high-entropy, extrapolation is refused."""
        pm = self._make_pair_matcher()

        lhs_keys = list(range(0, 8000, 40))
        rhs_keys = list(range(0, 8000, 40))
        lhs = self._make_frames(lhs_keys, prefix="lhs")
        rhs = self._make_frames(rhs_keys, prefix="rhs")

        boundary = (4000, 4000)

        def smart_mock(path: str) -> bool:
            if path.startswith("/lhs/"):
                return False   # LHS gap: all low-entropy
            if path.startswith("/rhs/"):
                return True    # RHS gap: all high-entropy
            return False

        with patch.object(PairMatcher, '_is_rich', side_effect=smart_mock):
            result = pm._extrapolate_through_low_entropy(
                lhs, rhs, boundary, (6000, 6000), ratio=1.0,
                direction=-1, entered_low_entropy=True,
            )

        # RHS is all high-entropy → should refuse
        self.assertEqual(result, boundary)

    # ---- _snap_to_edges ----

    def test_snap_within_4_frames_snaps(self):
        """Pairs within 4 frames of edge should snap to video boundary."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # threshold = 4 * 1000 / 25 = 160ms

        lhs_frames = self._make_frames([0, 40, 80, 100, 5000, 9900, 9960, 10000])
        rhs_frames = self._make_frames([0, 40, 80, 100, 5000, 9900, 9960, 10000])

        matching_pairs = [(100, 100), (5000, 5000), (9900, 9900)]

        with patch.object(video_utils, 'get_video_duration', return_value=10000):
            result = pm._snap_to_edges(matching_pairs, lhs_frames, rhs_frames)

        # 100ms is within 160ms threshold → snap to 0
        self.assertEqual(result[0], (0, 0))
        # 10000 - 9900 = 100ms is within 160ms threshold → snap to duration
        self.assertEqual(result[-1], (10000, 10000))

    def test_snap_beyond_4_frames_no_snap(self):
        """Pairs beyond 4 frames of edge should NOT snap."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)
        # threshold = 160ms

        lhs_frames = self._make_frames([0, 200, 5000, 9700, 10000])
        rhs_frames = self._make_frames([0, 200, 5000, 9700, 10000])

        matching_pairs = [(200, 200), (5000, 5000), (9700, 9700)]

        with patch.object(video_utils, 'get_video_duration', return_value=10000):
            result = pm._snap_to_edges(matching_pairs, lhs_frames, rhs_frames)

        # 200ms > 160ms → no snap
        self.assertEqual(result[0], (200, 200))
        # 10000 - 9700 = 300ms > 160ms → no snap
        self.assertEqual(result[-1], (9700, 9700))

    def test_snap_asymmetric_fps(self):
        """With different FPS per side, thresholds apply independently."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=50.0)
        # LHS threshold = 4 * 1000 / 25 = 160ms
        # RHS threshold = 4 * 1000 / 50 = 80ms

        lhs_frames = self._make_frames([0, 100, 5000, 10000])
        rhs_frames = self._make_frames([0, 90, 5000, 10000])

        matching_pairs = [(100, 90), (5000, 5000)]

        with patch.object(video_utils, 'get_video_duration', return_value=10000):
            result = pm._snap_to_edges(matching_pairs, lhs_frames, rhs_frames)

        # LHS: 100ms <= 160ms → snaps
        self.assertEqual(result[0][0], 0)
        # RHS: 90ms > 80ms → does NOT snap
        self.assertEqual(result[0][1], 90)

    # ---- _look_for_boundaries: look_ahead robustness ----

    @staticmethod
    def _mock_phash_always_match():
        """Return a context manager that replaces PhashCache with an always-matching stub."""
        class FakeHash:
            def __sub__(self, other): return 0
            def __abs__(self): return 0
        class FakePhashCache:
            def __init__(self, hash_size=16): pass
            def get(self, path): return FakeHash()
        return patch('twotone.tools.melt.pair_matcher.PhashCache', FakePhashCache)

    def test_boundary_search_survives_transient_dark_frame(self):
        """A single dark frame at a scene cut should not stop boundary search."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        lhs_keys = list(range(0, 10000, 40))
        rhs_keys = list(range(0, 10000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        dark_timestamps = {3000}

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts not in dark_timestamps

        first = (5000, 5000)
        last = (8000, 8000)

        with (
            patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich),
            patch.object(pm, '_edge_content_matches', return_value=True),
            self._mock_phash_always_match(),
        ):
            result_first, _ = pm._look_for_boundaries(
                lhs, rhs, first, last, cutoff=16, extrapolate=False,
            )

        self.assertLess(result_first[0], 3000,
            "Boundary should extend past the transient dark frame at 3000ms")

    def test_boundary_search_stops_at_sustained_dark_zone(self):
        """A sustained dark zone (>1.5s of dark frames) should stop the search."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        lhs_keys = list(range(0, 10000, 40))
        rhs_keys = list(range(0, 10000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        dark_zone = {ts for ts in lhs_keys if ts <= 2500}

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts not in dark_zone

        first = (5000, 5000)
        last = (8000, 8000)

        with (
            patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich),
            patch.object(pm, '_edge_content_matches', return_value=True),
            self._mock_phash_always_match(),
        ):
            result_first, _ = pm._look_for_boundaries(
                lhs, rhs, first, last, cutoff=16, extrapolate=False,
            )

        self.assertGreaterEqual(result_first[0], 2500,
            "Boundary should not extend into a sustained dark zone")

    def test_boundary_search_consecutive_scene_cuts_dont_stop(self):
        """Two brief dark frames close together should not stop the search."""
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        lhs_keys = list(range(0, 10000, 40))
        rhs_keys = list(range(0, 10000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        dark_timestamps = {3000, 3200}

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts not in dark_timestamps

        first = (5000, 5000)
        last = (8000, 8000)

        with (
            patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich),
            patch.object(pm, '_edge_content_matches', return_value=True),
            self._mock_phash_always_match(),
        ):
            result_first, _ = pm._look_for_boundaries(
                lhs, rhs, first, last, cutoff=16, extrapolate=False,
            )

        self.assertLess(result_first[0], 3000,
            "Boundary should extend past two isolated dark frames")

    def test_boundary_search_jumps_over_mid_movie_dark_zone(self):
        """A dark zone in the middle of matching content should be jumped over.

        This reproduces the Jumanji scenario: content matches before and after
        a sustained dark zone (scene transition), so the walk should continue
        past the dark zone rather than declaring it a boundary.
        """
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        # 500 frames at 40ms intervals: 0..19960ms
        lhs_keys = list(range(0, 20000, 40))
        rhs_keys = list(range(0, 20000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        # Sustained dark zone from 8000ms..10000ms (2s, ~50 frames)
        dark_zone = {ts for ts in lhs_keys if 8000 <= ts <= 10000}

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts not in dark_zone

        first = (14000, 14000)
        last = (18000, 18000)

        with (
            patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich),
            patch.object(pm, '_edge_content_matches', return_value=True),
            self._mock_phash_always_match(),
        ):
            result_first, _ = pm._look_for_boundaries(
                lhs, rhs, first, last, cutoff=16, extrapolate=False,
            )

        # The walk should have jumped over the dark zone at 8000-10000ms
        # and found matches before it, reaching near the video start.
        self.assertLess(result_first[0], 8000,
            "Boundary should jump over mid-movie dark zone and find earlier matches")

    def test_boundary_search_stops_when_content_differs_after_dark_zone(self):
        """When content on the other side of a dark zone does NOT match,
        the walk should stop (dark zone is the real boundary).
        """
        pm = self._make_pair_matcher(lhs_fps=25.0, rhs_fps=25.0)

        lhs_keys = list(range(0, 20000, 40))
        rhs_keys = list(range(0, 20000, 40))
        lhs = self._make_frames(lhs_keys)
        rhs = self._make_frames(rhs_keys)

        # Dark zone from 8000ms..10000ms
        dark_zone = {ts for ts in lhs_keys if 8000 <= ts <= 10000}

        def mock_is_rich(path: str) -> bool:
            ts = int(path.split("/")[-1].replace(".png", ""))
            return ts not in dark_zone

        first = (14000, 14000)
        last = (18000, 18000)

        class SelectivePhashCache:
            """Phash that matches for ts > 10000 but fails for ts <= 8000."""
            def __init__(self, hash_size=16): pass
            def get(self, path):
                ts = int(path.split("/")[-1].replace(".png", ""))
                return type('Hash', (), {
                    'value': ts,
                    '__sub__': lambda s, o: 0 if s.value > 10000 or o.value > 10000 else 99,
                    '__abs__': lambda s: s,
                    '__lt__': lambda s, o: s.value < (o if isinstance(o, (int, float)) else o.value),
                    '__le__': lambda s, o: s.value <= (o if isinstance(o, (int, float)) else o.value),
                    '__gt__': lambda s, o: s.value > (o if isinstance(o, (int, float)) else o.value),
                    '__int__': lambda s: s.value,
                    '__float__': lambda s: float(s.value),
                })()

        with (
            patch.object(PairMatcher, '_is_rich', side_effect=mock_is_rich),
            patch.object(pm, '_edge_content_matches', return_value=True),
            patch('twotone.tools.melt.pair_matcher.PhashCache', SelectivePhashCache),
        ):
            result_first, _ = pm._look_for_boundaries(
                lhs, rhs, first, last, cutoff=16, extrapolate=False,
            )

        # Content doesn't match after the dark zone — walk should stop at/near it
        self.assertGreaterEqual(result_first[0], 8000,
            "Boundary should stop at dark zone when content doesn't match on the other side")


if __name__ == '__main__':
    unittest.main()
