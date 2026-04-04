
import logging
import os
import tempfile
import types
import unittest

from collections.abc import Iterator
from functools import partial
from itertools import permutations
from pathlib import Path

from twotone.tools.utils import generic_utils, video_utils
from twotone.tools.melt.melt import DEFAULT_TOLERANCE_MS, MeltAnalyzer, MeltPerformer, StaticSource
from twotone.tools.utils.files_utils import ScopedDirectory
from common import (
    TwoToneTestCase,
    FileCache,
    get_audio,
    get_video,
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


_FAKE_PROCESS_OK = types.SimpleNamespace(returncode=0, stdout='', stderr='')


class MeltTestBase(TwoToneTestCase):
    """Base class for melt integration tests.

    Generates and caches sample video files and edge-case PairMatcher fixtures.
    """

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
        V = "10"  # bump to invalidate all edge fixtures

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
        data = video_utils.get_video_data(input_path)
        width = int(data["video"][0]["width"])
        height = int(data["video"][0]["height"])
        fps = data["video"][0]["fps"]
        fps_float = generic_utils.fps_str_to_float(fps)
        has_audio = len(data.get("audio", [])) > 0

        black_path = os.path.join(td, "black_intro.mkv")
        args = ["-y",
                "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps_float}:d={black_seconds}"]
        if has_audio:
            args += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={black_seconds}"]
            args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                     "-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100", "-shortest", black_path]
        else:
            args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", black_path]
        run_ffmpeg(args, expected_path=black_path)

        reencoded_path = os.path.join(td, "reencoded_input.mkv")
        args = ["-y", "-i", input_path,
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-r", str(fps_float), "-vf", f"fps={fps_float},scale={width}:{height}"]
        if has_audio:
            args += ["-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100"]
        else:
            args += ["-an"]
        args.append(reencoded_path)
        run_ffmpeg(args, expected_path=reencoded_path)

        return self._concat_videos([black_path, reencoded_path], output_path, tmp_dir=td)

    def _append_black(self, input_path: str, output_path: str, black_seconds: float, tmp_dir: str | None = None) -> str:
        """Append *black_seconds* of black video+silence after *input_path*."""
        td = tmp_dir or self.wd.path
        data = video_utils.get_video_data(input_path)
        width = int(data["video"][0]["width"])
        height = int(data["video"][0]["height"])
        fps = data["video"][0]["fps"]
        fps_float = generic_utils.fps_str_to_float(fps)
        has_audio = len(data.get("audio", [])) > 0

        reencoded_path = os.path.join(td, "reencoded_input_outro.mkv")
        args = ["-y", "-i", input_path,
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-r", str(fps_float), "-vf", f"fps={fps_float},scale={width}:{height}"]
        if has_audio:
            args += ["-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100"]
        else:
            args += ["-an"]
        args.append(reencoded_path)
        run_ffmpeg(args, expected_path=reencoded_path)

        black_path = os.path.join(td, "black_outro.mkv")
        args = ["-y",
                "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps_float}:d={black_seconds}"]
        if has_audio:
            args += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={black_seconds}"]
            args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                     "-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100", "-shortest", black_path]
        else:
            args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", black_path]
        run_ffmpeg(args, expected_path=black_path)

        return self._concat_videos([reencoded_path, black_path], output_path, tmp_dir=td)

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
        """Reencode a video to specific resolution/fps so it can be concatenated."""
        args = [
            "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-vf", f"fps={fps},scale={width}:{height}",
            "-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100",
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
        run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", filelist,
                    "-c:v", "copy", "-c:a", "aac",
                    output_path],
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
        intro_reenc = os.path.join(td, f"intro_reenc_{Path(output_path).stem}.mkv")
        trim_args = ["-t", str(intro_seconds)] if intro_seconds else []
        args = ["-y", "-i", intro_source] + trim_args + [
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-vf", f"fps={fps},scale={width}:{height}",
            "-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100", "-shortest",
            intro_reenc,
        ]
        run_ffmpeg(args, expected_path=intro_reenc)

        # Reencode content
        content_reenc = os.path.join(td, f"content_reenc_{Path(output_path).stem}.mkv")
        self._reencode_for_concat(content_path, content_reenc, width, height, fps)

        return self._concat_videos([intro_reenc, content_reenc], output_path, tmp_dir=td)

    def _append_video(self, content_path: str, outro_source: str, output_path: str, outro_seconds: float | None = None, tmp_dir: str | None = None) -> str:
        """Append content from *outro_source* after *content_path*."""
        td = tmp_dir or self.wd.path
        data = video_utils.get_video_data(content_path)
        width = int(data["video"][0]["width"])
        height = int(data["video"][0]["height"])
        fps = generic_utils.fps_str_to_float(data["video"][0]["fps"])

        content_reenc = os.path.join(td, f"content_reenc_outro_{Path(output_path).stem}.mkv")
        self._reencode_for_concat(content_path, content_reenc, width, height, fps)

        outro_reenc = os.path.join(td, f"outro_reenc_{Path(output_path).stem}.mkv")
        trim_args = ["-t", str(outro_seconds)] if outro_seconds else []
        args = ["-y", "-i", outro_source] + trim_args + [
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-vf", f"fps={fps},scale={width}:{height}",
            "-c:a", "pcm_s16le", "-ac", "2", "-ar", "44100", "-shortest",
            outro_reenc,
        ]
        run_ffmpeg(args, expected_path=outro_reenc)

        return self._concat_videos([content_reenc, outro_reenc], output_path, tmp_dir=td)
