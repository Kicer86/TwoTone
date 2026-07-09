
import numpy as np
import os
import wave
import unittest

from dataclasses import dataclass, replace
from itertools import combinations, permutations, product
from parameterized import parameterized
from pathlib import Path
from typing import ClassVar

from twotone.tools.melt.melt import MeltAnalyzer, MeltPerformer, StaticSource
from twotone.tools.melt.melt_cache import MeltCache
from twotone.tools.utils import generic_utils, video_utils

from common import (
    TwoToneTestCase,
    FileCache,
    get_video,
    hashes,
    run_ffmpeg,
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    audio_start_offset: bool
    video_start_offset: bool
    audio_end_trim: bool
    video_end_trim: bool
    extension: str
    speed: float
    width: int
    height: int

    @property
    def pixel_area(self) -> int:
        return self.width * self.height


def _make_variant_specs() -> list[VariantSpec]:
    specs = []
    containers = ("mkv", "mp4", "mov", "mkv")
    speeds = (1.0, 1.0, 1.02, 1.0, 0.98, 1.0, 1.015, 1.0)

    for index, flags in enumerate(product((False, True), repeat=4)):
        audio_start, video_start, audio_end, video_end = flags
        # Variant name tags:
        # as/vs = audio/video stream start; ae/ve = audio/video stream end.
        # O = represented by a container offset, R = real media is present,
        # T = stream is trimmed before the canonical end.
        tags = [
            "asO" if audio_start else "asR",
            "vsO" if video_start else "vsR",
            "aeT" if audio_end else "aeR",
            "veT" if video_end else "veR",
        ]
        specs.append(
            VariantSpec(
                name=f"v{index:02d}_{'_'.join(tags)}",
                audio_start_offset=audio_start,
                video_start_offset=video_start,
                audio_end_trim=audio_end,
                video_end_trim=video_end,
                extension=containers[index % len(containers)],
                speed=speeds[index % len(speeds)],
                width=1280 - index * 2,
                height=720,
            )
        )

    return specs


VARIANTS = _make_variant_specs()
VARIANT_BY_NAME = {spec.name: spec for spec in VARIANTS}
FRAME_DRIFT_VARIANTS = [
    replace(
        spec,
        name=f"fd{index:02d}_{'_'.join(spec.name.split('_')[1:])}",
        speed=1.0,
    )
    for index, spec in enumerate(VARIANTS)
]
FRAME_DRIFT_VARIANT_BY_NAME = {spec.name: spec for spec in FRAME_DRIFT_VARIANTS}
FRAME_DRIFT_FPS_BY_NAME = {
    spec.name: 25 if index % 2 == 0 else 23
    for index, spec in enumerate(FRAME_DRIFT_VARIANTS)
}
PAIR_CASES = [
    (f"{lhs.name}__{rhs.name}", lhs.name, rhs.name)
    for lhs, rhs in permutations(VARIANTS, 2)
]
FRAME_DRIFT_PAIR_CASES = []
for index, (first, second) in enumerate(combinations(FRAME_DRIFT_VARIANTS, 2)):
    normal, drift = (first, second) if index % 2 == 0 else (second, first)
    FRAME_DRIFT_PAIR_CASES.extend([
        (
            f"{normal.name}__frame_drift_{drift.name}",
            normal.name,
            False,
            drift.name,
            True,
        ),
        (
            f"frame_drift_{drift.name}__{normal.name}",
            drift.name,
            True,
            normal.name,
            False,
        ),
    ])


class AudioAlignmentTest(TwoToneTestCase):
    """Verify melt keeps alternate audio streams sample-aligned.

    The cached inputs are visually equivalent Big Buck Bunny variants with
    different stream start offsets, trailing stream trims, containers, speeds,
    and minimally different resolutions.  The unique resolution ordering makes
    melt's base video choice deterministic for every pair.

    The frame-drift group keeps the same stream offset/trim/container/resolution
    matrix but fixes speed at 1.0 and gives one side real extra or dropped video
    frames via 24<->25/23 fps resampling.  That isolates frame-number drift
    detection from ordinary temporal speedup/slowdown.

    The constant-offset group drops the first few real frames (video and audio)
    with no compensating container offset — a rip missing a few leading frames.
    Unlike the vsO/asO variants, whose trimmed start is restored by a container
    start offset, here the whole timeline genuinely starts early, so the shared
    content sits at a constant frame offset between the pair and the alternate
    audio must be shifted by that offset in the output.
    """

    CACHE_VERSION = "1"
    FRAME_DRIFT_CACHE_VERSION = "1"
    CONSTANT_OFFSET_CACHE_VERSION = "2"
    CONSTANT_OFFSET_FRAMES = 5
    BLACK_INTRO_SECONDS = 0.5
    BLACK_OUTRO_SECONDS = 0.5
    BEEP_DURATION_SECONDS = 0.12
    BEEP_FREQUENCY = 880
    SAMPLE_RATE = 48000
    FPS = 24

    AUDIO_ALIGNMENT_TOLERANCE_SECONDS = 0.0071
    EXPECTED_POSITION_TOLERANCE_SECONDS = 0.013
    OUTPUT_DURATION_TOLERANCE_MS = 42

    source_video: ClassVar[str]
    source_duration_seconds: ClassVar[float]
    total_duration_seconds: ClassVar[float]
    beep_times: ClassVar[list[float]]
    beep_centers: ClassVar[list[float]]
    canonical_video: ClassVar[str]
    variant_paths: ClassVar[dict[str, str]]
    frame_reference_variant_paths: ClassVar[dict[str, str]]
    frame_drift_variant_paths: ClassVar[dict[str, str]]
    constant_offset_variant_paths: ClassVar[dict[str, str]]
    melt_cache: ClassVar[MeltCache | None]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.source_video = get_video("big_buck_bunny_720p_10mb.mp4")
        cls.source_duration_seconds = cls._duration_seconds(cls.source_video)
        cls.total_duration_seconds = (
            cls.source_duration_seconds
            + cls.BLACK_INTRO_SECONDS
            + cls.BLACK_OUTRO_SECONDS
        )
        cls.beep_times = cls._build_beep_times(cls.total_duration_seconds)
        cls.beep_centers = [
            value + cls.BEEP_DURATION_SECONDS / 2
            for value in cls.beep_times
        ]

        file_cache = FileCache("TwoToneTests")
        cls.canonical_video = str(file_cache.get_or_generate(
            "audio_align_canonical_bbb",
            cls.CACHE_VERSION,
            "mkv",
            cls._generate_canonical_video,
        ))

        cls.variant_paths = {
            spec.name: str(file_cache.get_or_generate(
                f"audio_align_{spec.name}",
                cls.CACHE_VERSION,
                spec.extension,
                lambda out_path, spec=spec: cls._generate_variant(spec, out_path),
            ))
            for spec in VARIANTS
        }

        cls.frame_reference_variant_paths = {
            spec.name: str(file_cache.get_or_generate(
                f"audio_align_frame_reference_{spec.name}",
                cls.FRAME_DRIFT_CACHE_VERSION,
                spec.extension,
                lambda out_path, spec=spec: cls._generate_variant(spec, out_path),
            ))
            for spec in FRAME_DRIFT_VARIANTS
        }

        cls.frame_drift_variant_paths = {
            spec.name: str(file_cache.get_or_generate(
                f"audio_align_frame_drift_{spec.name}",
                cls.FRAME_DRIFT_CACHE_VERSION,
                spec.extension,
                lambda out_path, spec=spec: cls._generate_frame_drift_variant(spec, out_path),
            ))
            for spec in FRAME_DRIFT_VARIANTS
        }

        # The two widths pick which side of a pair with the full-size v00
        # variant (1280x720) becomes the base video: "small" loses, "large" wins.
        cls.constant_offset_variant_paths = {
            name: str(file_cache.get_or_generate(
                f"audio_align_constant_offset_{name}",
                cls.CONSTANT_OFFSET_CACHE_VERSION,
                "mkv",
                lambda out_path, width=width: cls._generate_constant_offset_variant(width, out_path),
            ))
            for name, width in (("small", 1278), ("large", 1282))
        }

        cache_dir = Path(file_cache.base_dir) / "audio_alignment_melt_cache"
        cls.melt_cache = MeltCache(str(cache_dir), cls.logger.getChild("MeltCache"))

    @staticmethod
    def _duration_seconds(path: str) -> float:
        duration_ms = video_utils.get_video_duration(path)
        if duration_ms is None:
            raise RuntimeError(f"Could not probe duration for {path}")
        return duration_ms / 1000

    @classmethod
    def _build_beep_times(cls, total_duration_seconds: float) -> list[float]:
        result = []
        current = 1.0
        while current + cls.BEEP_DURATION_SECONDS < total_duration_seconds - 1.0:
            result.append(round(current, 3))
            current += 5.25
        return result

    @classmethod
    def _generate_canonical_video(cls, out_path: Path) -> None:
        beep_expr = "+".join(
            f"between(t\\,{t:.6f}\\,{t + cls.BEEP_DURATION_SECONDS:.6f})"
            for t in cls.beep_times
        )
        red_expr = "+".join(
            f"between(t\\,{t - cls.BLACK_INTRO_SECONDS:.6f}\\,"
            f"{t - cls.BLACK_INTRO_SECONDS + cls.BEEP_DURATION_SECONDS:.6f})"
            for t in cls.beep_times
            if t >= cls.BLACK_INTRO_SECONDS
        )
        filter_complex = (
            f"[1:v]fps={cls.FPS},scale=1280:720,format=yuv420p,"
            f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.90:t=fill:"
            f"enable='{red_expr}'[main];"
            "[0:v][main][2:v]concat=n=3:v=1:a=0[v]"
        )

        run_ffmpeg(
            [
                "-y",
                "-f", "lavfi",
                "-i", f"color=c=black:s=1280x720:r={cls.FPS}:d={cls.BLACK_INTRO_SECONDS}",
                "-i", cls.source_video,
                "-f", "lavfi",
                "-i", f"color=c=black:s=1280x720:r={cls.FPS}:d={cls.BLACK_OUTRO_SECONDS}",
                "-f", "lavfi",
                "-i",
                (
                    f"aevalsrc=0.75*sin(2*PI*{cls.BEEP_FREQUENCY}*t)"
                    f"*({beep_expr}):s={cls.SAMPLE_RATE}:d={cls.total_duration_seconds:.6f}"
                ),
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "3:a:0",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "flac",
                "-sample_fmt", "s16",
                str(out_path),
            ],
            expected_path=str(out_path),
        )

    @classmethod
    def _generate_variant(cls, spec: VariantSpec, out_path: Path) -> None:
        video_start = cls.BLACK_INTRO_SECONDS if spec.video_start_offset else 0.0
        audio_start = cls.BLACK_INTRO_SECONDS if spec.audio_start_offset else 0.0
        video_end = (
            cls.total_duration_seconds - cls.BLACK_OUTRO_SECONDS
            if spec.video_end_trim
            else cls.total_duration_seconds
        )
        audio_end = (
            cls.total_duration_seconds - cls.BLACK_OUTRO_SECONDS
            if spec.audio_end_trim
            else cls.total_duration_seconds
        )
        video_offset = cls.BLACK_INTRO_SECONDS / spec.speed if spec.video_start_offset else 0.0
        audio_offset = cls.BLACK_INTRO_SECONDS / spec.speed if spec.audio_start_offset else 0.0

        filter_complex = (
            f"[0:v]trim=start={video_start:.6f}:end={video_end:.6f},"
            # Speed variants keep frame indexes stable; only frame duration/effective FPS changes.
            f"setpts=N/({cls.FPS}*{spec.speed:.8f})/TB,"
            f"scale={spec.width}:{spec.height},"
            f"setpts=PTS+{video_offset:.8f}/TB[v];"
            f"[0:a]atrim=start={audio_start:.6f}:end={audio_end:.6f},"
            "asetpts=PTS-STARTPTS,"
            f"atempo={spec.speed:.8f},"
            f"asetpts=PTS+{audio_offset:.8f}/TB[a]"
        )

        run_ffmpeg(
            [
                "-y",
                "-i", cls.canonical_video,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-fps_mode", "passthrough",
                "-enc_time_base:v", "1:1000000",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-ar", str(cls.SAMPLE_RATE),
                "-ac", "1",
                str(out_path),
            ],
            expected_path=str(out_path),
        )

    @classmethod
    def _generate_frame_drift_variant(cls, spec: VariantSpec, out_path: Path) -> None:
        video_start = cls.BLACK_INTRO_SECONDS if spec.video_start_offset else 0.0
        audio_start = cls.BLACK_INTRO_SECONDS if spec.audio_start_offset else 0.0
        video_end = (
            cls.total_duration_seconds - cls.BLACK_OUTRO_SECONDS
            if spec.video_end_trim
            else cls.total_duration_seconds
        )
        audio_end = (
            cls.total_duration_seconds - cls.BLACK_OUTRO_SECONDS
            if spec.audio_end_trim
            else cls.total_duration_seconds
        )
        video_offset = cls.BLACK_INTRO_SECONDS / spec.speed if spec.video_start_offset else 0.0
        audio_offset = cls.BLACK_INTRO_SECONDS / spec.speed if spec.audio_start_offset else 0.0
        drift_fps = FRAME_DRIFT_FPS_BY_NAME[spec.name]

        filter_complex = (
            f"[0:v]trim=start={video_start:.6f}:end={video_end:.6f},"
            "setpts=PTS-STARTPTS,"
            f"fps={drift_fps},"
            # Unlike _generate_variant(), this really creates/drops frames before
            # speed scaling.  A 24 fps source therefore maps frame 24 to frame 25
            # for 25 fps drift variants, or frame 24 to frame 23 for 23 fps ones.
            f"setpts=PTS/{spec.speed:.8f},"
            f"scale={spec.width}:{spec.height},"
            f"setpts=PTS+{video_offset:.8f}/TB[v];"
            f"[0:a]atrim=start={audio_start:.6f}:end={audio_end:.6f},"
            "asetpts=PTS-STARTPTS,"
            f"atempo={spec.speed:.8f},"
            f"asetpts=PTS+{audio_offset:.8f}/TB[a]"
        )

        run_ffmpeg(
            [
                "-y",
                "-i", cls.canonical_video,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-fps_mode", "passthrough",
                "-enc_time_base:v", "1:1000000",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-ar", str(cls.SAMPLE_RATE),
                "-ac", "1",
                str(out_path),
            ],
            expected_path=str(out_path),
        )

    @classmethod
    def _generate_constant_offset_variant(cls, width: int, out_path: Path) -> None:
        trim_seconds = cls.CONSTANT_OFFSET_FRAMES / cls.FPS
        filter_complex = (
            f"[0:v]trim=start_frame={cls.CONSTANT_OFFSET_FRAMES},"
            # Rebase timestamps to zero: the dropped frames leave no container
            # offset behind, the content itself starts earlier than canonical.
            f"setpts=N/{cls.FPS}/TB,"
            f"scale={width}:720[v];"
            f"[0:a]atrim=start={trim_seconds:.6f},"
            "asetpts=PTS-STARTPTS,"
            # Even at 1.0 atempo displaces audio by ~11 ms; every fixture built
            # by _generate_variant carries that displacement, so this one must
            # run the same chain or pairs against them inherit the difference.
            "atempo=1.0[a]"
        )

        run_ffmpeg(
            [
                "-y",
                "-i", cls.canonical_video,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-fps_mode", "passthrough",
                "-enc_time_base:v", "1:1000000",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-ar", str(cls.SAMPLE_RATE),
                "-ac", "1",
                str(out_path),
            ],
            expected_path=str(out_path),
        )

    @staticmethod
    def _pick_expected_base(lhs: VariantSpec, rhs: VariantSpec) -> VariantSpec:
        return max(
            (lhs, rhs),
            key=lambda spec: (spec.pixel_area, spec.width, spec.height, spec.speed),
        )

    @staticmethod
    def _stream_start_time(path: str, stream_type: str, stream_index: int) -> float:
        info = video_utils.get_video_full_info(path)
        streams = [
            stream
            for stream in info["streams"]
            if stream.get("codec_type") == stream_type
        ]
        value = streams[stream_index].get("start_time", 0.0)
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _stream_duration_seconds(stream: dict) -> float | None:
        value = stream.get("duration")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

        tag_duration = stream.get("tags", {}).get("DURATION")
        if tag_duration is not None:
            return generic_utils.time_to_ms(tag_duration) / 1000

        return None

    @classmethod
    def _playback_end_ms(cls, path: str) -> int | None:
        info = video_utils.get_video_full_info(path)
        ends = []
        for stream in info["streams"]:
            if stream.get("codec_type") not in ("audio", "video"):
                continue
            duration = cls._stream_duration_seconds(stream)
            if duration is None:
                continue
            start = float(stream.get("start_time", 0.0) or 0.0)
            ends.append(round((start + duration) * 1000))

        if ends:
            return max(ends)
        return video_utils.get_video_duration(path)

    @classmethod
    def _expected_beep_centers(cls, spec: VariantSpec) -> list[float]:
        return [value / spec.speed for value in cls.beep_centers]

    @classmethod
    def _expected_duration_ms(cls, spec: VariantSpec) -> int:
        ends = []
        for start_offset, end_trim in (
            (spec.video_start_offset, spec.video_end_trim),
            (spec.audio_start_offset, spec.audio_end_trim),
        ):
            start = cls.BLACK_INTRO_SECONDS if start_offset else 0.0
            end = (
                cls.total_duration_seconds - cls.BLACK_OUTRO_SECONDS
                if end_trim
                else cls.total_duration_seconds
            )
            offset = cls.BLACK_INTRO_SECONDS if start_offset else 0.0
            ends.append((offset + end - start) / spec.speed)
        return round(max(ends) * 1000)

    @staticmethod
    def _read_mono_wav(path: str) -> tuple[int, np.ndarray]:
        with wave.open(path, "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())

        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return sample_rate, samples / 32768.0

    @classmethod
    def _detect_beep_centers(cls, wav_path: str) -> list[float]:
        sample_rate, samples = cls._read_mono_wav(wav_path)
        if samples.size == 0:
            return []

        power = samples * samples
        window = max(1, round(sample_rate * 0.002))
        kernel = np.ones(window, dtype=np.float32) / window
        envelope = np.convolve(power, kernel, mode="same")
        peak = float(np.max(envelope))
        if peak <= 0:
            return []

        active = envelope > peak * 0.08
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            return []

        starts = [int(active_indices[0])]
        ends = []
        for previous, current in zip(active_indices, active_indices[1:]):
            if current - previous > 1:
                ends.append(int(previous))
                starts.append(int(current))
        ends.append(int(active_indices[-1]))

        min_gap = round(sample_rate * 0.020)
        merged: list[tuple[int, int]] = []
        for start, end in zip(starts, ends):
            if merged and start - merged[-1][1] <= min_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        min_duration = sample_rate * 0.030
        centers = [
            (start + end) / 2 / sample_rate
            for start, end in merged
            if end - start >= min_duration
        ]
        return centers

    def _run_melt_pair(self, lhs_path: str, rhs_path: str) -> str:
        # Feed melt the shared cached variant paths directly. Copying them into the
        # per-test working dir (as add_to_test_dir does on Windows) would give every
        # parameterized case a unique resolved path, defeating MeltCache — which keys
        # entries by resolved input path — and bloating the cache with single-use
        # entries. Melt only reads its inputs here, so sharing the fixtures is safe.
        file1 = lhs_path
        file2 = rhs_path

        interruption = generic_utils.InterruptibleProcess()
        duplicates = StaticSource(interruption)
        duplicates.add_entry("Video", file1)
        duplicates.add_entry("Video", file2)
        duplicates.add_metadata(file1, "audio_lang", "eng")
        duplicates.add_metadata(file2, "audio_lang", "pol")

        output_dir = os.path.join(self.wd.path, "output")
        os.makedirs(output_dir)

        logger = self.logger.getChild("Melter")
        analyzer = MeltAnalyzer(logger, duplicates, self.workspace, True, 100)
        duplicates_raw = duplicates.collect_duplicates()
        plan = analyzer.analyze_duplicates({
            title: list(files)
            for title, files in duplicates_raw.items()
        })

        performer = MeltPerformer(
            logger,
            interruption,
            self.workspace,
            output_dir,
            100,
            cache=self.melt_cache,
        )
        performer.process_duplicates(plan)

        output_files = list(hashes(output_dir).keys())
        self.assertEqual(len(output_files), 1)
        return output_files[0]

    def _extract_audio_centers(self, output_file: str, audio_index: int) -> list[float]:
        audio_start = self._stream_start_time(output_file, "audio", audio_index)
        wav_path = os.path.join(self.wd.path, f"track_{audio_index}.wav")
        run_ffmpeg(
            [
                "-y",
                "-i", output_file,
                "-map", f"0:a:{audio_index}",
                "-ac", "1",
                "-ar", str(self.SAMPLE_RATE),
                "-c:a", "pcm_s16le",
                wav_path,
            ],
            expected_path=wav_path,
        )
        return [
            audio_start + value
            for value in self._detect_beep_centers(wav_path)
        ]

    def _assert_audio_tracks_aligned(
        self,
        first_centers: list[float],
        second_centers: list[float],
    ) -> None:
        self.assertEqual(
            len(first_centers),
            len(self.beep_centers),
            f"Unexpected beep count in first audio track: {first_centers}",
        )
        self.assertEqual(
            len(second_centers),
            len(self.beep_centers),
            f"Unexpected beep count in second audio track: {second_centers}",
        )

        offsets = [
            abs(first - second)
            for first, second in zip(first_centers, second_centers)
        ]
        max_offset = max(offsets, default=0.0)
        self.assertLessEqual(
            max_offset,
            self.AUDIO_ALIGNMENT_TOLERANCE_SECONDS,
            f"Audio tracks are not aligned closely enough; "
            f"max offset: {max_offset:.6f}s, offsets: {offsets}",
        )

    def _assert_expected_positions(
        self,
        expected: list[float],
        actual: list[float],
    ) -> None:
        for index, (expected_time, actual_time) in enumerate(zip(expected, actual)):
            self.assertAlmostEqual(
                expected_time,
                actual_time,
                delta=self.EXPECTED_POSITION_TOLERANCE_SECONDS,
                msg=(
                    f"Beep #{index} expected near {expected_time:.6f}s, "
                    f"detected at {actual_time:.6f}s"
                ),
            )

    def _assert_melt_pair_alignment(
        self,
        lhs: VariantSpec,
        rhs: VariantSpec,
        lhs_path: str,
        rhs_path: str,
    ) -> None:
        expected_base = self._pick_expected_base(lhs, rhs)

        output_file = self._run_melt_pair(lhs_path, rhs_path)

        output_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_data["tracks"]["audio"]), 2)

        actual_duration = self._playback_end_ms(output_file)
        expected_duration = self._expected_duration_ms(expected_base)
        if actual_duration is None:
            self.fail("Could not determine output playback duration")
        self.assertAlmostEqual(
            expected_duration,
            actual_duration,
            delta=self.OUTPUT_DURATION_TOLERANCE_MS,
        )

        first_centers = self._extract_audio_centers(output_file, 0)
        second_centers = self._extract_audio_centers(output_file, 1)
        self._assert_audio_tracks_aligned(first_centers, second_centers)

        expected_centers = self._expected_beep_centers(expected_base)
        merged_centers = [
            (first + second) / 2
            for first, second in zip(first_centers, second_centers)
        ]
        self._assert_expected_positions(expected_centers, merged_centers)

    @parameterized.expand(PAIR_CASES)
    def test_audio_alignment_after_melt(self, _case_name: str, lhs_name: str, rhs_name: str):
        lhs = VARIANT_BY_NAME[lhs_name]
        rhs = VARIANT_BY_NAME[rhs_name]
        self._assert_melt_pair_alignment(
            lhs,
            rhs,
            self.variant_paths[lhs.name],
            self.variant_paths[rhs.name],
        )

    @parameterized.expand([
        ("offset_as_source", "small"),
        ("offset_as_base", "large"),
    ])
    def test_audio_alignment_with_uncompensated_frame_offset(
        self,
        _case_name: str,
        offset_variant: str,
    ):
        """A pair shifted by a few real frames must have the alternate audio
        moved by that offset in the output.

        PairMatcher detects the shift as a global-linear constant frame offset
        and the audio strategy is passthrough; the offset must survive into the
        muxed track placement, in both directions: the shifted file providing
        the alternate audio (sync delay) and the shifted file being the base
        (the alternate audio starts before the base timeline).
        """
        full_path = self.variant_paths["v00_asR_vsR_aeR_veR"]
        offset_path = self.constant_offset_variant_paths[offset_variant]
        offset_seconds = self.CONSTANT_OFFSET_FRAMES / self.FPS
        # The base video defines the output timeline; when the shifted file
        # wins, every beep sits one offset earlier than in the canonical cut.
        base_shift = -offset_seconds if offset_variant == "large" else 0.0

        output_file = self._run_melt_pair(full_path, offset_path)

        output_data = video_utils.get_video_data_mkvmerge(output_file)
        self.assertEqual(len(output_data["tracks"]["audio"]), 2)

        first_centers = self._extract_audio_centers(output_file, 0)
        second_centers = self._extract_audio_centers(output_file, 1)
        self._assert_audio_tracks_aligned(first_centers, second_centers)

        expected_centers = [value + base_shift for value in self.beep_centers]
        self._assert_expected_positions(expected_centers, first_centers)
        self._assert_expected_positions(expected_centers, second_centers)

    @parameterized.expand(FRAME_DRIFT_PAIR_CASES)
    def test_audio_alignment_after_melt_with_frame_drift(
        self,
        _case_name: str,
        lhs_name: str,
        lhs_frame_drift: bool,
        rhs_name: str,
        rhs_frame_drift: bool,
    ):
        lhs = FRAME_DRIFT_VARIANT_BY_NAME[lhs_name]
        rhs = FRAME_DRIFT_VARIANT_BY_NAME[rhs_name]
        lhs_paths = self.frame_drift_variant_paths if lhs_frame_drift else self.frame_reference_variant_paths
        rhs_paths = self.frame_drift_variant_paths if rhs_frame_drift else self.frame_reference_variant_paths
        self._assert_melt_pair_alignment(
            lhs,
            rhs,
            lhs_paths[lhs.name],
            rhs_paths[rhs.name],
        )


if __name__ == '__main__':
    unittest.main()
