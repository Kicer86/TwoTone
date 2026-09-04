"""Run-scoped media analysis shared by planning, validation, and execution."""

import enum
import logging
import os
import re

from dataclasses import dataclass

from tqdm import tqdm

from . import generic_utils, video_utils
from .files_utils import Workspace
from .generic_utils import InterruptibleProcess


_SCENE_FRAME_RE = re.compile(
    r"^frame:\d+\s+pts:\S+\s+pts_time:([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
)
_PROGRESS_TIME_RE = re.compile(r"^out_time_ms=(\d+)$")
_STATS_FRAME_RE = re.compile(
    r"^(\d+)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+))$"
)
_PROGRESS_PREFIXES = (
    "bitrate=",
    "drop_frames=",
    "dup_frames=",
    "fps=",
    "frame=",
    "out_time=",
    "out_time_ms=",
    "out_time_us=",
    "progress=",
    "speed=",
    "stream_",
    "total_size=",
)
_IDENTITY_SAMPLE_COUNT = 7


class MediaAnalysisFeature(enum.IntFlag):
    NONE = 0
    IDENTITY_SAMPLES = enum.auto()
    SCENE_CHANGES = enum.auto()
    FRAME_TIMESTAMPS = enum.auto()
    VALIDATE_STREAMS = enum.auto()

    MATCHING = SCENE_CHANGES | FRAME_TIMESTAMPS


def identity_timestamps(duration_ms: int, fps: float) -> tuple[int, ...]:
    if duration_ms <= 0 or fps <= 0:
        return (0,)

    last_timestamp = max(0, duration_ms - max(1, round(1000 / fps)))
    return tuple(sorted({
        round(last_timestamp * index / (_IDENTITY_SAMPLE_COUNT - 1))
        for index in range(_IDENTITY_SAMPLE_COUNT)
    }))


@dataclass(frozen=True)
class VideoSample:
    target_ms: int
    timestamp_ms: int
    frame_id: int
    path: str


@dataclass(frozen=True)
class VideoScanResult:
    path: str
    features: MediaAnalysisFeature
    frames: dict[int, dict]
    scene_changes: tuple[int, ...]
    identity_samples: tuple[VideoSample, ...]
    decode_error: str | None

    def supports(self, features: MediaAnalysisFeature) -> bool:
        return self.features & features == features

    @property
    def validated_all_streams(self) -> bool:
        return self.supports(MediaAnalysisFeature.VALIDATE_STREAMS)

    def frames_copy(self) -> dict[int, dict]:
        """Return mutable frame metadata isolated from other consumers."""
        return {timestamp: info.copy() for timestamp, info in self.frames.items()}


class MediaAnalysisSession:
    """Provide media-analysis data"""

    _SCENE_THRESHOLD = 0.3

    def __init__(
        self,
        workspace: Workspace,
        interruption: InterruptibleProcess,
        logger: logging.Logger,
        *,
        validate_all_streams: bool,
    ) -> None:
        self.workspace = workspace
        self.interruption = interruption
        self.logger = logger
        self.validate_all_streams = validate_all_streams
        self._cache: dict[tuple[object, ...], VideoScanResult] = {}
        self._path_results: dict[str, VideoScanResult] = {}

    def scan(
        self,
        path: str,
        *,
        duration_ms: int,
        fps: float,
        label: str,
        features: MediaAnalysisFeature,
    ) -> VideoScanResult:
        real_path = os.path.realpath(path)
        key = self._file_key(real_path)
        requested_features = features
        if self.validate_all_streams:
            requested_features |= MediaAnalysisFeature.VALIDATE_STREAMS

        if requested_features == MediaAnalysisFeature.NONE:
            raise ValueError("At least one media analysis feature must be requested")

        cached = self._cache.get(key)
        if cached is not None and cached.supports(requested_features):
            self.logger.info("Media scan for %s restored from this run's cache.", label)
            self._path_results[real_path] = cached
            return cached

        missing_features = requested_features
        if cached is not None:
            missing_features &= ~cached.features

        scanned = self._scan(real_path, duration_ms, fps, label, missing_features)
        result = self._merge_results(cached, scanned) if cached is not None else scanned
        self._cache[key] = result
        self._path_results[real_path] = result
        return result

    def result_for(self, path: str) -> VideoScanResult | None:
        return self._path_results.get(os.path.realpath(path))

    def results_for(self, paths: set[str]) -> dict[str, VideoScanResult]:
        return {
            path: result
            for path in paths
            if (result := self.result_for(path)) is not None
        }

    @staticmethod
    def _file_key(path: str) -> tuple[object, ...]:
        stat = os.stat(path)
        return (
            path,
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        )

    def _scan(
        self,
        path: str,
        duration_ms: int,
        fps: float,
        label: str,
        features: MediaAnalysisFeature,
    ) -> VideoScanResult:
        scan_dir = self.workspace.unique_dir("media_scan")
        frame_stats_path = os.path.join(scan_dir, "frames.txt")
        sample_stats_path = os.path.join(scan_dir, "identity.txt")
        sample_pattern = os.path.join(scan_dir, "identity_%08d.png")
        target_timestamps = identity_timestamps(duration_ms, fps)
        sample_select = self._sample_select_expression(target_timestamps)

        branches: list[str] = []
        if features & MediaAnalysisFeature.FRAME_TIMESTAMPS:
            branches.append("vframes")
        elif features & MediaAnalysisFeature.VALIDATE_STREAMS:
            branches.append("vvalidate")
        if features & MediaAnalysisFeature.SCENE_CHANGES:
            branches.append("vscenes")
        if features & MediaAnalysisFeature.IDENTITY_SAMPLES:
            branches.append("vsamples")

        filter_parts: list[str] = []
        if len(branches) > 1:
            outputs = "".join(f"[{branch}]" for branch in branches)
            filter_parts.append(f"[0:v:0]split={len(branches)}{outputs}")

        def branch_source(name: str) -> str:
            return f"[{name}]" if len(branches) > 1 else "[0:v:0]"

        if len(branches) == 1 and branches[0] in {"vframes", "vvalidate"}:
            filter_parts.append(f"[0:v:0]null[{branches[0]}]")

        if features & MediaAnalysisFeature.SCENE_CHANGES:
            filter_parts.append(
                f"{branch_source('vscenes')}select='gt(scene,{self._SCENE_THRESHOLD})',"
                "metadata=mode=print:file='pipe\\:2',nullsink"
            )

        if features & MediaAnalysisFeature.IDENTITY_SAMPLES:
            filter_parts.append(
                f"{branch_source('vsamples')}select='{sample_select}',scale=960:-2[identity]"
            )

        args = [
            "-v", "error",
            "-nostats",
            "-progress", "pipe:2",
        ]

        if features & MediaAnalysisFeature.VALIDATE_STREAMS:
            args.append("-xerror")

        args.extend(["-i", path])

        if filter_parts:
            args.extend(["-filter_complex", ";".join(filter_parts)])

        needs_null_output = bool(
            features
            & (MediaAnalysisFeature.FRAME_TIMESTAMPS | MediaAnalysisFeature.VALIDATE_STREAMS)
        ) or not features & MediaAnalysisFeature.IDENTITY_SAMPLES

        if needs_null_output:
            if features & MediaAnalysisFeature.FRAME_TIMESTAMPS:
                args.extend(["-map", "[vframes]"])
            elif features & MediaAnalysisFeature.VALIDATE_STREAMS:
                args.extend(["-map", "[vvalidate]"])
            elif features & MediaAnalysisFeature.SCENE_CHANGES:
                args.extend(["-map", "0:v:0"])

        if features & MediaAnalysisFeature.VALIDATE_STREAMS:
            args.extend([
                "-map", "0:v?",
                "-map", "-0:v:0",
                "-map", "0:a?",
            ])
        elif needs_null_output:
            args.append("-an")

        if needs_null_output:
            args.extend(["-sn", "-dn", "-fps_mode", "vfr"])
            if features & MediaAnalysisFeature.FRAME_TIMESTAMPS:
                args.extend([
                    "-stats_enc_pre:v:0", frame_stats_path,
                    "-stats_enc_pre_fmt:v:0", "{ni} {ti}",
                ])
            args.extend(["-f", "null", "-"])

        if features & MediaAnalysisFeature.IDENTITY_SAMPLES:
            args.extend([
                "-map", "[identity]",
                "-an", "-sn", "-dn",
                "-fps_mode", "vfr",
                "-stats_enc_pre:v:0", sample_stats_path,
                "-stats_enc_pre_fmt:v:0", "{ni} {ti}",
                sample_pattern,
            ])

        scene_timestamps: list[int] = []
        duration_s = duration_ms / 1000
        last_progress_s = 0.0
        timestamp_correction_ms = video_utils._showinfo_timestamp_correction_ms(
            path,
            logger=self.logger,
        )

        progress = tqdm(
            total=duration_s,
            desc=f"Scanning media: {label}",
            unit="s",
            **generic_utils.get_tqdm_defaults(),
        )

        def on_line(line: str) -> None:
            nonlocal last_progress_s

            stripped = line.strip()
            scene_match = _SCENE_FRAME_RE.match(stripped)
            if scene_match:
                timestamp_ms = video_utils._showinfo_timestamp_ms(
                    scene_match.group(1),
                    timestamp_correction_ms,
                )
                scene_timestamps.append(timestamp_ms)
                return

            progress_match = _PROGRESS_TIME_RE.match(stripped)
            if progress_match:
                current_s = int(progress_match.group(1)) / 1_000_000
                delta = min(duration_s, current_s) - last_progress_s
                if delta > 0:
                    progress.update(delta)
                    last_progress_s += delta

        process, stderr_lines = video_utils._start_ffmpeg_streaming(
            args,
            self.interruption,
            on_line=on_line,
            logger=self.logger,
        )
        if last_progress_s < duration_s:
            progress.update(duration_s - last_progress_s)
        progress.close()

        frames = (
            self._read_frames(frame_stats_path, timestamp_correction_ms)
            if features & MediaAnalysisFeature.FRAME_TIMESTAMPS
            else {}
        )
        if features & MediaAnalysisFeature.IDENTITY_SAMPLES:
            sample_entries = self._read_frame_entries(sample_stats_path, timestamp_correction_ms)
            sample_files = sorted(
                os.path.join(scan_dir, filename)
                for filename in os.listdir(scan_dir)
                if filename.startswith("identity_") and filename.endswith(".png")
            )
            samples = self._build_samples(
                target_timestamps,
                sample_entries,
                sample_files,
                frames,
            )
        else:
            samples = ()

        decode_error = self._decode_error(process.returncode, stderr_lines)
        return VideoScanResult(
            path=path,
            features=features,
            frames=frames,
            scene_changes=tuple(sorted(set(scene_timestamps))),
            identity_samples=samples,
            decode_error=decode_error,
        )

    @staticmethod
    def _merge_results(
        cached: VideoScanResult,
        scanned: VideoScanResult,
    ) -> VideoScanResult:
        errors = tuple(
            dict.fromkeys(
                error
                for error in (cached.decode_error, scanned.decode_error)
                if error is not None
            )
        )
        return VideoScanResult(
            path=scanned.path,
            features=cached.features | scanned.features,
            frames=(
                scanned.frames
                if scanned.supports(MediaAnalysisFeature.FRAME_TIMESTAMPS)
                else cached.frames
            ),
            scene_changes=(
                scanned.scene_changes
                if scanned.supports(MediaAnalysisFeature.SCENE_CHANGES)
                else cached.scene_changes
            ),
            identity_samples=(
                scanned.identity_samples
                if scanned.supports(MediaAnalysisFeature.IDENTITY_SAMPLES)
                else cached.identity_samples
            ),
            decode_error=" | ".join(errors) if errors else None,
        )

    @staticmethod
    def _sample_select_expression(timestamps_ms: tuple[int, ...]) -> str:
        parts = ["isnan(prev_pts)"]
        for timestamp_ms in timestamps_ms[1:]:
            timestamp_s = timestamp_ms / 1000
            parts.append(
                f"lt(prev_pts*TB,{timestamp_s:.6f})*gte(t,{timestamp_s:.6f})"
            )
        return "+".join(parts)

    @staticmethod
    def _read_frame_entries(path: str, correction_ms: int) -> list[tuple[int, int]]:
        try:
            with open(path, encoding="utf-8") as file:
                lines = file.readlines()
        except OSError:
            return []

        entries: list[tuple[int, int]] = []
        for line in lines:
            match = _STATS_FRAME_RE.match(line.strip())
            if match is None:
                continue
            frame_id = int(match.group(1))
            timestamp_ms = video_utils._showinfo_timestamp_ms(
                match.group(2),
                correction_ms,
            )
            entries.append((frame_id, timestamp_ms))
        return entries

    @classmethod
    def _read_frames(cls, path: str, correction_ms: int) -> dict[int, dict]:
        return {
            timestamp_ms: {"frame_id": frame_id, "path": None}
            for frame_id, timestamp_ms in cls._read_frame_entries(path, correction_ms)
        }

    @staticmethod
    def _build_samples(
        targets: tuple[int, ...],
        entries: list[tuple[int, int]],
        files: list[str],
        frames: dict[int, dict],
    ) -> tuple[VideoSample, ...]:
        candidates = [
            (frame_id, timestamp_ms, path)
            for (frame_id, timestamp_ms), path in zip(entries, files)
        ]
        samples: list[VideoSample] = []
        for target_ms in targets:
            if not candidates:
                break
            frame_id, timestamp_ms, path = min(
                candidates,
                key=lambda candidate: abs(candidate[1] - target_ms),
            )
            candidates.remove((frame_id, timestamp_ms, path))
            if frames:
                frame_timestamp = min(frames, key=lambda candidate: abs(candidate - timestamp_ms))
                frame_id = int(frames[frame_timestamp]["frame_id"])
                frames[frame_timestamp]["path"] = path
                timestamp_ms = frame_timestamp
            samples.append(VideoSample(target_ms, timestamp_ms, frame_id, path))
        return tuple(samples)

    @staticmethod
    def _decode_error(returncode: int, stderr_lines: list[str]) -> str | None:
        errors = [
            line.strip()
            for line in stderr_lines
            if line.strip()
            and not line.startswith(_PROGRESS_PREFIXES)
            and _SCENE_FRAME_RE.match(line.strip()) is None
            and not line.startswith("lavfi.scene_score=")
        ]
        if returncode == 0 and not errors:
            return None

        if errors:
            return " | ".join(dict.fromkeys(errors[-3:]))
        return f"ffmpeg exited with code {returncode}"
