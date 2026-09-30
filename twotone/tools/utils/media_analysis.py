"""Run-scoped media analysis shared by planning, validation, and execution."""

import enum
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from tqdm import tqdm

from . import generic_utils, video_utils
from .files_utils import Workspace
from .generic_utils import InterruptibleProcess

_SCENE_FRAME_RE = re.compile(
    r"^frame:\d+\s+pts:\S+\s+pts_time:([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
)
_PROGRESS_TIME_RE = re.compile(r"^out_time_ms=(\d+)$")
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


_FEATURE_LABELS = (
    (MediaAnalysisFeature.IDENTITY_SAMPLES, "identity_samples"),
    (MediaAnalysisFeature.SCENE_CHANGES, "scene_changes"),
    (MediaAnalysisFeature.FRAME_TIMESTAMPS, "frame_timestamps"),
    (MediaAnalysisFeature.VALIDATE_STREAMS, "validate_streams"),
)


def _format_features(features: MediaAnalysisFeature) -> str:
    names = [name for feature, name in _FEATURE_LABELS if features & feature]
    return ", ".join(names) if names else "none"


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
class MediaAnalysisRequest:
    path: str
    label: str
    features: MediaAnalysisFeature


@dataclass(frozen=True)
class MediaProbeResult:
    path: str
    data: dict
    error: str | None
    normalized_data: dict = field(default_factory=dict)

    @property
    def has_audio(self) -> bool:
        return any(stream.get("codec_type") == "audio" for stream in self.data.get("streams", []))

    @property
    def has_video(self) -> bool:
        return any(stream.get("codec_type") == "video" for stream in self.data.get("streams", []))

    @property
    def has_decodable_stream(self) -> bool:
        return self.has_audio or self.has_video

    @property
    def primary_video_track(self) -> dict | None:
        normalized_tracks = self.normalized_data.get("video", [])
        tracks_by_index = {
            track.get("tid"): track
            for track in normalized_tracks
        }
        for stream in self.data.get("streams", []):
            if (
                stream.get("codec_type") == "video"
                and not stream.get("disposition", {}).get("attached_pic", 0)
            ):
                track = tracks_by_index.get(stream.get("index"))
                if track is not None:
                    return track
        return None


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


class PersistentMediaAnalysisCache(Protocol):
    def load_scene_changes(self, video_path: str) -> list[int] | None: ...

    def save_scene_changes(self, video_path: str, scenes: list[int]) -> None: ...

    def load_frame_probes(self, video_path: str) -> dict[int, dict] | None: ...

    def save_frame_probes(self, video_path: str, probes: dict[int, dict]) -> None: ...


class MediaAnalysisSession:
    """Collect and reuse requested media-analysis data throughout one tool run."""

    _SCENE_THRESHOLD = 0.3

    def __init__(
        self,
        workspace: Workspace,
        interruption: InterruptibleProcess,
        logger: logging.Logger,
    ) -> None:
        self.workspace = workspace
        self.interruption = interruption
        self.logger = logger
        self._cache: dict[tuple[object, ...], VideoScanResult] = {}
        self._probe_cache: dict[tuple[object, ...], MediaProbeResult] = {}
        self._path_results: dict[str, VideoScanResult] = {}
        self._persistent_cache: PersistentMediaAnalysisCache | None = None

    def set_persistent_cache(self, cache: PersistentMediaAnalysisCache) -> None:
        self._persistent_cache = cache
        self.logger.debug(
            "Persistent media analysis cache configured: %s.",
            type(cache).__name__,
        )

    def probe(self, path: str) -> MediaProbeResult:
        real_path = os.path.realpath(path)
        self.logger.debug("Media probe requested for %s.", real_path)
        key = self._file_key(real_path)
        cached = self._probe_cache.get(key)
        if cached is not None:
            self.logger.debug("Media probe restored from this run's cache: %s", path)
            self._log_probe_result("Media probe cache hit", cached)
            return cached

        self.logger.debug("Running ffprobe for media metadata: %s.", real_path)
        try:
            data = video_utils.get_video_full_info(
                real_path,
                logger=self.logger,
                show_progress=True,
                progress_description="Reading media metadata",
            )
            normalized_data = video_utils.normalize_video_data(data)
            result = MediaProbeResult(
                path=real_path,
                data=data,
                error=None,
                normalized_data=normalized_data,
            )
        except RuntimeError as error:
            result = MediaProbeResult(real_path, {}, str(error))

        self._probe_cache[key] = result
        self._log_probe_result("Media probe completed", result)
        return result

    def fulfill(
        self,
        request: MediaAnalysisRequest,
        *,
        raise_on_error: bool = True,
    ) -> VideoScanResult:
        """Collect the requested feature set, reusing all available cached data."""
        path = request.path
        label = request.label
        features = request.features
        real_path = os.path.realpath(path)
        key = self._file_key(real_path)
        if features == MediaAnalysisFeature.NONE:
            raise ValueError("At least one media analysis feature must be requested")

        self.logger.debug(
            "Fulfilling media analysis request for %s (%s): features=[%s].",
            label,
            real_path,
            _format_features(features),
        )

        cached = self._cache.get(key)
        if cached is None:
            self.logger.debug("Media analysis run cache for %s is empty.", label)
        else:
            self._log_scan_result("Media analysis run cache contains data", label, cached)

        missing_features = features & ~(cached.features if cached is not None else MediaAnalysisFeature.NONE)
        persistent = self._restore_persistent(real_path, missing_features)
        if persistent is not None:
            self._log_scan_result("Persistent media analysis cache restored data", label, persistent)
            cached = self._merge_results(cached, persistent) if cached is not None else persistent
            self._cache[key] = cached
        elif self._persistent_cache is not None and missing_features:
            self.logger.debug(
                "Persistent media analysis cache for %s did not provide=[%s].",
                label,
                _format_features(missing_features),
            )

        if cached is not None and cached.supports(features):
            self.logger.debug("Media scan for %s restored from cache.", label)
            self._log_scan_result("Media analysis satisfied without FFmpeg", label, cached)
            self._path_results[real_path] = cached
            return self._handle_scan_error(cached, label, raise_on_error)

        missing_features = features
        if cached is not None:
            missing_features &= ~cached.features

        self.logger.debug(
            "Media analysis for %s requires a fresh scan: missing=[%s].",
            label,
            _format_features(missing_features),
        )
        scanned = self._scan(real_path, label, missing_features)
        self._log_scan_result("Fresh media analysis collected", label, scanned)
        self._store_persistent(scanned)
        result = self._merge_results(cached, scanned) if cached is not None else scanned
        self._cache[key] = result
        self._path_results[real_path] = result
        self._log_scan_result("Media analysis satisfied", label, result)
        return self._handle_scan_error(result, label, raise_on_error)

    @staticmethod
    def _handle_scan_error(
        result: VideoScanResult,
        label: str,
        raise_on_error: bool,
    ) -> VideoScanResult:
        if raise_on_error and result.decode_error is not None:
            raise RuntimeError(
                f"Media analysis failed for {label}: {result.decode_error}"
            )
        return result

    def _restore_persistent(
        self,
        path: str,
        requested_features: MediaAnalysisFeature,
    ) -> VideoScanResult | None:
        if self._persistent_cache is None:
            return None

        features = MediaAnalysisFeature.NONE
        scenes: tuple[int, ...] = ()
        frames: dict[int, dict] = {}
        if requested_features & MediaAnalysisFeature.SCENE_CHANGES:
            cached_scenes = self._persistent_cache.load_scene_changes(path)
            if cached_scenes is not None:
                features |= MediaAnalysisFeature.SCENE_CHANGES
                scenes = tuple(cached_scenes)
        if requested_features & MediaAnalysisFeature.FRAME_TIMESTAMPS:
            cached_frames = self._persistent_cache.load_frame_probes(path)
            if cached_frames is not None:
                features |= MediaAnalysisFeature.FRAME_TIMESTAMPS
                frames = cached_frames

        if features == MediaAnalysisFeature.NONE:
            return None
        return VideoScanResult(path, features, frames, scenes, (), None)

    def _store_persistent(self, result: VideoScanResult) -> None:
        if self._persistent_cache is None:
            return
        if result.decode_error is not None:
            self.logger.debug(
                "Persistent media analysis cache not updated for %s because the scan failed: %s.",
                result.path,
                result.decode_error,
            )
            return
        if result.supports(MediaAnalysisFeature.SCENE_CHANGES):
            self._persistent_cache.save_scene_changes(result.path, list(result.scene_changes))
            self.logger.debug(
                "Persistent media analysis cache stored %d scene changes for %s.",
                len(result.scene_changes),
                result.path,
            )
        if result.supports(MediaAnalysisFeature.FRAME_TIMESTAMPS):
            frame_probes = {
                timestamp: {**info, "path": None}
                for timestamp, info in result.frames.items()
            }
            self._persistent_cache.save_frame_probes(result.path, frame_probes)
            self.logger.debug(
                "Persistent media analysis cache stored %d frame timestamps for %s.",
                len(frame_probes),
                result.path,
            )

    def result_for(self, path: str) -> VideoScanResult | None:
        real_path = os.path.realpath(path)
        result = self._path_results.get(real_path)
        if result is None:
            self.logger.debug("Media analysis result lookup for %s: miss.", real_path)
        else:
            self._log_scan_result("Media analysis result lookup", real_path, result)
        return result

    def _log_probe_result(self, action: str, result: MediaProbeResult) -> None:
        streams = result.data.get("streams", [])
        streams = streams if isinstance(streams, list) else []
        has_video = any(
            isinstance(stream, dict) and stream.get("codec_type") == "video"
            for stream in streams
        )
        has_audio = any(
            isinstance(stream, dict) and stream.get("codec_type") == "audio"
            for stream in streams
        )
        self.logger.debug(
            "%s for %s: streams=%d, video=%s, audio=%s, error=%s.",
            action,
            result.path,
            len(streams),
            has_video,
            has_audio,
            result.error or "none",
        )

    def _log_scan_result(
        self,
        action: str,
        label: str,
        result: VideoScanResult,
    ) -> None:
        self.logger.debug(
            "%s for %s: features=[%s], frames=%d, scene_changes=%d, "
            "identity_samples=%d, decode_error=%s.",
            action,
            label,
            _format_features(result.features),
            len(result.frames),
            len(result.scene_changes),
            len(result.identity_samples),
            result.decode_error or "none",
        )

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

    @staticmethod
    def _progress_description(
        label: str,
        features: MediaAnalysisFeature,
    ) -> str:
        purposes = []
        if features & MediaAnalysisFeature.VALIDATE_STREAMS:
            purposes.append("decode validation")
        if features & MediaAnalysisFeature.MATCHING:
            purposes.append("timeline matching")
        if features & MediaAnalysisFeature.IDENTITY_SAMPLES:
            purposes.append("sample comparison")

        if len(purposes) > 1:
            return f"Analyzing input {label} ({', '.join(purposes)})"
        if purposes == ["decode validation"]:
            return f"Checking input {label} for decoding errors"
        if purposes == ["timeline matching"]:
            return f"Analyzing input {label} for timeline matching"
        if purposes == ["sample comparison"]:
            return f"Sampling input {label} for comparison"
        return f"Analyzing input {label}"

    def _scan(
        self,
        path: str,
        label: str,
        features: MediaAnalysisFeature,
    ) -> VideoScanResult:
        probe = self.probe(path)
        primary_video = probe.primary_video_track or {}
        duration_ms = primary_video.get("length")
        fps_value = primary_video.get("fps")
        try:
            fps = generic_utils.fps_str_to_float(str(fps_value))
        except (TypeError, ValueError):
            fps = None

        if features & MediaAnalysisFeature.IDENTITY_SAMPLES:
            valid_duration = isinstance(duration_ms, int) and duration_ms > 0
            valid_fps = fps is not None and fps > 0
            if not valid_duration or not valid_fps:
                raise ValueError(
                    f"Cannot collect identity samples for {label}: probe did not report "
                    "a positive video duration and frame rate"
                )
            assert isinstance(duration_ms, int)
            assert fps is not None
            target_timestamps = identity_timestamps(duration_ms, fps)
            sample_select = self._sample_select_expression(target_timestamps)
        else:
            target_timestamps = ()
            sample_select = ""

        if (
            features == MediaAnalysisFeature.VALIDATE_STREAMS
            and (probe.error is not None or not probe.has_decodable_stream)
        ):
            self.logger.debug(
                "FFmpeg media scan skipped for %s: validation-only request, "
                "decodable_stream=%s, probe_error=%s.",
                label,
                probe.has_decodable_stream,
                probe.error or "none",
            )
            return VideoScanResult(
                path=path,
                features=features,
                frames={},
                scene_changes=(),
                identity_samples=(),
                decode_error=probe.error,
            )

        has_primary_video = probe.has_video
        scan_dir = self.workspace.unique_dir("media_scan")
        frame_stats_path = os.path.join(scan_dir, "frames.txt")
        scene_stats_path = os.path.join(scan_dir, "scenes.txt")
        sample_stats_path = os.path.join(scan_dir, "identity.txt")
        sample_pattern = os.path.join(scan_dir, "identity_%08d.png")

        branches: list[str] = []
        if features & MediaAnalysisFeature.FRAME_TIMESTAMPS:
            branches.append("vframes")
        elif (
            features != MediaAnalysisFeature.VALIDATE_STREAMS
            and features & MediaAnalysisFeature.VALIDATE_STREAMS
            and has_primary_video
        ):
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
                f"{branch_source('vscenes')}select='gt(scene,{self._SCENE_THRESHOLD})'[scenes]"
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

        needs_null_output = bool(features & (
            MediaAnalysisFeature.FRAME_TIMESTAMPS
            | MediaAnalysisFeature.SCENE_CHANGES
            | MediaAnalysisFeature.VALIDATE_STREAMS
        ))

        scene_video_stream_index: int | None = None
        if needs_null_output:
            mapped_video_streams = 0
            if features & MediaAnalysisFeature.FRAME_TIMESTAMPS:
                args.extend(["-map", "[vframes]"])
                mapped_video_streams += 1
            elif "vvalidate" in branches:
                args.extend(["-map", "[vvalidate]"])
                mapped_video_streams += 1
            if features & MediaAnalysisFeature.SCENE_CHANGES:
                scene_video_stream_index = mapped_video_streams
                args.extend(["-map", "[scenes]"])

        if features & MediaAnalysisFeature.VALIDATE_STREAMS:
            if has_primary_video:
                args.extend(["-map", "0:V?"])
                if "vframes" in branches or "vvalidate" in branches:
                    args.extend(["-map", "-0:v:0?"])
            args.extend(["-map", "0:a?"])
        elif needs_null_output:
            args.append("-an")

        if needs_null_output:
            args.extend(["-sn", "-dn", "-fps_mode", "vfr"])
            if features & MediaAnalysisFeature.FRAME_TIMESTAMPS:
                args.extend([
                    # Keep decoded timestamps instead of quantizing them to nominal FPS.
                    "-enc_time_base:v:0", "filter",
                    "-stats_enc_pre:v:0", frame_stats_path,
                    "-stats_enc_pre_fmt:v:0", "{ni} {pts} {tb}",
                ])
            if scene_video_stream_index is not None:
                args.extend([
                    f"-enc_time_base:v:{scene_video_stream_index}", "filter",
                    f"-stats_enc_pre:v:{scene_video_stream_index}", scene_stats_path,
                    f"-stats_enc_pre_fmt:v:{scene_video_stream_index}", "{ni} {pts} {tb}",
                ])
            args.extend(["-f", "null", "-"])

        if features & MediaAnalysisFeature.IDENTITY_SAMPLES:
            args.extend([
                "-map", "[identity]",
                "-an", "-sn", "-dn",
                "-fps_mode", "vfr",
                "-enc_time_base:v:0", "filter",
                "-stats_enc_pre:v:0", sample_stats_path,
                "-stats_enc_pre_fmt:v:0", "{ni} {pts} {tb}",
                sample_pattern,
            ])

        self.logger.debug(
            "FFmpeg media scan pipeline for %s: features=[%s], video_branches=[%s], "
            "identity_targets=%d, validates_all_streams=%s.",
            label,
            _format_features(features),
            ", ".join(branches) if branches else "none",
            len(target_timestamps),
            bool(features & MediaAnalysisFeature.VALIDATE_STREAMS),
        )

        duration_s = duration_ms / 1000 if duration_ms is not None and duration_ms > 0 else None
        last_progress_s = 0.0
        timestamp_correction_ms = self._timestamp_correction_ms(probe)

        progress = tqdm(
            total=duration_s,
            desc=self._progress_description(label, features),
            unit="s",
            **generic_utils.get_tqdm_defaults(),
        )

        def on_line(line: str) -> None:
            nonlocal last_progress_s

            stripped = line.strip()
            progress_match = _PROGRESS_TIME_RE.match(stripped)
            if progress_match:
                current_s = int(progress_match.group(1)) / 1_000_000
                bounded_current_s = min(duration_s, current_s) if duration_s is not None else current_s
                delta = bounded_current_s - last_progress_s
                if delta > 0:
                    progress.update(delta)
                    last_progress_s += delta

        process, stderr_lines = video_utils._start_ffmpeg_streaming(
            args,
            self.interruption,
            on_line=on_line,
            logger=self.logger,
        )
        decode_error = self._decode_error(process.returncode, stderr_lines)
        if (
            decode_error is None
            and duration_s is not None
            and last_progress_s < duration_s
        ):
            progress.update(duration_s - last_progress_s)
        progress.close()

        scene_timestamps = (
            self._read_scene_timestamps(scene_stats_path, timestamp_correction_ms)
            if features & MediaAnalysisFeature.SCENE_CHANGES
            else []
        )
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

        return VideoScanResult(
            path=path,
            features=features,
            frames=frames,
            scene_changes=tuple(sorted(set(scene_timestamps))),
            identity_samples=samples,
            decode_error=decode_error,
        )

    @staticmethod
    def _timestamp_correction_ms(probe: MediaProbeResult) -> int:
        try:
            start_time = float(probe.data.get("format", {}).get("start_time") or 0.0)
        except (TypeError, ValueError):
            return 0
        return min(0, round(start_time * 1000))

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
        return video_utils._read_frame_entries(path, correction_ms)

    @classmethod
    def _read_frames(cls, path: str, correction_ms: int) -> dict[int, dict]:
        return {
            timestamp_ms: {"frame_id": frame_id, "path": None}
            for frame_id, timestamp_ms in cls._read_frame_entries(path, correction_ms)
        }

    @classmethod
    def _read_scene_timestamps(cls, path: str, correction_ms: int) -> list[int]:
        return [
            timestamp_ms
            for _frame_id, timestamp_ms in cls._read_frame_entries(path, correction_ms)
        ]

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
        unmatched_targets = list(targets)
        for frame_id, timestamp_ms, path in candidates:
            if not unmatched_targets:
                break
            target_ms = min(
                unmatched_targets,
                key=lambda target: abs(timestamp_ms - target),
            )
            unmatched_targets.remove(target_ms)
            if frames:
                frame_timestamp = min(frames, key=lambda candidate: abs(candidate - timestamp_ms))
                frame_id = int(frames[frame_timestamp]["frame_id"])
                frames[frame_timestamp]["path"] = path
                timestamp_ms = frame_timestamp
            samples.append(VideoSample(target_ms, timestamp_ms, frame_id, path))
        return tuple(sorted(samples, key=lambda sample: sample.target_ms))

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
