#!/usr/bin/env python3
"""Trace one ``test_audio_alignment_after_melt`` case.

The script is intentionally outside pytest's normal collection path.  Run it in
two environments and compare the generated reports to locate the first point at
which the environments diverge.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve()
TESTS_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TESTS_DIR))

import numpy as np  # noqa: E402

from common import add_to_test_dir, hashes, run_ffmpeg  # noqa: E402
from melt.test_audio_alignment import (  # noqa: E402
    AudioAlignmentTest,
    VARIANT_BY_NAME,
)
from twotone.tools.melt.melt import MeltAnalyzer, MeltPerformer, PairMatcher, StaticSource  # noqa: E402
from twotone.tools.melt.melt_cache import MeltCache  # noqa: E402
from twotone.tools.utils import files_utils, generic_utils, process_utils, video_utils  # noqa: E402


Json = dict[str, Any] | list[Any] | str | int | float | bool | None


def _jsonable(value: Any) -> Json:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _stable_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_capture(text: str, limit: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
        "length": len(text),
    }
    if len(text) <= limit:
        result["text"] = text
    else:
        half = max(1, limit // 2)
        result["head"] = text[:half]
        result["tail"] = text[-half:]
        result["truncated"] = True
    return result


class PathNormalizer:
    def __init__(self) -> None:
        self._paths: dict[str, str] = {}

    def register(self, alias: str, path: str | Path | None) -> None:
        if path is None:
            return
        resolved = os.path.abspath(os.fspath(path))
        self._paths[resolved] = alias

    def normalize_text(self, value: str) -> str:
        result = value
        for path, alias in sorted(self._paths.items(), key=lambda item: len(item[0]), reverse=True):
            result = result.replace(path, alias)
        result = re.sub(r"tmp_\d+_", "tmp_{pid}_", result)
        result = re.sub(r"AudioAlignmentDebug-[^/\\\s]+", "AudioAlignmentDebug-{case}", result)
        return result

    def normalize_value(self, value: Any) -> Json:
        value = _jsonable(value)
        if isinstance(value, str):
            return self.normalize_text(value)
        if isinstance(value, list):
            return [self.normalize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self.normalize_value(item) for key, item in value.items()}
        return value


class Recorder:
    def __init__(self, normalizer: PathNormalizer, process_output_limit: int) -> None:
        self.normalizer = normalizer
        self.process_output_limit = process_output_limit
        self.phase = "init"
        self.commands: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def use_phase(self, phase: str) -> Iterator[None]:
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def add_event(self, name: str, payload: Any) -> None:
        raw_payload = _jsonable(payload)
        self.events.append({
            "index": len(self.events),
            "phase": self.phase,
            "name": name,
            "payload": raw_payload,
            "normalized_payload": self.normalizer.normalize_value(raw_payload),
        })

    def record_process_call(
        self,
        process: str,
        args_before: list[str],
        args_after: list[str],
        show_progress: bool,
        cwd: str | None,
        started_at: float,
        result: process_utils.ProcessResult | None,
        exception: BaseException | None,
    ) -> None:
        command: dict[str, Any] = {
            "index": len(self.commands),
            "phase": self.phase,
            "process": process,
            "args_before": args_before,
            "args_after": args_after,
            "normalized_args_after": self.normalizer.normalize_value(args_after),
            "normalized_command": [process, *self.normalizer.normalize_value(args_after)],
            "show_progress": show_progress,
            "cwd": cwd,
            "normalized_cwd": self.normalizer.normalize_value(cwd),
            "elapsed_seconds": round(time.monotonic() - started_at, 6),
        }
        if result is not None:
            command["returncode"] = result.returncode
            command["stdout"] = _text_capture(result.stdout, self.process_output_limit)
            command["stderr"] = _text_capture(result.stderr, self.process_output_limit)
        if exception is not None:
            command["exception"] = repr(exception)
        self.commands.append(command)


@contextlib.contextmanager
def _instrument(recorder: Recorder) -> Iterator[None]:
    original_start_process = process_utils.start_process

    def start_process_wrapper(
        process: str,
        args: list[str],
        show_progress: bool = False,
        logger: logging.Logger | None = None,
        cwd: str | None = None,
    ) -> process_utils.ProcessResult:
        args_before = list(args)
        started_at = time.monotonic()
        result: process_utils.ProcessResult | None = None
        exception: BaseException | None = None
        try:
            result = original_start_process(process, args, show_progress, logger, cwd)
            return result
        except BaseException as err:
            exception = err
            raise
        finally:
            recorder.record_process_call(
                process,
                args_before,
                list(args),
                show_progress,
                cwd,
                started_at,
                result,
                exception,
            )

    original_detect_scenes_for = PairMatcher._detect_scenes_for

    def detect_scenes_for_wrapper(self: PairMatcher, video_path: str, label: str) -> list[int]:
        result = original_detect_scenes_for(self, video_path, label)
        recorder.add_event("pair_matcher.detect_scenes", {
            "label": label,
            "path": video_path,
            "scene_changes_ms": result,
        })
        return result

    original_probe_frames_for = PairMatcher._probe_frames_for

    def probe_frames_for_wrapper(self: PairMatcher, video_path: str, label: str) -> dict[int, dict]:
        result = original_probe_frames_for(self, video_path, label)
        recorder.add_event("pair_matcher.probe_frames", {
            "label": label,
            "path": video_path,
            "count": len(result),
            "timestamps_ms": list(result.keys()),
            "frame_ids": {timestamp: info.get("frame_id") for timestamp, info in result.items()},
        })
        return result

    original_extract_scene_frames = PairMatcher._extract_scene_frames

    def extract_scene_frames_wrapper(
        self: PairMatcher,
        lhs_scene_changes: list[int],
        rhs_scene_changes: list[int],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        result = original_extract_scene_frames(self, lhs_scene_changes, rhs_scene_changes)
        recorder.add_event("pair_matcher.extract_scene_frames", {
            "lhs_scene_ranges": result[0],
            "rhs_scene_ranges": result[1],
        })
        return result

    original_match_key_frames = PairMatcher._match_key_frames

    def match_key_frames_wrapper(self: PairMatcher, *args: Any, **kwargs: Any) -> list[tuple[int, int]]:
        result = original_match_key_frames(self, *args, **kwargs)
        recorder.add_event("pair_matcher.match_key_frames", {
            "pairs": result,
            "count": len(result),
        })
        return result

    original_constant_offset = PairMatcher.try_constant_offset_extrapolation

    def constant_offset_wrapper(self: PairMatcher, *args: Any, **kwargs: Any) -> list[tuple[int, int]] | None:
        result = original_constant_offset(self, *args, **kwargs)
        recorder.add_event("pair_matcher.constant_offset_extrapolation", {
            "result": result,
            "count": len(result) if result is not None else 0,
        })
        return result

    original_refine_boundaries = PairMatcher._extract_and_refine_boundaries

    def refine_boundaries_wrapper(self: PairMatcher, *args: Any, **kwargs: Any) -> list[tuple[int, int]]:
        result = original_refine_boundaries(self, *args, **kwargs)
        recorder.add_event("pair_matcher.refine_boundaries", {
            "pairs": result,
            "count": len(result),
        })
        return result

    original_snap_to_edges = PairMatcher.snap_to_edges

    def snap_to_edges_wrapper(self: PairMatcher, *args: Any, **kwargs: Any) -> list[tuple[int, int]]:
        result = original_snap_to_edges(self, *args, **kwargs)
        recorder.add_event("pair_matcher.snap_to_edges", {
            "pairs": result,
            "count": len(result),
        })
        return result

    original_create_segments_mapping = PairMatcher.create_segments_mapping

    def create_segments_mapping_wrapper(self: PairMatcher) -> Any:
        result = original_create_segments_mapping(self)
        recorder.add_event("pair_matcher.final_mapping", {
            "lhs_path": self.lhs_path,
            "rhs_path": self.rhs_path,
            "lhs_fps": self.lhs_fps,
            "rhs_fps": self.rhs_fps,
            "relation": result.relation.value,
            "mapping": result.mapping,
            "mapping_count": len(result.mapping),
            "lhs_all_frame_count": len(result.lhs_all_frames),
            "rhs_all_frame_count": len(result.rhs_all_frames),
        })
        return result

    original_analyze_group = MeltAnalyzer._analyze_group

    def analyze_group_wrapper(self: MeltAnalyzer, files: list[str], ids: dict[str, int], title: str) -> Any:
        result = original_analyze_group(self, files, ids, title)
        recorder.add_event("melt_analyzer.analyze_group", {
            "title": title,
            "files": files,
            "ids": ids,
            "plan_details": result[0],
            "issue": result[1],
        })
        return result

    original_prepare_stream_entries = MeltPerformer._prepare_stream_entries

    def prepare_stream_entries_wrapper(self: MeltPerformer, *args: Any, **kwargs: Any) -> Any:
        result = original_prepare_stream_entries(self, *args, **kwargs)
        recorder.add_event("melt_performer.prepare_stream_entries", {
            "entries": result.entries,
            "input_files": sorted(result.input_files),
        })
        return result

    original_patch_mismatched_audio = MeltPerformer._patch_mismatched_audio

    def patch_mismatched_audio_wrapper(self: MeltPerformer, *args: Any, **kwargs: Any) -> Any:
        recorder.add_event("melt_performer.patch_mismatched_audio.begin", {
            "video_path_base": args[0],
            "audio_stream": args[1],
            "base_duration": args[2],
            "file_ids": args[3],
        })
        result = original_patch_mismatched_audio(self, *args, **kwargs)
        recorder.add_event("melt_performer.patch_mismatched_audio.end", {
            "result": result,
        })
        return result

    original_patch_audio_constant_offset = MeltPerformer.patch_audio_constant_offset

    def patch_audio_constant_offset_wrapper(self: MeltPerformer, *args: Any, **kwargs: Any) -> int:
        segment_pairs = args[4]
        seg = self._segment_range(segment_pairs)
        source_dur = seg.rhs_end - seg.rhs_start
        target_dur = seg.lhs_end - seg.lhs_start
        video_ratio = target_dur / source_dur if source_dur else 1.0
        fps_ratio = source_dur / target_dur if target_dur else 1.0
        recorder.add_event("melt_performer.patch_audio_constant_offset.begin", {
            "wd": args[0],
            "base_video": args[1],
            "source_video": args[2],
            "output_path": args[3],
            "segment_pairs": segment_pairs,
            "segment_range": seg,
            "source_duration_ms": source_dur,
            "target_duration_ms": target_dur,
            "video_ratio": video_ratio,
            "fps_ratio": fps_ratio,
            "needs_scaling": self._needs_fps_scaling(fps_ratio),
            "use_silence": kwargs.get("use_silence", False),
        })
        result = original_patch_audio_constant_offset(self, *args, **kwargs)
        recorder.add_event("melt_performer.patch_audio_constant_offset.end", {
            "sync_offset_ms": result,
        })
        return result

    original_patch_audio_segment = MeltPerformer._patch_audio_segment

    def patch_audio_segment_wrapper(self: MeltPerformer, *args: Any, **kwargs: Any) -> Any:
        recorder.add_event("melt_performer.patch_audio_segment.begin", {
            "wd": args[0],
            "base_video": args[1],
            "source_video": args[2],
            "output_path": args[3],
            "segment_pairs": args[4],
            "segment_count": args[5],
            "use_silence": kwargs.get("use_silence", False),
        })
        result = original_patch_audio_segment(self, *args, **kwargs)
        recorder.add_event("melt_performer.patch_audio_segment.end", {})
        return result

    original_track_sync_offset = MeltPerformer._track_sync_offset_ms

    def track_sync_offset_wrapper(self: MeltPerformer, *args: Any, **kwargs: Any) -> int | None:
        result = original_track_sync_offset(self, *args, **kwargs)
        recorder.add_event("melt_performer.track_sync_offset", {
            "path": args[0],
            "stream_type": args[1],
            "stream_index": args[2],
            "desired_start_ms": args[3],
            "sync_offset_ms": result,
        })
        return result

    original_normalize_audio = MeltPerformer._normalize_audio_for_mkvmerge

    def normalize_audio_wrapper(self: MeltPerformer, *args: Any, **kwargs: Any) -> Any:
        result = original_normalize_audio(self, *args, **kwargs)
        recorder.add_event("melt_performer.normalize_audio_for_mkvmerge", {
            "source_path": args[0],
            "stream_index": args[1],
            "result": result,
        })
        return result

    original_trim_audio = MeltPerformer._trim_audio_to_timeline_end

    def trim_audio_wrapper(self: MeltPerformer, *args: Any, **kwargs: Any) -> Any:
        result = original_trim_audio(self, *args, **kwargs)
        recorder.add_event("melt_performer.trim_audio_to_timeline_end", {
            "source_path": args[0],
            "stream_index": args[1],
            "desired_end_ms": args[2],
            "result": result,
        })
        return result

    original_build_mkvmerge_args = MeltPerformer.build_mkvmerge_args

    def build_mkvmerge_args_wrapper(self: MeltPerformer, *args: Any, **kwargs: Any) -> list[str]:
        result = original_build_mkvmerge_args(self, *args, **kwargs)
        recorder.add_event("melt_performer.build_mkvmerge_args", {
            "output_path": args[0],
            "streams_list_sorted": args[1],
            "attachments": args[2],
            "preferred_audio": args[3],
            "required_input_files": sorted(args[4]),
            "args": result,
        })
        return result

    # Keep melt's scratch directories (matching/audio_extraction/…) on disk so the
    # intermediate FLAC/AAC files survive into the uploaded artifact and can be
    # measured with the CI toolchain.  Without this they are rmtree'd on scope exit.
    def scoped_dir_keep_exit(self: files_utils.ScopedDirectory, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(files_utils.ScopedDirectory, "__exit__", scoped_dir_keep_exit))
        stack.enter_context(patch.object(process_utils, "start_process", start_process_wrapper))
        stack.enter_context(patch.object(PairMatcher, "_detect_scenes_for", detect_scenes_for_wrapper))
        stack.enter_context(patch.object(PairMatcher, "_probe_frames_for", probe_frames_for_wrapper))
        stack.enter_context(patch.object(PairMatcher, "_extract_scene_frames", extract_scene_frames_wrapper))
        stack.enter_context(patch.object(PairMatcher, "_match_key_frames", match_key_frames_wrapper))
        stack.enter_context(patch.object(PairMatcher, "try_constant_offset_extrapolation", constant_offset_wrapper))
        stack.enter_context(patch.object(PairMatcher, "_extract_and_refine_boundaries", refine_boundaries_wrapper))
        stack.enter_context(patch.object(PairMatcher, "snap_to_edges", snap_to_edges_wrapper))
        stack.enter_context(patch.object(PairMatcher, "create_segments_mapping", create_segments_mapping_wrapper))
        stack.enter_context(patch.object(MeltAnalyzer, "_analyze_group", analyze_group_wrapper))
        stack.enter_context(patch.object(MeltPerformer, "_prepare_stream_entries", prepare_stream_entries_wrapper))
        stack.enter_context(patch.object(MeltPerformer, "_patch_mismatched_audio", patch_mismatched_audio_wrapper))
        stack.enter_context(patch.object(MeltPerformer, "patch_audio_constant_offset", patch_audio_constant_offset_wrapper))
        stack.enter_context(patch.object(MeltPerformer, "_patch_audio_segment", patch_audio_segment_wrapper))
        stack.enter_context(patch.object(MeltPerformer, "_track_sync_offset_ms", track_sync_offset_wrapper))
        stack.enter_context(patch.object(MeltPerformer, "_normalize_audio_for_mkvmerge", normalize_audio_wrapper))
        stack.enter_context(patch.object(MeltPerformer, "_trim_audio_to_timeline_end", trim_audio_wrapper))
        stack.enter_context(patch.object(MeltPerformer, "build_mkvmerge_args", build_mkvmerge_args_wrapper))
        yield


def _run_tool_version(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        output = result.stdout or result.stderr
        return {
            "command": command,
            "returncode": result.returncode,
            "first_line": output.splitlines()[0] if output.splitlines() else "",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as err:
        return {
            "command": command,
            "error": repr(err),
        }


def _environment_report() -> dict[str, Any]:
    tools = {
        "ffmpeg": ["ffmpeg", "-version"],
        "ffprobe": ["ffprobe", "-version"],
        "mkvmerge": ["mkvmerge", "--version"],
        "mkvextract": ["mkvextract", "--version"],
    }
    packages = {}
    for package in ("numpy", "opencv-python", "scikit-learn", "parameterized", "pytest"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "tools": {
            tool: {
                "path": shutil.which(tool),
                "version": _run_tool_version(command),
            }
            for tool, command in tools.items()
        },
        "packages": packages,
    }


def _media_report(path: str, label: str, logger: logging.Logger) -> dict[str, Any]:
    return {
        "label": label,
        "path": path,
        "size": os.path.getsize(path),
        "sha256": _sha256_file(path),
        "ffprobe": video_utils.get_video_full_info(path, logger=logger),
        "mkvmerge": video_utils.get_video_data_mkvmerge(path, enrich=True, logger=logger),
    }


def _stream_start_time(path: str, stream_type: str, stream_index: int) -> float:
    return AudioAlignmentTest._stream_start_time(path, stream_type, stream_index)


def _extract_audio_centers(output_file: str, audio_index: int, work_dir: str) -> dict[str, Any]:
    audio_start = _stream_start_time(output_file, "audio", audio_index)
    wav_path = os.path.join(work_dir, f"diagnostic_track_{audio_index}.wav")
    run_ffmpeg(
        [
            "-y",
            "-i", output_file,
            "-map", f"0:a:{audio_index}",
            "-ac", "1",
            "-ar", str(AudioAlignmentTest.SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            wav_path,
        ],
        expected_path=wav_path,
    )
    decoded_centers = AudioAlignmentTest._detect_beep_centers(wav_path)
    timeline_centers = [round(audio_start + value, 12) for value in decoded_centers]
    return {
        "audio_start_time": audio_start,
        "wav_path": wav_path,
        "decoded_centers": [round(value, 12) for value in decoded_centers],
        "timeline_centers": timeline_centers,
    }


def _audio_alignment_report(output_file: str, work_dir: str) -> dict[str, Any]:
    first = _extract_audio_centers(output_file, 0, work_dir)
    second = _extract_audio_centers(output_file, 1, work_dir)
    offsets = [
        round(abs(lhs - rhs), 12)
        for lhs, rhs in zip(first["timeline_centers"], second["timeline_centers"])
    ]
    return {
        "track_0": first,
        "track_1": second,
        "offsets": offsets,
        "max_offset": max(offsets, default=0.0),
        "tolerance": AudioAlignmentTest.AUDIO_ALIGNMENT_TOLERANCE_SECONDS,
        "passes_tolerance": max(offsets, default=0.0) <= AudioAlignmentTest.AUDIO_ALIGNMENT_TOLERANCE_SECONDS,
    }


def _audio_stream_meta(path: str, audio_index: int, logger: logging.Logger | None = None) -> dict[str, Any]:
    """Container-level priming metadata for one audio stream, as the CI toolchain reports it."""
    info = video_utils.get_video_full_info(path, logger=logger)
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if audio_index >= len(audio_streams):
        return {"present": False}
    stream = audio_streams[audio_index]
    return {
        "present": True,
        "codec_name": stream.get("codec_name"),
        "start_time": stream.get("start_time"),
        "start_pts": stream.get("start_pts"),
        "initial_padding": stream.get("initial_padding"),
        "nb_samples": stream.get("nb_samples") or stream.get("nb_read_samples"),
        "duration": stream.get("duration"),
        "time_base": stream.get("time_base"),
    }


def _measure_audio_landing(
    path: str,
    audio_index: int,
    work_dir: str,
    label: str,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Decode one audio stream with the active (CI) ffmpeg and report where its content lands.

    Captures the first detected beep centre and the leading-silence offset, alongside the
    container priming metadata, so a single pair run shows *at which pipeline stage* a
    sample-level shift (e.g. AAC encoder delay) enters — measured entirely by CI tools.
    """
    result: dict[str, Any] = {"label": label, "path": path, "audio_index": audio_index}
    result["stream"] = _audio_stream_meta(path, audio_index, logger=logger)
    if not result["stream"].get("present"):
        return result

    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{label}_a{audio_index}")
    wav_path = os.path.join(work_dir, f"verify_{safe_label}.wav")
    try:
        run_ffmpeg(
            [
                "-y", "-i", path,
                "-map", f"0:a:{audio_index}",
                "-ac", "1", "-ar", str(AudioAlignmentTest.SAMPLE_RATE),
                "-c:a", "pcm_s16le", wav_path,
            ],
            expected_path=wav_path,
        )
    except Exception as err:  # noqa: BLE001 - diagnostics must never abort the run
        result["decode_error"] = repr(err)
        return result

    _sample_rate, samples = AudioAlignmentTest._read_mono_wav(wav_path)
    centers = AudioAlignmentTest._detect_beep_centers(wav_path)
    loud = np.flatnonzero(np.abs(samples) > 0.02)
    result["first_beep_center"] = round(centers[0], 12) if centers else None
    result["beep_count"] = len(centers)
    result["leading_silence_ms"] = (
        round(float(loud[0]) / AudioAlignmentTest.SAMPLE_RATE * 1000, 3) if loud.size else None
    )
    return result


def _stage_verification(
    output_file: str,
    rhs_work_path: str,
    work_dir: str,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Trace beep landing through every pipeline stage, measured with the CI toolchain.

    Goal: localise where the patched track gains its offset.  Compare ``rhs_input`` →
    intermediate ``.flac``/``.mka`` (pre-mkvmerge) → ``output_track_1`` (post-mkvmerge).
    If the ``.mka`` already lands late, the encoder step leaks; if it is clean but the
    output is late, the mux/extract step does.
    """
    def _curated(patterns: list[str], limit: int) -> list[str]:
        found: list[str] = []
        for pattern in patterns:
            found.extend(sorted(glob.glob(os.path.join(work_dir, pattern), recursive=True)))
        unique = list(dict.fromkeys(found))
        return unique[:limit]

    mka_files = _curated(["**/*.mka"], limit=8)
    flac_files = _curated(
        ["**/source_trimmed*.flac", "**/source_scaled*.flac", "**/v2_audio*.flac", "**/scaled_*.flac"],
        limit=8,
    )

    report: dict[str, Any] = {
        "rhs_input": _measure_audio_landing(rhs_work_path, 0, work_dir, "rhs_input", logger=logger),
        "intermediate_flac": [
            _measure_audio_landing(path, 0, work_dir, f"flac::{os.path.basename(path)}", logger=logger)
            for path in flac_files
        ],
        "intermediate_mka": [
            _measure_audio_landing(path, 0, work_dir, f"mka::{os.path.basename(path)}", logger=logger)
            for path in mka_files
        ],
        "output_track_0": _measure_audio_landing(output_file, 0, work_dir, "output_track_0", logger=logger),
        "output_track_1": _measure_audio_landing(output_file, 1, work_dir, "output_track_1", logger=logger),
    }
    return report


def _extraction_candidates(
    source_path: str,
    work_dir: str,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Probe several source-audio extraction recipes and report where each lands.

    The +21 ms regression originates when the source AAC is decoded to a raw FLAC
    intermediate: some ffmpeg builds retain the AAC encoder-delay priming as real
    samples, and FLAC has no field to compensate it.  This runs candidate recipes on
    the source with the active (CI) toolchain so we can pick one that strips the
    priming consistently (leading silence identical to local) instead of guessing.
    """
    meta = _audio_stream_meta(source_path, 0, logger=logger)
    info = video_utils.get_video_full_info(source_path, logger=logger)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    try:
        sample_rate = int(audio.get("sample_rate") or 48000)
    except (TypeError, ValueError):
        sample_rate = 48000
    try:
        pad = int(meta.get("initial_padding") or 0)
    except (TypeError, ValueError):
        pad = 0
    pad_s = pad / sample_rate if sample_rate else 0.0

    candidates: list[dict[str, Any]] = [
        {"name": "baseline_atrim_asetpts", "pre_input": [], "filt": "atrim=start=0,asetpts=PTS-STARTPTS"},
        {"name": "atrim_no_asetpts", "pre_input": [], "filt": "atrim=start=0"},
        {"name": "plain_decode", "pre_input": [], "filt": None},
        {"name": "aresample_async", "pre_input": [], "filt": "aresample=async=1,asetpts=PTS-STARTPTS"},
        {"name": "input_seek_0", "pre_input": ["-ss", "0"], "filt": None},
    ]
    if pad_s > 0:
        candidates.append({
            "name": "explicit_pad_trim",
            "pre_input": [],
            "filt": f"atrim=start={pad_s:.6f},asetpts=PTS-STARTPTS",
        })
        # Deterministic across ffmpeg builds: disable automatic priming skip
        # (always keep it), then trim exactly initial_padding ourselves.
        candidates.append({
            "name": "skip_manual_pad_trim",
            "pre_input": ["-flags2", "+skip_manual"],
            "filt": f"atrim=start={pad_s:.6f},asetpts=PTS-STARTPTS",
        })

    cand_dir = os.path.join(work_dir, "extraction_candidates")
    os.makedirs(cand_dir, exist_ok=True)

    results: list[dict[str, Any]] = []
    for cand in candidates:
        out_path = os.path.join(cand_dir, f"{cand['name']}.flac")
        args = ["-y", *cand["pre_input"], "-i", source_path, "-map", "0:a:0"]
        if cand["filt"]:
            args += ["-filter:a", cand["filt"]]
        args += ["-sample_fmt", "s32", "-c:a", "flac", out_path]
        entry: dict[str, Any] = {"name": cand["name"], "ffmpeg_args": args}
        try:
            run_ffmpeg(args, expected_path=out_path)
            entry["measurement"] = _measure_audio_landing(
                out_path, 0, cand_dir, f"cand::{cand['name']}", logger=logger,
            )
        except Exception as err:  # noqa: BLE001 - diagnostics must never abort the run
            entry["error"] = repr(err)
        results.append(entry)

    return {
        "source_stream": meta,
        "source_sample_rate": sample_rate,
        "encoder_delay_ms": round(pad_s * 1000, 3),
        "candidates": results,
    }


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite or choose another path.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("AudioAlignmentDebug")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(output_dir / "twotone-debug.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger


def _select_cache(mode: str, output_dir: Path, logger: logging.Logger) -> MeltCache | None:
    if mode == "none":
        return None
    if mode == "test":
        return AudioAlignmentTest.melt_cache
    cache_dir = output_dir / "melt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return MeltCache(str(cache_dir), logger.getChild("MeltCache"))


def _run_melt_pair(
    lhs_path: str,
    rhs_path: str,
    work_dir: Path,
    melt_output_dir: Path,
    cache: MeltCache | None,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], str]:
    file1 = add_to_test_dir(str(work_dir), lhs_path)
    file2 = add_to_test_dir(str(work_dir), rhs_path)

    interruption = generic_utils.InterruptibleProcess()
    duplicates = StaticSource(interruption)
    duplicates.add_entry("Video", file1)
    duplicates.add_entry("Video", file2)
    duplicates.add_metadata(file1, "audio_lang", "eng")
    duplicates.add_metadata(file2, "audio_lang", "pol")

    analyzer = MeltAnalyzer(logger.getChild("Analyzer"), duplicates, str(work_dir), True, 100)
    duplicates_raw = duplicates.collect_duplicates()
    plan = analyzer.analyze_duplicates({
        title: list(files)
        for title, files in duplicates_raw.items()
    })

    performer = MeltPerformer(
        logger.getChild("Performer"),
        interruption,
        str(work_dir),
        str(melt_output_dir),
        100,
        cache=cache,
    )
    performer.process_duplicates(plan)
    return plan, file2


def _finalize_report(report: dict[str, Any], recorder: Recorder) -> None:
    for command in recorder.commands:
        command["normalized_args_after"] = recorder.normalizer.normalize_value(command["args_after"])
        command["normalized_command"] = [
            command["process"],
            *command["normalized_args_after"],
        ]
        command["normalized_cwd"] = recorder.normalizer.normalize_value(command["cwd"])
    report["events"] = recorder.events
    report["commands"] = recorder.commands
    report["hashes"] = {
        "events": _stable_hash([event["normalized_payload"] for event in recorder.events]),
        "melt_commands": _stable_hash(_command_signatures(report, phase_prefix="melt_")),
        "all_commands": _stable_hash(_command_signatures(report)),
    }


def run(args: argparse.Namespace) -> int:
    lhs_spec = VARIANT_BY_NAME[args.lhs]
    rhs_spec = VARIANT_BY_NAME[args.rhs]
    case_name = f"{lhs_spec.name}__{rhs_spec.name}"
    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir, args.overwrite)

    normalizer = PathNormalizer()
    recorder = Recorder(normalizer, args.process_output_limit)
    normalizer.register("{repo}", REPO_ROOT)
    normalizer.register("{output_dir}", output_dir)

    logger = _setup_logger(output_dir)
    report: dict[str, Any] = {
        "schema_version": 1,
        "case": {
            "name": case_name,
            "lhs": dataclasses.asdict(lhs_spec),
            "rhs": dataclasses.asdict(rhs_spec),
            "cache_mode": args.cache_mode,
        },
        "environment": _environment_report(),
    }

    try:
        with _instrument(recorder):
            with recorder.use_phase("setup_inputs"):
                AudioAlignmentTest.setUpClass()

            lhs_path = AudioAlignmentTest.variant_paths[lhs_spec.name]
            rhs_path = AudioAlignmentTest.variant_paths[rhs_spec.name]
            expected_base = AudioAlignmentTest._pick_expected_base(lhs_spec, rhs_spec)
            report["expected"] = {
                "base_variant": dataclasses.asdict(expected_base),
                "duration_ms": AudioAlignmentTest._expected_duration_ms(expected_base),
                "beep_centers": [
                    round(value, 12)
                    for value in AudioAlignmentTest._expected_beep_centers(expected_base)
                ],
            }

            work_dir = output_dir / "work"
            melt_output_dir = output_dir / "melt_output"
            work_dir.mkdir(parents=True, exist_ok=True)
            melt_output_dir.mkdir(parents=True, exist_ok=True)
            normalizer.register("{work_dir}", work_dir)
            normalizer.register("{melt_output_dir}", melt_output_dir)
            normalizer.register("{canonical_video}", AudioAlignmentTest.canonical_video)
            normalizer.register("{lhs_variant_cache}", lhs_path)
            normalizer.register("{rhs_variant_cache}", rhs_path)

            cache = _select_cache(args.cache_mode, output_dir, logger)
            if cache is not None and args.cache_mode == "test":
                normalizer.register("{test_melt_cache}", cache.cache_dir)
            elif cache is not None:
                normalizer.register("{fresh_melt_cache}", output_dir / "melt_cache")

            with recorder.use_phase("input_metadata"):
                report["inputs"] = {
                    "canonical": _media_report(AudioAlignmentTest.canonical_video, "canonical", logger),
                    "lhs": _media_report(lhs_path, lhs_spec.name, logger),
                    "rhs": _media_report(rhs_path, rhs_spec.name, logger),
                }

            with recorder.use_phase("melt_run"):
                plan, rhs_work_path = _run_melt_pair(
                    lhs_path,
                    rhs_path,
                    work_dir,
                    melt_output_dir,
                    cache,
                    logger,
                )
                normalizer.register("{rhs_work_file}", rhs_work_path)
                report["analysis_plan"] = _jsonable(plan)

            with recorder.use_phase("output_analysis"):
                output_hashes = hashes(str(melt_output_dir))
                report["output_files"] = [
                    {
                        "path": path,
                        "sha256": _sha256_file(path),
                        "md5": md5,
                        "size": os.path.getsize(path),
                    }
                    for path, md5 in sorted(output_hashes.items())
                ]
                if len(output_hashes) == 1:
                    output_file = next(iter(output_hashes.keys()))
                    normalizer.register("{output_file}", output_file)
                    report["output"] = {
                        "media": _media_report(output_file, "output", logger),
                        "playback_end_ms": AudioAlignmentTest._playback_end_ms(output_file),
                        "audio_alignment": _audio_alignment_report(output_file, str(work_dir)),
                        "stage_verification": _stage_verification(
                            output_file, rhs_work_path, str(work_dir), logger,
                        ),
                        "extraction_candidates": _extraction_candidates(
                            rhs_work_path, str(work_dir), logger,
                        ),
                    }
                else:
                    report["output_error"] = f"Expected exactly one output file, got {len(output_hashes)}"
    except Exception as err:
        report["error"] = repr(err)
        report["traceback"] = traceback.format_exc()
        logger.exception("Diagnostic run failed")
    finally:
        _finalize_report(report, recorder)
        report_path = output_dir / "debug_report.json"
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(_jsonable(report), file, indent=2, sort_keys=True)

    print(f"Debug report: {output_dir / 'debug_report.json'}")
    print(f"Detailed log:  {output_dir / 'twotone-debug.log'}")
    if "error" in report:
        return 1
    return 0


def _get_path(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            return None
    return current


def _command_signatures(report: dict[str, Any], phase_prefix: str | None = None) -> list[dict[str, Any]]:
    signatures = []
    for command in report.get("commands", []):
        phase = command.get("phase")
        if phase_prefix is not None and not str(phase).startswith(phase_prefix):
            continue
        signatures.append({
            "phase": phase,
            "process": command.get("process"),
            "args": command.get("normalized_args_after"),
            "returncode": command.get("returncode"),
        })
    return signatures


def _event_payloads(report: dict[str, Any], name: str) -> list[Any]:
    return [
        event.get("normalized_payload")
        for event in report.get("events", [])
        if event.get("name") == name
    ]


def _compare_field(left: dict[str, Any], right: dict[str, Any], path: str) -> dict[str, Any]:
    left_value = _get_path(left, path)
    right_value = _get_path(right, path)
    same = left_value == right_value
    item: dict[str, Any] = {
        "name": path,
        "same": same,
        "left_hash": _stable_hash(left_value),
        "right_hash": _stable_hash(right_value),
    }
    if not same:
        item["left"] = left_value
        item["right"] = right_value
    return item


def _first_diff(left: list[Any], right: list[Any]) -> dict[str, Any] | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return {
                "index": index,
                "left": left[index],
                "right": right[index],
            }
    if len(left) != len(right):
        return {
            "index": limit,
            "left": left[limit] if limit < len(left) else None,
            "right": right[limit] if limit < len(right) else None,
        }
    return None


def compare(args: argparse.Namespace) -> int:
    with open(args.left_report, encoding="utf-8") as file:
        left = json.load(file)
    with open(args.right_report, encoding="utf-8") as file:
        right = json.load(file)

    checks = [
        _compare_field(left, right, "case"),
        _compare_field(left, right, "inputs.canonical.sha256"),
        _compare_field(left, right, "inputs.lhs.sha256"),
        _compare_field(left, right, "inputs.rhs.sha256"),
        _compare_field(left, right, "analysis_plan"),
        _compare_field(left, right, "output.audio_alignment.track_0.timeline_centers"),
        _compare_field(left, right, "output.audio_alignment.track_1.timeline_centers"),
        _compare_field(left, right, "output.audio_alignment.offsets"),
    ]

    event_names = [
        "pair_matcher.detect_scenes",
        "pair_matcher.probe_frames",
        "pair_matcher.match_key_frames",
        "pair_matcher.constant_offset_extrapolation",
        "pair_matcher.final_mapping",
        "melt_performer.patch_audio_constant_offset.begin",
        "melt_performer.patch_audio_constant_offset.end",
        "melt_performer.build_mkvmerge_args",
    ]
    for name in event_names:
        left_events = _event_payloads(left, name)
        right_events = _event_payloads(right, name)
        item = {
            "name": f"events.{name}",
            "same": left_events == right_events,
            "left_hash": _stable_hash(left_events),
            "right_hash": _stable_hash(right_events),
        }
        diff = _first_diff(left_events, right_events)
        if diff is not None:
            item["first_difference"] = diff
        checks.append(item)

    for phase_name, phase_prefix in (
        ("commands.setup_inputs", "setup_inputs"),
        ("commands.input_metadata", "input_metadata"),
        ("commands.melt_run", "melt_run"),
        ("commands.output_analysis", "output_analysis"),
    ):
        left_commands = _command_signatures(left, phase_prefix=phase_prefix)
        right_commands = _command_signatures(right, phase_prefix=phase_prefix)
        item = {
            "name": phase_name,
            "same": left_commands == right_commands,
            "left_hash": _stable_hash(left_commands),
            "right_hash": _stable_hash(right_commands),
        }
        diff = _first_diff(left_commands, right_commands)
        if diff is not None:
            item["first_difference"] = diff
        checks.append(item)

    result = {
        "left_report": args.left_report,
        "right_report": args.right_report,
        "same": all(item["same"] for item in checks),
        "checks": checks,
    }

    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(output + "\n")
    print(output)
    return 0 if result["same"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one diagnostic case")
    run_parser.add_argument("--lhs", default="v00_asR_vsR_aeR_veR", choices=sorted(VARIANT_BY_NAME))
    run_parser.add_argument("--rhs", default="v01_asR_vsR_aeR_veT", choices=sorted(VARIANT_BY_NAME))
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--cache-mode", choices=("fresh", "test", "none"), default="fresh")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--process-output-limit", type=int, default=20000)
    run_parser.set_defaults(func=run)

    compare_parser = subparsers.add_parser("compare", help="compare two debug_report.json files")
    compare_parser.add_argument("left_report")
    compare_parser.add_argument("right_report")
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(func=compare)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
