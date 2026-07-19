"""Shared, cached integrity checks for files selected by an analyzed plan."""

import enum
import json
import logging
import os
import re
import shlex

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import generic_utils, process_utils


_CACHE_VERSION = 1


class ValidationMode(enum.Enum):
    OFF = "off"
    FAST = "fast"
    FULL = "full"


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    repair_suggestion: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]
    checked_count: int
    cached_count: int

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def render(self, logger: logging.Logger) -> None:
        for issue in self.issues:
            logger.error("Input validation failed for %s: %s", issue.path, issue.message)
            if issue.repair_suggestion:
                logger.error("Suggested repair: %s", issue.repair_suggestion)


class InputValidator:
    """Validate every unique regular input file and cache results by file identity."""

    def __init__(self, mode: ValidationMode, logger: logging.Logger, cache_dir: str | None = None) -> None:
        self.mode = mode
        self.logger = logger
        self.cache_path = Path(cache_dir or generic_utils.get_twotone_config_dir()) / "input_validation.json"

    def validate(self, paths: Iterable[str]) -> ValidationReport:
        if self.mode == ValidationMode.OFF:
            return ValidationReport((), 0, 0)

        cache = self._load_cache()
        unique_paths = sorted({os.path.realpath(path) for path in paths})
        if unique_paths:
            self.logger.info(
                "Validating %d input file(s) with %s validation.",
                len(unique_paths),
                self.mode.value,
            )
        issues: list[ValidationIssue] = []
        checked_count = 0
        cached_count = 0
        changed = False
        for index, path in enumerate(unique_paths, start=1):
            if not os.path.isfile(path):
                issues.append(ValidationIssue(path, "Input file no longer exists or is not a regular file."))
                continue
            key = self._cache_key(path)
            cached = cache.get(key)
            if cached is not None:
                cached_count += 1
                self.logger.info("Input validation %d/%d: using cached result for %s.", index, len(unique_paths), path)
                if cached["issue"]:
                    issues.append(ValidationIssue(path, cached["issue"], cached.get("repair_suggestion")))
                continue

            checked_count += 1
            self.logger.info("Input validation %d/%d: checking %s.", index, len(unique_paths), path)
            issue = self._validate_file(path)
            cache[key] = {"issue": issue.message if issue else None, "repair_suggestion": issue.repair_suggestion if issue else None}
            changed = True
            if issue:
                issues.append(issue)

        if changed:
            self._save_cache(cache)
        report = ValidationReport(tuple(issues), checked_count, cached_count)
        if unique_paths:
            self.logger.info(
                "Input validation complete: %d checked, %d cached, %d issue(s).",
                report.checked_count,
                report.cached_count,
                len(report.issues),
            )
        return report

    def _validate_file(self, path: str) -> ValidationIssue | None:
        probe = process_utils.start_process(
            "ffprobe",
            ["-v", "error", "-show_error", "-show_format", "-show_streams", "-of", "json", path],
            show_progress=True,
            progress_description="Reading media metadata",
            logger=self.logger,
        )
        if probe.returncode != 0:
            return ValidationIssue(path, self._summarize_error(probe.stderr or probe.stdout))
        try:
            probe_data = json.loads(probe.stdout)
        except json.JSONDecodeError:
            return ValidationIssue(path, "ffprobe returned invalid metadata.")
        has_decodable_stream = any(
            stream.get("codec_type") in {"audio", "video"}
            for stream in probe_data.get("streams", [])
        )
        if self.mode == ValidationMode.FULL and has_decodable_stream:
            decode = process_utils.start_process(
                "ffmpeg",
                ["-v", "error", "-stats", "-xerror", "-i", path, "-map", "0:v?", "-map", "0:a?", "-f", "null", "-"],
                show_progress=True,
                progress_description="Decoding media for validation",
                logger=self.logger,
            )
            if decode.returncode != 0:
                return ValidationIssue(
                    path,
                    self._describe_decode_failure(probe_data, decode.stderr or decode.stdout),
                    self._repair_suggestion(path, probe_data),
                )
        return None

    @staticmethod
    def _repair_suggestion(path: str, probe_data: dict) -> str:
        audio_streams = [stream for stream in probe_data.get("streams", []) if stream.get("codec_type") == "audio"]
        codec = audio_streams[0].get("codec_name", "aac") if audio_streams else "aac"
        repaired_path = f"{os.path.splitext(path)[0]}.repaired.mkv"
        return (
            "ffmpeg -fflags +genpts+discardcorrupt -err_detect ignore_err "
            f"-i {shlex.quote(path)} "
            "-map 0:v:0 -c:v copy "
            f"-map 0:a:0 -af aresample=async=1:first_pts=0 -c:a {codec} "
            f"-avoid_negative_ts make_zero {shlex.quote(repaired_path)}"
        )

    @staticmethod
    def _summarize_error(output: str) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return "The media tool rejected this input."
        diagnostic_lines = [
            line
            for line in lines
            if re.search(
                r"invalid data|corrupt|incomplete|error submitting|error processing|decode error|non[- ]monoton",
                line,
                re.IGNORECASE,
            )
        ]
        selected = diagnostic_lines or lines[-3:]
        return " | ".join(dict.fromkeys(selected[:3]))

    def _describe_decode_failure(self, probe_data: dict, output: str) -> str:
        streams = [
            self._stream_description(stream)
            for stream in probe_data.get("streams", [])
            if stream.get("codec_type") in {"audio", "video"}
        ]
        stream_details = ", ".join(streams) if streams else "no decodable streams reported"
        return f"Full decode failed ({stream_details}): {self._summarize_error(output)}"

    @staticmethod
    def _stream_description(stream: dict) -> str:
        stream_index = stream.get("index", "?")
        stream_type = stream.get("codec_type", "stream")
        codec = stream.get("codec_name", "unknown codec")
        details = [f"{stream_type} #{stream_index}: {codec}"]
        if stream.get("sample_rate"):
            details.append(f"{stream['sample_rate']} Hz")
        if stream.get("channels"):
            details.append(f"{stream['channels']} channels")
        if stream.get("width") and stream.get("height"):
            details.append(f"{stream['width']}x{stream['height']}")
        return " ".join(details)

    def _cache_key(self, path: str) -> str:
        stat = os.stat(path)
        return json.dumps({
            "version": _CACHE_VERSION,
            "mode": self.mode.value,
            "path": path,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }, sort_keys=True)

    def _load_cache(self) -> dict[str, dict[str, str | None]]:
        try:
            with self.cache_path.open(encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self, cache: dict[str, dict[str, str | None]]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.cache_path.with_suffix(".tmp")
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(cache, file, sort_keys=True)
            os.replace(temporary_path, self.cache_path)
        except OSError as error:
            self.logger.warning("Could not save input-validation cache: %s", error)
