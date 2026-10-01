"""Shared, cached integrity checks for files selected by an analyzed plan."""

import enum
import json
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from . import generic_utils, media_analysis

_CACHE_VERSION = 2


class ValidationMode(enum.Enum):
    OFF = "off"
    FAST = "fast"
    FULL = "full"


@dataclass(frozen=True)
class InputValidationPolicy:
    """Describe the runtime behavior and dependencies of an input-validation mode."""

    mode: ValidationMode

    @property
    def enabled(self) -> bool:
        return self.mode != ValidationMode.OFF

    @property
    def validate_all_streams(self) -> bool:
        return self.mode == ValidationMode.FULL

    def required_tools(self) -> set[str]:
        if self.enabled:
            tools = {"ffprobe"}
            if self.validate_all_streams:
                tools.add("ffmpeg")
        else:
            tools = set()
        return tools


@dataclass(frozen=True)
class InputValidationTarget:
    path: str
    reference: str


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    reference: str | None = None


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
            logger.error(
                "Input validation failed for %s: %s",
                issue.reference or issue.path,
                issue.message,
            )


class InputValidator:
    """Validate every unique regular input file and cache results by file identity."""

    def __init__(
        self,
        policy: InputValidationPolicy,
        logger: logging.Logger,
        cache_dir: str | None = None,
        *,
        media_analysis_session: media_analysis.MediaAnalysisSession,
    ) -> None:
        self.policy = policy
        self.logger = logger
        self.cache_path = Path(cache_dir or generic_utils.get_twotone_config_dir()) / "input_validation.json"
        self.media_analysis = media_analysis_session

    def validate(
        self,
        inputs: Iterable[str | InputValidationTarget],
    ) -> ValidationReport:
        if not self.policy.enabled:
            return ValidationReport((), 0, 0)

        cache = self._load_cache()
        targets_by_path: dict[str, InputValidationTarget] = {}
        for input_value in inputs:
            target = (
                input_value
                if isinstance(input_value, InputValidationTarget)
                else InputValidationTarget(input_value, input_value)
            )
            real_path = os.path.realpath(target.path)
            targets_by_path.setdefault(
                real_path,
                InputValidationTarget(real_path, target.reference),
            )
        targets = tuple(targets_by_path.values())

        results: dict[str, ValidationIssue | None] = {}
        pending_checks: list[tuple[InputValidationTarget, str]] = []
        cached_count = 0
        for target in targets:
            path = target.path
            if not os.path.isfile(path):
                results[path] = ValidationIssue(
                    path,
                    "Input file no longer exists or is not a regular file.",
                    target.reference,
                )
                continue

            key = self._cache_key(path)
            cached = cache.get(key)
            has_fresh_decode = self._has_analysis_decode(path)
            if cached is not None and not has_fresh_decode:
                cached_count += 1
                cached_issue = cached["issue"]
                results[path] = (
                    ValidationIssue(path, cached_issue, target.reference)
                    if cached_issue is not None
                    else None
                )
            else:
                pending_checks.append((target, key))

        if cached_count:
            file_label = "input file" if cached_count == 1 else "input files"
            self.logger.info(
                "Loaded validation results from an earlier run for %d unchanged %s.",
                cached_count,
                file_label,
            )

        checked_count = len(pending_checks)
        if checked_count:
            file_label = "input file" if checked_count == 1 else "input files"
            if self.policy.validate_all_streams:
                self.logger.info(
                    "Checking %d %s for media errors.",
                    checked_count,
                    file_label,
                )
            else:
                self.logger.info(
                    "Checking metadata for %d %s.",
                    checked_count,
                    file_label,
                )

        for target, key in pending_checks:
            path = target.path
            self.logger.info("Checking input %s.", target.reference)
            issue = self._validate_file(target)
            results[path] = issue
            cache[key] = {"issue": issue.message if issue else None}

        if pending_checks:
            self._save_cache(cache)

        issues = tuple(
            issue
            for target in targets
            if (issue := results[target.path]) is not None
        )
        report = ValidationReport(issues, checked_count, cached_count)
        if targets:
            for target in targets:
                if results[target.path] is not None:
                    self.logger.warning("Input %s is invalid.", target.reference)
                else:
                    self.logger.info("Input %s is valid.", target.reference)

            self.logger.debug(
                "Input validation statistics: checked=%d, cached=%d, issues=%d.",
                report.checked_count,
                report.cached_count,
                len(report.issues),
            )
            if report.is_valid:
                self.logger.info("All input files are valid.")

        return report

    def _has_analysis_decode(self, path: str) -> bool:
        result = self.media_analysis.result_for(path)
        return result is not None and result.validated_all_streams

    def _validate_file(self, target: InputValidationTarget) -> ValidationIssue | None:
        path = target.path
        probe = self.media_analysis.probe(path)
        if probe.error is not None:
            return ValidationIssue(
                path,
                self._summarize_error(probe.error),
                target.reference,
            )

        if self.policy.validate_all_streams and probe.has_decodable_stream:
            request = media_analysis.MediaAnalysisRequest(
                path=path,
                label=target.reference,
                features=media_analysis.MediaAnalysisFeature.VALIDATE_STREAMS,
            )
            scan = self.media_analysis.fulfill(request, raise_on_error=False)
            if scan.decode_error is not None:
                return ValidationIssue(
                    path,
                    self._describe_decode_failure(
                        probe.data,
                        scan.decode_error,
                    ),
                    target.reference,
                )
        return None

    @staticmethod
    def _summarize_error(output: str) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return "The media tool rejected this input."
        diagnostic_lines = [
            line
            for line in lines
            if re.search(
                r"invalid data|corrupt|incomplete|ended prematurely|error submitting|error processing|decode error|non[- ]monoton",
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
            "mode": self.policy.mode.value,
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
                json.dump(cache, file, sort_keys=True, indent=4)
            os.replace(temporary_path, self.cache_path)
        except OSError as error:
            self.logger.warning("Could not save input-validation cache: %s", error)
