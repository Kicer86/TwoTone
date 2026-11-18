import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..utils import input_validation, language_utils, media_analysis
from .melt_common import (
    AttachmentRef,
    AudioStreamRef,
    MeltInputFiles,
    StreamType,
    SubtitleStreamRef,
    VideoStreamRef,
    stream_short_details,
)


@dataclass(frozen=True)
class MeltGroupPlan:
    input_files: MeltInputFiles = field(default_factory=lambda: MeltInputFiles(()))
    output_name: str = "output"
    video_streams: tuple[VideoStreamRef, ...] | None = None
    audio_streams: tuple[AudioStreamRef, ...] | None = None
    subtitle_streams: tuple[SubtitleStreamRef, ...] | None = None
    attachments: tuple[AttachmentRef, ...] = ()
    chapter_source: str | None = None
    audio_prod_lang: str | None = None
    files_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    alignment_paths: tuple[str, ...] = ()
    media_analysis_requests: tuple[media_analysis.MediaAnalysisRequest, ...] = ()
    issue: str | None = None


@dataclass(frozen=True)
class MeltTitlePlan:
    title: str = "<unknown>"
    groups: tuple[MeltGroupPlan, ...] = ()
    skipped_groups: tuple[MeltGroupPlan, ...] = ()


@dataclass
class MeltPlan:
    items: tuple[MeltTitlePlan, ...]
    output_dir: str

    def is_empty(self) -> bool:
        return not any(item.groups for item in self.items)

    def input_files(self) -> set[str]:
        return {
            path
            for item in self.items
            for group in item.groups
            for path in group.input_files.paths
        }

    def input_validation_targets(
        self,
    ) -> tuple[input_validation.InputValidationTarget, ...]:
        planned_groups = [
            (item, group_index, group)
            for item in self.items
            for group_index, group in enumerate(item.groups, start=1)
        ]
        include_context = len(planned_groups) > 1
        targets: list[input_validation.InputValidationTarget] = []
        for item, group_index, group in planned_groups:
            input_files = group.input_files
            multiple_candidates = len(item.groups) > 1
            for path in input_files.paths:
                reference = f"#{input_files.id_for(path)}"
                if include_context:
                    context = item.title
                    if multiple_candidates:
                        context += f", candidate #{group_index}"
                    reference += f" ({context})"
                if group.video_streams is None and group.audio_streams is None:
                    stream_indices = None
                else:
                    selected_streams = (
                        (group.video_streams or ())
                        + (group.audio_streams or ())
                    )
                    stream_indices = tuple(sorted({
                        stream.ffprobe_stream_index
                        for stream in selected_streams
                        if stream.path == path
                    }))
                targets.append(
                    input_validation.InputValidationTarget(
                        path,
                        reference,
                        stream_indices,
                    ),
                )
        return tuple(targets)

    def media_analysis_requests(self) -> tuple[media_analysis.MediaAnalysisRequest, ...]:
        return tuple(
            request
            for item in self.items
            for group in item.groups
            for request in group.media_analysis_requests
        )

    def render(self, logger: logging.Logger) -> None:
        if not self.items:
            logger.info("No titles to melt.")
            return

        planned_items = [item for item in self.items if item.groups]
        if not planned_items:
            logger.info("No suitable candidates found for melting.")
            return

        total_outputs = sum(len(item.groups) for item in planned_items)
        total_files = sum(
            len(group.input_files.paths)
            for item in planned_items
            for group in item.groups
        )

        logger.info(
            "Planned melt: %d output(s) from %d input file(s).",
            total_outputs,
            total_files,
        )
        logger.info("Output directory: %s", self.output_dir)

        self._render_planned(logger, planned_items)

    @staticmethod
    def _format_track_line(stype: StreamType, stream: dict[str, Any], used: bool, override_lang: str | None = None) -> str:
        tid = stream.get("tid", "?")
        name = stream.get("name")
        lang = override_lang or stream.get("language")
        lang_name = language_utils.language_name(lang) if lang else "unknown"
        details = stream_short_details(stype, stream)
        parts = []
        if stype != "video" or lang_name != "unknown":
            parts.append(lang_name)
        if name:
            parts.append(str(name))
        if details:
            parts.append(details)
        suffix = ", ".join(parts) if parts else "-"
        flag = "used" if used else "skip"
        return f"#{tid} ({flag}): {suffix}"

    @staticmethod
    def _collect_selected(
        group: MeltGroupPlan,
    ) -> tuple[dict[str, dict[str, set[int]]], dict[str, set[int]], dict[str, dict[tuple[str, int], str]]]:
        selected: dict[str, dict[str, set[int]]] = {
            stype: defaultdict(set) for stype in ("video", "audio", "subtitle")
        }
        lang_overrides: dict[str, dict[tuple[str, int], str]] = {
            stype: {} for stype in ("video", "audio", "subtitle")
        }
        stream_groups: tuple[
            tuple[
                StreamType,
                Iterable[VideoStreamRef | AudioStreamRef | SubtitleStreamRef],
            ],
            ...,
        ] = (
            ("video", group.video_streams or ()),
            ("audio", group.audio_streams or ()),
            ("subtitle", group.subtitle_streams or ()),
        )
        for stype, selected_streams in stream_groups:
            for stream in selected_streams:
                selected[stype][stream.path].add(stream.mkvmerge_track_id)
                if stream.language is not None:
                    lang_overrides[stype][
                        (stream.path, stream.mkvmerge_track_id)
                    ] = stream.language

        selected_attachments: defaultdict[str, set[int]] = defaultdict(set)
        for attachment in group.attachments:
            selected_attachments[attachment.path].add(
                attachment.mkvmerge_attachment_id,
            )

        return selected, selected_attachments, lang_overrides

    def _render_planned(self, logger: logging.Logger, planned_items: list[MeltTitlePlan]) -> None:
        logger.info("Planned outputs:")
        for item in planned_items:
            groups = item.groups
            file_count = sum(len(group.input_files.paths) for group in groups)
            logger.info("Title: %s (%d file(s))", item.title, file_count)
            for idx, group in enumerate(groups, start=1):
                output_path = os.path.join(
                    self.output_dir,
                    item.title,
                    f"{group.output_name}.mkv",
                )
                if len(groups) > 1:
                    logger.info("  Output %d:", idx)
                    prefix = "    "
                else:
                    prefix = "  "
                logger.info("%sOutput: %s", prefix, output_path)

                input_files = group.input_files
                if input_files.paths:
                    references = ", ".join(
                        f"#{input_files.id_for(path)}"
                        for path in input_files.paths
                    )
                    logger.info("%sInputs: %s", prefix, references)

                self._render_group_streams(logger, group, prefix)
                if group.chapter_source:
                    logger.info(
                        "%sChapters: used from file #%d",
                        prefix,
                        input_files.id_for(group.chapter_source),
                    )

    def _render_group_streams(
        self,
        logger: logging.Logger,
        group: MeltGroupPlan,
        prefix: str,
    ) -> None:
        files_details = group.files_details
        if not files_details:
            return

        logger.info("%sStreams:", prefix)
        selected, selected_attachments, lang_overrides = self._collect_selected(group)
        for path in group.input_files.paths:
            details = files_details.get(path)
            if not details:
                continue
            logger.info("%s  File #%d:", prefix, group.input_files.id_for(path))
            tracks = details.get("tracks", {})
            for stype in ("video", "audio", "subtitle"):
                streams = tracks.get(stype, [])
                if not streams:
                    continue
                logger.info("%s    %s:", prefix, stype)
                selected_ids = selected.get(stype, {}).get(path, set())
                stype_lang_overrides = lang_overrides.get(stype, {})
                for stream in streams:
                    tid = stream.get("tid")
                    used = tid in selected_ids
                    override_lang = stype_lang_overrides.get((path, tid))
                    logger.info("%s      %s", prefix, self._format_track_line(stype, stream, used, override_lang))

            attachments = details.get("attachments", [])
            if attachments:
                logger.info("%s    attachments:", prefix)
                selected_ids = selected_attachments.get(path, set())
                for attachment in attachments:
                    tid = attachment.get("tid", "?")
                    name = attachment.get("file_name", "-")
                    used = tid in selected_ids
                    flag = "used" if used else "skip"
                    logger.info("%s      #%s (%s): %s", prefix, tid, flag, name)
