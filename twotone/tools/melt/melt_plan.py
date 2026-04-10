import logging
import os

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..utils import language_utils
from .melt_common import StreamType, stream_short_details


@dataclass
class MeltPlan:
    items: list[dict[str, Any]]
    output_dir: str

    def is_empty(self) -> bool:
        return not any(item.get("groups") for item in self.items)

    def render(self, logger: logging.Logger) -> None:
        if not self.items:
            logger.info("No titles to melt.")
            return

        planned_items = [item for item in self.items if item.get("groups")]
        if not planned_items:
            logger.info("No suitable candidates found for melting.")
            return

        total_outputs = sum(len(item.get("groups", [])) for item in planned_items)
        total_files = sum(
            len(group.get("files", []))
            for item in planned_items
            for group in item.get("groups", [])
        )

        logger.info(
            "Planned melt: %d output(s) from %d input file(s).",
            total_outputs,
            total_files,
        )
        logger.info("Output directory: %s", self.output_dir)

        self._render_planned(logger, planned_items)

    _stream_short_details = staticmethod(stream_short_details)

    @staticmethod
    def _format_track_line(stype: StreamType, stream: dict[str, Any], used: bool, override_lang: str | None = None) -> str:
        tid = stream.get("tid", "?")
        name = stream.get("name")
        lang = override_lang or stream.get("language")
        lang_name = language_utils.language_name(lang) if lang else "unknown"
        details = MeltPlan._stream_short_details(stype, stream)
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
        group: dict[str, Any],
    ) -> tuple[dict[str, dict[str, set[int]]], dict[str, set[int]], dict[str, dict[tuple[str, int], str]]]:
        selected: dict[str, dict[str, set[int]]] = {
            stype: defaultdict(set) for stype in ("video", "audio", "subtitle")
        }
        lang_overrides: dict[str, dict[tuple[str, int], str]] = {
            stype: {} for stype in ("video", "audio", "subtitle")
        }
        streams = group.get("streams", {})
        for stype in ("video", "audio", "subtitle"):
            for path, tid, lang in streams.get(stype, []):
                selected[stype][path].add(tid)
                if lang is not None:
                    lang_overrides[stype][(path, tid)] = lang

        selected_attachments: defaultdict[str, set[int]] = defaultdict(set)
        for path, tid in group.get("attachments", []):
            selected_attachments[path].add(tid)

        return selected, selected_attachments, lang_overrides

    def _render_planned(self, logger: logging.Logger, planned_items: list[dict[str, Any]]) -> None:
        logger.info("Planned outputs:")
        for item in planned_items:
            title = item.get("title", "<unknown>")
            groups = item.get("groups", [])
            file_count = sum(len(group.get("files", [])) for group in groups)
            logger.info("Title: %s (%d file(s))", title, file_count)
            for idx, group in enumerate(groups, start=1):
                output_name = group.get("output_name", "output")
                output_path = os.path.join(self.output_dir, title, f"{output_name}.mkv")
                if len(groups) > 1:
                    logger.info("  Output %d:", idx)
                    prefix = "    "
                else:
                    prefix = "  "
                logger.info("%sOutput: %s", prefix, output_path)

                files = group.get("files", [])
                if files:
                    logger.info("%sInputs:", prefix)
                    for file_idx, path in enumerate(files, start=1):
                        logger.info("%s  #%d: %s", prefix, file_idx, path)

                self._render_group_streams(logger, group, files, prefix)

    def _render_group_streams(
        self,
        logger: logging.Logger,
        group: dict[str, Any],
        files: list[str],
        prefix: str
    ) -> None:
        files_details = group.get("files_details", {})
        if not files_details:
            return

        logger.info("%sStreams:", prefix)
        selected, selected_attachments, lang_overrides = self._collect_selected(group)
        for file_idx, path in enumerate(files, start=1):
            details = files_details.get(path)
            if not details:
                continue
            logger.info("%s  File #%d:", prefix, file_idx)
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


