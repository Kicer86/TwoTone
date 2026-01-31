import logging
import os
import re

from dataclasses import dataclass
from typing import Any, Dict, List

from ..utils import generic_utils, language_utils


@dataclass
class MeltPlan:
    items: List[Dict[str, Any]]
    output_dir: str

    def is_empty(self) -> bool:
        return not self.items

    def render(self, logger: logging.Logger) -> None:
        if not self.items:
            logger.info("No titles to melt.")
            return

        visible_items = [
            item for item in self.items
            if item.get("groups") or item.get("skipped_groups")
        ]
        if not visible_items:
            logger.info("No suitable candidates found for melting.")
            return

        planned_items = [item for item in visible_items if item.get("groups")]
        skipped_items = [item for item in visible_items if item.get("skipped_groups")]

        def stream_short_details(stype: str, stream: Dict[str, Any]) -> str:
            if stype == "video":
                fps = stream.get("fps")
                width = stream.get("width")
                height = stream.get("height")
                bitrate = stream.get("bitrate")
                details = []
                if width and height:
                    details.append(f"{width}x{height}")
                if fps:
                    try:
                        fps_val = generic_utils.fps_str_to_float(str(fps))
                        details.append(f"{fps_val:.3f}fps")
                    except Exception:
                        details.append(f"{fps}fps")
                if bitrate:
                    details.append(f"{bitrate}bps")
                return ", ".join(details)
            if stype == "audio":
                channels = stream.get("channels")
                sample_rate = stream.get("sample_rate")
                details = []
                if channels:
                    details.append(f"{channels}ch")
                if sample_rate:
                    details.append(f"{sample_rate}Hz")
                return ", ".join(details)
            if stype == "subtitle":
                fmt = stream.get("format") or stream.get("codec")
                return str(fmt) if fmt else ""
            return ""

        def format_track_line(stype: str, stream: Dict[str, Any], used: bool) -> str:
            tid = stream.get("tid", "?")
            name = stream.get("name")
            lang = stream.get("language")
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

        def collect_selected(group: Dict[str, Any]) -> tuple[dict[str, dict[str, set[int]]], dict[str, set[int]]]:
            selected: dict[str, dict[str, set[int]]] = {
                "video": {},
                "audio": {},
                "subtitle": {},
            }
            streams = group.get("streams", {})
            for stype in ("video", "audio", "subtitle"):
                for path, tid, _ in streams.get(stype, []):
                    selected[stype].setdefault(path, set()).add(tid)

            selected_attachments: dict[str, set[int]] = {}
            for path, tid in group.get("attachments", []):
                selected_attachments.setdefault(path, set()).add(tid)

            return selected, selected_attachments

        total_outputs = sum(len(item.get("groups", [])) for item in planned_items)
        total_files = sum(
            len(group.get("files", []))
            for item in planned_items
            for group in item.get("groups", [])
        )
        skipped_sets = sum(len(item.get("skipped_groups", [])) for item in skipped_items)
        skipped_files = sum(
            len(group.get("files", []))
            for item in skipped_items
            for group in item.get("skipped_groups", [])
        )

        if planned_items:
            logger.info(
                "Planned melt: %d output(s) from %d input file(s).",
                total_outputs,
                total_files,
            )
        else:
            logger.info("No outputs planned.")
        logger.info("Output directory: %s", self.output_dir)

        if planned_items:
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

                    files_details = group.get("files_details", {})
                    if files_details:
                        logger.info("%sStreams:", prefix)
                        selected, selected_attachments = collect_selected(group)
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
                                for stream in streams:
                                    tid = stream.get("tid")
                                    used = tid in selected_ids
                                    logger.info("%s      %s", prefix, format_track_line(stype, stream, used))

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

        if skipped_sets:
            logger.info("Skipped candidates: %d set(s), %d file(s).", skipped_sets, skipped_files)
