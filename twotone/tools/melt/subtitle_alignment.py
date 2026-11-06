"""Text-subtitle extraction and transformation for melt alignment."""

import logging
import os
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from typing import Any

import pysubs2

from ..utils import files_utils, process_utils, subtitles_utils, video_utils
from . import matching_analysis
from .melt_common import SubtitleStreamRef
from .pair_matcher import MappingRelation, PairMatcher, SegmentsMappingResult
from .track_timeline import TrackTimeline


class SubtitleAlignment:
    """Validate, extract, retime, and serialize one subtitle track."""

    def __init__(self, workspace: files_utils.Workspace, logger: logging.Logger) -> None:
        self.workspace = workspace
        self.logger = logger

    @staticmethod
    def track_details(
        stream: SubtitleStreamRef,
        files_details: Mapping[str, Any],
    ) -> dict[str, Any]:
        tracks = files_details.get(stream.path, {}).get("tracks", {}).get("subtitle", [])
        return next(
            (track for track in tracks if track.get("tid") == stream.mkvmerge_track_id),
            {},
        )

    @staticmethod
    def validate_text_track(
        subtitle_stream: SubtitleStreamRef,
        file_ids: Mapping[str, int],
        files_details: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str, str]:
        details = SubtitleAlignment.track_details(subtitle_stream, files_details)
        subtitle_format = str(details.get("format") or details.get("codec") or "unknown")
        normalized_format = subtitle_format.lower()
        image_format_markers = ("pgs", "vobsub", "dvd_subtitle", "dvb_subtitle")
        description = (
            f"subtitle track #{subtitle_stream.mkvmerge_track_id} "
            f"from file #{file_ids[subtitle_stream.path]}"
        )
        if any(marker in normalized_format for marker in image_format_markers):
            raise RuntimeError(
                f"Cannot align {description}: image-based subtitle format "
                f"'{subtitle_format}' is not supported."
            )
        return details, subtitle_format, description

    @staticmethod
    def map_timestamp(
        timestamp_ms: int,
        mapping: Sequence[tuple[int, int]],
        relation: MappingRelation,
    ) -> int:
        if len(mapping) < 2:
            raise RuntimeError("Subtitle timeline mapping needs at least two anchors.")

        if relation is MappingRelation.GLOBAL_LINEAR:
            lhs_start, rhs_start = mapping[0]
            lhs_end, rhs_end = mapping[-1]
        else:
            rhs_values = [pair[1] for pair in mapping]
            index = min(max(0, bisect_right(rhs_values, timestamp_ms) - 1), len(mapping) - 2)
            lhs_start, rhs_start = mapping[index]
            lhs_end, rhs_end = mapping[index + 1]

        rhs_span = rhs_end - rhs_start
        if rhs_span <= 0:
            raise RuntimeError("Subtitle timeline mapping is not strictly monotonic.")
        lhs_span = lhs_end - lhs_start
        return round(lhs_start + (timestamp_ms - rhs_start) * lhs_span / rhs_span)

    @classmethod
    def retime_content(
        cls,
        content: pysubs2.SSAFile,
        matching: SegmentsMappingResult,
        *,
        source_mapping_to_video_ms: int,
        output_mapping_to_video_ms: int,
        source_description: str,
    ) -> None:
        mapping = matching_analysis.strict_mapping(matching.mapping)
        if len(mapping) < 2:
            raise RuntimeError(
                f"Cannot align {source_description}: fewer than two monotonic mapping anchors."
            )

        first_rhs = mapping[0][1]
        last_rhs = mapping[-1][1]
        frame_ms = 1000 / matching.rhs_fps
        edge_tolerance_ms = round(PairMatcher._MAX_BOUNDARY_ERROR_FRAMES * frame_ms)
        rhs_frame_times = sorted(matching.rhs_all_frames)
        accepted_start = first_rhs
        accepted_end = last_rhs
        if rhs_frame_times:
            if first_rhs - rhs_frame_times[0] <= edge_tolerance_ms:
                accepted_start = rhs_frame_times[0]
            if rhs_frame_times[-1] - last_rhs <= edge_tolerance_ms:
                accepted_end = round(rhs_frame_times[-1] + frame_ms)

        for index, event in enumerate(content, start=1):
            source_start = event.start - source_mapping_to_video_ms
            source_end = event.end - source_mapping_to_video_ms
            if source_start < accepted_start or source_end > accepted_end:
                raise RuntimeError(
                    f"Cannot align {source_description}: subtitle event #{index} "
                    f"({event.start}-{event.end} ms) extends beyond verified shared content "
                    f"({accepted_start + source_mapping_to_video_ms}-"
                    f"{accepted_end + source_mapping_to_video_ms} ms)."
                )

            mapped_start = cls.map_timestamp(
                min(max(source_start, first_rhs), last_rhs),
                mapping,
                matching.relation,
            )
            mapped_end = cls.map_timestamp(
                min(max(source_end, first_rhs), last_rhs),
                mapping,
                matching.relation,
            )
            event.start = mapped_start + output_mapping_to_video_ms
            event.end = mapped_end + output_mapping_to_video_ms
            if event.end <= event.start:
                raise RuntimeError(
                    f"Cannot align {source_description}: subtitle event #{index} "
                    "has no positive duration after retiming."
                )

    def align(
        self,
        subtitle_stream: SubtitleStreamRef,
        matching: SegmentsMappingResult,
        *,
        details: dict[str, Any],
        subtitle_format: str,
        description: str,
        source_mapping_to_video_ms: int,
        output_mapping_to_video_ms: int,
    ) -> tuple[SubtitleStreamRef, dict[str, Any]]:
        extension = os.path.splitext(subtitle_stream.path)[1].lower()
        if extension in TrackTimeline._MKVMERGE_PRESERVES_START_EXTENSIONS:
            extracted_base = self.workspace.unique_file("subtitle_source")
            extracted = video_utils.extract_subtitle_to_temp(
                subtitle_stream.path,
                [subtitle_stream.mkvmerge_track_id],
                extracted_base,
                logger=self.logger,
            ).get(subtitle_stream.mkvmerge_track_id)
        else:
            extracted = self.workspace.unique_file("subtitle_source", "ass")
            process_utils.raise_on_error(
                process_utils.start_process(
                    "ffmpeg",
                    [
                        "-y",
                        "-i", subtitle_stream.path,
                        "-map", f"0:{subtitle_stream.ffprobe_stream_index}",
                        "-c:s", "ass",
                        extracted,
                    ],
                    logger=self.logger,
                )
            )
        if extracted is None or not os.path.exists(extracted):
            raise RuntimeError(f"Cannot align {description}: subtitle extraction failed.")

        content = subtitles_utils.open_subtitle_file(extracted, logger=self.logger)
        if content is None:
            raise RuntimeError(
                f"Cannot align {description}: extracted format '{subtitle_format}' "
                "cannot be parsed as text subtitles."
            )

        self.retime_content(
            content,
            matching,
            source_mapping_to_video_ms=source_mapping_to_video_ms,
            output_mapping_to_video_ms=output_mapping_to_video_ms,
            source_description=description,
        )
        output_path = self.workspace.unique_file("aligned_subtitle", "ass")
        content.save(output_path, format_="ass", encoding="utf-8")
        return (
            SubtitleStreamRef(output_path, 0, 0, subtitle_stream.language),
            details,
        )
