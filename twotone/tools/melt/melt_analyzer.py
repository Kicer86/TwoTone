import logging
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from tqdm import tqdm

from ..utils import files_utils, generic_utils, language_utils, video_utils
from .attachments_picker import AttachmentsPicker
from .duplicates_source import DuplicatesSource
from .streams_picker import StreamsPicker
from .melt_common import (
    AttachmentRef,
    AudioStreamRef,
    MeltInputFiles,
    StreamType,
    SubtitleStreamRef,
    VideoStreamRef,
    _is_length_mismatch,
    stream_short_details,
)
from .pair_matcher import PairMatcher


class UnsupportedMeltInputError(RuntimeError):
    """Raised when an input group contains an element Melt cannot model yet."""


@dataclass(frozen=True)
class AlignmentRequirement:
    path: str
    issue: str


class MeltAnalyzer:
    def __init__(
        self,
        logger: logging.Logger,
        duplicates_source: DuplicatesSource,
        workspace: files_utils.Workspace,
        allow_video_timeline_mismatch: bool,
    ) -> None:
        self.logger = logger
        self.duplicates_source = duplicates_source
        self.workspace = workspace
        self.allow_video_timeline_mismatch = allow_video_timeline_mismatch
        self.input_paths: tuple[str, ...] = ()

    def analyze_duplicates(self, duplicates: dict[str, list[str]]) -> list[dict[str, Any]]:
        base_plan = self._prepare_duplicates_set(duplicates)

        analysis_plan: list[dict[str, Any]] = []
        for item in tqdm(base_plan, desc="Titles", unit="title", **generic_utils.get_tqdm_defaults()):
            title = item["title"]
            groups = item["groups"]

            analyzed_groups: list[dict[str, Any]] = []
            skipped_groups: list[dict[str, Any]] = []
            if len(groups) > 1:
                groups_iter = tqdm(groups, desc="Candidates", unit="set", position=1, **generic_utils.get_tqdm_defaults())
            else:
                groups_iter = groups
            for group in groups_iter:
                files = group["files"]
                output_name = group["output_name"]

                input_files = MeltInputFiles(files, self.input_paths)
                self._log_group_inputs(title, input_files)

                # analysis for group
                try:
                    plan_details, issue, _ = self._analyze_group(files, input_files.ids, title)
                except UnsupportedMeltInputError as err:
                    plan_details = None
                    issue = str(err)
                if plan_details is None:
                    self._log_group_issue(issue or "Unknown issue.")
                    skipped_groups.append({
                        "files": files,
                        "output_name": output_name,
                        "issue": issue or "Unknown issue.",
                    })
                else:
                    analyzed_groups.append({
                        "files": files,
                        "output_name": output_name,
                        **plan_details,
                    })

            analysis_plan.append({
                "title": title,
                "groups": analyzed_groups,
                "skipped_groups": skipped_groups,
            })

        return analysis_plan

    @staticmethod
    def _pick_track_by_tid(streams: Sequence[dict[str, Any]], tid: int) -> dict[str, Any]:
        track = next((item for item in streams if item.get("tid") == tid), None)
        if track is None:
            raise RuntimeError(f"Track #{tid} not found.")
        return track

    @staticmethod
    def _pick_primary_video_track(streams: Sequence[dict[str, Any]], file_id: int) -> dict[str, Any]:
        for track in streams:
            if not track.get("attached_pic", False):
                return track
        raise RuntimeError(f"No video track found in file #{file_id}.")

    def _print_file_details(self, file: str, details: dict[str, Any], ids: dict[str, int]) -> None:
        def formatter(key: str, value: Any) -> str:
            if key == "fps":
                try:
                    fps = generic_utils.fps_str_to_float(str(value))
                    return f"{fps:.3f}"
                except Exception:
                    return str(value)
            if key == "length":
                return generic_utils.ms_to_time(value) if value else "-"
            return str(value) if value else "-"

        def show(key: str) -> bool:
            return key != "tid"

        file_id = ids[file]
        self.logger.debug(f"File #{file_id} details:")
        tracks = details["tracks"]
        attachments = details["attachments"]

        for stream_type, streams in tracks.items():
            self.logger.debug(f"  {stream_type}: {len(streams)} track(s)")
            for stream in streams:
                lang_name = language_utils.language_name(stream.get("language"))
                short = stream_short_details(stream_type, stream)

                info = lang_name
                if short:
                    info += f" ({short})"

                sid = stream.get("tid")
                self.logger.debug(f"    #{sid}: {info}")

        for attachment in attachments:
            file_name = attachment["file_name"]
            self.logger.debug(f"  attachment: {file_name}")

        # more details for debug
        for stream_type, streams in tracks.items():
            self.logger.debug(f"\t{stream_type}:")

            for stream in streams:
                sid = stream.get("tid")
                self.logger.debug(f"\t#{sid}:")
                for key, value in stream.items():
                    if show(key):
                        key_title = key + ":"
                        self.logger.debug(
                            f"\t\t{key_title:<16}{formatter(key, value)}")

    def _print_streams_details(
        self,
        ids: dict[str, int],
        all_streams: Iterable[
            tuple[StreamType, Iterable[VideoStreamRef | AudioStreamRef | SubtitleStreamRef]]
        ],
        tracks: dict[str, dict],
    ) -> None:
        for stype, type_stream in all_streams:
            for stream in type_stream:
                path = stream.path
                tid = stream.mkvmerge_track_id
                language = language_utils.language_name(stream.language)

                stream_details = None
                track_infos = tracks.get(path, {}).get(stype, [])
                for info in track_infos:
                    if info.get("tid") == tid:
                        stream_details = stream_short_details(stype, info)
                        break

                extra = f" ({stream_details})" if stream_details else ""

                file_id = ids[path]
                self.logger.debug(f"{stype} track #{tid}: {language} from file #{file_id}{extra}")

    def _print_attachments_details(self, ids: dict[str, int], all_attachments: Iterable[AttachmentRef]) -> None:
        for stream in all_attachments:
            path = stream.path
            tid = stream.mkvmerge_attachment_id

            file_id = ids[path]
            self.logger.debug(f"Attachment ID #{tid} from file #{file_id}")

    @staticmethod
    def _validate_supported_elements(raw_details: dict[str, dict[str, Any]], ids: dict[str, int]) -> None:
        thumbnails: list[tuple[str, str]] = []

        for path, details in raw_details.items():
            for track in details.get("tracks", []):
                track_type = track.get("type")
                if track_type not in ("video", "audio", "subtitle", "subtitles"):
                    raise UnsupportedMeltInputError(
                        f"File #{ids[path]} contains unsupported track type '{track_type}' (not supported yet)."
                    )

            for attachment in details.get("attachments", []):
                file_name = attachment.get("file_name", f"attachment #{attachment.get('id', '?')}")
                content_type = attachment.get("content_type")
                if isinstance(content_type, str) and content_type.startswith("image/"):
                    thumbnails.append((path, file_name))
                else:
                    raise UnsupportedMeltInputError(
                        f"File #{ids[path]} contains unsupported attachment '{file_name}' "
                        f"with content type '{content_type}' (not supported yet)."
                    )

        if len(thumbnails) > 1:
            locations = ", ".join(f"'{name}' from file #{ids[path]}" for path, name in thumbnails)
            raise UnsupportedMeltInputError(
                f"Melt input group contains {len(thumbnails)} thumbnails, which are not supported yet; "
                f"at most one thumbnail is supported: {locations}"
            )

    def _probe_inputs(self, files: Sequence[str], ids: dict[str, int]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        raw_details = {
            file: video_utils.get_video_full_info_mkvmerge(file, logger=self.logger)
            for file in files
        }
        self._validate_supported_elements(raw_details, ids)

        details_full = {
            file: video_utils.get_video_data_mkvmerge(
                file,
                enrich=True,
                logger=self.logger,
                _mkvmerge_info=raw_details[file],
            )
            for file in files
        }
        for file, details in details_full.items():
            details["chapters"] = raw_details[file].get("chapters", [])
        attachments = {file: info["attachments"] for file, info in details_full.items()}
        tracks = {file: info["tracks"] for file, info in details_full.items()}
        return details_full, attachments, tracks

    def _prepare_duplicates_set(self, duplicates: dict[str, list[str]]) -> list[dict[str, Any]]:
        """Prepare groups of duplicate files and output names per title.

        Returns a plan in the form:
        [
          {"title": str, "groups": [{"files": [str,...], "output_name": str}, ...]},
          ...
        ]
        """
        def process_entries(entries: list[str]) -> list[tuple[list[str], str]]:
            # Returns list of: (group of duplicates, output base name)

            def file_without_ext(path: str) -> str:
                dir, name, _ = files_utils.split_path(path)
                return os.path.join(dir, name)

            def collect_media_files(dir_path: str) -> list[str]:
                media_files = video_utils.collect_video_files(dir_path, self.duplicates_source.interruption)
                media_files.sort()
                return media_files

            if all(os.path.isdir(p) for p in entries):
                dirs = entries

                if len(dirs) == 1:
                    # Special case: single dir → treat all files as one group of duplicates
                    dir_path = dirs[0]
                    media_files = collect_media_files(dir_path)
                    output_name = file_without_ext(os.path.relpath(media_files[0], dir_path)) if media_files else "output"
                    return [(media_files, output_name)]

                # Multiple dirs → group matching files by position
                files_per_dir = []
                for dir_path in dirs:
                    files_per_dir.append(collect_media_files(dir_path))

                lengths = [len(files) for files in files_per_dir]
                if len(set(lengths)) != 1:
                    raise RuntimeError(f"Input directories have different counts of video files: {lengths}")

                sorted_file_lists = [list(entry) for entry in zip(*files_per_dir)]
                first_file_fullnames = [os.path.relpath(path[0], dirs[0]) for path in sorted_file_lists]
                first_file_names = [file_without_ext(path) for path in first_file_fullnames]

                return [(files_group, output_name) for files_group, output_name in zip(sorted_file_lists, first_file_names)]

            else:
                # List of individual files
                first_file_fullname = os.path.basename(entries[0])
                first_file_name = Path(first_file_fullname).stem
                return [(entries, first_file_name)]

        plan: list[dict[str, Any]] = []
        for title, entries in duplicates.items():
            files_groups = process_entries(entries)
            item = {
                "title": title,
                "groups": [{"files": files, "output_name": output_name} for files, output_name in files_groups]
            }
            plan.append(item)

        return plan

    def _pick_streams(
        self,
        tracks: dict[str, Any],
        ids: dict[str, int],
    ) -> tuple[list[VideoStreamRef], list[AudioStreamRef], list[SubtitleStreamRef]]:
        picker_wd = self.workspace.unique_dir("stream_picker")
        streams_picker = StreamsPicker(
            self.logger,
            self.duplicates_source,
            picker_wd,
        )
        return streams_picker.pick_streams(tracks, ids)

    @staticmethod
    def _has_chapters(details: dict[str, Any]) -> bool:
        return bool(details.get("chapters", []))

    def _pick_chapter_source(
        self,
        files_details: dict[str, dict[str, Any]],
        tracks: dict[str, Any],
        video_streams: Sequence[VideoStreamRef],
        ids: dict[str, int],
    ) -> str | None:
        """Pick chapters only when their source timeline already matches base video."""
        base_video = video_streams[0]
        base_path = base_video.path
        if self._has_chapters(files_details[base_path]):
            return base_path

        base_track = self._pick_track_by_tid(
            tracks[base_path]["video"], base_video.mkvmerge_track_id,
        )
        base_length = base_track.get("length")
        for path, details in files_details.items():
            if path == base_path or not self._has_chapters(details):
                continue

            file_id = ids[path]
            source_track = self._pick_primary_video_track(tracks[path]["video"], file_id)
            if source_track.get("length") == base_length:
                return path

            self.logger.debug(
                "Not copying chapters from file #%d because its video length differs from the base video.",
                file_id,
            )

        return None

    def _validate_input_files(
        self,
        tracks: dict[str, Any],
        ids: dict[str, int],
        video_streams: list[VideoStreamRef],
        audio_streams: list[AudioStreamRef],
        subtitle_streams: list[SubtitleStreamRef],
    ) -> str | None:
        # Validate subtitle lengths against the selected base video.

        # Base length for detailed checks
        base_video = video_streams[0]
        v_path = base_video.path
        v_tid = base_video.mkvmerge_track_id
        base_file_id = ids[v_path]
        base_track = self._pick_track_by_tid(tracks[v_path]["video"], v_tid)
        base_length = base_track["length"]

        # Subtitle mismatch (unsupported)
        for subtitle_stream in subtitle_streams:
            path = subtitle_stream.path
            file_id = ids[path]
            length = self._pick_primary_video_track(tracks[path]["video"], file_id)["length"]

            if _is_length_mismatch(base_length, length):
                base_fmt = generic_utils.ms_to_time(base_length) if base_length else "?"
                other_fmt = generic_utils.ms_to_time(length) if length else "?"
                self.logger.debug(
                    f"Subtitles stream from file #{file_id} has length different than length of video stream from file #{base_file_id}. "
                    "This is not supported yet"
                )

                return (f"Subtitle length mismatch between #{file_id} ({other_fmt}) and #{base_file_id} ({base_fmt}) (unsupported).")

        return None

    def _log_group_inputs(self, title: str, input_files: MeltInputFiles) -> None:
        self.logger.info("Title %s: input files:", title)
        input_files.render(self.logger, prefix="  ")

    def _log_group_issue(self, issue: str) -> None:
        self.logger.warning("%s", issue)

    def _find_alignment_requirements(
        self,
        tracks: dict[str, Any],
        ids: dict[str, int],
        video_streams: list[VideoStreamRef],
        audio_streams: list[AudioStreamRef],
        subtitle_streams: list[SubtitleStreamRef],
    ) -> list[AlignmentRequirement]:
        stream_paths = {stream.path for stream in (video_streams + audio_streams + subtitle_streams)}

        if len(stream_paths) <= 1:
            return []

        base_path = video_streams[0].path
        base_file_id = ids[base_path]
        base_length = self._pick_primary_video_track(tracks[base_path]["video"], base_file_id).get("length")
        requirements: list[AlignmentRequirement] = []

        for path in sorted(stream_paths - {base_path}, key=ids.__getitem__):
            file_id = ids[path]
            self.logger.info("Checking video alignment: #%d ↔ #%d", base_file_id, file_id)
            length = self._pick_primary_video_track(tracks[path]["video"], file_id).get("length")

            if _is_length_mismatch(base_length, length):
                issue = f"Video length mismatch between #{file_id} and #{base_file_id} (use --allow-video-timeline-mismatch)."
            elif base_length is None or length is None:
                continue
            else:
                with self.workspace.scoped_dir("equal_length_content") as matching_wd:
                    matcher = PairMatcher(
                        self.duplicates_source.interruption, matching_wd, base_path, path,
                        self.logger.getChild("PairMatcher"), lhs_label=f"#{base_file_id}", rhs_label=f"#{file_id}",
                    )
                    if matcher.has_identical_timeline_content():
                        continue

                issue = f"Video content mismatch between #{file_id} and #{base_file_id} (use --allow-video-timeline-mismatch)."

            requirements.append(AlignmentRequirement(path, issue))

        return requirements

    def _analyze_group(
        self,
        files: list[str],
        ids: dict[str, int],
        title: str,
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
        # Probe inputs and print details
        details_full, attachments, tracks = self._probe_inputs(files, ids)
        for file, file_details in details_full.items():
            self._print_file_details(file, file_details, ids)

        # Pick streams
        try:
            video_streams, audio_streams, subtitle_streams = self._pick_streams(tracks, ids)
        except RuntimeError as err:
            self.logger.debug(err)
            return None, str(err), details_full

        if not video_streams:
            self.logger.debug("No video streams found.")
            return None, "No video streams found.", details_full

        requirements = self._find_alignment_requirements(
            tracks, ids, video_streams, audio_streams, subtitle_streams,
        )
        if requirements:
            if self.allow_video_timeline_mismatch:
                for requirement in requirements:
                    self.logger.debug(
                        "%s Continuing due to allow-video-timeline-mismatch; full content matching runs during processing.",
                        requirement.issue,
                    )
            else:
                return None, "\n".join(requirement.issue for requirement in requirements), details_full

        # Validate and compute audio patch requirements
        issue = self._validate_input_files(tracks, ids, video_streams, audio_streams, subtitle_streams)
        if issue:
            return None, issue, details_full

        chapter_source = self._pick_chapter_source(details_full, tracks, video_streams, ids)

        # Attachments picking
        picked_attachments = AttachmentsPicker(self.logger).pick_attachments(attachments)
        audio_prod_lang = self.duplicates_source.get_metadata_for(video_streams[0].path).get("audio_prod_lang")

        # Present proposed output
        self.logger.debug("Streams used to create output video file:")
        self._print_streams_details(
            ids,
            (
                ("video", video_streams),
                ("audio", audio_streams),
                ("subtitle", subtitle_streams),
            ),
            tracks,
        )
        self._print_attachments_details(ids, picked_attachments)

        # Prepare plan entity
        return {
            "streams": {
                "video": video_streams,
                "audio": audio_streams,
                "subtitle": subtitle_streams,
            },
            "attachments": picked_attachments,
            "chapter_source": chapter_source,
            "audio_prod_lang": audio_prod_lang,
            "files_details": details_full,
            "alignment_paths": sorted({requirement.path for requirement in requirements}),
        }, None, details_full
