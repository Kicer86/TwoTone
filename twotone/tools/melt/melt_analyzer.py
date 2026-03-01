import logging
import os

from pathlib import Path
from typing import Any, Iterable, Sequence
from tqdm import tqdm

from ..utils import files_utils, generic_utils, language_utils, video_utils
from .attachments_picker import AttachmentsPicker
from .duplicates_source import DuplicatesSource
from .streams_picker import StreamsPicker
from .melt_common import _is_length_mismatch


class MeltAnalyzer:
    def __init__(
        self,
        logger: logging.Logger,
        duplicates_source: DuplicatesSource,
        wd: str,
        allow_length_mismatch: bool,
        tolerance_ms: int,
    ) -> None:
        self.logger = logger
        self.duplicates_source = duplicates_source
        self.wd = wd
        self.allow_length_mismatch = allow_length_mismatch
        self.tolerance_ms = tolerance_ms
        self.base_path: str | None = None

    @staticmethod
    def _stream_short_details(stype: str, stream: dict[str, Any]) -> str:
        def fmt_fps(value: str) -> str | None:
            try:
                fps = generic_utils.fps_str_to_float(str(value))
            except Exception:
                return None

            if abs(fps - round(fps)) < 0.01:
                return str(int(round(fps)))
            return f"{fps:.2f}"

        if stype == "video":
            width = stream.get("width")
            height = stream.get("height")
            fps = stream.get("fps")
            codec = stream.get("codec")
            length = stream.get("length")
            length_formatted = generic_utils.ms_to_time(length) if length else None
            details = []
            if width and height:
                fps_val = fmt_fps(fps) if fps else None
                if fps_val:
                    details.append(f"{width}x{height}@{fps_val}")
                else:
                    details.append(f"{width}x{height}")
            elif fps:
                fps_val = fmt_fps(fps)
                if fps_val:
                    details.append(f"{fps_val}fps")
            if codec:
                details.append(codec)

            if length_formatted:
                details.append(f"duration: {length_formatted}")

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
            fmt = stream.get("format")
            return fmt or ""
        return ""

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
                short = self._stream_short_details(stream_type, stream)

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
        all_streams: Iterable[tuple[str, Iterable[tuple[str, int, str | None]]]],
        tracks: dict[str, dict],
    ) -> None:
        for stype, type_stream in all_streams:
            for stream in type_stream:
                path = stream[0]
                tid = stream[1]
                language = language_utils.language_name(stream[2])

                stream_details = None
                track_infos = tracks.get(path, {}).get(stype, [])
                for info in track_infos:
                    if info.get("tid") == tid:
                        stream_details = self._stream_short_details(stype, info)
                        break

                extra = f" ({stream_details})" if stream_details else ""

                file_id = ids[path]
                self.logger.debug(f"{stype} track #{tid}: {language} from file #{file_id}{extra}")

    def _print_attachments_details(self, ids: dict[str, int], all_attachments: Iterable[tuple[str, int]]) -> None:
        for stream in all_attachments:
            path = stream[0]
            tid = stream[1]

            file_id = ids[path]
            self.logger.debug(f"Attachment ID #{tid} from file #{file_id}")

    @staticmethod
    def _probe_inputs(files: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        details_full = {file: video_utils.get_video_data_mkvmerge(file, enrich=True) for file in files}
        attachments = {file: info["attachments"] for file, info in details_full.items()}
        tracks = {file: info["tracks"] for file, info in details_full.items()}
        return details_full, attachments, tracks

    @staticmethod
    def _prepare_duplicates_set(duplicates: dict[str, list[str]]) -> list[dict[str, Any]]:
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

            if all(os.path.isdir(p) for p in entries):
                dirs = entries

                if len(dirs) == 1:
                    # Special case: single dir → treat all files as one group of duplicates
                    dir_path = dirs[0]
                    media_files = [
                        os.path.join(root, file)
                        for root, _, filenames in os.walk(dir_path)
                        for file in filenames
                        if video_utils.is_video(file)
                    ]
                    media_files.sort()
                    output_name = file_without_ext(os.path.relpath(media_files[0], dir_path)) if media_files else "output"
                    return [(media_files, output_name)]

                # Multiple dirs → group matching files by position
                files_per_dir = []
                for dir_path in dirs:
                    media_files = [
                        os.path.join(root, file)
                        for root, _, filenames in os.walk(dir_path)
                        for file in filenames
                        if video_utils.is_video(file)
                    ]
                    media_files.sort()
                    files_per_dir.append(media_files)

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
    ) -> tuple[list[tuple[str, int, str | None]], list[tuple[str, int, str | None]], list[tuple[str, int, str | None]]]:
        picker_wd = os.path.join(self.wd, "stream_picker")
        streams_picker = StreamsPicker(
            self.logger,
            self.duplicates_source,
            picker_wd,
        )
        return streams_picker.pick_streams(tracks, ids)

    def _validate_input_files(
        self,
        tracks: dict[str, Any],
        ids: dict[str, int],
        video_streams: list[tuple[str, int, str | None]],
        audio_streams: list[tuple[str, int, str | None]],
        subtitle_streams: list[tuple[str, int, str | None]],
    ) -> str | None:
        # Validate lengths across used files

        # Base length for detailed checks
        v_path, v_tid, _ = video_streams[0]
        base_file_id = ids[v_path]
        base_track = self._pick_track_by_tid(tracks[v_path]["video"], v_tid)
        base_length = base_track["length"]

        # Subtitle mismatch (unsupported)
        for path, _, _ in subtitle_streams:
            file_id = ids[path]
            length = self._pick_primary_video_track(tracks[path]["video"], file_id)["length"]
            if _is_length_mismatch(base_length, length, self.tolerance_ms):
                self.logger.debug(
                    f"Subtitles stream from file #{file_id} has length different than length of video stream from file {v_path}. "
                    "This is not supported yet"
                )
                return f"Subtitle length mismatch between #{file_id} and #{base_file_id} (unsupported)."

        # Audio lengths valdiation
        for path, tid, _ in audio_streams:
            file_id = ids[path]
            length = self._pick_primary_video_track(tracks[path]["video"], file_id)["length"]
            if _is_length_mismatch(base_length, length, self.tolerance_ms):
                base_file_id = ids[v_path]
                self.logger.debug(
                    f"Audio stream from file #{file_id} has length different than length of video stream from file #{base_file_id}. "
                    "Check for --allow-length-mismatch option to allow this."
                )

                if self.allow_length_mismatch:
                    self.logger.debug("Audio length mismatch detected; audio will be time-adjusted during processing.")

                else:
                    return f"Audio length mismatch between #{file_id} and #{base_file_id} (use --allow-length-mismatch)."

        return None

    def _log_group_issue(self, title: str, issue: str, files: Sequence[str]) -> None:
        self.logger.warning("Title %s: %s", title, issue)
        for path in files:
            self.logger.warning("  %s", self._format_group_path(path))

    def _format_group_path(self, path: str) -> str:
        if not self.base_path:
            return path
        try:
            return os.path.relpath(path, self.base_path)
        except ValueError:
            return path

    def _validate_group_lengths(
        self,
        tracks: dict[str, Any],
        ids: dict[str, int],
        title: str,
        files: Sequence[str],
    ) -> str | None:
        base_length = None
        base_file_id = None

        for path, info in tracks.items():
            try:
                track = self._pick_primary_video_track(info["video"], ids[path])
            except Exception:
                continue

            length = track.get("length")
            if length is None:
                continue

            if base_length is None:
                base_length = length
                base_file_id = ids[path]
                continue

            if _is_length_mismatch(base_length, length, self.tolerance_ms):
                issue = f"Video length mismatch between #{ids[path]} and #{base_file_id} (use --allow-length-mismatch)."
                if self.allow_length_mismatch:
                    self.logger.debug(f"{issue} Continuing due to allow-length-mismatch.")
                    continue
                return issue

        return None

    def _analyze_group(
        self,
        files: list[str],
        ids: dict[str, int],
        title: str,
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
        # Probe inputs and print details
        details_full, attachments, tracks = self._probe_inputs(files)
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

        # If all streams come from a single file, length mismatches are irrelevant.
        stream_paths = {path for path, _, _ in (video_streams + audio_streams + subtitle_streams)}
        if len(stream_paths) > 1:
            # Only validate lengths for files from which streams are actually used
            used_tracks = {path: info for path, info in tracks.items() if path in stream_paths}
            length_issue = self._validate_group_lengths(used_tracks, ids, title, files)
            if length_issue:
                return None, length_issue, details_full

        # Validate and compute audio patch requirements
        issue = self._validate_input_files(tracks, ids, video_streams, audio_streams, subtitle_streams)
        if issue:
            return None, issue, details_full

        # Attachments picking
        picked_attachments = AttachmentsPicker(self.logger).pick_attachments(attachments)
        audio_prod_lang = self.duplicates_source.get_metadata_for(video_streams[0][0]).get("audio_prod_lang")

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
            "audio_prod_lang": audio_prod_lang,
            "files_details": details_full,
        }, None, details_full

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

                ids = {file: i + 1 for i, file in enumerate(files)}

                # analysis for group
                plan_details, issue, files_details = self._analyze_group(files, ids, title)
                if plan_details is None:
                    self._log_group_issue(title, issue or "Unknown issue.", files)
                    skipped_groups.append({
                        "files": files,
                        "output_name": output_name,
                        "issue": issue or "Unknown issue.",
                        "files_details": files_details,
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
