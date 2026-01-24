import argparse
import logging
import os
import re
import uuid

import pycountry
from dataclasses import dataclass

from overrides import override
from tqdm import tqdm

from .tool import Plan, Tool
from twotone.tools.utils import generic_utils, language_utils, process_utils, subtitles_utils, video_utils


_LANG_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")

# Heuristic: tokens that are common audio labels but not languages.
_AUDIO_LABEL_STOPWORDS = {
    "audio",
    "commentary",
    "director",
    "dub",
    "dubbed",
    "dialog",
    "dialogue",
    "mono",
    "stereo",
    "surround",
    "track",
    "voice",
    "vocals",
    "mix",
    "mixdown",
    "aac",
    "ac3",
    "eac3",
    "dts",
    "truehd",
    "atmos",
    "flac",
    "mp3",
    "opus",
    "pcm",
    "lpcm",
}


@dataclass
class LanguageFixPlanItem:
    path: str
    subtitles_missing: list[int]
    audio_missing: list[int]
    subtitle_updates: dict[int, str]
    audio_updates: dict[int, str]


@dataclass
class LanguageFixPlan:
    items: list[LanguageFixPlanItem]
    include_audio: bool
    base_path: str | None = None

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def render(self, logger: logging.Logger) -> None:
        if not self.items:
            logger.info("No missing track languages found.")
            return

        files_with_updates = sum(1 for item in self.items if item.subtitle_updates or item.audio_updates)
        subtitles_updates_total = sum(len(item.subtitle_updates) for item in self.items)
        audio_updates_total = sum(len(item.audio_updates) for item in self.items)

        logger.info(
            "Planned updates for %d files (subtitles: %d, audio: %d).",
            files_with_updates,
            subtitles_updates_total,
            audio_updates_total,
        )
        if files_with_updates < len(self.items):
            logger.debug(
                "Files with missing languages but no detections: %d.",
                len(self.items) - files_with_updates,
            )
        for item in self.items:
            has_sub_updates = bool(item.subtitle_updates)
            has_audio_updates = bool(item.audio_updates)

            if not has_sub_updates and not has_audio_updates:
                logger.debug("File: %s", _format_path(item.path, self.base_path))
                continue

            logger.info("File: %s", _format_path(item.path, self.base_path))
            if has_sub_updates:
                for line in self._format_track_lines(
                    "subtitles",
                    list(item.subtitle_updates.keys()),
                    item.subtitle_updates,
                    show_unknown=False,
                ):
                    logger.info(line)
            if self.include_audio and has_audio_updates:
                for line in self._format_track_lines(
                    "audio",
                    list(item.audio_updates.keys()),
                    item.audio_updates,
                    show_unknown=False,
                ):
                    logger.info(line)

    @staticmethod
    def _format_track_lines(label: str, tids: list[int], updates: dict[int, str], show_unknown: bool) -> list[str]:
        if not tids:
            return [f"  {label}: -"]

        lines = [f"  {label}:"]
        for tid in tids:
            lang = updates.get(tid)
            if lang:
                lines.append(f"    #{tid} -> {lang}")
            elif show_unknown:
                lines.append(f"    #{tid} -> unknown")
        return lines


class LanguageFixerTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger("TwoTone.language_fix")
        self.working_dir = ""
        self._include_audio = True
        self._base_path: str | None = None
        self._interruption: generic_utils.InterruptibleProcess | None = None

    @override
    def setup_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--no-audio",
            action="store_true",
            help="Do not attempt audio language detection.",
        )
        parser.add_argument(
            "videos_path",
            nargs=1,
            help="Path with videos to analyze.",
        )

    @override
    def analyze(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str) -> Plan:
        self._include_audio = not args.no_audio
        self._base_path = os.path.abspath(args.videos_path[0])
        self._set_context(logger, working_dir)
        process_utils.ensure_tools_exist(["mkvmerge", "mkvextract", "ffprobe"], logger)

        self.logger.info("Searching for files with missing track languages")
        raw_items = self._scan_directory(args.videos_path[0], include_audio=self._include_audio)
        plan_items: list[LanguageFixPlanItem] = []

        for item in tqdm(
            raw_items,
            desc="Detecting languages",
            unit="video",
            **generic_utils.get_tqdm_defaults(),
        ):
            self._check_for_stop()
            video_path = item["path"]
            subtitles_missing = item["missing_subtitles"]
            audio_missing = item["missing_audio"]

            subtitle_updates = self._detect_subtitle_languages(
                video_path,
                subtitles_missing,
                log_detection=False,
            )
            audio_updates: dict[int, str] = {}
            if self._include_audio:
                tracks = self._get_tracks(video_path)
                audio_updates = self._detect_audio_languages(
                    tracks,
                    audio_missing,
                    log_detection=False,
                )

            plan_items.append(
                LanguageFixPlanItem(
                    path=video_path,
                    subtitles_missing=subtitles_missing,
                    audio_missing=audio_missing,
                    subtitle_updates=subtitle_updates,
                    audio_updates=audio_updates,
                )
            )

        return LanguageFixPlan(items=plan_items, include_audio=self._include_audio, base_path=self._base_path)

    @override
    def perform(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str, plan: Plan) -> None:
        self._set_context(logger, working_dir)

        if plan.is_empty():
            self.logger.info("No analysis results, nothing to fix.")
            return

        if not isinstance(plan, LanguageFixPlan):
            self.logger.info("Unsupported plan type, nothing to fix.")
            return

        self._base_path = plan.base_path
        self._apply_plan(plan.items)
        self.logger.info("Done")

    def _apply_plan(self, items: list[LanguageFixPlanItem]) -> None:
        self.logger.info("Fixing track languages")

        for item in tqdm(items, desc="Fixing", unit="video", **generic_utils.get_tqdm_defaults()):
            self._check_for_stop()

            video_path = item.path
            tracks = self._get_tracks(video_path)
            current_langs = {track["tid"]: track["language"] for track in tracks}

            subtitle_updates = {
                tid: lang for tid, lang in item.subtitle_updates.items() if current_langs.get(tid) is None
            }
            audio_updates = {
                tid: lang for tid, lang in item.audio_updates.items() if current_langs.get(tid) is None
            }

            if not subtitle_updates and not audio_updates:
                self.logger.warning("No languages could be detected; skipping file.")
                continue

            self.logger.info("Processing %s", _format_path(video_path, self._base_path))
            if subtitle_updates:
                for line in LanguageFixPlan._format_track_lines(
                    "subtitles",
                    list(subtitle_updates.keys()),
                    subtitle_updates,
                    show_unknown=False,
                ):
                    self.logger.info(line)
            if audio_updates:
                for line in LanguageFixPlan._format_track_lines(
                    "audio",
                    list(audio_updates.keys()),
                    audio_updates,
                    show_unknown=False,
                ):
                    self.logger.info(line)

            updates: dict[int, str] = {}
            updates.update(subtitle_updates)
            updates.update(audio_updates)

            if not self._apply_language_updates(video_path, updates):
                self.logger.warning("Failed to update track languages.")

    def _scan_directory(self, path: str, include_audio: bool) -> list[dict]:
        self.logger.debug(f"Finding MKV files in {path}")
        video_files = []

        for cd, _, files in os.walk(path, followlinks=True):
            for file in files:
                self._check_for_stop()
                if file.lower().endswith(".mkv"):
                    video_files.append(os.path.join(cd, file))

        results: list[dict] = []
        self.logger.debug("Analysing files")
        for video in tqdm(video_files, desc="Analysing videos", unit="video", **generic_utils.get_tqdm_defaults()):
            self._check_for_stop()
            missing = self._collect_missing_tracks(video, include_audio)
            if missing is not None:
                results.append(missing)

        return results

    def _set_context(self, logger: logging.Logger, working_dir: str) -> None:
        self.logger = logger
        self.working_dir = working_dir
        self._interruption = generic_utils.InterruptibleProcess()

    def _check_for_stop(self) -> None:
        if self._interruption is not None:
            self._interruption._check_for_stop()

    def _get_tracks(self, video_path: str) -> list[dict]:
        info = video_utils.get_video_full_info_mkvmerge(video_path)
        tracks: list[dict] = []

        for track in info.get("tracks", []):
            track_type = track.get("type")
            tid = track.get("id")
            props = track.get("properties", {})

            language = props.get("language")
            if language == "und":
                language = None
            if language:
                try:
                    language = language_utils.unify_lang(language)
                except Exception:
                    language = None

            name = props.get("track_name")
            codec = track.get("codec")
            codec_id = props.get("codec_id")

            tracks.append({
                "type": track_type,
                "tid": tid,
                "language": language,
                "name": name,
                "codec": codec,
                "codec_id": codec_id,
            })

        return tracks

    def _collect_missing_tracks(self, video_path: str, include_audio: bool) -> dict | None:
        tracks = self._get_tracks(video_path)

        missing_subtitles: list[int] = []
        webvtt_skipped: set[str] = set()
        pgs_skipped: set[str] = set()
        for track in tracks:
            if track["type"] not in ("subtitle", "subtitles"):
                continue
            if track["language"] is not None:
                continue
            if self._is_webvtt_track(track):
                codec_id = track.get("codec_id") or track.get("codec") or "unknown"
                webvtt_skipped.add(str(codec_id))
                continue
            if self._is_pgs_track(track):
                codec_id = track.get("codec_id") or track.get("codec") or "unknown"
                pgs_skipped.add(str(codec_id))
                continue
            missing_subtitles.append(track["tid"])

        missing_audio: list[int] = []
        if include_audio:
            missing_audio = [
                track["tid"]
                for track in tracks
                if track["type"] == "audio" and track["language"] is None
            ]

        if webvtt_skipped:
            formats = ", ".join(sorted(webvtt_skipped))
            self.logger.warning(
                "WebVTT subtitles are not supported for language detection in %s (codec_id: %s)",
                _format_path(video_path, self._base_path),
                formats,
            )
        if pgs_skipped:
            formats = ", ".join(sorted(pgs_skipped))
            self.logger.warning(
                "PGS subtitles are not supported for language detection in %s (codec_id: %s)",
                _format_path(video_path, self._base_path),
                formats,
            )

        if not missing_subtitles and not missing_audio:
            return None

        return {
            "path": video_path,
            "missing_subtitles": missing_subtitles,
            "missing_audio": missing_audio,
        }

    @staticmethod
    def _is_webvtt_track(track: dict) -> bool:
        codec = str(track.get("codec") or "").lower()
        codec_id = str(track.get("codec_id") or "").lower()
        if "webvtt" not in f"{codec} {codec_id}":
            return False
        return not codec_id.startswith("s_text/webvtt")

    @staticmethod
    def _is_pgs_track(track: dict) -> bool:
        codec = str(track.get("codec") or "").lower()
        codec_id = str(track.get("codec_id") or "").lower()
        return "pgs" in f"{codec} {codec_id}" or "hdmv_pgs_subtitle" in f"{codec} {codec_id}"

    def _guess_audio_language(self, label: str) -> str | None:
        if not label:
            return None

        for token in _LANG_TOKEN_RE.findall(label):
            token_low = token.lower()
            if token_low in _AUDIO_LABEL_STOPWORDS:
                continue

            if language_utils.is_valid_lang_code(token_low):
                try:
                    return language_utils.unify_lang(token_low)
                except Exception:
                    pass

            try:
                lang_info = pycountry.languages.lookup(token_low)
            except LookupError:
                continue

            code = getattr(lang_info, "alpha_3", None) or getattr(lang_info, "alpha_2", None)
            if code:
                try:
                    return language_utils.unify_lang(code)
                except Exception:
                    pass

        return None

    def _detect_subtitle_languages(self, video_path: str, missing_subtitles: list[int], *, log_detection: bool = True) -> dict[int, str]:
        if not missing_subtitles:
            return {}

        base_tmp = os.path.join(self.working_dir, f"subtitle_{uuid.uuid4().hex}")
        tid_to_path = video_utils.extract_subtitle_to_temp(video_path, missing_subtitles, base_tmp, logger=self.logger)

        detected: dict[int, str] = {}
        for tid, path in tid_to_path.items():
            self._check_for_stop()
            if not os.path.exists(path):
                self.logger.warning(f"Subtitle track #{tid} extraction failed for {_format_path(video_path, self._base_path)}")
                continue

            try:
                encoding = subtitles_utils.file_encoding(path)
                detected_lang = subtitles_utils.guess_subtitle_language(path, encoding)
                if detected_lang:
                    try:
                        unified = language_utils.unify_lang(detected_lang)
                    except Exception:
                        self.logger.debug(
                            f"Unrecognized subtitle language '{detected_lang}' for {_format_path(video_path, self._base_path)} track #{tid}"
                        )
                        continue
                    detected[tid] = unified
                    if log_detection:
                        self.logger.info(f"Detected subtitle language for track #{tid}: {unified}")
            except Exception as e:
                self.logger.debug(
                    f"Subtitle language detection failed for {_format_path(video_path, self._base_path)} track #{tid}: {e}"
                )
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

        return detected

    def _detect_audio_languages(self, tracks: list[dict], missing_audio: list[int], *, log_detection: bool = True) -> dict[int, str]:
        if not missing_audio:
            return {}

        audio_tracks = {track["tid"]: track for track in tracks if track["type"] == "audio"}
        detected: dict[int, str] = {}

        for tid in missing_audio:
            self._check_for_stop()
            track = audio_tracks.get(tid)
            if not track:
                continue

            name = track.get("name")
            if not name:
                continue

            detected_lang = self._guess_audio_language(name)
            if detected_lang:
                detected[tid] = detected_lang
                if log_detection:
                    self.logger.info(f"Detected audio language for track #{tid}: {detected_lang}")

        return detected

    def _apply_language_updates(self, video_path: str, updates: dict[int, str]) -> bool:
        if not updates:
            return False

        output_path = f"{video_path}.langfix.mkv"
        if os.path.exists(output_path):
            os.remove(output_path)

        args = ["-o", output_path]
        for tid, lang in sorted(updates.items()):
            args.extend(["--language", f"{tid}:{lang}"])
        args.append(video_path)

        status = process_utils.start_process("mkvmerge", args)
        if status.returncode != 0:
            self.logger.error(f"mkvmerge failed for {_format_path(video_path, self._base_path)}: {status.stderr}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        if not os.path.exists(output_path):
            self.logger.error(f"mkvmerge did not create output file for {_format_path(video_path, self._base_path)}")
            return False

        os.replace(output_path, video_path)
        return True


def _format_path(path: str, base_path: str | None) -> str:
    if not base_path:
        return path

    try:
        base = os.path.abspath(base_path)
        target = os.path.abspath(path)
    except OSError:
        return path

    try:
        if os.path.commonpath([base, target]) != base:
            return path
    except ValueError:
        return path

    rel = os.path.relpath(target, base)
    return rel or path
