import argparse
import logging
import os
import re
import uuid

import pycountry

from overrides import override
from tqdm import tqdm

from .tool import Tool
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


class LanguageFixerTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger("TwoTone.language_fix")
        self.working_dir = ""
        self._analysis_results: list[dict] | None = None
        self._include_audio = True
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
    def analyze(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str) -> None:
        self._analysis_results = None
        self._include_audio = not args.no_audio
        self._set_context(logger, working_dir)
        process_utils.ensure_tools_exist(["mkvmerge", "mkvextract", "ffprobe"], logger)

        self.logger.info("Searching for files with missing track languages")
        self._analysis_results = self._scan_directory(args.videos_path[0], include_audio=self._include_audio)

    @override
    def perform(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str) -> None:
        self._set_context(logger, working_dir)

        items = self._analysis_results
        self._analysis_results = None

        if items is None or len(items) == 0:
            self.logger.info("No analysis results, nothing to fix.")
            return

        self._fix_files(items, include_audio=self._include_audio)
        self.logger.info("Done")

    def _fix_files(self, items: list[dict], include_audio: bool) -> None:
        self._print_missing(items)
        self.logger.info("Fixing track languages")

        for item in tqdm(items, desc="Fixing", unit="video", leave=False, smoothing=0.1, mininterval=.2, disable=generic_utils.hide_progressbar()):
            self._check_for_stop()

            video_path = item["path"]
            tracks = self._get_tracks(video_path)
            current_langs = {track["tid"]: track["language"] for track in tracks}

            missing_subtitles = [tid for tid in item["missing_subtitles"] if current_langs.get(tid) is None]
            missing_audio: list[int] = []
            if include_audio:
                missing_audio = [tid for tid in item["missing_audio"] if current_langs.get(tid) is None]

            self.logger.info(f"Processing {video_path}")

            updates: dict[int, str] = {}
            updates.update(self._detect_subtitle_languages(video_path, missing_subtitles))
            if include_audio:
                updates.update(self._detect_audio_languages(tracks, missing_audio))

            if not updates:
                self.logger.warning("No languages could be detected; skipping file.")
                continue

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
        for video in tqdm(video_files, desc="Analysing videos", unit="video", leave=False, smoothing=0.1, mininterval=.2, disable=generic_utils.hide_progressbar()):
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

            tracks.append({
                "type": track_type,
                "tid": tid,
                "language": language,
                "name": name,
            })

        return tracks

    def _collect_missing_tracks(self, video_path: str, include_audio: bool) -> dict | None:
        tracks = self._get_tracks(video_path)

        missing_subtitles = [
            track["tid"]
            for track in tracks
            if track["type"] in ("subtitle", "subtitles") and track["language"] is None
        ]

        missing_audio: list[int] = []
        if include_audio:
            missing_audio = [
                track["tid"]
                for track in tracks
                if track["type"] == "audio" and track["language"] is None
            ]

        if not missing_subtitles and not missing_audio:
            return None

        return {
            "path": video_path,
            "missing_subtitles": missing_subtitles,
            "missing_audio": missing_audio,
        }

    def _print_missing(self, items: list[dict]) -> None:
        self.logger.info(f"Found {len(items)} files with missing track languages:")
        for item in items:
            subs_count = len(item["missing_subtitles"])
            audio_count = len(item["missing_audio"])
            self.logger.info(f"{item['path']} (subtitles: {subs_count}, audio: {audio_count})")

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

    def _detect_subtitle_languages(self, video_path: str, missing_subtitles: list[int]) -> dict[int, str]:
        if not missing_subtitles:
            return {}

        base_tmp = os.path.join(self.working_dir, f"subtitle_{uuid.uuid4().hex}")
        tid_to_path = video_utils.extract_subtitle_to_temp(video_path, missing_subtitles, base_tmp, logger=self.logger)

        detected: dict[int, str] = {}
        for tid, path in tid_to_path.items():
            if not os.path.exists(path):
                self.logger.warning(f"Subtitle track #{tid} extraction failed for {video_path}")
                continue

            try:
                encoding = subtitles_utils.file_encoding(path)
                detected_lang = subtitles_utils.guess_language(path, encoding)
                if detected_lang:
                    try:
                        unified = language_utils.unify_lang(detected_lang)
                    except Exception:
                        self.logger.debug(f"Unrecognized subtitle language '{detected_lang}' for {video_path} track #{tid}")
                        continue
                    detected[tid] = unified
                    self.logger.info(f"Detected subtitle language for track #{tid}: {unified}")
            except Exception as e:
                self.logger.debug(f"Subtitle language detection failed for {video_path} track #{tid}: {e}")
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

        return detected

    def _detect_audio_languages(self, tracks: list[dict], missing_audio: list[int]) -> dict[int, str]:
        if not missing_audio:
            return {}

        audio_tracks = {track["tid"]: track for track in tracks if track["type"] == "audio"}
        detected: dict[int, str] = {}

        for tid in missing_audio:
            track = audio_tracks.get(tid)
            if not track:
                continue

            name = track.get("name")
            if not name:
                continue

            detected_lang = self._guess_audio_language(name)
            if detected_lang:
                detected[tid] = detected_lang
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
            self.logger.error(f"mkvmerge failed for {video_path}: {status.stderr}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        if not os.path.exists(output_path):
            self.logger.error(f"mkvmerge did not create output file for {video_path}")
            return False

        os.replace(output_path, video_path)
        return True
