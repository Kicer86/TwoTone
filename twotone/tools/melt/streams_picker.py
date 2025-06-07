
import logging

from typing import Dict, List, Optional, Tuple

from ..utils2 import languages
from .duplicates_source import DuplicatesSource

class StreamsPicker:
    def __init__(self, logger: logging.Logger, duplicates_source: DuplicatesSource):
        self.logger = logger
        self.duplicates_source = duplicates_source


    @staticmethod
    def _iter_starting_with(d: dict, start_key) -> Tuple:
        """Yield (key, value) pairs starting from start_key, then the rest in order."""
        if start_key not in d:
            raise KeyError(f"{start_key} not found in dictionary")

        yielded = set()

        # Yield the starting key first
        yield start_key, d[start_key]
        yielded.add(start_key)

        # Yield the rest in order
        for k, v in d.items():
            if k not in yielded:
                yield k, v

    def _pick_best_video(self, files_details: Dict[str, Dict]) -> List[Tuple[str, int, str]]:
        best_file = None
        best_stream = None

        # todo: handle many video streams
        default_video_stream = 0

        for file, details in files_details.items():
            if best_file is None:
                best_file = file
                best_stream = details
                continue

            lhs_video_stream = best_stream["video"][default_video_stream]
            rhs_video_stream = details["video"][default_video_stream]

            if lhs_video_stream["width"] < rhs_video_stream["width"] and lhs_video_stream["height"] < rhs_video_stream["height"]:
                best_file = file
                best_stream = details
            elif lhs_video_stream["width"] > rhs_video_stream["width"] and lhs_video_stream["height"] > rhs_video_stream["height"]:
                continue
            else:
                # equal or mixed width/height
                if eval(lhs_video_stream["fps"]) < eval(rhs_video_stream["fps"]):
                    best_file = file
                    best_stream = details

        return (best_file, default_video_stream, None)

    def _pick_unique_streams(
        self,
        files_details: Dict[str, Dict],
        best_file: str,
        stream_type: str,
        unique_keys: List[str],
        preference_keys: List[str],
        fallback_languages: Optional[Dict[str, str]] = None,
        override_languages: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, int, str]]:
        """
        Select unique streams of a given type across multiple files.

        Args:
            files_details: Mapping from file path to metadata.
            best_file: File to prioritize when collecting streams.
            stream_type: Stream type to extract (e.g., "audio", "subtitles").
            unique_keys: Keys that define stream uniqueness (first must be 'language').
            preference_keys: Keys used to resolve ties (e.g., ["bitrate"]).
            fallback_languages: Optional per-file fallback if stream["language"] is None.
            override_languages: Optional per-file override to force stream["language"].

        Returns:
            List of (file_path, stream_id, language) tuples.
        """

        assert unique_keys and unique_keys[0] == "language", "First unique_key must be 'language'"

        stream_index = {}

        for path, details in StreamsPicker._iter_starting_with(files_details, best_file):
            for index, stream in enumerate(details.get(stream_type, [])):
                # Determine language
                lang = stream.get("language")
                if override_languages and path in override_languages:
                    original_lang = lang
                    lang = override_languages[path]
                    tid = stream["tid"]
                    if original_lang:
                        self.logger.info(f"Overriding {stream_type} stream #{tid} language {original_lang} with {lang} for file {path}")
                    else:
                        self.logger.info(f"Setting {stream_type} stream #{tid} language to {lang} for file {path}")
                elif (not lang) and fallback_languages and path in fallback_languages:
                    original_lang = lang
                    lang = fallback_languages[path]
                    tid = stream["tid"]
                    self.logger.info(f"Setting {stream_type} stream #{tid} language to {lang} for file {path}")

                # Build a modified copy of the stream for comparison
                stream_view = stream.copy()
                stream_view["language"] = lang

                # Build unique key based on stream view
                unique_key = tuple(stream_view.get(k) for k in unique_keys)
                language = unique_key[0] if unique_key else "default"

                current = {"file": path, "index": index, "details": stream}

                if unique_key not in stream_index:
                    stream_index[unique_key] = current
                    continue

                existing = stream_index[unique_key]
                existing_view = existing["details"].copy()
                existing_lang = existing_view.get("language")
                if override_languages and existing["file"] in override_languages:
                    existing_lang = override_languages[existing["file"]]
                elif (not existing_lang) and fallback_languages and existing["file"] in fallback_languages:
                    existing_lang = fallback_languages[existing["file"]]
                existing_view["language"] = existing_lang

                if preference_keys:
                    better = False
                    for key in preference_keys:
                        old = existing_view.get(key)
                        new = stream_view.get(key)
                        if old is None or new is None:
                            continue
                        if new > old:
                            better = True
                            break
                        elif new < old:
                            break
                    if better:
                        stream_index[unique_key] = current

        # Flatten result
        result = []
        for unique_key, entry in stream_index.items():
            index = entry["index"]
            path = entry["file"]
            language = unique_key[0]
            result.append((path, index, language))

        return result


    def pick_streams(self, files_details: Dict):
        # pick video stream
        video_stream = self._pick_best_video(files_details)
        video_stream_path = video_stream[0]

        # pick audio streams
        forced_audio_language = {path: self.duplicates_source.get_metadata_for(path).get("audio_lang") for path in files_details}
        forced_audio_language = {path: languages.unify_lang(lang) for path, lang in forced_audio_language.items() if lang}
        audio_streams = self._pick_unique_streams(files_details, video_stream_path, "audio", ["language", "channels"], ["sample_rate"], override_languages=forced_audio_language)

        # pick subtitle streams
        forced_subtitle_language = {path: self.duplicates_source.get_metadata_for(path).get("subtitle_lang") for path in files_details}
        forced_subtitle_language = {path: lang for path, lang in forced_subtitle_language.items() if lang}
        subtitle_streams = self._pick_unique_streams(files_details, video_stream_path, "subtitle", ["language"], [], override_languages=forced_subtitle_language)

        return [video_stream], audio_streams, subtitle_streams
