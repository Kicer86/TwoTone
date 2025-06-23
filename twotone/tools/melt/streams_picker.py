
import logging

from collections import defaultdict
from functools import cmp_to_key
from typing import Dict, List, Optional, Tuple

from ..utils import files_utils, language_utils
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

    @staticmethod
    def _pick_best_file_candidate(files_details: Dict[str, Dict]):
        """
            Function returns file with most streams.
        """

        filtered_videos = files_details.copy()
        for stream_type in ["video", "audio", "subtitle"]:
            max_videos = max(len(streams.get(stream_type, [])) for _, streams in filtered_videos.items())
            filtered_videos = {k: v for k, v in files_details.items() if len(v.get(stream_type, [])) == max_videos}

            if len(filtered_videos) == 1:
                return list(filtered_videos.keys())[0]

        # no specific candidate, return first
        return list(filtered_videos.keys())[0]

    def _pick_best_video(self, files_details: Dict[str, Dict]) -> List[Tuple[str, int, str]]:
        best_file = None
        best_stream = None

        # todo: handle many video streams
        default_video_stream = 0

        # todo: this shouldn't be there.
        # _pick_best_video should most likely be merged with _pick_unique_streams and most_promising_video should be provided from the caller
        most_promising_video = StreamsPicker._pick_best_file_candidate(files_details)

        for file, details in StreamsPicker._iter_starting_with(files_details, most_promising_video):
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

        paths_common_prefix = files_utils.get_common_prefix(files_details)

        def get_language(stream, path) -> str:
            printable_path = files_utils.get_printable_path(path, paths_common_prefix)
            lang = stream.get("language")
            if override_languages and path in override_languages:
                original_lang = lang
                lang = override_languages[path]
                tid = stream["tid"]
                if original_lang:
                    self.logger.info(f"Overriding {stream_type} stream #{tid} language {original_lang} with {lang} for file {printable_path}")
                else:
                    self.logger.info(f"Setting {stream_type} stream #{tid} language to {lang} for file {printable_path}")
            elif (not lang) and fallback_languages and path in fallback_languages:
                original_lang = lang
                lang = fallback_languages[path]
                tid = stream["tid"]
                self.logger.info(f"Setting {stream_type} stream #{tid} language to {lang} for file {printable_path}")

            return lang

        stream_index = defaultdict(lambda: defaultdict(list))

        # organize all streams by unique_key and file
        for path, details in StreamsPicker._iter_starting_with(files_details, best_file):
            for index, stream in enumerate(details.get(stream_type, [])):
                # Build a modified copy of the stream for comparison
                stream_view = stream.copy()

                # Determine language
                lang = get_language(stream, path)
                stream_view["language"] = lang

                # Build unique key based on stream view
                unique_key = tuple(stream_view.get(k) for k in unique_keys)
                language = unique_key[0] if unique_key else "default"

                current = {"file": path, "index": index, "details": stream_view}

                stream_index[unique_key][path].append(current)

        # process collected streams
        picked_streams = []
        for key, file_streams in stream_index.items():

            # from all files providing streams with given 'key' use those with most entries
            max_entries = max(len(details) for details in file_streams.values())
            files_with_most_entries = {k: v for k, v in file_streams.items() if len(v) == max_entries}

            if len(files_with_most_entries) == 1:
                first_file_streams = list(files_with_most_entries.values())[0]

                for stream in first_file_streams:
                    picked_streams.append((key, stream))
                continue

            # two or more files provide streams of the same uniqness. choose better ones
            def preference_sorting(lhs, rhs):
                lhs_details = lhs["details"]
                rhs_details = rhs["details"]

                for key in preference_keys:
                    lhs_value = lhs_details.get(key)
                    rhs_value = rhs_details.get(key)
                    if lhs_value is None or rhs is None:
                        continue
                    if lhs_value > rhs_value:
                        return -1
                    elif lhs_value < rhs_value:
                        return 1

                return 0

            # sort lists of details for each file
            file_streams = {k: sorted(v,  key=cmp_to_key(preference_sorting)) for k, v in file_streams.items()}

            # pick best details
            best_file_streams = max(
                file_streams.items(),
                key=cmp_to_key(lambda a, b: preference_sorting(a[1][0], b[1][0]))   # compare best ([0]) items for each value([1]) in dicts
            )

            for stream in best_file_streams[1]:
                picked_streams.append((key, stream))

        # Flatten result
        result = []
        for unique_key, entry in picked_streams:
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
        forced_audio_language = {path: language_utils.unify_lang(lang) for path, lang in forced_audio_language.items() if lang}
        audio_streams = self._pick_unique_streams(files_details, video_stream_path, "audio", ["language", "channels"], ["sample_rate"], override_languages=forced_audio_language)

        # pick subtitle streams
        forced_subtitle_language = {path: self.duplicates_source.get_metadata_for(path).get("subtitle_lang") for path in files_details}
        forced_subtitle_language = {path: lang for path, lang in forced_subtitle_language.items() if lang}
        subtitle_streams = self._pick_unique_streams(files_details, video_stream_path, "subtitle", ["language"], [], override_languages=forced_subtitle_language)

        return [video_stream], audio_streams, subtitle_streams
