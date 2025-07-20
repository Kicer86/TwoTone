
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


    def _pick_streams(
        self,
        files_details: Dict[str, Dict],
        best_file: str,
        stream_type: str,
        unique_keys: List[str],
        preference,
        fallback_languages: Optional[Dict[str, str]] = None,
        override_languages: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, int, Optional[str]]]:
        """Pick best streams of ``stream_type`` from ``files_details``.

        ``unique_keys`` determines the grouping for uniqueness. ``preference`` is
        a callable accepting two stream dictionaries and returning ``-1`` if the
        first argument should be preferred, ``1`` if the second is better and
        ``0`` otherwise.
        """

        paths_common_prefix = files_utils.get_common_prefix(files_details)

        def get_language(stream, path) -> Optional[str]:
            printable_path = files_utils.get_printable_path(path, paths_common_prefix)
            lang = stream.get("language")
            if override_languages and path in override_languages:
                original_lang = lang
                lang = override_languages[path]
                tid = stream.get("tid")
                if original_lang:
                    self.logger.info(
                        f"Overriding {stream_type} stream #{tid} language {original_lang} with {lang} for file {printable_path}"
                    )
                else:
                    self.logger.info(
                        f"Setting {stream_type} stream #{tid} language to {lang} for file {printable_path}"
                    )
            elif (not lang) and fallback_languages and path in fallback_languages:
                original_lang = lang
                lang = fallback_languages[path]
                tid = stream.get("tid")
                self.logger.info(
                    f"Setting {stream_type} stream #{tid} language to {lang} for file {printable_path}"
                )

            return lang

        stream_index = defaultdict(lambda: defaultdict(list))

        # organize all streams by unique_key and file
        for path, details in StreamsPicker._iter_starting_with(files_details, best_file):
            for stream in details.get(stream_type, []):
                # Build a modified copy of the stream for comparison
                stream_view = stream.copy()

                # Determine language if available
                lang = get_language(stream, path)
                if lang is not None:
                    stream_view["language"] = lang

                # Build unique key based on stream view
                unique_key = tuple(stream_view.get(k) for k in unique_keys)

                # put tid into top layer for easier access
                tid = stream_view["tid"]

                current = {"file": path, "tid": tid, "details": stream_view}

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
                return preference(lhs["details"], rhs["details"])

            # sort lists of details for each file
            file_streams = {k: sorted(v, key=cmp_to_key(preference_sorting)) for k, v in file_streams.items()}

            # pick best details
            best_file_streams = max(
                file_streams.items(),
                key=cmp_to_key(lambda a, b: preference_sorting(a[1][0], b[1][0]))
            )

            for stream in best_file_streams[1]:
                picked_streams.append((key, stream))

        # Flatten result
        result = []
        for unique_key, entry in picked_streams:
            tid = entry["tid"]
            path = entry["file"]
            language = entry["details"].get("language")
            result.append((path, tid, language))

        return result


    def pick_streams(self, files_details: Dict):
        # video preference comparator
        def video_cmp(lhs: Dict, rhs: Dict) -> int:
            if lhs.get("width") > rhs.get("width") and lhs.get("height") > rhs.get("height"):
                return 1
            if lhs.get("width") < rhs.get("width") and lhs.get("height") < rhs.get("height"):
                return -1

            lhs_fps = eval(str(lhs.get("fps", "0")))
            rhs_fps = eval(str(rhs.get("fps", "0")))

            if lhs_fps > rhs_fps:
                return 1
            if lhs_fps < rhs_fps:
                return -1

            return 0

        # comparator based on keys
        def cmp_by_keys(keys):
            def _cmp(lhs: Dict, rhs: Dict) -> int:
                for key in keys:
                    lhs_value = lhs.get(key)
                    rhs_value = rhs.get(key)
                    if lhs_value is None or rhs_value is None:
                        continue
                    if lhs_value > rhs_value:
                        return -1
                    if lhs_value < rhs_value:
                        return 1
                return 0
            return _cmp

        #collect video streams (path and index) which are attached_pics so we can drop them later as not handled now
        attached_pics = [(file_path, index) for (file_path, details) in files_details.items() for index, vd in enumerate(details["video"]) if vd.get("attached_pic", False)]

        best_file_candidate = StreamsPicker._pick_best_file_candidate(files_details)
        video_streams = self._pick_streams(files_details, best_file_candidate, "video", [], video_cmp)
        video_streams = [video_stream for video_stream in video_streams if (video_stream[0], video_stream[1]) not in attached_pics]
        video_stream = video_streams[0]
        video_stream_path = video_stream[0]

        # pick audio streams
        forced_audio_language = {path: self.duplicates_source.get_metadata_for(path).get("audio_lang") for path in files_details}
        forced_audio_language = {path: language_utils.unify_lang(lang) for path, lang in forced_audio_language.items() if lang}
        audio_streams = self._pick_streams(
            files_details,
            video_stream_path,
            "audio",
            ["language", "channels"],
            cmp_by_keys(["sample_rate"]),
            override_languages=forced_audio_language,
        )

        # pick subtitle streams
        forced_subtitle_language = {path: self.duplicates_source.get_metadata_for(path).get("subtitle_lang") for path in files_details}
        forced_subtitle_language = {path: lang for path, lang in forced_subtitle_language.items() if lang}
        subtitle_streams = self._pick_streams(
            files_details,
            video_stream_path,
            "subtitle",
            ["language"],
            lambda a, b: 0,
            override_languages=forced_subtitle_language,
        )

        # results
        return video_streams, audio_streams, subtitle_streams
