
import logging
import os
import re

from collections import defaultdict
from functools import cmp_to_key
from typing import Any, Dict, Generator, List, Optional, Tuple

from ..utils import files_utils, language_utils
from ..utils import subtitles_utils
from .duplicates_source import DuplicatesSource

# precompiled regex for fast language guessing
_RE_LANG_ALL = re.compile(r"(?i)(?:([a-z]{2,3})(?=dub))|(?<![a-z])([a-z]{2,3})(?![a-z])")

class StreamsPicker:
    def __init__(self, logger: logging.Logger, duplicates_source: DuplicatesSource, wd: str, allow_language_guessing: bool = False):
        self.logger = logger
        self.duplicates_source = duplicates_source
        self.allow_language_guessing = allow_language_guessing
        self.wd = wd


    @staticmethod
    def _iter_starting_with(d: dict, start_key) -> Generator[Any, Any, Any]:
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
        files_ids: Dict[str, int],
        stream_type: str,
        unique_keys: List[str],
        preference,
        get_language
    ) -> List[Tuple[str, int, Optional[str]]]:
        """Pick best streams of ``stream_type`` from ``files_details``.

        ``unique_keys`` determines the grouping for uniqueness. ``preference`` is
        a callable accepting two stream dictionaries and returning ``-1`` if the
        first argument should be preferred, ``1`` if the second is better and
        ``0`` otherwise.
        ``get_language`` is a functor returning language for given stream and path.
        """

        stream_index = defaultdict(lambda: defaultdict(list))

        # organize all streams by unique_key and file
        for path, details in StreamsPicker._iter_starting_with(files_details, best_file):
            for stream in details.get(stream_type, []):
                # Build a modified copy of the stream for comparison
                stream_view = stream.copy()

                # Determine language if available
                lang = get_language(stream, stream_type, path)
                stream_view["language"] = lang

                # Build unique key based on stream view
                unique_key = tuple(stream_view.get(k) for k in unique_keys)

                # all components of key are valid?
                key_is_valid = all(c for c in unique_key)

                if not key_is_valid:
                    missing_properties = [name for name, value in zip(unique_keys, unique_key) if value is None]
                    missing_properties_str = ", ".join(missing_properties)

                    id = files_ids[path]
                    raise RuntimeError(f"Could not properly build stream information of type {stream_type} for file #{id}. "
                                       f"Missing properties: {missing_properties_str}")

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


    def pick_streams(self, files_details: Dict, ids: Dict[str, int]):
        # video preference comparator
        def video_cmp(lhs: Dict, rhs: Dict) -> int:
            if lhs["width"] > rhs["width"] and lhs["height"] > rhs["height"]:
                return 1
            if lhs["width"] < rhs["width"] and lhs["height"] < rhs["height"]:
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

        def get_language(
                stream,
                stream_type,
                path,
                override_languages : Optional[Dict[str, str]] = None,
                fallback_languages : Optional[Dict[str, str]] = None) -> Optional[str]:
            id = ids[path]
            lang = stream.get("language")

            if lang == "und":
                lang = None

            if override_languages and path in override_languages:
                original_lang = lang
                lang = override_languages[path]
                tid = stream.get("tid")
                if original_lang:
                    self.logger.info(f"Overriding {stream_type} stream #{tid} language {original_lang} with {lang} for file #{id}")
                else:
                    self.logger.info(f"Setting {stream_type} stream #{tid} language to {lang} for file #{id}")
            elif (not lang) and fallback_languages and path in fallback_languages:
                original_lang = lang
                lang = fallback_languages[path]
                tid = stream.get("tid")
                self.logger.info(f"Setting {stream_type} stream #{tid} language to {lang} for file #{id}")
            elif self.allow_language_guessing and lang is None and stream_type == "audio":
                _, stem, _ = files_utils.split_path(path)
                file_name_low = stem.lower()

                best = None
                best_rank = 99
                for m in _RE_LANG_ALL.finditer(file_name_low):
                    for gi, val in enumerate(m.groups(), start=1):
                        if val and language_utils.is_valid_lang_code(val):
                            rank = gi
                            cand = val
                            if rank < best_rank:
                                best_rank = rank
                                best = cand
                            break

                if best:
                    try:
                        lang = language_utils.unify_lang(best)
                        lang_name = language_utils.language_name(lang)
                        self.logger.warning(f"Guessed audio language: {lang_name} for file #{id}")
                    except Exception:
                        pass
            elif lang is None and stream_type == "subtitle":
                # Extract subtitle stream to a temporary file via subtitles_utils
                # and guess language using its utilities.
                tid = stream.get("tid")
                base_tmp = os.path.join(self.wd, "tmp_subtitle")
                tid_to_path = subtitles_utils.extract_subtitle_to_temp(path, [tid], base_tmp, logger=self.logger)
                tmp_path = tid_to_path.get(tid)

                if tmp_path:
                    try:
                        encoding = subtitles_utils.file_encoding(tmp_path)
                        detected_lang = subtitles_utils.guess_language(tmp_path, encoding)
                        if detected_lang:
                            try:
                                lang = language_utils.unify_lang(detected_lang)
                                lang_name = language_utils.language_name(lang)
                                self.logger.warning(f"Detected subtitle language: {lang_name} for file #{id}, track #{tid}")
                            except Exception:
                                pass
                    except Exception as e:
                        self.logger.debug(f"Subtitle language detection failed for file #{id}, tid {tid}: {e}")

            return lang

        #collect video streams (path and index) which are attached_pics so we can drop them later as not handled now
        attached_pics = [(file_path, index) for (file_path, details) in files_details.items() for index, vd in enumerate(details["video"]) if vd.get("attached_pic", False)]

        best_file_candidate = StreamsPicker._pick_best_file_candidate(files_details)
        video_streams = self._pick_streams(
            files_details,
            best_file_candidate,
            ids,
            "video",
            [],
            video_cmp,
            get_language = lambda stream, stream_type, file: get_language(stream, stream_type, file)
        )
        video_streams = [video_stream for video_stream in video_streams if (video_stream[0], video_stream[1]) not in attached_pics]
        video_stream = video_streams[0]
        video_stream_path = video_stream[0]

        # pick audio streams
        forced_audio_language = {path: self.duplicates_source.get_metadata_for(path).get("audio_lang") for path in files_details}
        forced_audio_language = {path: language_utils.unify_lang(lang) for path, lang in forced_audio_language.items() if lang}
        audio_streams = self._pick_streams(
            files_details,
            video_stream_path,
            ids,
            "audio",
            ["language", "channels"],
            cmp_by_keys(["sample_rate"]),
            get_language = lambda stream, stream_type, file: get_language(stream, stream_type, file, override_languages = forced_audio_language)
        )

        # pick subtitle streams
        forced_subtitle_language = {path: self.duplicates_source.get_metadata_for(path).get("subtitle_lang") for path in files_details}
        forced_subtitle_language = {path: lang for path, lang in forced_subtitle_language.items() if lang}
        subtitle_streams = self._pick_streams(
            files_details,
            video_stream_path,
            ids,
            "subtitle",
            ["language"],
            lambda a, b: 0,
            get_language = lambda stream, stream_type, file: get_language(stream, stream_type, file, override_languages = forced_subtitle_language)
        )

        # results
        return video_streams, audio_streams, subtitle_streams
