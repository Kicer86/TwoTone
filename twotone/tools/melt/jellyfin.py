
import logging
import os
import time
import requests

from collections import defaultdict
from overrides import override
from typing import Dict, List, Tuple

from ..utils import generic_utils, language_utils
from ..utils.tmdb_cache import TmdbCache
from .duplicates_source import DuplicatesSource


class JellyfinSource(DuplicatesSource):
    def __init__(self, interruption: generic_utils.InterruptibleProcess, url: str, token: str, path_fix: Tuple[str, str] | None, logger: logging.Logger | None = None) -> None:
        super().__init__(interruption)

        self.url = url
        self.token = token
        self.path_fix = path_fix
        # allow injecting a logger for better control in callers/tests
        self.logger = logger or logging.getLogger(__name__)
        self.tmdb_id_by_path: Dict[str, str] = {}
        self.tmdb_cache = TmdbCache(logger=self.logger)

        self.last_tmdb_request: float = 0.0
        self.tmdb_api_key = os.getenv("TMDB_API_KEY")

    def _fix_path(self, path: str) -> str:
        fixed_path = path
        if self.path_fix:
            pfrom = self.path_fix[0]
            pto = self.path_fix[1]

            if path.startswith(pfrom):
                fixed_path = f"{pto}{path[len(pfrom):]}"
            else:
                self.logger.error(f"Could not replace \"{pfrom}\" in \"{path}\"")

        return fixed_path


    @override
    def collect_duplicates(self) -> Dict[str, List[str]]:
        endpoint = f"{self.url}"
        headers = {
            "X-Emby-Token": self.token
        }

        paths_by_id = defaultdict(lambda: defaultdict(list))

        def fetchItems(params: Dict[str, str] = {}) -> None:
            self.interruption._check_for_stop()
            params.update({"fields": "Path,ProviderIds"})

            response = requests.get(endpoint + "/Items", headers=headers, params=params)
            if response.status_code != 200:
                raise RuntimeError("No access")

            responseJson = response.json()
            items = responseJson["Items"]

            for item in items:
                name = item["Name"]
                id = item["Id"]
                type = item["Type"]

                if type == "Folder":
                    fetchItems(params={"parentId": id})
                elif type == "Movie":
                    providers = item["ProviderIds"]
                    path = item["Path"]
                    fixed_path = self._fix_path(path)

                    tmdb_id = providers.get("Tmdb")
                    if tmdb_id:
                        self.tmdb_id_by_path[fixed_path] = tmdb_id

                    for provider, id in providers.items():
                        # ignore collection ID
                        if provider != "TmdbCollection":
                            paths_by_id[provider][id].append((name, fixed_path))

        fetchItems()
        duplicates = {}

        for provider, ids in paths_by_id.items():
            for id, data in ids.items():
                if len(data) > 1:
                    names, fixed_paths = zip(*data)

                    # all names should be the same
                    same = all(x == names[0] for x in names)

                    if same:
                        all_paths_are_files = all(os.path.isfile(path) for path in fixed_paths)
                        name = names[0]

                        if not all_paths_are_files:
                            self.logger.warning(f"Some paths for title {name} are not files:")
                            for path in fixed_paths:
                                self.logger.warning(f"\t{path}")
                            self.logger.warning("Skipping title")
                        else:
                            duplicates[name] = fixed_paths
                    else:
                        names_str = '\n'.join(names)
                        paths_str = '\n'.join(fixed_paths)
                        self.logger.warning(f"Different names for the same movie ({provider}: {id}):\n{names_str}.\nJellyfin files:\n{paths_str}\nSkipping.")

        return duplicates

    @override
    def get_metadata_for(self, path: str) -> Dict[str, str]:
        tmdb_id = self.tmdb_id_by_path.get(path)

        if not tmdb_id:
            return {"audio_prod_lang": None}

        cached_lang = self.tmdb_cache.get(tmdb_id, "audio_prod_lang")
        if cached_lang is not None:
            return {"audio_prod_lang": cached_lang}

        if not self.tmdb_api_key:
            self.logger.error("TMDB_API_KEY not set")
            return {"audio_prod_lang": None}

        now = time.time()
        delta = now - self.last_tmdb_request
        if delta < 0.3:
            wait = 0.3 - delta
            self.logger.warning(f"TMDB API limit reached. Waiting {wait:.2f}s.")
            time.sleep(wait)

        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": self.tmdb_api_key}
        )
        self.last_tmdb_request = time.time()

        if response.status_code != 200:
            self.logger.error(f"TMDB request failed for id {tmdb_id}: {response.status_code}")
            return {"audio_prod_lang": None}

        data = response.json()
        lang = data.get("original_language")
        lang = language_utils.unify_lang(lang) if lang else None
        self.tmdb_cache.set(tmdb_id, "audio_prod_lang", lang)

        return {"audio_prod_lang": lang}
