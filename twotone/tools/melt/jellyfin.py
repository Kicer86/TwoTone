
import json
import logging
import os
import time
import requests

from collections import defaultdict
from overrides import override
from typing import Dict, List, Tuple

from ..utils import generic_utils
from .duplicates_source import DuplicatesSource


logger = logging.getLogger("TwoTone.melt.jellyfin")


class JellyfinSource(DuplicatesSource):
    def __init__(self, interruption: generic_utils.InterruptibleProcess, url: str, token: str, path_fix: Tuple[str, str] | None) -> None:
        super().__init__(interruption)

        self.url = url
        self.token = token
        self.path_fix = path_fix
        self.tmdb_id_by_path: Dict[str, str] = {}

        cache_dir = generic_utils.get_twotone_working_dir()
        os.makedirs(cache_dir, exist_ok=True)
        self.tmdb_cache_file = os.path.join(cache_dir, "tmdb_cache.json")
        try:
            with open(self.tmdb_cache_file, "r", encoding="utf-8") as f:
                self.tmdb_cache = json.load(f)
        except FileNotFoundError:
            self.tmdb_cache = {}

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
                logger.error(f"Could not replace \"{pfrom}\" in \"{path}\"")

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
                            logger.warning(f"Some paths for title {name} are not files:")
                            for path in fixed_paths:
                                logger.warning(f"\t{path}")
                            logger.warning("Skipping title")
                        else:
                            duplicates[name] = fixed_paths
                    else:
                        names_str = '\n'.join(names)
                        paths_str = '\n'.join(fixed_paths)
                        logger.warning(f"Different names for the same movie ({provider}: {id}):\n{names_str}.\nJellyfin files:\n{paths_str}\nSkipping.")

        return duplicates

    @override
    def get_metadata_for(self, path: str) -> Dict[str, str]:
        tmdb_id = self.tmdb_id_by_path.get(path)

        if not tmdb_id:
            return {"audio_prod_lang": None}

        if tmdb_id in self.tmdb_cache:
            return {"audio_prod_lang": self.tmdb_cache[tmdb_id]}

        if not self.tmdb_api_key:
            logging.error("TMDB_API_KEY not set")
            return {"audio_prod_lang": None}

        now = time.time()
        delta = now - self.last_tmdb_request
        if delta < 0.3:
            wait = 0.3 - delta
            logging.warning(f"TMDB API limit reached. Waiting {wait:.2f}s.")
            time.sleep(wait)

        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": self.tmdb_api_key}
        )
        self.last_tmdb_request = time.time()

        if response.status_code != 200:
            logging.error(f"TMDB request failed for id {tmdb_id}: {response.status_code}")
            return {"audio_prod_lang": None}

        data = response.json()
        lang = data.get("original_language")
        self.tmdb_cache[tmdb_id] = lang
        try:
            with open(self.tmdb_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.tmdb_cache, f)
        except OSError as exc:
            logging.warning(f"Could not write TMDB cache: {exc}")

        return {"audio_prod_lang": lang}
