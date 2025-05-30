
import argparse
import json
import logging
import re
import requests

from collections import defaultdict
from overrides import override
from typing import Dict, List, Tuple

from . import utils
from .tool import Tool


class DuplicatesSource:
    def __init__(self, interruption: utils.InterruptibleProcess):
        self.interruption = interruption

    def collect_duplicates(self) -> Dict[str, List[str]]:
        pass


def _split_path_fix(value: str) -> List[str]:
    pattern = r'"((?:[^"\\]|\\.)*?)"'

    matches = re.findall(pattern, value)
    return [match.replace(r'\"', '"') for match in matches]


class JellyfinSource(DuplicatesSource):
    def __init__(self, interruption: utils.InterruptibleProcess, url: str, token: str, path_fix: Tuple[str, str]):
        super().__init__(interruption)

        self.url = url
        self.token = token
        self.path_fix = path_fix

    def _fix_path(self, path: str) -> str:
        fixed_path = path
        if self.path_fix:
            pfrom = self.path_fix[0]
            pto = self.path_fix[1]

            if path.startswith(pfrom):
                fixed_path = f"{pto}{path[len(pfrom):]}"
            else:
                logging.error(f"Could not replace \"{pfrom}\" in \"{path}\"")

        return fixed_path


    def collect_duplicates(self) -> Dict[str, List[str]]:
        endpoint = f"{self.url}"
        headers = {
            "X-Emby-Token": self.token
        }

        paths_by_id = defaultdict(lambda: defaultdict(list))

        def fetchItems(params: Dict[str, str] = {}):
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

                    for provider, id in providers.items():
                        # ignore collection ID
                        if provider != "TmdbCollection":
                            paths_by_id[provider][id].append((name, path))

        fetchItems()
        duplicates = {}

        for provider, ids in paths_by_id.items():
            for id, data in ids.items():
                if len(data) > 1:
                    names, paths = zip(*data)

                    # all names should be the same
                    same = all(x == names[0] for x in names)

                    if same:
                        fixed_paths = [self._fix_path(path) for path in paths]

                        name = names[0]
                        duplicates[name] = fixed_paths
                    else:
                        names_str = '\n'.join(names)
                        paths_str = '\n'.join(fixed_paths)
                        logging.warning(f"Different names for the same movie ({provider}: {id}):\n{names_str}.\nJellyfin files:\n{paths_str}\nSkipping.")

        return duplicates


class Melter():
    def __init__(self, logger, interruption: utils.InterruptibleProcess, duplicates_source: DuplicatesSource):
        self.logger = logger
        self.interruption = interruption
        self.duplicates_source = duplicates_source


    def _process_duplicates(self, duplicates: Dict[str, List[str]]):
        pass


    def melt(self):
        self.logger.info("Finding duplicates")
        duplicates = self.duplicates_source.collect_duplicates()
        self._process_duplicates(duplicates)
        print(json.dumps(duplicates, indent=4))


class RequireJellyfinServer(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, "jellyfin_server", None) is None:
            parser.error(
                f"{option_string} requires --jellyfin-server to be specified")
        setattr(namespace, self.dest, values)


class MeltTool(Tool):
    @override
    def setup_parser(self, parser: argparse.ArgumentParser):
        jellyfin_group = parser.add_argument_group("Jellyfin source")
        jellyfin_group.add_argument('--jellyfin-server',
                                    help='URL to the Jellyfin server which will be used as a source of video files duplicates')
        jellyfin_group.add_argument('--jellyfin-token',
                                    action=RequireJellyfinServer,
                                    help='Access token (http://server:8096/web/#/dashboard/keys)')
        jellyfin_group.add_argument('--jellyfin-path-fix',
                                    action=RequireJellyfinServer,
                                    help='Specify a replacement pattern for file paths to ensure "melt" can access Jellyfin video files.\n\n'
                                        '"Melt" requires direct access to video files. If Jellyfin is not running on the same machine as "melt,"\n'
                                        'you must set up network access to Jellyfin’s video storage and specify how paths should be resolved.\n\n'
                                        'For example, suppose Jellyfin runs on a Linux machine where the video library is stored at "/srv/videos" (a shared directory).\n'
                                        'If "melt" is running on another Linux machine that accesses this directory remotely at "/mnt/shared_videos,"\n'
                                        'you need to map "/srv/videos" (Jellyfin’s path) to "/mnt/shared_videos" (the path accessible on the machine running "melt").\n\n'
                                        'In this case, use: --jellyfin-path-fix "/srv/videos","/mnt/shared_videos" to define the replacement pattern.')


    @override
    def run(self, args, no_dry_run: bool, logger: logging.Logger):
        interruption = utils.InterruptibleProcess()

        data_source = None
        if args.jellyfin_server:
            path_fix = _split_path_fix(args.jellyfin_path_fix) if args.jellyfin_path_fix else None

            if path_fix and len(path_fix) != 2:
                raise ValueError(f"Invalid content for --jellyfin-path-fix argument. Got: {path_fix}")

            data_source = JellyfinSource(interruption=interruption,
                                         url=args.jellyfin_server,
                                         token=args.jellyfin_token,
                                         path_fix=path_fix)

        melter = Melter(logger, interruption, data_source, live_run = no_dry_run)
        melter.melt()
