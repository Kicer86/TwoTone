
import argparse
import cv2 as cv
import imagehash
import json
import logging
import numpy as np
import os
import re
import requests
import shutil
import tempfile

from collections import defaultdict
from overrides import override
from PIL import Image
from scipy.stats import entropy
from typing import Any, Callable, Dict, List, Tuple

from . import utils
from .tool import Tool
from .utils2 import process, video


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

                    fixed_paths = [self._fix_path(path) for path in paths]

                    # all names should be the same
                    same = all(x == names[0] for x in names)

                    if same:
                        name = names[0]
                        duplicates[name] = fixed_paths
                    else:
                        names_str = '\n'.join(names)
                        paths_str = '\n'.join(fixed_paths)
                        logging.warning(f"Different names for the same movie ({provider}: {id}):\n{names_str}.\nJellyfin files:\n{paths_str}\nSkipping.")

        return duplicates


class Melter():
    def __init__(self, logger, interruption: utils.InterruptibleProcess, duplicates_source: DuplicatesSource, live_run: bool):
        self.logger = logger
        self.interruption = interruption
        self.duplicates_source = duplicates_source
        self.live_run = live_run


    @staticmethod
    def _frame_entropy(path: str) -> float:
        pil_image = Image.open(path)
        image = np.array(pil_image.convert("L"))
        histogram, _ = np.histogram(image, bins=256, range=(0, 256))
        histogram = histogram / float(np.sum(histogram))
        e = entropy(histogram)
        return e


    @staticmethod
    def _filter_low_detailed(scenes: Dict[int, Dict]):
        valuable_scenes = { timestamp: info for timestamp, info in scenes.items() if Melter._frame_entropy(info["path"]) > 4}
        return valuable_scenes


    @staticmethod
    def _generate_phashes(scenes: Dict[int, Dict], since = None, to = None) -> Dict[int, Dict]:
        # this function's logic may not handle both statements in the way user expects
        assert since == None or to == None
        for timestamp, info in scenes.items():
            if (not since or timestamp >= since) and (not to or timestamp <= to):
                path = info["path"]
                img = Image.open(path).convert('L').resize((256, 256))
                img_hash = imagehash.phash(img, hash_size=16)
                info["hash"] = img_hash


    @staticmethod
    def _look_for_boundaries(lhs: Dict[int, Dict], rhs: Dict[int, Dict], first: Tuple[int, int], last: Tuple[int, int], cutoff: float):
        def _collect(data_set, first, last):
            front = []
            back = []
            for timestamp in data_set:
                if timestamp < first:
                    front.append(timestamp)
                elif timestamp > last:
                    back.append(timestamp)

            return front, back

        lhs_front_timestamps, lhs_back_timestamps = _collect(lhs, first[0], last[0])
        rhs_front_timestamps, rhs_back_timestamps = _collect(rhs, first[1], last[1])

        # update hashes
        Melter._generate_phashes(lhs, to = lhs_front_timestamps[-1])
        Melter._generate_phashes(rhs, to = rhs_front_timestamps[-1])
        Melter._generate_phashes(lhs, since = lhs_back_timestamps[0])
        Melter._generate_phashes(rhs, since = rhs_back_timestamps[0])

        l = len(lhs_front_timestamps) - 1
        r = len(rhs_front_timestamps) - 1


        def get_hash(indices, dataset, idx):
            if 0 <= idx < len(indices):
                timestamp = indices[idx]
                return dataset[timestamp]["hash"]
            return ImageHash()

        def find_common_frame(lhs_indices, rhs_indices, lhs_dataset, rhs_dataset, start_l, start_r, cutoff, direction=-1):
            """
            Finds a common frame between two video hash sequences.

            Args:
                lhs_indices, rhs_indices: Timestamp indices for the videos.
                lhs_dataset, rhs_dataset: Hash datasets for the videos.
                start_l, start_r: Starting indices (usually identified scene change).
                cutoff: Threshold for frame hash difference.
                direction: Direction of search (-1 for backward, +1 for forward).

            Returns:
                Tuple (l, r): indices of last common frame.
            """
            l, r = start_l, start_r
            last_matching_timestamps = ()

            while 0 <= l < len(lhs_indices) and 0 <= r < len(rhs_indices):
                current_vs_current = abs(get_hash(lhs_indices, lhs_dataset, l) - get_hash(rhs_indices, rhs_dataset, r))
                next_left_vs_current_right = abs(get_hash(lhs_indices, lhs_dataset, l + direction) - get_hash(rhs_indices, rhs_dataset, r))
                current_left_vs_next_right = abs(get_hash(lhs_indices, lhs_dataset, l) - get_hash(rhs_indices, rhs_dataset, r + direction))

                if current_vs_current <= min(next_left_vs_current_right, current_left_vs_next_right) and current_vs_current <= cutoff:
                    l += direction
                    r += direction
                    last_matching_timestamps = (lhs_indices[l], rhs_indices[r])
                elif current_left_vs_next_right < next_left_vs_current_right and current_left_vs_next_right <= cutoff:
                    r += direction
                    last_matching_timestamps = (lhs_indices[l], rhs_indices[r])
                elif next_left_vs_current_right <= cutoff:
                    l += direction
                    last_matching_timestamps = (lhs_indices[l], rhs_indices[r])
                else:
                    break

            return last_matching_timestamps

        first_common_frame = find_common_frame(
            lhs_front_timestamps, rhs_front_timestamps, lhs, rhs,
            start_l = len(lhs_front_timestamps) - 1,
            start_r = len(rhs_front_timestamps) - 1,
            cutoff = cutoff,
            direction = -1
        )

        last_common_frame = find_common_frame(
            lhs_back_timestamps, rhs_back_timestamps, lhs, rhs,
            start_l = 0,
            start_r = 0,
            cutoff = cutoff,
            direction = +1
        )

        print(f"First: L: {lhs[first_common_frame[0]]["path"]} R: {rhs[first_common_frame[1]]["path"]}")
        print(f"Last:  L: {lhs[last_common_frame[0]]["path"]} R: {rhs[last_common_frame[1]]["path"]}")

        return first_common_frame, last_common_frame


    @staticmethod
    def _match_pairs(lhs: Dict[int, Dict], rhs: Dict[int, Dict]):
        # calculate PHashes
        Melter._generate_phashes(lhs)
        Melter._generate_phashes(rhs)

        # Generate initial set of candidates using generated phashes
        pairs_candidates = defaultdict(list)

        for lhs_timestamp, lhs_info in lhs.items():
            lhs_hash = lhs_info["hash"]
            for rhs_timestamp, rhs_info in rhs.items():
                rhs_hash = rhs_info["hash"]
                distance = abs(lhs_hash - rhs_hash)
                pairs_candidates[lhs_timestamp].append((distance, rhs_timestamp))

        # Pick best candidates
        best_candidates = []
        used_rhs_timestamps = set()
        for lhs_timestamp, candidates in pairs_candidates.items():
            candidates.sort()

            # look for first unused candidate
            for candidate in candidates:
                diff, best_rhs_candidate = candidate

                if best_rhs_candidate not in used_rhs_timestamps:
                    # mark as used
                    used_rhs_timestamps.add(best_rhs_candidate)

                    # append diff, lhs timestamp, rhs timestamp
                    best_candidates.append((diff, lhs_timestamp, best_rhs_candidate))
                    break

        # sort best candidates by diff
        best_candidates.sort()

        #find median to know where to cut off
        m = len(best_candidates) // 2
        cutoff = best_candidates[m][0]
        best_candidates = [c for c in best_candidates if c[0] <= cutoff]

        # build pairs structure
        pairs = [(candidate[1], candidate[2]) for candidate in best_candidates]

        pairs.sort()
        print([(lhs[pair[0]]["path"], rhs[pair[1]]["path"]) for pair in pairs])

        return pairs


    @staticmethod
    def _match_scenes(lhs_scenes: Dict[int, Any], rhs_scenes: Dict[int, Any], comp: Callable[[Any, Any], bool]) -> List[Tuple[int, int]]:
        # O^2 solution, but it should do
        matches = []

        for lhs_timestamp, lhs_info in lhs_scenes.items():
            lhs_hash = lhs_info["hash"]
            for rhs_timestamp, rhs_info in rhs_scenes.items():
                rhs_hash = rhs_info["hash"]
                if comp(lhs_hash, rhs_hash):
                    matches.append((lhs_timestamp, rhs_timestamp))
                    break

        return matches


    @staticmethod
    def _get_frames_for_timestamps(timestamps: List[int], frames_info: Dict[int, Dict]) -> List[str]:
        frame_files = {timestamp: info for timestamp, info in frames_info.items() if timestamp in timestamps}

        return frame_files


    def _create_segments_mapping(self, lhs: str, rhs: str) -> List[Tuple[int, int]]:
        with tempfile.TemporaryDirectory() as wd:
            lhs_scene_changes = video.detect_scene_changes(lhs, threshold=0.3)
            rhs_scene_changes = video.detect_scene_changes(rhs, threshold=0.3)

            if len(lhs_scene_changes) == 0 or len(rhs_scene_changes) == 0:
                return

            lhs_wd = os.path.join(wd, "lhs")
            rhs_wd = os.path.join(wd, "rhs")

            lhs_all_wd = os.path.join(lhs_wd, "all")
            rhs_all_wd = os.path.join(rhs_wd, "all")
            lhs_key_wd = os.path.join(lhs_wd, "key")
            rhs_key_wd = os.path.join(rhs_wd, "key")

            for d in [lhs_wd,
                      rhs_wd,
                      lhs_all_wd,
                      rhs_all_wd,
                      lhs_key_wd,
                      rhs_key_wd]:
                os.makedirs(d)

            # extract all scenes
            lhs_all_frames = video.extract_all_frames(lhs, lhs_all_wd, scale = 0.5) # (256,256))
            rhs_all_frames = video.extract_all_frames(rhs, rhs_all_wd, scale = 0.5) # (256,256))

            # extract key frames (as 'key' a scene change frame is meant)
            lhs_key_frames = Melter._get_frames_for_timestamps(lhs_scene_changes, lhs_all_frames)
            rhs_key_frames = Melter._get_frames_for_timestamps(rhs_scene_changes, rhs_all_frames)

            # copy key frames
            for src, dst in [ (lhs_key_frames, lhs_key_wd), (rhs_key_frames, rhs_key_wd) ]:
                for src_info in src.values():
                    shutil.copy2(src_info["path"], dst)

            # pick frames with descent entropy (remove single color frames etc)
            #lhs_useful_key_frames = Melter._filter_low_detailed(lhs_key_frames)
            #rhs_useful_key_frames = Melter._filter_low_detailed(rhs_key_frames)

            # find matching pairs
            matching_pairs = Melter._match_pairs(lhs_key_frames, rhs_key_frames)

            # try to locate first and last common frames
            Melter._look_for_boundaries(lhs_all_frames, rhs_all_frames, matching_pairs[0], matching_pairs[-1], 80)

            # calculate finerprint for each frame
            lhs_hashes = Melter._generate_hashes(lhs_key_frames)
            rhs_hashes = Melter._generate_hashes(rhs_key_frames)

            # find similar scenes
            hash_algo = cv.img_hash.BlockMeanHash().create()
            matching_scenes = Melter._match_scenes(lhs_hashes, rhs_hashes, lambda l, r: hash_algo.compare(l, r) < 20)

            matching_files = [(lhs_key_frames[lhs_timestamp]["path"], rhs_key_frames[rhs_timestamp]["path"])  for lhs_timestamp, rhs_timestamp in matching_scenes]

            return matching_scenes


    def _process_duplicates(self, files: List[str]):
        mapping = self._create_segments_mapping(files[0], files[1])
        return


        video_details = [video.get_video_data2(video_file) for video_file in files]
        video_lengths = {video.video_tracks[0].length for video in video_details}

        if len(video_lengths) == 1:
            # all files are of the same lenght
            # remove all but first one
            logging.info("Removing exact duplicates. Leaving one copy")
            if self.live_run:
                for file in files[1:]:
                    os.remove(file)
        else:
            logging.warning("Videos have different lengths, skipping")


    def _process_duplicates_set(self, duplicates: Dict[str, List[str]]):
        for title, files in duplicates.items():
            logging.info(f"Analyzing duplicates for {title}")

            self._process_duplicates(files)


    def melt(self):
        self.logger.info("Finding duplicates")
        duplicates = self.duplicates_source.collect_duplicates()
        self._process_duplicates_set(duplicates)
        #print(json.dumps(duplicates, indent=4))


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
