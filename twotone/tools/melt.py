
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
from .utils2 import files, process, video


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
        histogram, _ = np.histogram(image, bins = 256, range=(0, 256))
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
                img = Image.open(path)
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
            return imagehash.ImageHash()

        def find_common_frame(lhs_indices, rhs_indices, lhs_dataset, rhs_dataset, start_l, start_r, cutoff, direction=-1):
            """
            Finds a common frame between two video hash sequences considering up to two frames in the given direction.

            Args:
                lhs_indices, rhs_indices: Timestamp indices for the videos.
                lhs_dataset, rhs_dataset: Hash datasets for the videos.
                start_l, start_r: Starting indices (usually identified scene change).
                cutoff: Threshold for frame hash difference.
                direction: Direction of search (-1 for backward, +1 for forward).

            Returns:
                Tuple (lhs_timestamp, rhs_timestamp): timestamps of last common frame.
            """
            l, r = start_l, start_r
            last_matching_timestamps = (lhs_indices[l], rhs_indices[r])

            offsets = [direction, 2 * direction]

            while 0 <= l < len(lhs_indices) and 0 <= r < len(rhs_indices):
                min_diff = float('inf')
                best_candidates = (0, 0)

                for dl in offsets:
                    for dr in offsets:
                        if dl == 0 and dr == 0:
                            continue

                        lhs_candidate = l + dl
                        rhs_candidate = r + dr
                        lh = get_hash(lhs_indices, lhs_dataset, lhs_candidate)
                        rh = get_hash(rhs_indices, rhs_dataset, rhs_candidate)
                        diff = abs(lh - rh)

                        if diff < min_diff:
                            min_diff = diff
                            best_candidates = (lhs_candidate, rhs_candidate)

                if min_diff <= cutoff:
                    l = best_candidates[0]
                    r = best_candidates[1]

                    if 0 <= l < len(lhs_indices) and 0 <= r < len(rhs_indices):
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
    def _generate_matching_frames(lhs: Dict[int, Dict], rhs: Dict[int, Dict]):
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

        # find median to know where to cut off
        m = len(best_candidates) // 2
        cutoff = best_candidates[m][0]
        best_candidates = [c for c in best_candidates if c[0] <= cutoff]

        # build pairs structure
        pairs = [(candidate[1], candidate[2]) for candidate in best_candidates]

        return pairs


    @staticmethod
    def _match_pairs(lhs: Dict[int, Dict], rhs: Dict[int, Dict]):
        pairs = Melter._generate_matching_frames(lhs, rhs)

        pairs.sort()
        print([(lhs[pair[0]]["path"], rhs[pair[1]]["path"]) for pair in pairs])

        # validate pace
        prev_pair = None
        pace = []
        for pair in pairs:
            if prev_pair:
                diff = (pair[0] - prev_pair[0], pair[1] - prev_pair[1])
                pace.append(diff[0]/diff[1])

            prev_pair = pair

        print(pace)

        return pairs

    @staticmethod
    def _find_most_matching_pair(lhs: Dict[int, Dict], rhs: Dict[int, Dict]):
        pairs = Melter._generate_matching_frames(lhs, rhs)

        return pairs[0]

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


    @staticmethod
    def _replace_path(frames_info: Dict[int, Dict], dir: str) -> Dict[int, Dict]:
        result = {}
        for timestamp, info in frames_info.items():
            path = info["path"]
            _, file, ext = files.split_path(path)
            new_path = os.path.join(dir, file + "." + ext)

            result[timestamp] = info
            result[timestamp]["path"] = new_path

        return result


    @staticmethod
    def _normalize_frames(frames_info: Dict[int, Dict], wd: str) -> Dict[int, str]:
        def crop_5_percent(image: Image.Image) -> Image.Image:
            width, height = image.size
            dx = int(width * 0.05)
            dy = int(height * 0.05)

            return image.crop((dx, dy, width - dx, height - dy))

        result = {}
        for timestamp, info in frames_info.items():
            path = info["path"]
            img = Image.open(path).convert('L')
            img = crop_5_percent(img)
            img = img.resize((256, 256))
            _, file, ext = files.split_path(path)
            new_path = os.path.join(wd, file + "." + ext)
            img.save(new_path)

        return Melter._replace_path(frames_info, wd)


    @staticmethod
    def _compute_overlap(im1, im2, h):
        # Warp second image onto first
        warped_im2 = cv.warpPerspective(im2, h, (im1.shape[1], im1.shape[0]))

        # Find overlapping region mask
        gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(warped_im2, cv.COLOR_BGR2GRAY)

        mask1 = (gray1 > 0).astype(np.uint8)
        mask2 = (gray2 > 0).astype(np.uint8)
        overlap_mask = cv.bitwise_and(mask1, mask2)

        # Find bounding rectangle of overlapping mask
        x, y, w, h = cv.boundingRect(overlap_mask)
        return (x, y, w, h)


    @staticmethod
    def _rect_center(rect):
        x, y, w, h = rect
        return np.array([x + w/2, y + h/2, w, h])


    @staticmethod
    def _filter_outlier_rects(rects, threshold=2.0):
        centers = np.array([Melter._rect_center(r) for r in rects])
        median_center = np.median(centers, axis=0)
        deviations = np.linalg.norm(centers - median_center, axis=1)
        median_dev = np.median(deviations)
        if median_dev == 0:
            median_dev = 1  # Avoid div by zero
        filtered_rects = [rect for rect, dev in zip(rects, deviations) if dev / median_dev < threshold]
        return filtered_rects


    @staticmethod
    def _intersection_rect(rects):
        x1 = max(r[0] for r in rects)
        y1 = max(r[1] for r in rects)
        x2 = min(r[0]+r[2] for r in rects)
        y2 = min(r[1]+r[3] for r in rects)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2-x1, y2-y1)


    @staticmethod
    def _find_common_crop(pairs):
        overlaps1 = []
        overlaps2 = []

        for path1, path2 in pairs:
            im1 = cv.imread(path1)
            im2 = cv.imread(path2)

            orb = cv.ORB_create(1000)  # Increased points for better precision
            kp1, des1 = orb.detectAndCompute(im1, None)
            kp2, des2 = orb.detectAndCompute(im2, None)

            if des1 is None or des2 is None:
                continue

            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            if len(matches) < 3:  # Affine requires at least 3 points
                continue

            matches = sorted(matches, key=lambda x: x.distance)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # Use affine transformation (no perspective distortion)
            h_matrix, inliers = cv.estimateAffinePartial2D(pts2, pts1, cv.RANSAC)
            if h_matrix is None:
                continue

            overlap1 = Melter._compute_overlap(im1, im2, np.vstack([h_matrix, [0,0,1]]))
            overlap2 = Melter._compute_overlap(im2, im1, np.vstack([cv.invertAffineTransform(h_matrix), [0,0,1]]))

            overlaps1.append(overlap1)
            overlaps2.append(overlap2)

        overlaps1_filtered = Melter._filter_outlier_rects(overlaps1)
        overlaps2_filtered = Melter._filter_outlier_rects(overlaps2)

        common_crop1 = Melter._intersection_rect(overlaps1_filtered)
        common_crop2 = Melter._intersection_rect(overlaps2_filtered)

        return common_crop1, common_crop2


    @staticmethod
    def _apply_crop(src_dir, dst_dir, crop):
        if crop is None:
            raise ValueError("No common crop found.")
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        x, y, w, h = crop
        for fname in os.listdir(src_dir):
            path = os.path.join(src_dir, fname)
            img = cv.imread(path)
            cropped = img[y:y+h, x:x+w]
            cv.imwrite(os.path.join(dst_dir, fname), cropped)


    @staticmethod
    def _resize_dirs_to_smallest(dir1, dir2):
        # Get dimensions of the first image in each directory
        def get_dimensions(directory):
            first_img_path = next(os.path.join(directory, fname) for fname in os.listdir(directory))
            img = cv.imread(first_img_path)
            h, w = img.shape[:2]
            return w, h

        w1, h1 = get_dimensions(dir1)
        w2, h2 = get_dimensions(dir2)

        target_w = min(w1, w2)
        target_h = min(h1, h2)

        def resize_directory(directory):
            for fname in os.listdir(directory):
                path = os.path.join(directory, fname)
                img = cv.imread(path)
                resized = cv.resize(img, (target_w, target_h), interpolation=cv.INTER_AREA)
                cv.imwrite(path, resized)

        resize_directory(dir1)
        resize_directory(dir2)

    @staticmethod
    def _apply_final_border_crop(directory, crop_percent=0.02):
        for fname in os.listdir(directory):
            path = os.path.join(directory, fname)
            img = cv.imread(path)
            h, w = img.shape[:2]
            dx = int(w * crop_percent)
            dy = int(h * crop_percent)
            cropped = img[dy:h-dy, dx:w-dx]
            cv.imwrite(path, cropped)


    @staticmethod
    def _crop_both_sets(pairs, dir1, dir2, output_dir1, output_dir2, final_crop_percent=0.02):
        crop1, crop2 = Melter._find_common_crop(pairs) # Melter.find_common_image_part_naive(*pairs[0]) # Melter._find_common_crop(pairs)

        # Initial robust crop
        Melter._apply_crop(dir1, output_dir1, crop1)
        Melter._apply_crop(dir2, output_dir2, crop2)

        # Resize both directories to match smallest dimensions
        Melter._resize_dirs_to_smallest(output_dir1, output_dir2)

        # Final border crop
        #Melter._apply_final_border_crop(output_dir1, final_crop_percent)
        #Melter._apply_final_border_crop(output_dir2, final_crop_percent)


    def _create_segments_mapping(self, lhs: str, rhs: str) -> List[Tuple[int, int]]:
        with tempfile.TemporaryDirectory() as wd:
            lhs_scene_changes = video.detect_scene_changes(lhs, threshold = 0.3)
            rhs_scene_changes = video.detect_scene_changes(rhs, threshold = 0.3)

            if len(lhs_scene_changes) == 0 or len(rhs_scene_changes) == 0:
                return

            lhs_wd = os.path.join(wd, "lhs")
            rhs_wd = os.path.join(wd, "rhs")

            lhs_all_wd = os.path.join(lhs_wd, "all")
            rhs_all_wd = os.path.join(rhs_wd, "all")
            lhs_normalized_wd = os.path.join(lhs_wd, "norm")
            rhs_normalized_wd = os.path.join(rhs_wd, "norm")
            lhs_normalized_cropped_wd = os.path.join(lhs_wd, "norm_cropped")
            rhs_normalized_cropped_wd = os.path.join(rhs_wd, "norm_cropped")
            lhs_key_wd = os.path.join(lhs_wd, "key")
            rhs_key_wd = os.path.join(rhs_wd, "key")
            lhs_key_cropped_wd = os.path.join(lhs_wd, "key_cropped")
            rhs_key_cropped_wd = os.path.join(rhs_wd, "key_cropped")

            for d in [lhs_wd,
                      rhs_wd,
                      lhs_all_wd,
                      rhs_all_wd,
                      lhs_normalized_wd,
                      rhs_normalized_wd,
                      lhs_normalized_cropped_wd,
                      rhs_normalized_cropped_wd,
                      lhs_key_wd,
                      rhs_key_wd,
                      lhs_key_cropped_wd,
                      rhs_key_cropped_wd]:
                os.makedirs(d)

            # extract all scenes
            lhs_all_frames = video.extract_all_frames(lhs, lhs_all_wd, scale = 0.5)
            rhs_all_frames = video.extract_all_frames(rhs, rhs_all_wd, scale = 0.5)

            # normalize frames. This could be done in previous step, however for some videos ffmpeg fails to save some of the frames when using 256x256 resolution. Who knows why...
            lhs_normalized_frames = Melter._normalize_frames(lhs_all_frames, lhs_normalized_wd)
            rhs_normalized_frames = Melter._normalize_frames(rhs_all_frames, rhs_normalized_wd)

            # extract key frames (as 'key' a scene change frame is meant)
            lhs_key_frames = Melter._get_frames_for_timestamps(lhs_scene_changes, lhs_normalized_frames)
            rhs_key_frames = Melter._get_frames_for_timestamps(rhs_scene_changes, rhs_normalized_frames)

            # copy key frames
            for src, dst in [ (lhs_key_frames, lhs_key_wd), (rhs_key_frames, rhs_key_wd) ]:
                for src_info in src.values():
                    shutil.copy2(src_info["path"], dst)

            # pick frames with descent entropy (remove single color frames etc)
            #lhs_useful_key_frames = Melter._filter_low_detailed(lhs_key_frames)
            #rhs_useful_key_frames = Melter._filter_low_detailed(rhs_key_frames)

            # find best pair
            matching_pair = Melter._find_most_matching_pair(lhs_key_frames, rhs_key_frames)
            matching_pairs_paths = [(lhs_normalized_frames[pair[0]]["path"], rhs_normalized_frames[pair[1]]["path"]) for pair in [matching_pair]]

            # crop frames basing on best match
            Melter._crop_both_sets(
                pairs = matching_pairs_paths,
                dir1 = lhs_normalized_wd,
                dir2 = rhs_normalized_wd,
                output_dir1 = lhs_normalized_cropped_wd,
                output_dir2 = rhs_normalized_cropped_wd
            )

            lhs_normalized_cropped_frames = Melter._replace_path(lhs_normalized_frames, lhs_normalized_cropped_wd)
            rhs_normalized_cropped_frames = Melter._replace_path(rhs_normalized_frames, rhs_normalized_cropped_wd)

            # extract key frames from cropped images (as 'key' a scene change frame is meant)
            lhs_key_cropped_frames = Melter._get_frames_for_timestamps(lhs_scene_changes, lhs_normalized_cropped_frames)
            rhs_key_cropped_frames = Melter._get_frames_for_timestamps(rhs_scene_changes, rhs_normalized_cropped_frames)

            # copy key frames from cropped
            for src, dst in [ (lhs_key_cropped_frames, lhs_key_cropped_wd), (rhs_key_cropped_frames, rhs_key_cropped_wd) ]:
                for src_info in src.values():
                    shutil.copy2(src_info["path"], dst)

            # look for all pairs now
            matching_pairs = Melter._match_pairs(lhs_key_cropped_frames, rhs_key_cropped_frames)

            # try to locate first and last common frames
            Melter._look_for_boundaries(lhs_normalized_frames, rhs_normalized_frames, matching_pairs[0], matching_pairs[-1], 90)

            # calculate finerprint for each frame
            #lhs_hashes = Melter._generate_hashes(lhs_key_frames)
            #rhs_hashes = Melter._generate_hashes(rhs_key_frames)

            # find similar scenes
            #hash_algo = cv.img_hash.BlockMeanHash().create()
            #matching_scenes = Melter._match_scenes(lhs_hashes, rhs_hashes, lambda l, r: hash_algo.compare(l, r) < 20)

            #matching_files = [(lhs_key_frames[lhs_timestamp]["path"], rhs_key_frames[rhs_timestamp]["path"])  for lhs_timestamp, rhs_timestamp in matching_scenes]

            return matching_pairs


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
