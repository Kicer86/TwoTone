
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
from sklearn.linear_model import RANSACRegressor, LinearRegression
from typing import Any, Callable, Dict, List, Tuple

from . import utils
from .tool import Tool
from .utils2 import files, process, video


type FramesInfo = Dict[int, Dict[str, str]]


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
    class PhashCache:
        def __init__(self, hash_size: int = 16):
            self.hash_size = hash_size
            self._memory_cache: dict[str, imagehash.ImageHash] = {}

        def get(self, image_path: str) -> imagehash.ImageHash:
            if image_path in self._memory_cache:
                return self._memory_cache[image_path]

            with Image.open(image_path) as img:
                phash = imagehash.phash(img, hash_size=self.hash_size)

            self._memory_cache[image_path] = phash
            return phash


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
    def _filter_low_detailed(scenes: FramesInfo):
        valuable_scenes = { timestamp: info for timestamp, info in scenes.items() if Melter._frame_entropy(info["path"]) > 4}
        return valuable_scenes


    @staticmethod
    def _look_for_boundaries(lhs: FramesInfo, rhs: FramesInfo, first: Tuple[int, int], last: Tuple[int, int], cutoff: float):
        def estimate_fps(timestamps: List[int]) -> float:
            if len(timestamps) < 2:
                return 1.0
            diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:]) if t2 > t1]
            return 1.0 / (sum(diffs) / len(diffs)) if diffs else 1.0

        def find_boundary(lhs_set, rhs_set, lhs_ts, rhs_ts, lhs_fps, rhs_fps, pace, direction):
            lhs_keys = sorted(lhs_set.keys())
            rhs_keys = sorted(rhs_set.keys())

            def get_nearest(t, keys):
                return min(keys, key=lambda k: abs(k - t)) if keys else t

            current = (lhs_ts, rhs_ts)
            allowed_misses = 3
            miss = 0

            phash = Melter.PhashCache()
            i = 0
            while True:
                time_step = round(i / lhs_fps) * direction
                next_lhs_ts = lhs_ts + time_step
                next_rhs_est = rhs_ts + time_step / pace
                i += 1

                if next_lhs_ts not in lhs_set:
                    break

                next_rhs_ts = get_nearest(next_rhs_est, rhs_keys)
                if abs(next_rhs_ts - next_rhs_est) > 10:
                    continue

                lhs_img_path = lhs_set[next_lhs_ts]["path"]
                rhs_img_path = rhs_set[next_rhs_ts]["path"]
                lhs_hash = phash.get(lhs_img_path)
                rhs_hash = phash.get(rhs_img_path)

                diff = abs(lhs_hash - rhs_hash)
                is_good = diff <= cutoff
                if is_good or miss < allowed_misses:
                    if is_good:
                        current = (next_lhs_ts, next_rhs_ts)
                        miss = 0
                    else:
                        miss += 1
                else:
                    break

            return current

        lhs_times = sorted(lhs.keys())
        rhs_times = sorted(rhs.keys())

        lhs_fps = estimate_fps(lhs_times)
        rhs_fps = estimate_fps(rhs_times)
        pace = (last[0] - first[0]) / (last[1] - first[1]) if (last[1] - first[1]) != 0 else 1.0

        refined_first = find_boundary(lhs, rhs, first[0], first[1], lhs_fps, rhs_fps, pace, direction=-1)
        refined_last = find_boundary(lhs, rhs, last[0], last[1], lhs_fps, rhs_fps, pace, direction=1)

        print(f"Refined First: L: {lhs[refined_first[0]]['path']} R: {rhs[refined_first[1]]['path']}")
        print(f"Refined Last:  L: {lhs[refined_last[0]]['path']} R: {rhs[refined_last[1]]['path']}")

        return refined_first, refined_last


    @staticmethod
    def _generate_matching_frames(lhs: FramesInfo, rhs: FramesInfo) -> List[Tuple[int, int]]:

        def compute_phash_candidates(lhs, rhs):
            phash = Melter.PhashCache()
            pairs_candidates = defaultdict(list)
            for lhs_timestamp, lhs_info in lhs.items():
                lhs_hash = phash.get(lhs_info["path"])
                for rhs_timestamp, rhs_info in rhs.items():
                    rhs_hash = phash.get(rhs_info["path"])
                    distance = abs(lhs_hash - rhs_hash)
                    pairs_candidates[lhs_timestamp].append((distance, rhs_timestamp))
            return pairs_candidates

        def select_best_candidates(pairs_candidates):
            best_candidates = []
            used_rhs_timestamps = set()
            for lhs_timestamp, candidates in pairs_candidates.items():
                candidates.sort()
                for diff, rhs_candidate in candidates:
                    if rhs_candidate not in used_rhs_timestamps:
                        used_rhs_timestamps.add(rhs_candidate)
                        best_candidates.append((diff, lhs_timestamp, rhs_candidate))
                        break
            best_candidates.sort()
            return sorted([(lhs, rhs) for _, lhs, rhs in best_candidates])

        def filter_with_ransac(initial_pairs):
            lhs_timestamps, rhs_timestamps = zip(*initial_pairs)
            lhs_array = np.array(lhs_timestamps).reshape(-1, 1)
            rhs_array = np.array(rhs_timestamps)

            ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=5000)
            ransac.fit(lhs_array, rhs_array)

            inlier_mask = ransac.inlier_mask_
            return [pair for pair, is_inlier in zip(initial_pairs, inlier_mask) if is_inlier]

        def calculate_ratios(pairs):
            ratios = []
            for i in range(len(pairs) - 1):
                lhs_diff = pairs[i + 1][0] - pairs[i][0]
                rhs_diff = pairs[i + 1][1] - pairs[i][1]
                if rhs_diff <= 0:
                    ratios.append(None)
                else:
                    ratios.append(lhs_diff / rhs_diff)
            return ratios

        def iterative_filter(pairs):
            pairs = pairs.copy()
            iteration = 0
            while True:
                ratios = calculate_ratios(pairs)
                valid_ratios = [r for r in ratios if r is not None and r > 0]
                if not valid_ratios:
                    break
                median_ratio = np.median(valid_ratios)
                mad = np.median(np.abs(valid_ratios - median_ratio))
                threshold = 3 * mad

                to_remove = set()
                for i, ratio in enumerate(ratios):
                    if ratio is None or ratio <= 0 or abs(ratio - median_ratio) > threshold:
                        to_remove.add(i + 1)

                if not to_remove:
                    break

                pairs = [pair for idx, pair in enumerate(pairs) if idx not in to_remove]
                iteration += 1

                if iteration > len(pairs):
                    break

            return pairs

        pairs_candidates = compute_phash_candidates(lhs, rhs)
        initial_pairs = select_best_candidates(pairs_candidates)
        filtered_pairs = filter_with_ransac(initial_pairs)
        final_pairs = iterative_filter(filtered_pairs)

        return final_pairs


    @staticmethod
    def _match_pairs(lhs: FramesInfo, rhs: FramesInfo) -> List[Tuple[int, int]]:
        pairs = Melter._generate_matching_frames(lhs, rhs)

        pairs.sort()

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
    def _get_frames_for_timestamps(timestamps: List[int], frames_info: FramesInfo) -> List[str]:
        frame_files = {timestamp: info for timestamp, info in frames_info.items() if timestamp in timestamps}

        return frame_files

    @staticmethod
    def _get_new_info(info: Dict[str, str], path: str) -> Dict[str, str]:
        new_info = info.copy()
        new_info["path"] = path
        return new_info


    @staticmethod
    def _normalize_frames(frames_info: FramesInfo, wd: str) -> Dict[int, str]:
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

            result[timestamp] = Melter._get_new_info(info, new_path)

        return result


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
    def _interpolate_crop_rects(timestamps, rects):
        """
        Given a list of timestamps and matching crop rects, return a function that interpolates
        a crop for any timestamp between and extrapolates outside the range.
        rect = (x, y, w, h)
        """

        timestamps = np.array(timestamps)
        rects = np.array(rects)

        def interpolate(t):
            if t <= timestamps[0]:
                return tuple(rects[0])
            elif t >= timestamps[-1]:
                return tuple(rects[-1])
            else:
                x = np.interp(t, timestamps, rects[:, 0])
                y = np.interp(t, timestamps, rects[:, 1])
                w = np.interp(t, timestamps, rects[:, 2])
                h = np.interp(t, timestamps, rects[:, 3])
                return int(round(x)), int(round(y)), int(round(w)), int(round(h))

        return interpolate


    @staticmethod
    def _find_interpolated_crop(pairs_with_timestamps, lhs_frames: FramesInfo, rhs_frames: FramesInfo):
        timestamps = []
        crops1 = []
        crops2 = []

        for lhs_t, rhs_t in pairs_with_timestamps:
            lhs_info = lhs_frames[lhs_t]
            rhs_info = rhs_frames[rhs_t]
            lhs_img = cv.imread(lhs_info["path"])
            rhs_img = cv.imread(rhs_info["path"])

            orb = cv.ORB_create(1000)
            kp1, des1 = orb.detectAndCompute(lhs_img, None)
            kp2, des2 = orb.detectAndCompute(rhs_img, None)
            if des1 is None or des2 is None:
                continue

            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            if len(matches) < 3:
                continue

            matches = sorted(matches, key=lambda x: x.distance)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            h_matrix, inliers = cv.estimateAffinePartial2D(pts2, pts1, cv.RANSAC)
            if h_matrix is None:
                continue

            overlap1 = Melter._compute_overlap(lhs_img, rhs_img, np.vstack([h_matrix, [0, 0, 1]]))
            overlap2 = Melter._compute_overlap(rhs_img, lhs_img, np.vstack([cv.invertAffineTransform(h_matrix), [0, 0, 1]]))

            timestamps.append(lhs_t)
            crops1.append(overlap1)
            crops2.append(overlap2)

        # Return interpolators
        return Melter._interpolate_crop_rects(timestamps, crops1), Melter._interpolate_crop_rects(timestamps, crops2)


    @staticmethod
    def _apply_crop_interpolated(frames: FramesInfo, dst_dir: str, crop_fn: Callable[[int], Tuple[int, int, int, int]]) -> FramesInfo:
        output_files = {}
        for timestamp, info in frames.items():
            path = info["path"]
            img = cv.imread(path)
            x, y, w, h = crop_fn(timestamp)
            cropped = img[y:y+h, x:x+w]
            cropped = cv.resize(cropped, (128, 128))
            dst_path = os.path.join(dst_dir, os.path.basename(path))
            cv.imwrite(dst_path, cropped)

            output_files[timestamp] = Melter._get_new_info(info, dst_path)

        return output_files


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
    def _crop_both_sets(
        pairs_with_timestamps: List[Tuple[int, int]],
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo,
        lhs_cropped_dir: str,
        rhs_cropped_dir: str,
        final_crop_percent: float = 0.02
    ) -> Tuple[FramesInfo, FramesInfo]:
        # Step 1: Get interpolated crop functions for both sets
        lhs_crop_fn, rhs_crop_fn = Melter._find_interpolated_crop(pairs_with_timestamps, lhs_frames, rhs_frames)

        # Step 2: Apply interpolated cropping to each frame
        lhs_cropped = Melter._apply_crop_interpolated(lhs_frames, lhs_cropped_dir, lhs_crop_fn)
        rhs_cropped = Melter._apply_crop_interpolated(rhs_frames, rhs_cropped_dir, rhs_crop_fn)

        # Step 3: Resize both output sets to same resolution (downscale to smaller one)
        #Melter._resize_dirs_to_smallest(out_dir1, out_dir2)

        return lhs_cropped, rhs_cropped


    @staticmethod
    def _calculate_cutoff(
        pairs_with_timestamps: List[Tuple[int, int]],
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo
    ) -> int:
        phash = Melter.PhashCache()
        max_phash = 0

        for lhs, rhs in pairs_with_timestamps:
            lhs_path = lhs_frames[lhs]["path"]
            rhs_path = rhs_frames[rhs]["path"]
            lhs_phash = phash.get(lhs_path)
            rhs_phash = phash.get(rhs_path)

            pdiff = abs(lhs_phash - rhs_phash)
            print(f"{lhs_path} vs {rhs_path}: {pdiff}")
            max_phash = max(max_phash, pdiff)

        return max_phash


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

            for d in [lhs_wd,
                      rhs_wd,
                      lhs_all_wd,
                      rhs_all_wd,
                      lhs_normalized_wd,
                      rhs_normalized_wd,
                      lhs_normalized_cropped_wd,
                      rhs_normalized_cropped_wd]:
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

            # find matching keys
            matching_pairs = Melter._match_pairs(lhs_key_frames, rhs_key_frames)

            prev_first, prev_last = None, None
            while True:
                # crop frames basing on matching ones
                lhs_normalized_cropped_frames, rhs_normalized_cropped_frames = Melter._crop_both_sets(
                    pairs_with_timestamps = matching_pairs,
                    lhs_frames = lhs_normalized_frames,
                    rhs_frames = rhs_normalized_frames,
                    lhs_cropped_dir = lhs_normalized_cropped_wd,
                    rhs_cropped_dir = rhs_normalized_cropped_wd
                )

                cutoff = Melter._calculate_cutoff(matching_pairs, lhs_normalized_cropped_frames, rhs_normalized_cropped_frames)

                # try to locate first and last common frames
                first, last = Melter._look_for_boundaries(lhs_normalized_cropped_frames, rhs_normalized_cropped_frames, matching_pairs[0], matching_pairs[-1], cutoff * 1.5)

                if first == prev_first and last == prev_last:
                    break
                else:
                    if first != prev_first:
                        matching_pairs = [first, *matching_pairs]
                        prev_first = first
                    if last != prev_last:
                        matching_pairs = [*matching_pairs, last]
                        prev_last = last

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
