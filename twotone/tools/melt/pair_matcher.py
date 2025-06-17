
import cv2 as cv
import logging
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from sklearn.linear_model import RANSACRegressor, LinearRegression
from typing import Callable, Dict, List, Tuple

from .debug_routines import DebugRoutines
from .phash_cache import PhashCache
from ..utils import files_utils, generic_utils, image_utils, video_utils

FramesInfo = Dict[int, Dict[str, str]]


class PairMatcher:
    def __init__(self, interruption: generic_utils.InterruptibleProcess, wd: str, lhs_path: str, rhs_path: str, logger: logging.Logger) -> None:
        self.interruption = interruption
        self.wd = os.path.join(wd, "pair_matcher")
        self.lhs_path = lhs_path
        self.rhs_path = rhs_path
        self.logger = logger
        self.phash = PhashCache()
        self.lhs_fps = eval(video_utils.get_video_data(lhs_path)["video"][0]["fps"])
        self.rhs_fps = eval(video_utils.get_video_data(rhs_path)["video"][0]["fps"])

        lhs_wd = os.path.join(self.wd, "lhs")
        rhs_wd = os.path.join(self.wd, "rhs")

        self.lhs_all_wd = os.path.join(lhs_wd, "all")
        self.rhs_all_wd = os.path.join(rhs_wd, "all")
        self.lhs_normalized_wd = os.path.join(lhs_wd, "norm")
        self.rhs_normalized_wd = os.path.join(rhs_wd, "norm")
        self.lhs_normalized_cropped_wd = os.path.join(lhs_wd, "norm_cropped")
        self.rhs_normalized_cropped_wd = os.path.join(rhs_wd, "norm_cropped")
        self.debug_wd = os.path.join(self.wd, "debug")

        for d in [lhs_wd,
                  rhs_wd,
                  self.lhs_all_wd,
                  self.rhs_all_wd,
                  self.lhs_normalized_wd,
                  self.rhs_normalized_wd,
                  self.lhs_normalized_cropped_wd,
                  self.rhs_normalized_cropped_wd,
                  self.debug_wd,
        ]:
            os.makedirs(d)

    def _normalize_frames(self, frames_info: FramesInfo, wd: str) -> FramesInfo:
        def crop_5_percent(image: cv.Mat) -> cv.Mat:
            height, width = image.shape
            dx = int(width * 0.05)
            dy = int(height * 0.05)

            image_cropped = image[dy:height - dy, dx:width - dx]

            return image_cropped

        def process_frame(item):
            timestamp, info = item
            self.interruption._check_for_stop()
            path = info["path"]
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            img = crop_5_percent(img)
            img = cv.resize(img, (256, 256), interpolation=cv.INTER_AREA)
            _, file, ext = files_utils.split_path(path)
            new_path = os.path.join(wd, file + "." + ext)
            cv.imwrite(new_path, img)

            return timestamp, PairMatcher._get_new_info(info, new_path)

        with ThreadPoolExecutor() as executor:
            results = executor.map(process_frame, frames_info.items())

        return dict(results)

    @staticmethod
    def calculate_ratio(pairs: List[Tuple[int, int]]) -> float:
        ratios = [(r[0] - l[0]) / (r[1] - l[1]) for l, r in zip(pairs[:-1], pairs[1:]) if (r[1] - l[1]) != 0]
        median_ratio = np.median(ratios)
        return float(median_ratio)

    @staticmethod
    def is_ratio_acceptable(ratio: float, perfect_ratio: float) -> bool:
        return abs(ratio - perfect_ratio) < 0.05 * perfect_ratio

    @staticmethod
    def _is_rich(frame_path: str) -> bool:
        return image_utils.image_entropy(frame_path) > 3.5


    @staticmethod
    def _get_new_info(info: Dict[str, str], path: str) -> Dict[str, str]:
        new_info = info.copy()
        new_info["path"] = path
        return new_info

    @staticmethod
    def _filter_low_detailed(scenes: FramesInfo) -> FramesInfo:
        valuable_scenes = {timestamp: info for timestamp, info in scenes.items() if PairMatcher._is_rich(info["path"])}
        return valuable_scenes

    @staticmethod
    def _get_frames_for_timestamps(timestamps: List[int], frames_info: FramesInfo) -> FramesInfo:
        frame_files = {timestamp: info for timestamp, info in frames_info.items() if timestamp in timestamps}

        return frame_files

    @staticmethod
    def filter_phash_outliers(phash: PhashCache, pairs: List[Tuple[int, int]], lhs_set: FramesInfo, rhs_set: FramesInfo) -> List[Tuple[int, int]]:
        dists = [abs(phash.get(lhs_set[l]["path"]) - phash.get(rhs_set[r]["path"])) for l, r in pairs]
        med = np.median(dists)
        mad = np.median(np.abs(dists - med))
        threshold = med + 1.5 * mad
        return [pair for pair, dist in zip(pairs, dists) if dist <= threshold]

    @staticmethod
    def summarize_pairs(phash: PhashCache, pairs: List[Tuple[int, int]], lhs: FramesInfo, rhs: FramesInfo, verbose: bool = False) -> str:
        distances = []
        for lhs_ts, rhs_ts in pairs:
            d = abs(phash.get(lhs[lhs_ts]["path"]) - phash.get(rhs[rhs_ts]["path"]))
            distances.append((d, lhs_ts, rhs_ts))

        arr = np.array([d[0] for d in distances])
        median = np.median(arr)
        mean = np.mean(arr)
        std = np.std(arr)
        max_val = np.max(arr)
        min_val = np.min(arr)

        # Identify the max pair
        max_entry = max(distances, key=lambda x: x[0])
        max_lhs_path = lhs[max_entry[1]]["path"]
        max_rhs_path = rhs[max_entry[2]]["path"]

        summary = (
            f"Pairs: {len(pairs)} | "
            f"Median: {median:.2f} | "
            f"Mean: {mean:.2f} | "
            f"Std Dev: {std:.2f} | "
            f"Min: {min_val} | "
            f"Max: {max_val} | "
            f"Max Pair: {max_lhs_path} <-> {max_rhs_path}"
        )

        if verbose:
            details = [
                f"  {lhs[lhs_ts]['path']} <-> {rhs[rhs_ts]['path']} | Diff: {dist}"
                for dist, lhs_ts, rhs_ts in distances
            ]
            summary += "\nDetailed pairs:" + "\n" + "\n".join(details)

        return summary

    @staticmethod
    def summarize_segments(pairs: List[Tuple[int, int]], lhs_fps: float, rhs_fps: float, verbose: bool = True) -> str:
        if len(pairs) < 2:
            return "Not enough pairs to build segments."

        pairs_sorted = sorted(pairs)
        segments = []

        lhs_frame_step = 1000.0 / lhs_fps
        rhs_frame_step = 1000.0 / rhs_fps
        max_quant_error = max(lhs_frame_step, rhs_frame_step)

        for (lhs1, rhs1), (lhs2, rhs2) in zip(pairs_sorted[:-1], pairs_sorted[1:]):
            lhs_delta = lhs2 - lhs1
            rhs_delta = rhs2 - rhs1
            if rhs_delta <= 0:
                continue
            ratio = lhs_delta / rhs_delta

            # Estimate ratio uncertainty
            min_delta = min(lhs_delta, rhs_delta)
            ratio_error = (2 * max_quant_error) / min_delta if min_delta > 0 else float("inf")
            confidence = "LOW" if ratio_error > 0.1 else "OK"

            segments.append((lhs1, lhs2, rhs1, rhs2, lhs_delta, rhs_delta, ratio, ratio_error, confidence))

        if not segments:
            return "No valid segments."

        ratios = np.array([s[6] for s in segments])
        out = [
            f"Segments: {len(segments)} | "
            f"Median ratio: {np.median(ratios):.4f} | "
            f"Mean ratio: {np.mean(ratios):.4f} | "
            f"Std Dev: {np.std(ratios):.4f} | "
            f"Min: {np.min(ratios):.4f} | Max: {np.max(ratios):.4f}"
        ]

        if verbose:
            out.append("\nDetailed segments:")
            for lhs1, lhs2, rhs1, rhs2, ldelta, rdelta, ratio, err, conf in segments:
                out.append(
                    f"  LHS {lhs1}->{lhs2} ({ldelta:4} ms), "
                    f"RHS {rhs1}->{rhs2} ({rdelta:4} ms), "
                    f"Ratio: {ratio:.4f}, "
                    f"Error~{err:.2%}, Confidence: {conf}"
                )

        return '\n'.join(out)

    @staticmethod
    def _compute_overlap(lhs_img: cv.typing.MatLike, rhs_img: cv.typing.MatLike, h) -> tuple[int, int, int, int]:
        # Expect images to be in the grayscale
        assert len(lhs_img.shape) == 2
        assert len(rhs_img.shape) == 2

        # Warp second image onto first
        warped_im2 = cv.warpPerspective(rhs_img, h, (lhs_img.shape[1], lhs_img.shape[0]))

        # Find overlapping region mask
        gray1 = lhs_img
        gray2 = warped_im2

        mask1 = (gray1 > 0).astype(np.uint8)
        mask2 = (gray2 > 0).astype(np.uint8)
        overlap_mask = cv.bitwise_and(mask1, mask2)

        # Find bounding rectangle of overlapping mask
        x, y, w, h = cv.boundingRect(overlap_mask)
        return (x, y, w, h)

    @staticmethod
    def _interpolate_crop_rects(timestamps_list: List[int], rects_list: List[Tuple[int, int, int, int]]) -> Callable[[int], Tuple[int, int, int, int]]:
        """
        Given a list of timestamps and matching crop rects, return a function that interpolates
        a crop for any timestamp between and extrapolates outside the range.
        rect = (x, y, w, h)
        """

        timestamps = np.array(timestamps_list)
        rects = np.array(rects_list)

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
    def _find_interpolated_crop(pairs_with_timestamps: List[Tuple[int, int]], lhs_frames: FramesInfo, rhs_frames: FramesInfo) -> Tuple[Callable[[int], Tuple[int, int, int, int]], Callable[[int], Tuple[int, int, int, int]]]:
        timestamps_lhs = []
        timestamps_rhs = []
        lhs_crops = []
        rhs_crops = []

        for lhs_t, rhs_t in pairs_with_timestamps:
            lhs_info = lhs_frames[lhs_t]
            rhs_info = rhs_frames[rhs_t]
            lhs_img = cv.imread(lhs_info["path"], cv.IMREAD_GRAYSCALE)
            rhs_img = cv.imread(rhs_info["path"], cv.IMREAD_GRAYSCALE)

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

            lhs_overlap = PairMatcher._compute_overlap(lhs_img, rhs_img, np.vstack([h_matrix, [0, 0, 1]]))
            rhs_overlap = PairMatcher._compute_overlap(rhs_img, lhs_img, np.vstack([cv.invertAffineTransform(h_matrix), [0, 0, 1]]))

            timestamps_lhs.append(lhs_t)
            timestamps_rhs.append(rhs_t)
            lhs_crops.append(lhs_overlap)
            rhs_crops.append(rhs_overlap)

        # Return interpolators
        return PairMatcher._interpolate_crop_rects(timestamps_lhs, lhs_crops), PairMatcher._interpolate_crop_rects(timestamps_rhs, rhs_crops)

    def _apply_crop_interpolated(self, frames: FramesInfo, dst_dir: str, crop_fn: Callable[[int], Tuple[int, int, int, int]]) -> FramesInfo:
        def _process_frame(item):
            timestamp, info = item
            self.interruption._check_for_stop()
            path = info["path"]
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            x, y, w, h = crop_fn(timestamp)
            cropped = img[y:y+h, x:x+w]
            cropped = cv.resize(cropped, (128, 128))
            dst_path = os.path.join(dst_dir, os.path.basename(path))
            cv.imwrite(dst_path, cropped)
            return timestamp, PairMatcher._get_new_info(info, dst_path)

        with ThreadPoolExecutor() as executor:
            results = executor.map(_process_frame, frames.items())

        return dict(results)

    def _three_before(self, timestamps: List[int], target: int) -> List[int]:
        timestamps = sorted(timestamps)
        idx = int(np.searchsorted(timestamps, target))
        return list(filter(lambda x: x in timestamps, timestamps[max(0, idx-3):idx]))

    def _nearest_three(self, timestamps: List[int], target: int) -> List[int]:
        timestamps = sorted(timestamps)
        idx = int(np.searchsorted(timestamps, target))
        return list(filter(lambda x: x in timestamps, timestamps[max(0, idx-1):idx+2]))

    def _best_phash_match(self, lhs_ts: int, rhs_ts_guess: int, lhs_all_set: FramesInfo, rhs_all_set: FramesInfo) -> Tuple[int, int] | None:
        lhs_near = self._nearest_three(list(lhs_all_set.keys()), lhs_ts)
        rhs_near = self._nearest_three(list(rhs_all_set.keys()), rhs_ts_guess)
        best = None
        best_dist = float("inf")
        for l in lhs_near:
            for r in rhs_near:
                if l in lhs_all_set and r in rhs_all_set:
                    d = abs(self.phash.get(lhs_all_set[l]["path"]) - self.phash.get(rhs_all_set[r]["path"]))
                    if d < best_dist:
                        best = (l, r)
                        best_dist = d
        return best

    def _build_matches(self, lhs: FramesInfo, rhs: FramesInfo) -> List[Tuple[int, int, int]]:
        lhs_items = list(lhs.items())
        rhs_items = list(rhs.items())

        all_matches = []
        for lhs_ts, lhs_info in lhs_items:
            lhs_hash = self.phash.get(lhs_info["path"])
            for rhs_ts, rhs_info in rhs_items:
                rhs_hash = self.phash.get(rhs_info["path"])
                distance = abs(lhs_hash - rhs_hash)
                all_matches.append((distance, lhs_ts, rhs_ts))

        all_matches.sort()
        return all_matches

    def _build_initial_candidates(self, lhs: FramesInfo, rhs: FramesInfo) -> List[Tuple[int, int]]:
        all_matches = self._build_matches(lhs, rhs)

        used_lhs = set()
        used_rhs = set()
        pairs = []

        for distance, lhs_ts, rhs_ts in all_matches:
            if lhs_ts not in used_lhs and rhs_ts not in used_rhs:
                pairs.append((lhs_ts, rhs_ts))
                used_lhs.add(lhs_ts)
                used_rhs.add(rhs_ts)

        return sorted(pairs)

    def _reject_outliers(self, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(pairs) < 3:
            return pairs

        lhs_vals, rhs_vals = zip(*pairs)
        lhs_array = np.array(lhs_vals).reshape(-1, 1)
        rhs_array = np.array(rhs_vals)
        model = RANSACRegressor(LinearRegression(), residual_threshold=5000)
        model.fit(lhs_array, rhs_array)
        inliers = model.inlier_mask_
        return [p for p, keep in zip(pairs, inliers) if keep]

    def _check_history(self, pair: Tuple[int, int], lhs_pool: FramesInfo, rhs_pool: FramesInfo, cutoff: float) -> bool:
        lhs_three = self._three_before(list(lhs_pool.keys()), pair[0])
        rhs_three = self._three_before(list(rhs_pool.keys()), pair[1])

        if len(lhs_three) < 3 and len(rhs_three) < 3:
            return True
        elif len(lhs_three) < 3 or len(rhs_three) < 3:
            # TODO: some logic needed here
            pass

        # at least one match before current pair is required
        lhs_frames = {l: lhs_pool[l] for l in lhs_three}
        rhs_frames = {r: rhs_pool[r] for r in rhs_three}
        matches = self._build_matches(lhs_frames, rhs_frames)

        if len(matches) > 0:
            best_match = matches[0][0]

            if best_match <= cutoff:
                return True

        return False

    def _extrapolate_matches(self, known_pairs: List[Tuple[int, int]], lhs_pool: FramesInfo, rhs_pool: FramesInfo, phash: PhashCache) -> List[Tuple[int, int]]:
        known_pairs.sort()
        lhs_used = {l for l, _ in known_pairs}
        rhs_used = {r for _, r in known_pairs}
        lhs_free = sorted(set(lhs_pool.keys()) - lhs_used)
        rhs_keys = sorted(rhs_pool.keys())

        if len(known_pairs) < 2:
            return known_pairs

        median_ratio = PairMatcher.calculate_ratio(known_pairs)
        first_known_pair = known_pairs[0]
        cutoff = self._calculate_cutoff(phash, known_pairs, lhs_pool, rhs_pool)

        new_pairs = []
        for l in lhs_free:
            expected_rhs = first_known_pair[1] + (l - first_known_pair[0]) / median_ratio
            nearest_rhs_candidates = self._nearest_three(rhs_keys, int(expected_rhs))

            for rhs_candidate in nearest_rhs_candidates:
                ratio = (l - first_known_pair[0]) / (rhs_candidate - first_known_pair[1]) if (rhs_candidate - first_known_pair[1]) != 0 else None
                if ratio and PairMatcher.is_ratio_acceptable(ratio, median_ratio):
                    if rhs_candidate not in rhs_used:
                        # make sure lhs and rhs_candidate are matching
                        lhs_path = lhs_pool[l]["path"]
                        rhs_path = rhs_pool[rhs_candidate]["path"]

                        pdiff = abs(phash.get(lhs_path) - phash.get(rhs_path))
                        phash_matching = pdiff < cutoff
                        matching = image_utils.are_images_similar(lhs_path, rhs_path)
                        if phash_matching and matching:
                            new_pairs.append((l, rhs_candidate))
                            rhs_used.add(rhs_candidate)
                            break
                        else:
                            pass

        return sorted(set(known_pairs + new_pairs))

    def _crop_both_sets(
        self,
        pairs_with_timestamps: List[Tuple[int, int]],
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo,
        lhs_cropped_dir: str,
        rhs_cropped_dir: str
    ) -> Tuple[FramesInfo, FramesInfo]:
        # Step 1: Get interpolated crop functions for both sets
        lhs_crop_fn, rhs_crop_fn = PairMatcher._find_interpolated_crop(pairs_with_timestamps, lhs_frames, rhs_frames)

        # Step 2: Apply interpolated cropping to each frame
        lhs_cropped = self._apply_crop_interpolated(lhs_frames, lhs_cropped_dir, lhs_crop_fn)
        rhs_cropped = self._apply_crop_interpolated(rhs_frames, rhs_cropped_dir, rhs_crop_fn)

        # Step 3: Resize both output sets to same resolution (downscale to smaller one)
        #Melter._resize_dirs_to_smallest(out_dir1, out_dir2)

        return lhs_cropped, rhs_cropped

    def _calculate_cutoff(
        self,
        phash: PhashCache,
        pairs: List[Tuple[int, int]],
        lhs: FramesInfo,
        rhs: FramesInfo
    ) -> int:
        distances = [abs(phash.get(lhs[lhs_ts]["path"]) - phash.get(rhs[rhs_ts]["path"])) for lhs_ts, rhs_ts in pairs]

        arr = np.array(distances)
        median = np.median(arr)
        std = np.std(arr)

        return median + std * 2

    def _make_pairs(self, lhs: FramesInfo, rhs: FramesInfo, lhs_all: FramesInfo, rhs_all: FramesInfo) -> List[Tuple[int, int]]:
        # Pipeline
        lhs = PairMatcher._filter_low_detailed(lhs)
        rhs = PairMatcher._filter_low_detailed(rhs)

        if not lhs or not rhs:
            return []

        initial = self._build_initial_candidates(lhs, rhs)
        self.logger.debug(f"Initial candidates:        {PairMatcher.summarize_pairs(self.phash, initial, lhs_all, rhs_all)}")

        stable = self._reject_outliers(initial)
        self.logger.debug(f"After linear matching:     {PairMatcher.summarize_pairs(self.phash, stable, lhs_all, rhs_all)}")

        stable = PairMatcher.filter_phash_outliers(self.phash, stable, lhs_all, rhs_all)
        self.logger.debug(f"Phash outlier elimination: {PairMatcher.summarize_pairs(self.phash, stable, lhs_all, rhs_all)}")

        extrapolated = self._extrapolate_matches(stable, lhs, rhs, self.phash)
        self.logger.debug(f"Extrapolation:             {PairMatcher.summarize_pairs(self.phash, extrapolated, lhs_all, rhs_all)}")

        extrapolated_refined = [self._best_phash_match(l, r, lhs_all, rhs_all) for l, r in extrapolated]
        self.logger.debug(f"Frame adjustment:          {PairMatcher.summarize_pairs(self.phash, extrapolated_refined, lhs_all, rhs_all)}")

        outliers_eliminated = PairMatcher.filter_phash_outliers(self.phash, extrapolated_refined, lhs_all, rhs_all)
        self.logger.debug(f"Phash outlier elimination: {PairMatcher.summarize_pairs(self.phash, outliers_eliminated, lhs_all, rhs_all)}")

        orb_filtered = [
            (lhs_ts, rhs_ts) for lhs_ts, rhs_ts in outliers_eliminated
            if image_utils.are_images_similar(lhs_all[lhs_ts]["path"], rhs_all[rhs_ts]["path"])
        ]
        self.logger.debug(f"After ORB elimination:     {PairMatcher.summarize_pairs(self.phash, orb_filtered, lhs_all, rhs_all)}")

        cutoff = self._calculate_cutoff(self.phash, orb_filtered, lhs_all, rhs_all)
        final = [pair for pair in orb_filtered if self._check_history(pair, lhs_all, rhs_all, cutoff)]
        self.logger.debug(f"After history analysis:    {PairMatcher.summarize_pairs(self.phash, final, lhs_all, rhs_all)}")

        unique_pairs = sorted(set(final))

        self.logger.debug(PairMatcher.summarize_segments(unique_pairs, self.lhs_fps, self.rhs_fps))

        return unique_pairs


    def _look_for_boundaries(self, lhs: FramesInfo, rhs: FramesInfo, first: Tuple[int, int], last: Tuple[int, int], cutoff: float, lookahead_seconds: float = 3.0):
        self.logger.debug("Improving boundaries")
        self.logger.debug(f"Current first: {first} and last: {last} pairs")
        phash = PhashCache()
        ratio = PairMatcher.calculate_ratio([first, last])

        def find_best_pair(lhs: FramesInfo, rhs: FramesInfo) -> Tuple[Tuple[int, int], int]:
            best_score = 1000
            best_pair = ()

            for lhs_ts, lhs_info in lhs.items():
                lhs_hash = phash.get(lhs_info["path"])
                options = [(abs(lhs_hash - phash.get(rhs_info["path"])), rhs_ts) for rhs_ts, rhs_info in rhs.items()]
                if not options:
                    continue

                options.sort()
                best_dist, best_rhs = options[0]
                pair_candidate = (lhs_ts, best_rhs)

                pair_ratio_to_first = PairMatcher.calculate_ratio([pair_candidate, first])
                pair_ratio_to_last = PairMatcher.calculate_ratio([pair_candidate, last])

                if best_dist < best_score and PairMatcher.is_ratio_acceptable(pair_ratio_to_first, ratio) and PairMatcher.is_ratio_acceptable(pair_ratio_to_last, ratio):
                    best_score = best_dist
                    best_pair = pair_candidate

            return best_pair, best_score

        def find_boundary(lhs_set: FramesInfo, rhs_set: FramesInfo, lhs_ts, rhs_ts, direction):
            lhs_keys = sorted(lhs_set.keys())
            rhs_keys = sorted(rhs_set.keys())

            lhs_idx = lhs_keys.index(lhs_ts)
            rhs_idx = rhs_keys.index(rhs_ts)

            current_pair = (lhs_ts, rhs_ts)

            step_lhs = int(self.lhs_fps * lookahead_seconds)
            step_rhs = int(self.rhs_fps * lookahead_seconds)

            while True:
                lhs_slice = slice(lhs_idx + direction, lhs_idx + direction * step_lhs, direction)
                rhs_slice = slice(rhs_idx + direction, rhs_idx + direction * step_rhs, direction)

                lhs_range = lhs_keys[lhs_slice]
                rhs_range = rhs_keys[rhs_slice]

                if not lhs_range or not rhs_range:
                    break

                lhs_candidates = {lhs_ts: lhs[lhs_ts] for lhs_ts in lhs_range if PairMatcher._is_rich(lhs[lhs_ts]["path"])}
                rhs_candidates = {rhs_ts: rhs[rhs_ts] for rhs_ts in rhs_range if PairMatcher._is_rich(rhs[rhs_ts]["path"])}

                best_pair, best_score = find_best_pair(lhs_candidates, rhs_candidates)

                if best_pair and best_score < cutoff:
                    if best_pair == current_pair:
                        break
                    else:
                        current_pair = best_pair
                        lhs_ts = current_pair[0]
                        rhs_ts = current_pair[1]
                        lhs_idx = lhs_keys.index(lhs_ts)
                        rhs_idx = rhs_keys.index(rhs_ts)

                        lhs_path = lhs_set[lhs_ts]["path"]
                        rhs_path = rhs_set[rhs_ts]["path"]
                else:
                    break

            return current_pair

        refined_first = find_boundary(lhs, rhs, first[0], first[1], direction=-1)
        self.logger.debug(f"Refined First: L: {lhs[refined_first[0]]['path']} R: {rhs[refined_first[1]]['path']}")

        refined_last = find_boundary(lhs, rhs, last[0], last[1], direction=1)
        self.logger.debug(f"Refined Last:  L: {lhs[refined_last[0]]['path']} R: {rhs[refined_last[1]]['path']}")

        return refined_first, refined_last


    def create_segments_mapping(self) -> Tuple[List[Tuple[int, int]], FramesInfo, FramesInfo]:
        lhs_scene_changes = video_utils.detect_scene_changes(self.lhs_path, threshold = 0.3)
        rhs_scene_changes = video_utils.detect_scene_changes(self.rhs_path, threshold = 0.3)

        if len(lhs_scene_changes) == 0 or len(rhs_scene_changes) == 0:
            raise RuntimeError("Not enought scene changes detected")

        # extract all scenes
        self.lhs_all_frames = video_utils.extract_all_frames(self.lhs_path, self.lhs_all_wd, scale = 0.5, format = "png")
        self.rhs_all_frames = video_utils.extract_all_frames(self.rhs_path, self.rhs_all_wd, scale = 0.5, format = "png")

        lhs_key_frames_str = [str(self.lhs_all_frames[lhs]["frame_id"]) for lhs in lhs_scene_changes]
        rhs_key_frames_str = [str(self.rhs_all_frames[rhs]["frame_id"]) for rhs in rhs_scene_changes]

        self.logger.debug(f"lhs key frames: {' '.join(lhs_key_frames_str)}")
        self.logger.debug(f"rhs key frames: {' '.join(rhs_key_frames_str)}")

        # normalize frames. This could have been done in the previous step, however for some videos ffmpeg fails to save some of the frames when using 256x256 resolution. Who knows why...
        lhs_normalized_frames = self._normalize_frames(self.lhs_all_frames, self.lhs_normalized_wd)
        rhs_normalized_frames = self._normalize_frames(self.rhs_all_frames, self.rhs_normalized_wd)

        # extract key frames (as 'key' a scene change frame is meant)
        lhs_key_frames = PairMatcher._get_frames_for_timestamps(lhs_scene_changes, lhs_normalized_frames)
        rhs_key_frames = PairMatcher._get_frames_for_timestamps(rhs_scene_changes, rhs_normalized_frames)

        debug = DebugRoutines(self.debug_wd, self.lhs_all_frames, self.rhs_all_frames)

        debug.dump_frames(lhs_key_frames, "lhs key frames")
        debug.dump_frames(rhs_key_frames, "rhs key frames")

        # find matching keys
        matching_pairs = self._make_pairs(lhs_key_frames, rhs_key_frames, lhs_normalized_frames, rhs_normalized_frames)
        debug.dump_matches(matching_pairs, "initial matching")
        self.logger.debug("Pairs summary after initial matching:")
        self.logger.debug(PairMatcher.summarize_pairs(self.phash, matching_pairs, self.lhs_all_frames, self.rhs_all_frames, verbose = True))

        # adjust images and rerun matching
        lhs_normalized_cropped_frames, rhs_normalized_cropped_frames = self._crop_both_sets(
            pairs_with_timestamps = matching_pairs,
            lhs_frames = lhs_normalized_frames,
            rhs_frames = rhs_normalized_frames,
            lhs_cropped_dir = self.lhs_normalized_cropped_wd,
            rhs_cropped_dir = self.rhs_normalized_cropped_wd
        )

        lhs_key_frames = PairMatcher._get_frames_for_timestamps(lhs_scene_changes, lhs_normalized_cropped_frames)
        rhs_key_frames = PairMatcher._get_frames_for_timestamps(rhs_scene_changes, rhs_normalized_cropped_frames)
        matching_pairs = self._make_pairs(lhs_key_frames, rhs_key_frames, lhs_normalized_cropped_frames, rhs_normalized_cropped_frames)
        debug.dump_matches(matching_pairs, "secondary matching")

        prev_first, prev_last = None, None
        while True:
            self.interruption._check_for_stop()
            # crop frames basing on matching ones
            lhs_normalized_cropped_frames, rhs_normalized_cropped_frames = self._crop_both_sets(
                pairs_with_timestamps = matching_pairs,
                lhs_frames = lhs_normalized_frames,
                rhs_frames = rhs_normalized_frames,
                lhs_cropped_dir = self.lhs_normalized_cropped_wd,
                rhs_cropped_dir = self.rhs_normalized_cropped_wd
            )

            first_lhs, first_rhs = matching_pairs[0]
            last_lhs, last_rhs = matching_pairs[-1]
            first_lhs_path = lhs_normalized_cropped_frames[first_lhs]["path"]
            first_rhs_path = rhs_normalized_cropped_frames[first_rhs]["path"]
            last_lhs_path = lhs_normalized_cropped_frames[last_lhs]["path"]
            last_rhs_path = rhs_normalized_cropped_frames[last_rhs]["path"]
            self.logger.debug(f"First pair: {first_lhs_path} {first_rhs_path}")
            self.logger.debug(f"Last pair:  {last_lhs_path} {last_rhs_path}")

            # use new PhashCache as normalized frames are being regenerated every time
            phash4normalized = PhashCache()
            self.logger.debug(f"Cropped and aligned:       {PairMatcher.summarize_pairs(phash4normalized, matching_pairs, lhs_normalized_cropped_frames, rhs_normalized_cropped_frames)}")

            cutoff = self._calculate_cutoff(phash4normalized, matching_pairs, lhs_normalized_cropped_frames, rhs_normalized_cropped_frames)

            # try to locate first and last common frames
            first, last = self._look_for_boundaries(lhs_normalized_cropped_frames, rhs_normalized_cropped_frames, matching_pairs[0], matching_pairs[-1], cutoff)

            if first == prev_first and last == prev_last:
                break
            else:
                if first != prev_first:
                    matching_pairs = [first, *matching_pairs]
                    prev_first = first
                if last != prev_last:
                    matching_pairs = [*matching_pairs, last]
                    prev_last = last

            debug.dump_matches(matching_pairs, "improving boundaries")

        self.logger.debug("Status after boundaries lookup:\n")
        self.logger.debug(PairMatcher.summarize_segments(matching_pairs, self.lhs_fps, self.rhs_fps))
        self.logger.debug(PairMatcher.summarize_pairs(phash4normalized, matching_pairs, self.lhs_all_frames, self.rhs_all_frames, verbose = True))

        return matching_pairs, self.lhs_all_frames, self.rhs_all_frames
