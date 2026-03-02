
import cv2 as cv
import logging
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from sklearn.linear_model import RANSACRegressor, LinearRegression
from typing import Callable

from .debug_routines import DebugRoutines
from .melt_common import FramesInfo
from .phash_cache import PhashCache
from ..utils import files_utils, generic_utils, image_utils, video_utils


class PairMatcher:
    def __init__(self, interruption: generic_utils.InterruptibleProcess, wd: str, lhs_path: str, rhs_path: str, logger: logging.Logger) -> None:
        self.interruption = interruption
        self.wd = os.path.join(wd, "pair_matcher")
        self.lhs_path = lhs_path
        self.rhs_path = rhs_path
        self.logger = logger
        self.phash = PhashCache()
        self.lhs_fps = generic_utils.fps_str_to_float(video_utils.get_video_data(lhs_path)["video"][0]["fps"])
        self.rhs_fps = generic_utils.fps_str_to_float(video_utils.get_video_data(rhs_path)["video"][0]["fps"])

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
        def crop_5_percent(image: cv.typing.MatLike) -> cv.typing.MatLike:
            height, width = image.shape
            dx = int(width * 0.05)
            dy = int(height * 0.05)

            image_cropped = image[dy:height - dy, dx:width - dx]

            return image_cropped

        def process_frame(item):
            timestamp, info = item
            self.interruption.check_for_stop()
            path = info["path"]
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read frame from {path}")
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
    def calculate_ratio(pairs: list[tuple[int, int]]) -> float:
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
    def _get_new_info(info: dict[str, str], path: str) -> dict[str, str]:
        new_info = info.copy()
        new_info["path"] = path
        return new_info

    @staticmethod
    def _filter_low_detailed(scenes: FramesInfo) -> FramesInfo:
        valuable_scenes = {timestamp: info for timestamp, info in scenes.items() if PairMatcher._is_rich(info["path"])}
        return valuable_scenes

    @staticmethod
    def _get_frames_for_timestamps(timestamps: list[int], frames_info: FramesInfo) -> FramesInfo:
        frame_files = {timestamp: info for timestamp, info in frames_info.items() if timestamp in timestamps}

        return frame_files

    @staticmethod
    def filter_phash_outliers(phash: PhashCache, pairs: list[tuple[int, int]], lhs_set: FramesInfo, rhs_set: FramesInfo) -> list[tuple[int, int]]:
        if len(pairs) <= 3:
            # Too few data points for reliable MAD-based outlier detection.
            # Let downstream filters (ORB, history check) handle quality control.
            return pairs

        dists_array = np.array([abs(phash.get(lhs_set[l]["path"]) - phash.get(rhs_set[r]["path"])) for l, r in pairs], dtype=float)
        med = float(np.median(dists_array))
        mad = float(np.median(np.abs(dists_array - med)))
        threshold = med + 1.5 * mad
        return [pair for pair, dist in zip(pairs, dists_array) if dist <= threshold]

    @staticmethod
    def summarize_pairs(phash: PhashCache, pairs: list[tuple[int, int]], lhs: FramesInfo, rhs: FramesInfo, verbose: bool = False) -> str:
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
    def summarize_segments(pairs: list[tuple[int, int]], lhs_fps: float, rhs_fps: float, verbose: bool = True) -> str:
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
    def _interpolate_crop_rects(timestamps_list: list[int], rects_list: list[tuple[int, int, int, int]]) -> Callable[[int], tuple[int, int, int, int]]:
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
    def _find_interpolated_crop(pairs_with_timestamps: list[tuple[int, int]], lhs_frames: FramesInfo, rhs_frames: FramesInfo) -> tuple[Callable[[int], tuple[int, int, int, int]], Callable[[int], tuple[int, int, int, int]]]:
        timestamps_lhs = []
        timestamps_rhs = []
        lhs_crops = []
        rhs_crops = []

        for lhs_t, rhs_t in pairs_with_timestamps:
            lhs_info = lhs_frames[lhs_t]
            rhs_info = rhs_frames[rhs_t]
            lhs_img = cv.imread(lhs_info["path"], cv.IMREAD_GRAYSCALE)
            rhs_img = cv.imread(rhs_info["path"], cv.IMREAD_GRAYSCALE)
            if lhs_img is None or rhs_img is None:
                continue

            orb = cv.ORB_create(1000)  # type: ignore[attr-defined]
            kp1, des1 = orb.detectAndCompute(lhs_img, None)
            kp2, des2 = orb.detectAndCompute(rhs_img, None)
            if des1 is None or des2 is None:
                continue

            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            if len(matches) < 3:
                continue

            matches = sorted(matches, key=lambda x: x.distance)
            pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

            h_matrix, inliers = cv.estimateAffinePartial2D(pts2, pts1, method=cv.RANSAC)
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

    def _apply_crop_interpolated(self, frames: FramesInfo, dst_dir: str, crop_fn: Callable[[int], tuple[int, int, int, int]]) -> FramesInfo:
        def _process_frame(item):
            timestamp, info = item
            self.interruption.check_for_stop()
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

    def _three_before(self, timestamps: list[int], target: int) -> list[int]:
        timestamps = sorted(timestamps)
        idx = int(np.searchsorted(timestamps, target))
        return list(filter(lambda x: x in timestamps, timestamps[max(0, idx-3):idx]))

    def _nearest_three(self, timestamps: list[int], target: int) -> list[int]:
        timestamps = sorted(timestamps)
        idx = int(np.searchsorted(timestamps, target))
        return list(filter(lambda x: x in timestamps, timestamps[max(0, idx-1):idx+2]))

    @staticmethod
    def _snap_to_nearest_frame(keys: list[int], target: int) -> int:
        """Return the timestamp from *keys* closest to *target*."""
        idx = int(np.searchsorted(keys, target))
        candidates = []
        if idx > 0:
            candidates.append(keys[idx - 1])
        if idx < len(keys):
            candidates.append(keys[idx])
        return min(candidates, key=lambda k: abs(k - target))

    def _best_phash_match(self, lhs_ts: int, rhs_ts_guess: int, lhs_all_set: FramesInfo, rhs_all_set: FramesInfo) -> tuple[int, int] | None:
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

    def _build_matches(self, lhs: FramesInfo, rhs: FramesInfo) -> list[tuple[int, int, int]]:
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

    def _build_initial_candidates(self, lhs: FramesInfo, rhs: FramesInfo) -> list[tuple[int, int]]:
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

    def _reject_outliers(self, pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(pairs) < 3:
            return pairs

        lhs_vals, rhs_vals = zip(*pairs)
        lhs_array = np.array(lhs_vals).reshape(-1, 1)
        rhs_array = np.array(rhs_vals)
        model = RANSACRegressor(LinearRegression(), residual_threshold=5000)
        model.fit(lhs_array, rhs_array)
        inliers = model.inlier_mask_
        return [p for p, keep in zip(pairs, inliers) if keep]

    def _check_history(self, pair: tuple[int, int], lhs_pool: FramesInfo, rhs_pool: FramesInfo, cutoff: float) -> bool:
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

    def _extrapolate_matches(self, known_pairs: list[tuple[int, int]], lhs_pool: FramesInfo, rhs_pool: FramesInfo, phash: PhashCache) -> list[tuple[int, int]]:
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
        pairs_with_timestamps: list[tuple[int, int]],
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo,
        lhs_cropped_dir: str,
        rhs_cropped_dir: str
    ) -> tuple[FramesInfo, FramesInfo]:
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
        pairs: list[tuple[int, int]],
        lhs: FramesInfo,
        rhs: FramesInfo
    ) -> int:
        distances = [abs(phash.get(lhs[lhs_ts]["path"]) - phash.get(rhs[rhs_ts]["path"])) for lhs_ts, rhs_ts in pairs]

        arr = np.array(distances)
        median = np.median(arr)
        std = np.std(arr)

        return median + std * 2

    def _make_pairs(self, lhs: FramesInfo, rhs: FramesInfo, lhs_all: FramesInfo, rhs_all: FramesInfo) -> list[tuple[int, int]]:
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

        extrapolated_refined: list[tuple[int, int]] = []
        for l, r in extrapolated:
            best_match = self._best_phash_match(l, r, lhs_all, rhs_all)
            if best_match is not None:
                extrapolated_refined.append(best_match)
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


    @staticmethod
    def _find_entropy_transition(frames: FramesInfo, start_ts: int, direction: int, entropy_threshold: float = 3.5) -> int | None:
        """Find the timestamp where entropy crosses *entropy_threshold*.

        Walks from *start_ts* in *direction* (+1 = forward, -1 = backward) and
        returns the first timestamp whose entropy is on the "rich" side of the
        threshold (i.e. above it when walking into content, below it when walking
        into darkness).  Returns ``None`` when no transition is found.
        """
        keys = sorted(frames.keys())
        try:
            idx = keys.index(start_ts)
        except ValueError:
            return None

        start_entropy = image_utils.image_entropy(frames[start_ts]["path"])
        looking_for_rich = start_entropy <= entropy_threshold

        while True:
            idx += direction
            if idx < 0 or idx >= len(keys):
                return None
            ts = keys[idx]
            e = image_utils.image_entropy(frames[ts]["path"])
            if looking_for_rich and e > entropy_threshold:
                return ts
            if not looking_for_rich and e <= entropy_threshold:
                # We walked from rich into low-entropy — return the previous
                # (last rich) frame.
                prev_idx = idx - direction
                return keys[prev_idx] if 0 <= prev_idx < len(keys) else None

        return None

    def _look_for_boundaries(self, lhs: FramesInfo, rhs: FramesInfo, first: tuple[int, int], last: tuple[int, int], cutoff: float, max_gap_seconds: float = 15.0):
        """Find the first and last common frame pair by walking outward from known matches.

        Uses the linear time mapping derived from *first* and *last* to predict
        where each LHS frame should appear in RHS, then walks from the current
        boundaries toward the edges of the video.  Allows configurable gaps of
        non-matching frames before giving up, instead of stopping on first miss.

        When the search enters a low-entropy region (dark frames, end credits),
        phash matching becomes unreliable.  In that case the boundary is
        extrapolated linearly from well-matched pairs, but only if both files
        show a consistent entropy transition at the predicted position.
        """
        self.logger.debug("Improving boundaries")
        self.logger.debug(f"Current first: {first} and last: {last}")
        phash = PhashCache()
        ratio = PairMatcher.calculate_ratio([first, last])

        # Ensure cutoff is not absurdly tight — with few calibration pairs or
        # cropped frames the computed cutoff can be as low as 4, which makes the
        # search unable to extend even a single frame.
        cutoff = max(cutoff, 16)

        def find_boundary(anchor: tuple[int, int], reference: tuple[int, int], direction: int) -> tuple[tuple[int, int], bool]:
            """Walk from *anchor* in *direction*, using linear prediction to find matches.

            Returns ``(best_pair, entered_low_entropy)`` where the flag indicates
            that the search stopped because it reached a low-entropy zone rather
            than exhausting the gap budget on high-entropy mismatches.
            """
            lhs_keys = sorted(lhs.keys())
            rhs_keys = sorted(rhs.keys())

            current_best = anchor
            max_gap = int(max_gap_seconds * self.lhs_fps)
            consecutive_misses = 0
            entered_low_entropy = False

            # Check roughly every half-second worth of frames
            step = max(1, int(self.lhs_fps * 0.5))

            start_idx = lhs_keys.index(anchor[0])
            i = 0

            while True:
                i += step
                idx = start_idx + direction * i
                if idx < 0 or idx >= len(lhs_keys):
                    break

                lhs_ts = lhs_keys[idx]

                # Check if we entered a low-entropy zone — phash is unreliable here
                if not PairMatcher._is_rich(lhs[lhs_ts]["path"]):
                    entered_low_entropy = True
                    break

                # Predict the corresponding RHS timestamp using the linear mapping
                predicted_rhs = int(anchor[1] + (lhs_ts - anchor[0]) / ratio)

                rhs_near = self._nearest_three(rhs_keys, predicted_rhs)

                best_dist = float('inf')
                best_pair: tuple[int, int] | None = None
                for rhs_ts in rhs_near:
                    if rhs_ts not in rhs or lhs_ts not in lhs:
                        continue
                    d = abs(phash.get(lhs[lhs_ts]["path"]) - phash.get(rhs[rhs_ts]["path"]))
                    if d < best_dist:
                        best_dist = d
                        best_pair = (lhs_ts, rhs_ts)

                if best_pair is not None and best_dist <= cutoff:
                    # Validate ratio against reference to avoid false positives
                    if reference != best_pair:
                        cand_ratio = PairMatcher.calculate_ratio([best_pair, reference])
                        if not PairMatcher.is_ratio_acceptable(cand_ratio, ratio):
                            consecutive_misses += step
                            if consecutive_misses >= max_gap:
                                break
                            continue

                    current_best = best_pair
                    consecutive_misses = 0
                else:
                    consecutive_misses += step
                    if consecutive_misses >= max_gap:
                        break

            return current_best, entered_low_entropy

        refined_first, first_low_entropy = find_boundary(first, last, direction=-1)
        self.logger.debug(f"Refined First: L: {lhs[refined_first[0]]['path']} R: {rhs[refined_first[1]]['path']}"
                          f"{' (stopped at low-entropy zone)' if first_low_entropy else ''}")

        refined_last, last_low_entropy = find_boundary(last, first, direction=1)
        self.logger.debug(f"Refined Last:  L: {lhs[refined_last[0]]['path']} R: {rhs[refined_last[1]]['path']}"
                          f"{' (stopped at low-entropy zone)' if last_low_entropy else ''}")

        # Extrapolate through low-entropy regions when possible
        refined_first = self._extrapolate_through_low_entropy(
            lhs, rhs, refined_first, refined_last, ratio, direction=-1,
            entered_low_entropy=first_low_entropy,
        )
        refined_last = self._extrapolate_through_low_entropy(
            lhs, rhs, refined_last, refined_first, ratio, direction=1,
            entered_low_entropy=last_low_entropy,
        )

        return refined_first, refined_last

    def _extrapolate_through_low_entropy(
        self,
        lhs: FramesInfo,
        rhs: FramesInfo,
        boundary: tuple[int, int],
        reference: tuple[int, int],
        ratio: float,
        direction: int,
        entered_low_entropy: bool,
    ) -> tuple[int, int]:
        """Extend *boundary* through a low-entropy zone using linear extrapolation.

        Instead of comparing individual frames (unreliable on dark/uniform
        content), we locate the entropy transition point (dark↔content) in each
        file independently, then check whether those transitions are consistent
        with the known time-mapping ratio.

        If they are — we adopt the transition points as the new boundary.
        If not — we keep the original *boundary* and emit a warning that the
        files likely have different intro/outro lengths.
        """
        if not entered_low_entropy:
            return boundary

        label = "start" if direction == -1 else "end"
        lhs_keys = sorted(lhs.keys())
        rhs_keys = sorted(rhs.keys())

        # Find where the entropy transition is in each file
        lhs_transition = PairMatcher._find_entropy_transition(lhs, boundary[0], direction)
        predicted_rhs_boundary = int(boundary[1] + (lhs_transition - boundary[0]) / ratio) if lhs_transition is not None else None
        rhs_transition = PairMatcher._find_entropy_transition(rhs, boundary[1], direction)

        if lhs_transition is None and rhs_transition is None:
            # Both files are low-entropy all the way to the edge (e.g. both
            # start/end with black).  Extrapolate to the edge.
            if direction == -1:
                ext_lhs = lhs_keys[0]
            else:
                ext_lhs = lhs_keys[-1]
            ext_rhs = int(boundary[1] + (ext_lhs - boundary[0]) / ratio)
            ext_rhs = PairMatcher._snap_to_nearest_frame(rhs_keys, ext_rhs)

            self.logger.warning(
                f"Boundary {label}: both files are low-entropy to the edge. "
                f"Extrapolating {label} boundary from ({boundary[0]}, {boundary[1]}) "
                f"to ({ext_lhs}, {ext_rhs}). Precision cannot be verified."
            )
            return (ext_lhs, ext_rhs)

        if lhs_transition is None or rhs_transition is None:
            self.logger.warning(
                f"Boundary {label}: entropy transition found in only one file "
                f"(LHS={'yes' if lhs_transition else 'no'}, RHS={'yes' if rhs_transition else 'no'}). "
                f"The files may have different {label} sequences. "
                f"Keeping boundary at ({boundary[0]}, {boundary[1]})."
            )
            return boundary

        # Both transitions found — check if they are consistent with the ratio
        if predicted_rhs_boundary is not None:
            transition_error_ms = abs(rhs_transition - predicted_rhs_boundary)
            # Allow up to 2 seconds of discrepancy for the transition positions
            max_transition_error_ms = 2000

            if transition_error_ms <= max_transition_error_ms:
                # Transitions are consistent — use the transition as the boundary
                # Pick the nearer-to-edge timestamp so we don't crop content
                if direction == -1:
                    new_lhs = min(lhs_transition, boundary[0])
                    new_rhs = int(boundary[1] + (new_lhs - boundary[0]) / ratio)
                else:
                    new_lhs = max(lhs_transition, boundary[0])
                    new_rhs = int(boundary[1] + (new_lhs - boundary[0]) / ratio)
                new_rhs = PairMatcher._snap_to_nearest_frame(rhs_keys, new_rhs)

                self.logger.info(
                    f"Boundary {label}: extrapolating through low-entropy zone. "
                    f"LHS transition at {lhs_transition}ms, RHS transition at {rhs_transition}ms "
                    f"(predicted {predicted_rhs_boundary}ms, error {transition_error_ms}ms). "
                    f"Moving boundary from ({boundary[0]}, {boundary[1]}) to ({new_lhs}, {new_rhs})."
                )
                return (new_lhs, new_rhs)
            else:
                self.logger.warning(
                    f"Boundary {label}: entropy transitions are inconsistent with time mapping. "
                    f"LHS transition at {lhs_transition}ms, RHS transition at {rhs_transition}ms "
                    f"(predicted {predicted_rhs_boundary}ms, error {transition_error_ms}ms > {max_transition_error_ms}ms). "
                    f"The files likely have different {label} sequences. "
                    f"Keeping boundary at ({boundary[0]}, {boundary[1]})."
                )
                return boundary

        return boundary


    def create_segments_mapping(self) -> tuple[list[tuple[int, int]], FramesInfo, FramesInfo]:
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

        prev_first, prev_last = None, None
        while True:
            self.interruption.check_for_stop()
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

            debug.dump_matches(matching_pairs, f"improving boundaries")

        self.logger.debug("Status after boundaries lookup:\n")
        self.logger.debug(PairMatcher.summarize_segments(matching_pairs, self.lhs_fps, self.rhs_fps))
        self.logger.debug(PairMatcher.summarize_pairs(phash4normalized, matching_pairs, self.lhs_all_frames, self.rhs_all_frames, verbose = True))

        # Final boundary search on uncropped normalized frames.
        # The cropped-frame search may fail near edges because the crop
        # interpolation extrapolates from the nearest known pair, which can
        # be significantly off when the spatial transformation changes over time.
        # Searching the uncropped frames avoids this limitation.
        phash_uncropped = PhashCache()
        uncropped_cutoff = self._calculate_cutoff(phash_uncropped, matching_pairs, lhs_normalized_frames, rhs_normalized_frames)
        final_first, final_last = self._look_for_boundaries(
            lhs_normalized_frames, rhs_normalized_frames,
            matching_pairs[0], matching_pairs[-1],
            uncropped_cutoff,
        )
        if final_first != matching_pairs[0]:
            matching_pairs = [final_first, *matching_pairs]
            self.logger.debug(f"Uncropped search extended first boundary to {final_first}")
        if final_last != matching_pairs[-1]:
            matching_pairs = [*matching_pairs, final_last]
            self.logger.debug(f"Uncropped search extended last boundary to {final_last}")

        debug.dump_matches(matching_pairs, "after uncropped boundary search")
        self.logger.debug("Final status:\n")
        self.logger.debug(PairMatcher.summarize_segments(matching_pairs, self.lhs_fps, self.rhs_fps))
        self.logger.debug(PairMatcher.summarize_pairs(phash_uncropped, matching_pairs, self.lhs_all_frames, self.rhs_all_frames, verbose = True))

        return matching_pairs, self.lhs_all_frames, self.rhs_all_frames
