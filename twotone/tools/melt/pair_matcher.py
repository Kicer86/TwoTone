
from typing import Any

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
        if not ratios:
            return float("nan")
        median_ratio = np.median(ratios)
        return float(median_ratio)

    @staticmethod
    def is_ratio_acceptable(ratio: float, perfect_ratio: float) -> bool:
        return abs(ratio - perfect_ratio) < 0.05 * perfect_ratio

    @staticmethod
    def coverage_summary(
        mappings: list[tuple[int, int]],
        lhs_duration_ms: int,
        rhs_duration_ms: int,
    ) -> dict[str, Any]:
        """Compute a human-readable coverage summary for matched files.

        Returns a dict with:
        - ``full_coverage``: True when first/last pairs are within one frame
          (40 ms) of both video edges
        - ``lhs_start_gap_s`` / ``rhs_start_gap_s``: unmatched seconds at start
        - ``lhs_end_gap_s`` / ``rhs_end_gap_s``: unmatched seconds at end
        - ``ratio``: speed ratio between the two files
        """
        first = mappings[0]
        last = mappings[-1]

        lhs_start_gap = first[0]
        rhs_start_gap = first[1]
        lhs_end_gap = max(0, lhs_duration_ms - last[0])
        rhs_end_gap = max(0, rhs_duration_ms - last[1])

        edge_tolerance_ms = 100  # ~2-3 frames; last extracted frame may not reach video end
        full_coverage = (
            lhs_start_gap <= edge_tolerance_ms
            and rhs_start_gap <= edge_tolerance_ms
            and lhs_end_gap <= edge_tolerance_ms
            and rhs_end_gap <= edge_tolerance_ms
        )

        return {
            "full_coverage": full_coverage,
            "lhs_start_gap_s": lhs_start_gap / 1000,
            "rhs_start_gap_s": rhs_start_gap / 1000,
            "lhs_end_gap_s": lhs_end_gap / 1000,
            "rhs_end_gap_s": rhs_end_gap / 1000,
            "ratio": PairMatcher.calculate_ratio(mappings),
        }

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
        if not pairs:
            return "Pairs: 0"

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
        if not pairs:
            return 16  # sensible default when no pairs available

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

    def _edge_content_matches(
        self,
        lhs: FramesInfo,
        rhs: FramesInfo,
        lhs_keys: list[int],
        rhs_keys: list[int],
        phash: "PhashCache",
        first: tuple[int, int],
        last: tuple[int, int],
        ratio: float,
        cutoff: float,
        direction: int,
    ) -> bool:
        """Check if both videos share the same content at one edge.

        *direction* = -1 checks the video start, +1 checks the video end.

        Compares the edge frame of LHS against the edge frame of RHS (not
        a prediction-based frame).  Validates with pHash, ratio consistency,
        and ORB.  Returns ``True`` when the edge pair looks like a valid match,
        meaning ``find_boundary`` should be able to reach the edge.
        """
        edge_lhs_ts = lhs_keys[0] if direction == -1 else lhs_keys[-1]
        edge_rhs_ts = rhs_keys[0] if direction == -1 else rhs_keys[-1]

        if edge_lhs_ts not in lhs or edge_rhs_ts not in rhs:
            return False

        d = abs(phash.get(lhs[edge_lhs_ts]["path"]) - phash.get(rhs[edge_rhs_ts]["path"]))
        if d > cutoff:
            return False

        # Ratio validation against the farther known pair
        reference = last if direction == -1 else first
        edge_pair = (edge_lhs_ts, edge_rhs_ts)
        if reference != edge_pair:
            cand_ratio = PairMatcher.calculate_ratio([edge_pair, reference])
            if not PairMatcher.is_ratio_acceptable(cand_ratio, ratio):
                return False

        # ORB verification on the half-resolution extracted frames
        lhs_path = self.lhs_all_frames[edge_lhs_ts]["path"] if edge_lhs_ts in self.lhs_all_frames else lhs[edge_lhs_ts]["path"]
        rhs_path = self.rhs_all_frames[edge_rhs_ts]["path"] if edge_rhs_ts in self.rhs_all_frames else rhs[edge_rhs_ts]["path"]
        if not image_utils.are_images_similar(lhs_path, rhs_path):
            return False

        return True

    def _look_for_boundaries(self, lhs: FramesInfo, rhs: FramesInfo, first: tuple[int, int], last: tuple[int, int], cutoff: float, max_gap_seconds: float = 15.0, extrapolate: bool = True):
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

        # When first == last (single pair) the ratio is NaN;
        # fall back to fps-based ratio.
        if np.isnan(ratio):
            ratio = self.lhs_fps / self.rhs_fps

        # Ensure cutoff is not absurdly tight — with few calibration pairs or
        # cropped frames the computed cutoff can be as low as 4, which makes the
        # search unable to extend even a single frame.
        cutoff = max(cutoff, 16)

        # --- Fast edge pre-check ---
        # Before the iterative search, check whether both videos share content
        # at each edge.  When they do, extend the gap budget for find_boundary
        # so it does not give up before reaching the video edge.  The anchor
        # and prediction logic are left untouched for maximum precision.
        lhs_keys = sorted(lhs.keys())
        rhs_keys = sorted(rhs.keys())

        first_gap_seconds = max_gap_seconds
        last_gap_seconds = max_gap_seconds

        for direction in [-1, 1]:
            label = "start" if direction == -1 else "end"
            matches = self._edge_content_matches(
                lhs, rhs, lhs_keys, rhs_keys, phash,
                first, last, ratio, cutoff, direction,
            )
            if not matches:
                self.logger.info(
                    f"Edge pre-check ({label}): edges do NOT visually match "
                    f"— using default gap budget {max_gap_seconds}s"
                )
                continue

            anchor = first if direction == -1 else last
            edge_lhs = lhs_keys[0] if direction == -1 else lhs_keys[-1]
            distance_s = abs(anchor[0] - edge_lhs) / 1000.0

            if direction == -1:
                first_gap_seconds = max(max_gap_seconds, distance_s + 2.0)
                self.logger.info(
                    f"Edge pre-check ({label}): edges match, "
                    f"extending gap budget to {first_gap_seconds:.1f}s "
                    f"(anchor={anchor[0]}ms, edge={edge_lhs}ms, distance={distance_s:.1f}s)"
                )
            else:
                last_gap_seconds = max(max_gap_seconds, distance_s + 2.0)
                self.logger.info(
                    f"Edge pre-check ({label}): edges match, "
                    f"extending gap budget to {last_gap_seconds:.1f}s "
                    f"(anchor={anchor[0]}ms, edge={edge_lhs}ms, distance={distance_s:.1f}s)"
                )

        def find_boundary(anchor: tuple[int, int], reference: tuple[int, int], direction: int, gap_seconds: float = max_gap_seconds) -> tuple[tuple[int, int], bool]:
            """Walk from *anchor* in *direction*, using linear prediction to find matches.

            Returns ``(best_pair, entered_low_entropy)`` where the flag indicates
            that the search stopped because it reached a low-entropy zone rather
            than exhausting the gap budget on high-entropy mismatches.
            """
            lhs_keys = sorted(lhs.keys())
            rhs_keys = sorted(rhs.keys())

            current_best = anchor
            max_gap = int(gap_seconds * self.lhs_fps)
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

                # Check if we entered a sustained low-entropy zone.
                # A single dark frame (e.g. at a scene transition) is treated
                # as a miss rather than halting the search — only a sustained
                # dark zone (e.g. a black intro/outro) triggers the
                # low-entropy path.
                if not PairMatcher._is_rich(lhs[lhs_ts]["path"]):
                    # Look ahead 1.5 seconds — short dark zones at scene
                    # transitions should not stop the search; only sustained
                    # black regions (intros/outros) should trigger the
                    # low-entropy path.
                    look_ahead = max(3, int(1.5 * self.lhs_fps))
                    sustained = True
                    for la in range(1, look_ahead + 1):
                        la_idx = idx + direction * la
                        if 0 <= la_idx < len(lhs_keys):
                            if PairMatcher._is_rich(lhs[lhs_keys[la_idx]]["path"]):
                                sustained = False
                                break
                        # past the edge — treat as sustained
                    if sustained:
                        # Instead of immediately stopping, try to jump past the
                        # dark zone.  A sustained dark zone in the middle of
                        # shared content is typically a scene transition (e.g. a
                        # fade-to-black between scenes), not the actual video
                        # boundary.  We scan past all dark frames, find the
                        # first rich frame on the other side, and try a phash
                        # match.  If it succeeds, the walk continues; if it
                        # fails (or we reach the video edge), we declare a
                        # low-entropy boundary.
                        jumped_pair: tuple[int, int] | None = None
                        jump_idx = idx + direction
                        while 0 <= jump_idx < len(lhs_keys):
                            if PairMatcher._is_rich(lhs[lhs_keys[jump_idx]]["path"]):
                                jump_ts = lhs_keys[jump_idx]
                                predicted_rhs = int(anchor[1] + (jump_ts - anchor[0]) / ratio)
                                rhs_near = self._nearest_three(rhs_keys, predicted_rhs)
                                bdist = float('inf')
                                bpair: tuple[int, int] | None = None
                                for rts in rhs_near:
                                    if rts not in rhs or jump_ts not in lhs:
                                        continue
                                    d = abs(phash.get(lhs[jump_ts]["path"]) - phash.get(rhs[rts]["path"]))
                                    if d < bdist:
                                        bdist = d
                                        bpair = (jump_ts, rts)
                                if bpair is not None and bdist <= cutoff:
                                    accept = True
                                    if reference != bpair:
                                        cand_ratio = PairMatcher.calculate_ratio([bpair, reference])
                                        accept = PairMatcher.is_ratio_acceptable(cand_ratio, ratio)
                                    if accept:
                                        jumped_pair = bpair
                                break
                            jump_idx += direction

                        if jumped_pair is not None:
                            dark_start = lhs_ts
                            dark_end = lhs_keys[jump_idx]
                            self.logger.info(
                                f"Jumped over dark zone {dark_start}ms → {dark_end}ms "
                                f"(pair: {jumped_pair})"
                            )
                            current_best = jumped_pair
                            consecutive_misses = 0
                            i = direction * (jump_idx - start_idx)
                            continue
                        else:
                            entered_low_entropy = True
                            break
                    # Transient dark frame — treat as a miss
                    consecutive_misses += step
                    if consecutive_misses >= max_gap:
                        break
                    continue

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

            # When the search reached the video edge without entering a
            # low-entropy zone, try matching the actual edge frame.  The coarse
            # step size may have skipped it.  We attempt this even for dark
            # edge frames — phash+ratio validation is sufficient to prevent
            # false matches.
            if not entered_low_entropy and consecutive_misses < max_gap:
                edge_idx = 0 if direction == -1 else len(lhs_keys) - 1
                edge_ts = lhs_keys[edge_idx]
                if edge_ts != current_best[0] and edge_ts in lhs:
                    predicted_rhs = int(anchor[1] + (edge_ts - anchor[0]) / ratio)
                    clamped = max(rhs_keys[0], min(rhs_keys[-1], predicted_rhs))
                    rhs_near = self._nearest_three(rhs_keys, clamped)
                    best_edge_dist = float('inf')
                    best_edge_pair: tuple[int, int] | None = None
                    for rts in rhs_near:
                        if rts not in rhs:
                            continue
                        d = abs(phash.get(lhs[edge_ts]["path"]) - phash.get(rhs[rts]["path"]))
                        if d < best_edge_dist:
                            best_edge_dist = d
                            best_edge_pair = (edge_ts, rts)
                    if best_edge_pair is not None and best_edge_dist <= cutoff:
                        accept = True
                        if reference != best_edge_pair:
                            cand_ratio = PairMatcher.calculate_ratio([best_edge_pair, reference])
                            accept = PairMatcher.is_ratio_acceptable(cand_ratio, ratio)
                        if accept:
                            current_best = best_edge_pair

            return current_best, entered_low_entropy

        refined_first, first_low_entropy = find_boundary(first, last, direction=-1, gap_seconds=first_gap_seconds)
        self.logger.info(
            f"Boundary start: walked {first[0]}ms → {refined_first[0]}ms "
            f"({'entered low-entropy zone' if first_low_entropy else 'gap budget or edge reached'}, "
            f"budget={first_gap_seconds:.1f}s)"
        )

        refined_last, last_low_entropy = find_boundary(last, first, direction=1, gap_seconds=last_gap_seconds)
        self.logger.info(
            f"Boundary end: walked {last[0]}ms → {refined_last[0]}ms "
            f"({'entered low-entropy zone' if last_low_entropy else 'gap budget or edge reached'}, "
            f"budget={last_gap_seconds:.1f}s)"
        )

        if extrapolate:
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
        """Extend *boundary* through a low-entropy zone to the video edge.

        When the boundary search stops at a low-entropy zone (black frames,
        end credits, etc.), this method checks whether the region between the
        current boundary and the video edge is consistently low-entropy in
        *both* files.  If so, the boundary is linearly extrapolated to the
        edge; otherwise it is kept unchanged (the files likely have different
        intro/outro content).

        The coarse step used by ``find_boundary`` may overshoot the exact
        content→dark transition by a few frames.  To compensate, this method
        walks from the boundary toward the edge, skips any high-entropy frames
        near the boundary (at most ``step * 3``), and then verifies that the
        remaining zone to the edge is entirely low-entropy.  The tolerance is
        ``step * 3`` rather than ``step`` because the RHS boundary is predicted
        via the linear ratio and may be off by more than the walk stride.
        """
        if not entered_low_entropy:
            return boundary

        label = "start" if direction == -1 else "end"
        lhs_keys = sorted(lhs.keys())
        rhs_keys = sorted(rhs.keys())
        step = max(1, int(self.lhs_fps * 0.5))

        # Determine LHS edge and gap frames between boundary and edge
        if direction == -1:
            edge_lhs = lhs_keys[0]
            lhs_gap = [k for k in lhs_keys if k < boundary[0]]
        else:
            edge_lhs = lhs_keys[-1]
            lhs_gap = [k for k in lhs_keys if k > boundary[0]]

        if not lhs_gap:
            return boundary

        # Walk from boundary toward edge — skip high-entropy content frames
        # the coarse step may have jumped over, then verify the remaining
        # zone to the edge is all low-entropy.
        from_boundary = sorted(lhs_gap, reverse=(direction == -1))
        le_start: int | None = None
        for i, ts in enumerate(from_boundary):
            if not PairMatcher._is_rich(lhs[ts]["path"]):
                le_start = i
                break

        if le_start is None:
            self.logger.warning(
                f"Boundary {label}: gap between boundary ({boundary[0]}ms) and "
                f"edge ({edge_lhs}ms) is all high-entropy in LHS. "
                f"Keeping boundary at ({boundary[0]}, {boundary[1]})."
            )
            return boundary

        max_skip = step * 3
        if le_start > max_skip:
            self.logger.warning(
                f"Boundary {label}: {le_start} high-entropy frames between "
                f"boundary ({boundary[0]}ms) and low-entropy zone (max "
                f"expected: {max_skip}). "
                f"Keeping boundary at ({boundary[0]}, {boundary[1]})."
            )
            return boundary

        remaining_lhs = from_boundary[le_start:]
        non_le_frames = [k for k in remaining_lhs if PairMatcher._is_rich(lhs[k]["path"])]
        noise_ratio = len(non_le_frames) / len(remaining_lhs) if remaining_lhs else 1.0
        if noise_ratio > 0.05:
            self.logger.warning(
                f"Boundary {label}: zone between boundary ({boundary[0]}ms) "
                f"and edge ({edge_lhs}ms) has {len(non_le_frames)}/{len(remaining_lhs)} "
                f"({noise_ratio:.1%}) high-entropy frames in LHS (first at "
                f"{non_le_frames[0]}ms, last at {non_le_frames[-1]}ms). "
                f"Cannot extrapolate through non-uniform zone. "
                f"Keeping boundary at ({boundary[0]}, {boundary[1]})."
            )
            return boundary
        elif non_le_frames:
            self.logger.info(
                f"Boundary {label}: tolerating {len(non_le_frames)}/{len(remaining_lhs)} "
                f"({noise_ratio:.1%}) sparse high-entropy frames in LHS zone "
                f"(boundary {boundary[0]}ms to edge {edge_lhs}ms)."
            )

        # Predict RHS edge position and clamp to valid range
        predicted_rhs = int(boundary[1] + (edge_lhs - boundary[0]) / ratio)
        clamped_rhs = max(rhs_keys[0], min(rhs_keys[-1], predicted_rhs))

        # Check RHS gap: same logic — skip content frames near boundary,
        # then verify remaining zone is all low-entropy.
        if direction == -1:
            rhs_gap = [k for k in rhs_keys if k < boundary[1] and k >= clamped_rhs]
        else:
            rhs_gap = [k for k in rhs_keys if k > boundary[1] and k <= clamped_rhs]

        if rhs_gap:
            from_boundary_rhs = sorted(rhs_gap, reverse=(direction == -1))
            rhs_le_start: int | None = None
            for i, ts in enumerate(from_boundary_rhs):
                if not PairMatcher._is_rich(rhs[ts]["path"]):
                    rhs_le_start = i
                    break

            if rhs_le_start is None:
                self.logger.warning(
                    f"Boundary {label}: RHS gap ({clamped_rhs}ms to "
                    f"{boundary[1]}ms) is all high-entropy. "
                    f"Keeping boundary at ({boundary[0]}, {boundary[1]})."
                )
                return boundary

            if rhs_le_start > max_skip:
                self.logger.warning(
                    f"Boundary {label}: {rhs_le_start} high-entropy RHS frames "
                    f"between boundary and low-entropy zone (max "
                    f"expected: {max_skip}). "
                    f"Keeping boundary at ({boundary[0]}, {boundary[1]})."
                )
                return boundary

            remaining_rhs = from_boundary_rhs[rhs_le_start:]
            rhs_non_le = [k for k in remaining_rhs if PairMatcher._is_rich(rhs[k]["path"])]
            rhs_noise_ratio = len(rhs_non_le) / len(remaining_rhs) if remaining_rhs else 1.0
            if rhs_noise_ratio > 0.05:
                self.logger.warning(
                    f"Boundary {label}: RHS zone has {len(rhs_non_le)}/{len(remaining_rhs)} "
                    f"({rhs_noise_ratio:.1%}) high-entropy frames — not contiguous. "
                    f"Keeping boundary at ({boundary[0]}, {boundary[1]})."
                )
                return boundary
            elif rhs_non_le:
                self.logger.info(
                    f"Boundary {label}: tolerating {len(rhs_non_le)}/{len(remaining_rhs)} "
                    f"({rhs_noise_ratio:.1%}) sparse high-entropy frames in RHS zone."
                )

        # Both files have consistently low-entropy gaps — extrapolate to edge
        new_rhs = PairMatcher._snap_to_nearest_frame(rhs_keys, clamped_rhs)

        self.logger.info(
            f"Boundary {label}: extrapolating through low-entropy zone from "
            f"({boundary[0]}, {boundary[1]}) to ({edge_lhs}, {new_rhs})."
        )
        return (edge_lhs, new_rhs)

    def _snap_to_edges(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_all_frames: FramesInfo,
        rhs_all_frames: FramesInfo,
        snap_frames: int = 4,
    ) -> list[tuple[int, int]]:
        """Snap first/last pair timestamps to video edges when within a few frames.

        When the first or last mapping pair is very close to a video edge
        (within *snap_frames* frames), keeping a tiny head or tail audio
        segment is pointless and may cause artifacts.  This method extends
        those pairs to the edge (timestamp 0 for start, video duration for
        end) so downstream audio patching skips the trivial segments.

        Synthetic entries are added to *lhs_all_frames* / *rhs_all_frames*
        for any newly created timestamps (pointing to the nearest real frame
        file) so debug routines remain functional.
        """
        lhs_threshold_ms = snap_frames * 1000 / self.lhs_fps
        rhs_threshold_ms = snap_frames * 1000 / self.rhs_fps

        lhs_duration = video_utils.get_video_duration(self.lhs_path)
        rhs_duration = video_utils.get_video_duration(self.rhs_path)

        lhs_keys = sorted(lhs_all_frames.keys())
        rhs_keys = sorted(rhs_all_frames.keys())

        first_l, first_r = matching_pairs[0]
        last_l, last_r = matching_pairs[-1]

        # --- Start edge ---
        new_first_l = 0 if first_l <= lhs_threshold_ms else first_l
        new_first_r = 0 if first_r <= rhs_threshold_ms else first_r

        # When one side snapped to edge but the other didn't (e.g. ratio
        # imprecision after extrapolation through a shared black intro),
        # check whether the un-snapped side is low-entropy all the way to
        # its edge AND the residual gap is small (≤ 2s — within ratio
        # prediction error).  Large gaps indicate genuinely different-length
        # intros/outros where snapping would be wrong.
        max_extend_ms = 2000
        if new_first_l == 0 and new_first_r != 0 and first_r <= max_extend_ms:
            gap_frames = [k for k in rhs_keys if k < first_r]
            if gap_frames:
                non_le = [k for k in gap_frames if PairMatcher._is_rich(rhs_all_frames[k]["path"])]
                if len(non_le) / len(gap_frames) <= 0.05:
                    self.logger.info(
                        f"Edge snap: RHS low-entropy from {first_r}ms to edge "
                        f"(0ms), extending RHS to 0"
                    )
                    new_first_r = 0
        elif new_first_r == 0 and new_first_l != 0 and first_l <= max_extend_ms:
            gap_frames = [k for k in lhs_keys if k < first_l]
            if gap_frames:
                non_le = [k for k in gap_frames if PairMatcher._is_rich(lhs_all_frames[k]["path"])]
                if len(non_le) / len(gap_frames) <= 0.05:
                    self.logger.info(
                        f"Edge snap: LHS low-entropy from {first_l}ms to edge "
                        f"(0ms), extending LHS to 0"
                    )
                    new_first_l = 0

        if (new_first_l, new_first_r) != (first_l, first_r):
            self.logger.info(
                f"Edge snap: first pair ({first_l}, {first_r}) → "
                f"({new_first_l}, {new_first_r})"
            )
            matching_pairs[0] = (new_first_l, new_first_r)
            if new_first_l not in lhs_all_frames:
                lhs_all_frames[new_first_l] = lhs_all_frames[lhs_keys[0]].copy()
            if new_first_r not in rhs_all_frames:
                rhs_all_frames[new_first_r] = rhs_all_frames[rhs_keys[0]].copy()

        # --- End edge ---
        new_last_l = lhs_duration if (lhs_duration - last_l) <= lhs_threshold_ms else last_l
        new_last_r = rhs_duration if (rhs_duration - last_r) <= rhs_threshold_ms else last_r

        # Same low-entropy extension for end edge.
        if new_last_l == lhs_duration and new_last_r != rhs_duration and (rhs_duration - last_r) <= max_extend_ms:
            gap_frames = [k for k in rhs_keys if k > last_r]
            if gap_frames:
                non_le = [k for k in gap_frames if PairMatcher._is_rich(rhs_all_frames[k]["path"])]
                if len(non_le) / len(gap_frames) <= 0.05:
                    self.logger.info(
                        f"Edge snap: RHS low-entropy from {last_r}ms to edge "
                        f"({rhs_duration}ms), extending RHS to duration"
                    )
                    new_last_r = rhs_duration
        elif new_last_r == rhs_duration and new_last_l != lhs_duration and (lhs_duration - last_l) <= max_extend_ms:
            gap_frames = [k for k in lhs_keys if k > last_l]
            if gap_frames:
                non_le = [k for k in gap_frames if PairMatcher._is_rich(lhs_all_frames[k]["path"])]
                if len(non_le) / len(gap_frames) <= 0.05:
                    self.logger.info(
                        f"Edge snap: LHS low-entropy from {last_l}ms to edge "
                        f"({lhs_duration}ms), extending LHS to duration"
                    )
                    new_last_l = lhs_duration

        if (new_last_l, new_last_r) != (last_l, last_r):
            self.logger.info(
                f"Edge snap: last pair ({last_l}, {last_r}) → "
                f"({new_last_l}, {new_last_r})"
            )
            matching_pairs[-1] = (new_last_l, new_last_r)
            if new_last_l not in lhs_all_frames:
                lhs_all_frames[new_last_l] = lhs_all_frames[lhs_keys[-1]].copy()
            if new_last_r not in rhs_all_frames:
                rhs_all_frames[new_last_r] = rhs_all_frames[rhs_keys[-1]].copy()

        return matching_pairs

    def create_segments_mapping(self) -> tuple[list[tuple[int, int]], FramesInfo, FramesInfo]:
        self.logger.info("Phase 1/5: Detecting scene changes")
        lhs_scene_changes = video_utils.detect_scene_changes(self.lhs_path, threshold=0.3, logger=self.logger, interruption=self.interruption)
        rhs_scene_changes = video_utils.detect_scene_changes(self.rhs_path, threshold=0.3, logger=self.logger, interruption=self.interruption)

        if len(lhs_scene_changes) == 0 or len(rhs_scene_changes) == 0:
            raise RuntimeError("Not enought scene changes detected")

        # extract all frames
        self.logger.info("Phase 2/5: Extracting all frames")
        self.lhs_all_frames = video_utils.extract_all_frames(self.lhs_path, self.lhs_all_wd, scale=(960, -2), format="png", logger=self.logger, interruption=self.interruption)
        self.rhs_all_frames = video_utils.extract_all_frames(self.rhs_path, self.rhs_all_wd, scale=(960, -2), format="png", logger=self.logger, interruption=self.interruption)

        lhs_key_frames_str = [str(self.lhs_all_frames[lhs]["frame_id"]) for lhs in lhs_scene_changes]
        rhs_key_frames_str = [str(self.rhs_all_frames[rhs]["frame_id"]) for rhs in rhs_scene_changes]

        self.logger.debug(f"lhs key frames: {' '.join(lhs_key_frames_str)}")
        self.logger.debug(f"rhs key frames: {' '.join(rhs_key_frames_str)}")

        # normalize frames. This could have been done in the previous step, however for some videos ffmpeg fails to save some of the frames when using 256x256 resolution. Who knows why...
        self.logger.info("Phase 3/5: Normalizing frames")
        lhs_normalized_frames = self._normalize_frames(self.lhs_all_frames, self.lhs_normalized_wd)
        rhs_normalized_frames = self._normalize_frames(self.rhs_all_frames, self.rhs_normalized_wd)

        # extract key frames (as 'key' a scene change frame is meant)
        lhs_key_frames = PairMatcher._get_frames_for_timestamps(lhs_scene_changes, lhs_normalized_frames)
        rhs_key_frames = PairMatcher._get_frames_for_timestamps(rhs_scene_changes, rhs_normalized_frames)

        debug = DebugRoutines(self.debug_wd, self.lhs_all_frames, self.rhs_all_frames)

        debug.dump_frames(lhs_key_frames, "lhs key frames")
        debug.dump_frames(rhs_key_frames, "rhs key frames")

        # find matching keys
        self.logger.info("Phase 4/5: Matching key frames")
        matching_pairs = self._make_pairs(lhs_key_frames, rhs_key_frames, lhs_normalized_frames, rhs_normalized_frames)
        debug.dump_matches(matching_pairs, "initial matching")
        self.logger.debug("Pairs summary after initial matching:")
        self.logger.debug(PairMatcher.summarize_pairs(self.phash, matching_pairs, self.lhs_all_frames, self.rhs_all_frames, verbose = True))

        if not matching_pairs:
            raise RuntimeError("No matching pairs found between the two files")

        self.logger.info("Phase 5/5: Refining boundaries")
        prev_first, prev_last = None, None
        iteration = 0
        while True:
            iteration += 1
            self.logger.info(f"Boundary refinement iteration {iteration} (cropped frames)")
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
            first, last = self._look_for_boundaries(lhs_normalized_cropped_frames, rhs_normalized_cropped_frames, matching_pairs[0], matching_pairs[-1], cutoff, extrapolate=False)

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

        self.logger.info(f"Boundary refinement converged after {iteration} iteration(s)")
        self.logger.debug("Status after boundaries lookup:\n")
        self.logger.debug(PairMatcher.summarize_segments(matching_pairs, self.lhs_fps, self.rhs_fps))
        self.logger.debug(PairMatcher.summarize_pairs(phash4normalized, matching_pairs, self.lhs_all_frames, self.rhs_all_frames, verbose = True))

        # Final boundary search on uncropped normalized frames.
        # The cropped-frame search may fail near edges because the crop
        # interpolation extrapolates from the nearest known pair, which can
        # be significantly off when the spatial transformation changes over time.
        # Searching the uncropped frames avoids this limitation.
        self.logger.info("Final boundary search (uncropped frames, with extrapolation)")
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

        # Snap near-edge pairs to exact video boundaries to avoid trivially
        # short head/tail audio segments (< 3 frames).
        matching_pairs = self._snap_to_edges(matching_pairs, self.lhs_all_frames, self.rhs_all_frames)

        return matching_pairs, self.lhs_all_frames, self.rhs_all_frames
