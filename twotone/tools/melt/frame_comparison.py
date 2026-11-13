"""Frame state and image preparation shared by melt matching stages.

This module owns the representation of one matching side and all operations
that turn decoded frames into comparison-space images.  Matching and boundary
algorithms consume these facilities; they do not need to know how images are
extracted, normalized, cropped, or cached.
"""

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
from tqdm import tqdm

from ..utils import files_utils, generic_utils, image_utils, video_utils
from .melt_common import FrameInfo, FramesInfo
from .phash_cache import PhashCache


# The adaptive pHash cutoff (median + 2*std of matched pairs) collapses to zero
# for pixel-perfect pairs.  Neighbour comparisons must still tolerate ordinary
# transfer and frame-selection noise.
MIN_PHASH_CUTOFF = 16
# Boundary-gap samples add prediction jitter and extrapolated-crop noise.  In
# the normalized shared-crop space, real degraded/cross-geometry fixtures reach
# ~74/~62 bits while divergent intros remain >=122; 96 preserves useful margin
# on both sides and matched pairs may still raise it adaptively.
MIN_BOUNDARY_GAP_PHASH_CUTOFF = 96
# Frames above this entropy contain enough texture for dependable pHash/ORB.
RICH_FRAME_ENTROPY = 3.5
# Only decisively black frames may veto a rich opposite side.  Flat-but-lit
# content can straddle the richer threshold across differently framed transfers
# and therefore falls through to pHash comparison.
BLACK_FRAME_ENTROPY = 2.0


@dataclass
class MatchingSide:
    """Run-scoped frame state and work directories for one input video."""

    path: str
    label: str
    fps: float
    duration_ms: int | None
    all_dir: str
    boundary_dir: str
    normalized_dir: str
    normalized_cropped_dir: str
    boundary_comparison_dir: str
    all_frames: FramesInfo = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        path: str,
        label: str,
        fps: float,
        duration_ms: int | None,
        working_dir: str,
    ) -> "MatchingSide":
        return cls(
            path=path,
            label=label,
            fps=fps,
            duration_ms=duration_ms,
            all_dir=os.path.join(working_dir, "all"),
            boundary_dir=os.path.join(working_dir, "boundary"),
            normalized_dir=os.path.join(working_dir, "norm"),
            normalized_cropped_dir=os.path.join(working_dir, "norm_cropped"),
            boundary_comparison_dir=os.path.join(working_dir, "boundary_cmp"),
        )

    def directories(self) -> tuple[str, ...]:
        return (
            self.all_dir,
            self.boundary_dir,
            self.normalized_dir,
            self.normalized_cropped_dir,
            self.boundary_comparison_dir,
        )


@dataclass
class VerifySide:
    """One video's frame sources for boundary-gap verification."""

    video_path: str
    raw_dir: str
    all_frames: FramesInfo
    normalized: FramesInfo
    crop_fn: Callable[[int], tuple[int, int, int, int]] | None
    comparison_dir: str
    comparison_cache: dict[int, str | None]


@dataclass
class BoundaryVerifyContext:
    """Pair-calibrated comparison state for boundary-gap verification."""

    lhs: VerifySide
    rhs: VerifySide
    phash: PhashCache
    cutoff: float


def extracted_path(info: FrameInfo) -> str:
    """Return an extracted frame path, failing loudly on incomplete state."""
    path = info["path"]
    if path is None:
        raise AssertionError("frame used before extraction")
    return path


def path_or_none(frames: FramesInfo, timestamp: int) -> str | None:
    info = frames.get(timestamp)
    return info["path"] if info else None


def with_path(info: FrameInfo, path: str) -> FrameInfo:
    result = info.copy()
    result["path"] = path
    return result


def assert_frames_extracted(frames: FramesInfo, context: str = "") -> None:
    for timestamp, info in frames.items():
        if info.get("path") is None:
            frame_id = info.get("frame_id", "?")
            raise AssertionError(
                f"Frame at {timestamp}ms (frame_id={frame_id}) has not been "
                f"extracted to disk.  This is a bug — the frame should "
                f"have been included in the extraction range.  "
                f"Context: {context}"
            )


def normalize_image(src_path: str, dst_path: str) -> None:
    """Bring one frame into the grayscale, border-cropped comparison space."""
    image = cv.imread(src_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read frame from {src_path}")
    height, width = image.shape
    dx = int(width * 0.05)
    dy = int(height * 0.05)
    image = image[dy:height - dy, dx:width - dx]
    image = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)
    cv.imwrite(dst_path, image)


def crop_image(
    src_path: str,
    dst_path: str,
    rect: tuple[int, int, int, int],
) -> None:
    """Crop one comparison-space frame to *rect* and rescale to 128x128."""
    image = cv.imread(src_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read frame from {src_path}")
    x, y, width, height = rect
    cropped = image[y:y + height, x:x + width]
    cropped = cv.resize(cropped, (128, 128))
    cv.imwrite(dst_path, cropped)


def is_rich(frame_path: str) -> bool:
    return image_utils.image_entropy(frame_path) > RICH_FRAME_ENTROPY


def filter_low_detail(frames: FramesInfo) -> FramesInfo:
    return {
        timestamp: info
        for timestamp, info in frames.items()
        if is_rich(extracted_path(info))
    }


def frames_for_timestamps(
    timestamps: list[int],
    frames: FramesInfo,
) -> FramesInfo:
    return {
        timestamp: info
        for timestamp, info in frames.items()
        if timestamp in timestamps
    }


def compute_overlap(
    lhs_image: cv.typing.MatLike,
    rhs_image: cv.typing.MatLike,
    transform,
) -> tuple[int, int, int, int]:
    assert len(lhs_image.shape) == 2
    assert len(rhs_image.shape) == 2

    warped_rhs = cv.warpPerspective(
        rhs_image,
        transform,
        (lhs_image.shape[1], lhs_image.shape[0]),
    )
    lhs_mask = (lhs_image > 0).astype(np.uint8)
    rhs_mask = (warped_rhs > 0).astype(np.uint8)
    x, y, width, height = cv.boundingRect(cv.bitwise_and(lhs_mask, rhs_mask))
    return x, y, width, height


def interpolate_crop_rects(
    timestamps_list: list[int],
    rects_list: list[tuple[int, int, int, int]],
) -> Callable[[int], tuple[int, int, int, int]]:
    timestamps = np.array(timestamps_list)
    rects = np.array(rects_list)

    def interpolate(timestamp: int) -> tuple[int, int, int, int]:
        if timestamp <= timestamps[0]:
            values = rects[0]
        elif timestamp >= timestamps[-1]:
            values = rects[-1]
        else:
            values = np.array([
                np.interp(timestamp, timestamps, rects[:, index])
                for index in range(4)
            ])
        return tuple(int(round(value)) for value in values)  # type: ignore[return-value]

    return interpolate


def find_interpolated_crop(
    pairs: list[tuple[int, int]],
    lhs_frames: FramesInfo,
    rhs_frames: FramesInfo,
) -> tuple[
    Callable[[int], tuple[int, int, int, int]],
    Callable[[int], tuple[int, int, int, int]],
] | None:
    lhs_timestamps: list[int] = []
    rhs_timestamps: list[int] = []
    lhs_crops: list[tuple[int, int, int, int]] = []
    rhs_crops: list[tuple[int, int, int, int]] = []

    for lhs_timestamp, rhs_timestamp in pairs:
        lhs_image = cv.imread(
            extracted_path(lhs_frames[lhs_timestamp]),
            cv.IMREAD_GRAYSCALE,
        )
        rhs_image = cv.imread(
            extracted_path(rhs_frames[rhs_timestamp]),
            cv.IMREAD_GRAYSCALE,
        )
        if lhs_image is None or rhs_image is None:
            continue

        orb = cv.ORB_create(1000)  # type: ignore[attr-defined]
        lhs_keypoints, lhs_descriptors = orb.detectAndCompute(lhs_image, None)
        rhs_keypoints, rhs_descriptors = orb.detectAndCompute(rhs_image, None)
        if lhs_descriptors is None or rhs_descriptors is None:
            continue

        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(lhs_descriptors, rhs_descriptors)
        if len(matches) < 3:
            continue

        matches = sorted(matches, key=lambda match: match.distance)
        lhs_points = np.array(
            [lhs_keypoints[match.queryIdx].pt for match in matches],
            dtype=np.float32,
        )
        rhs_points = np.array(
            [rhs_keypoints[match.trainIdx].pt for match in matches],
            dtype=np.float32,
        )
        transform, _ = cv.estimateAffinePartial2D(
            rhs_points,
            lhs_points,
            method=cv.RANSAC,
        )
        if transform is None:
            continue

        lhs_overlap = compute_overlap(
            lhs_image,
            rhs_image,
            np.vstack([transform, [0, 0, 1]]),
        )
        rhs_overlap = compute_overlap(
            rhs_image,
            lhs_image,
            np.vstack([cv.invertAffineTransform(transform), [0, 0, 1]]),
        )
        lhs_timestamps.append(lhs_timestamp)
        rhs_timestamps.append(rhs_timestamp)
        lhs_crops.append(lhs_overlap)
        rhs_crops.append(rhs_overlap)

    if not lhs_timestamps:
        return None
    return (
        interpolate_crop_rects(lhs_timestamps, lhs_crops),
        interpolate_crop_rects(rhs_timestamps, rhs_crops),
    )


def calculate_cutoff(
    phash: PhashCache,
    pairs: list[tuple[int, int]],
    lhs: FramesInfo,
    rhs: FramesInfo,
) -> float:
    if not pairs:
        return MIN_PHASH_CUTOFF
    distances = [
        abs(
            phash.get(extracted_path(lhs[lhs_timestamp]))
            - phash.get(extracted_path(rhs[rhs_timestamp]))
        )
        for lhs_timestamp, rhs_timestamp in pairs
    ]
    values = np.array(distances)
    return float(np.median(values) + np.std(values) * 2)


def floored_phash_cutoff(cutoff: float) -> float:
    return max(cutoff, MIN_PHASH_CUTOFF)


class FrameComparison:
    """Prepare, cache, and compare frame images for a pair of inputs."""

    def __init__(
        self,
        interruption: generic_utils.InterruptibleProcess,
        logger: logging.Logger,
        lhs: MatchingSide,
        rhs: MatchingSide,
    ) -> None:
        self.interruption = interruption
        self.logger = logger
        self.lhs = lhs
        self.rhs = rhs

    def normalize_frames(
        self,
        frames: FramesInfo,
        output_dir: str,
        desc: str = "Normalizing frames",
        prefix: str = "",
    ) -> FramesInfo:
        assert_frames_extracted(frames, f"normalize_frames({desc})")

        def process_frame(item: tuple[int, FrameInfo]) -> tuple[int, FrameInfo]:
            timestamp, info = item
            self.interruption.check_for_stop()
            source_path = extracted_path(info)
            _, filename, extension = files_utils.split_path(source_path)
            target_path = os.path.join(
                output_dir,
                prefix + filename + "." + extension,
            )
            normalize_image(source_path, target_path)
            return timestamp, with_path(info, target_path)

        with ThreadPoolExecutor() as executor:
            results_iter = executor.map(process_frame, frames.items())
            results = []
            with tqdm(
                total=len(frames),
                desc=desc,
                unit="frame",
                **generic_utils.get_tqdm_defaults(),
            ) as progress:
                for result in results_iter:
                    results.append(result)
                    progress.update(1)
        return dict(results)

    def apply_crop(
        self,
        frames: FramesInfo,
        output_dir: str,
        crop_fn: Callable[[int], tuple[int, int, int, int]],
    ) -> FramesInfo:
        def process_frame(item: tuple[int, FrameInfo]) -> tuple[int, FrameInfo]:
            timestamp, info = item
            self.interruption.check_for_stop()
            source_path = extracted_path(info)
            target_path = os.path.join(output_dir, os.path.basename(source_path))
            crop_image(source_path, target_path, crop_fn(timestamp))
            return timestamp, with_path(info, target_path)

        with ThreadPoolExecutor() as executor:
            return dict(executor.map(process_frame, frames.items()))

    def crop_both(
        self,
        pairs: list[tuple[int, int]],
        lhs_frames: FramesInfo,
        rhs_frames: FramesInfo,
        lhs_output_dir: str,
        rhs_output_dir: str,
    ) -> tuple[FramesInfo, FramesInfo]:
        assert_frames_extracted(lhs_frames, "crop_both(lhs)")
        assert_frames_extracted(rhs_frames, "crop_both(rhs)")
        crop_fns = find_interpolated_crop(pairs, lhs_frames, rhs_frames)
        if crop_fns is None:
            self.logger.debug(
                "No usable crop geometry found for any matched pair — "
                "comparing uncropped frames"
            )
            return lhs_frames, rhs_frames
        lhs_crop_fn, rhs_crop_fn = crop_fns
        return (
            self.apply_crop(lhs_frames, lhs_output_dir, lhs_crop_fn),
            self.apply_crop(rhs_frames, rhs_output_dir, rhs_crop_fn),
        )

    def build_boundary_verify_context(
        self,
        matching_pairs: list[tuple[int, int]],
        lhs_normalized_frames: FramesInfo,
        rhs_normalized_frames: FramesInfo,
        *,
        minimum_cutoff: int = MIN_BOUNDARY_GAP_PHASH_CUTOFF,
    ) -> BoundaryVerifyContext:
        crop_fns = find_interpolated_crop(
            matching_pairs,
            lhs_normalized_frames,
            rhs_normalized_frames,
        )
        if crop_fns is None:
            self.logger.debug(
                "Boundary verification: no usable crop geometry found for any "
                "matched pair — comparing uncropped normalized frames"
            )
            lhs_crop_fn = rhs_crop_fn = None
        else:
            lhs_crop_fn, rhs_crop_fn = crop_fns

        lhs_side = VerifySide(
            video_path=self.lhs.path,
            raw_dir=self.lhs.boundary_dir,
            all_frames=self.lhs.all_frames,
            normalized=lhs_normalized_frames,
            crop_fn=lhs_crop_fn,
            comparison_dir=self.lhs.boundary_comparison_dir,
            comparison_cache={},
        )
        rhs_side = VerifySide(
            video_path=self.rhs.path,
            raw_dir=self.rhs.boundary_dir,
            all_frames=self.rhs.all_frames,
            normalized=rhs_normalized_frames,
            crop_fn=rhs_crop_fn,
            comparison_dir=self.rhs.boundary_comparison_dir,
            comparison_cache={},
        )

        phash = PhashCache()
        lhs_comparison: FramesInfo = {}
        rhs_comparison: FramesInfo = {}
        for lhs_timestamp, rhs_timestamp in matching_pairs:
            lhs_image = self.comparison_image(lhs_side, lhs_timestamp)
            rhs_image = self.comparison_image(rhs_side, rhs_timestamp)
            if lhs_image is None or rhs_image is None:
                continue
            lhs_comparison[lhs_timestamp] = with_path(
                lhs_normalized_frames[lhs_timestamp],
                lhs_image,
            )
            rhs_comparison[rhs_timestamp] = with_path(
                rhs_normalized_frames[rhs_timestamp],
                rhs_image,
            )

        usable_pairs = [
            (lhs_timestamp, rhs_timestamp)
            for lhs_timestamp, rhs_timestamp in matching_pairs
            if lhs_timestamp in lhs_comparison
            and rhs_timestamp in rhs_comparison
        ]
        cutoff = max(
            calculate_cutoff(
                phash,
                usable_pairs,
                lhs_comparison,
                rhs_comparison,
            ),
            minimum_cutoff,
        )
        self.logger.debug(
            "Boundary verification cutoff: %.1f (calibrated on %d matched pair(s))",
            cutoff,
            len(usable_pairs),
        )
        return BoundaryVerifyContext(lhs_side, rhs_side, phash, cutoff)

    def comparison_image(
        self,
        side: VerifySide,
        timestamp: int,
    ) -> str | None:
        if timestamp in side.comparison_cache:
            return side.comparison_cache[timestamp]

        normalized_info = side.normalized.get(timestamp)
        raw_path = None
        if normalized_info is None:
            raw_path = self.ensure_boundary_image(
                side.video_path,
                side.raw_dir,
                side.all_frames,
                timestamp,
            )

        path: str | None = None
        try:
            if normalized_info is not None:
                normalized_path = normalized_info["path"]
            elif raw_path is None:
                normalized_path = None
            else:
                normalized_path = os.path.join(
                    side.comparison_dir,
                    f"n_{timestamp}.png",
                )
                normalize_image(raw_path, normalized_path)

            if normalized_path is not None:
                if side.crop_fn is None:
                    path = normalized_path
                else:
                    path = os.path.join(
                        side.comparison_dir,
                        f"c_{timestamp}.png",
                    )
                    crop_image(normalized_path, path, side.crop_fn(timestamp))
        except Exception as error:  # pragma: no cover - non-fatal preparation failure
            self.logger.debug(
                "Comparison image preparation failed for %dms: %s",
                timestamp,
                error,
            )
            path = None

        side.comparison_cache[timestamp] = path
        return path

    def boundary_content_matches(
        self,
        context: BoundaryVerifyContext,
        lhs_timestamp: int,
        rhs_timestamp: int,
    ) -> bool:
        lhs_path = self.comparison_image(context.lhs, lhs_timestamp)
        rhs_path = self.comparison_image(context.rhs, rhs_timestamp)
        if lhs_path is None or rhs_path is None:
            self.logger.debug(
                "Boundary gap sample %dms vs %dms: no comparison image (%s, %s)",
                lhs_timestamp,
                rhs_timestamp,
                lhs_path,
                rhs_path,
            )
            return False

        lhs_entropy = image_utils.image_entropy(lhs_path)
        rhs_entropy = image_utils.image_entropy(rhs_path)
        if lhs_entropy <= RICH_FRAME_ENTROPY and rhs_entropy <= RICH_FRAME_ENTROPY:
            return True
        if min(lhs_entropy, rhs_entropy) < BLACK_FRAME_ENTROPY:
            self.logger.debug(
                "Boundary gap sample %dms vs %dms: black vs content "
                "(entropies %.2f, %.2f)",
                lhs_timestamp,
                rhs_timestamp,
                lhs_entropy,
                rhs_entropy,
            )
            return False
        distance = abs(context.phash.get(lhs_path) - context.phash.get(rhs_path))
        self.logger.debug(
            "Boundary gap sample %dms vs %dms: phash distance %d vs cutoff %.1f",
            lhs_timestamp,
            rhs_timestamp,
            distance,
            context.cutoff,
        )
        return distance <= context.cutoff

    def prefetch(self, side: MatchingSide, timestamps: list[int]) -> None:
        self.prefetch_images(
            side.path,
            side.boundary_dir,
            side.all_frames,
            timestamps,
        )

    def prefetch_images(
        self,
        video_path: str,
        output_dir: str,
        frames: FramesInfo,
        timestamps: list[int],
    ) -> None:
        missing = sorted({
            int(frames[timestamp]["frame_id"])
            for timestamp in timestamps
            if timestamp in frames and not frames[timestamp].get("path")
        })
        if not missing:
            return
        video_utils.extract_frames_at_ranges(
            video_path,
            output_dir,
            [(frame_id, frame_id) for frame_id in missing],
            frames,
            scale=(960, -2),
            format="png",
            interruption=self.interruption,
            desc="Verifying boundary gap",
            logger=self.logger,
        )

    def ensure_boundary_image(
        self,
        video_path: str,
        output_dir: str,
        frames: FramesInfo,
        timestamp: int,
    ) -> str | None:
        info = frames.get(timestamp)
        if info is None:
            return None
        if info.get("path"):
            return info["path"]
        frame_id = int(info["frame_id"])
        video_utils.extract_frames_at_ranges(
            video_path,
            output_dir,
            [(frame_id, frame_id)],
            frames,
            scale=(960, -2),
            format="png",
            interruption=self.interruption,
            desc="Verifying boundary frame",
            logger=self.logger,
        )
        return path_or_none(frames, timestamp)
