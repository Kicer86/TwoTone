
import argparse
import logging
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from overrides import override
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Callable, List

from .tool import Tool
from twotone.tools.utils import files_utils, generic_utils, process_utils, video_utils


class Transcoder(generic_utils.InterruptibleProcess):
    def __init__(self, working_dir: str, logger: logging.Logger, live_run: bool = False, target_ssim: float = 0.98, codec: str = "libx265") -> None:
        super().__init__()
        self.logger = logger
        self.live_run = live_run
        self.target_ssim = target_ssim
        self.codec = codec
        self.working_dir = working_dir


    def _find_video_files(self, directory: str) -> list[str]:
        """Find video files with specified extensions."""
        video_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if video_utils.is_video(file):
                    video_files.append(os.path.join(root, file))
        return video_files


    def _calculate_quality(self, original: str, transcoded: str) -> float | None:
        """Calculate SSIM between original and transcoded video_utils."""
        args = [
            "-i", original, "-i", transcoded,
            "-lavfi", "ssim", "-f", "null", "-"
        ]

        result = process_utils.start_process("ffmpeg", args)
        ssim_line = [line for line in result.stderr.splitlines() if "All:" in line]

        if ssim_line:
            # Extract the SSIM value immediately after "All:"
            ssim_value = ssim_line[-1].split("All:")[1].split()[0]
            try:
                return float(ssim_value)
            except ValueError:
                self.logger.error(f"Failed to parse SSIM value: {ssim_value}")
                return None

        return None


    def _transcode_video(
        self,
        input_file: str,
        output_file: str,
        crf: int,
        preset: str,
        input_params: list[str] | None = None,
        output_params: list[str] | None = None,
        audio_codec: list[str] | None = None,
        show_progress: bool = False,
    ) -> None:
        """
        Encode video with a given CRF, preset, and extra parameters.
        By default audio is removed as in most cases this function is being used
        for finding optimal CRF and quite often audio may alter SSIM results
        (in most cases due to interfering with timestamps).
        """

        args = [
            "-v", "error", "-stats", "-nostdin",
            *(input_params or []),
            "-i", input_file,
            "-c:v", self.codec,
            "-crf", str(crf),
            "-preset", preset,
            "-profile:v", "main10",
            *(audio_codec or ["-an"]),
            *(output_params or []),
            output_file
        ]

        process_utils.raise_on_error(process_utils.start_process("ffmpeg", args, show_progress=show_progress))


    def _extract_segment(self, video_file: str, start_time: float, end_time: float, output_file: str) -> None:
        """ Extract video segment. Video is transcoded with lossless quality to rebuild damaged or troublesome videos """
        self._transcode_video(
            video_file,
            output_file,
            crf=0,
            preset="veryfast",
            input_params=["-ss", str(start_time), "-to", str(end_time)]
        )


    def _extract_segments(self, video_file: str, segments: list[tuple[float, float]], output_dir: str) -> list[str]:
        output_files = []
        _, filename, ext = files_utils.split_path(video_file)

        i = 0
        with logging_redirect_tqdm():
            for (start, end) in tqdm(segments, desc="Extracting scenes", unit="scene", leave=False, smoothing=0.1, mininterval=.2, disable=generic_utils.hide_progressbar()):
                self._check_for_stop()
                output_file = os.path.join(output_dir, f"{filename}.frag{i}.mp4")
                self._extract_segment(video_file, start, end, output_file)
                output_files.append(output_file)
                i += 1

        return output_files


    def _select_scenes(self, video_file: str, segment_duration: int = 5) -> list[tuple[float, float]]:
        """
        Select video segments around detected scene changes, merging nearby timestamps.

        Parameters:
            video_file (str): Path to the input video file.
            segment_duration (int): Minimum duration (in seconds) of each segment.

        Returns:
            list: Full paths to the generated video files.
        """

        # FFmpeg command to detect scene changes and log timestamps
        args = [
            "-i", video_file,
            "-vf", "select='gt(scene,0.4)',showinfo",
            "-vsync", "vfr", "-f", "null", "/dev/null"
        ]

        result = process_utils.start_process("ffmpeg", args)

        # Parse timestamps from the ffmpeg output
        timestamps = []
        showinfo_output = result.stderr
        for line in showinfo_output.splitlines():
            match = re.search(r"pts_time:(\d+(\.\d+)?)", line)
            if match:
                timestamps.append(float(match.group(1)))

        # Generate segments with padding
        segments = []
        for timestamp in timestamps:
            start = max(0, timestamp - segment_duration / 2)
            end = timestamp + segment_duration / 2
            segments.append((start, end))

        # # Merge overlapping segments
        merged_segments: List[tuple[float, float]] = []
        for start, end in sorted(segments):
            if not merged_segments or start > merged_segments[-1][1]:  # No overlap
                merged_segments.append((start, end))
            else:  # Overlap detected, merge
                merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], end))

        return merged_segments


    def _select_segments(self, video_file: str, segment_duration: int = 5) -> list[tuple[float, float]]:
        duration = video_utils.get_video_duration(video_file) / 1000
        num_segments = max(3, min(10, int(duration // 30)))

        if duration <= 0 or num_segments <= 0 or segment_duration <= 0:
            raise ValueError("Total length, number of segments, and segment length must all be positive.")
        if segment_duration > duration:
            raise ValueError("Segment length cannot exceed total length.")
        if num_segments * segment_duration > duration:
            raise ValueError("Total segments cannot fit within the total length.")

        step = (duration - segment_duration) / (num_segments - 1) if num_segments > 1 else 0

        segments = [(float(round(i * step)), float(round(i * step) + segment_duration)) for i in range(num_segments)]

        return segments


    def _bisection_search(self, eval_func: Callable[[int], float | None], min_value: int, max_value: int, target_condition: Callable[[float], bool]) -> tuple[int | None, float | None]:
        """
        Generic bisection search algorithm.

        Parameters:
            eval_func (callable): Function to evaluate the current value (e.g., CRF).
                                Should return a tuple (value, evaluation_result).
            min_value (int): Minimum value of the range to search.
            max_value (int): Maximum value of the range to search.
            target_condition (callable): Function to check if the evaluation result meets the desired condition.
                                        Should return True if the condition is met.

        Returns:
            Tuple[int, any]: The optimal value and its corresponding evaluation result.
        """
        best_value = None
        best_result = None

        while min_value <= max_value:
            mid_value = (min_value + max_value) // 2
            eval_result = eval_func(mid_value)

            if eval_result is not None and target_condition(eval_result):
                best_value = mid_value
                best_result = eval_result
                min_value = mid_value + 1
            else:
                max_value = mid_value - 1

        return best_value, best_result


    def _transcode_segment_and_compare(self, wd_dir: str, segment_file: str, crf: int) -> float | None:
        basename = os.path.basename(segment_file)
        transcoded_segment_output = os.path.join(wd_dir, basename)

        self._transcode_video(segment_file, transcoded_segment_output, crf, "veryfast", output_params = ["-vsync", "vfr"])

        quality = self._calculate_quality(segment_file, transcoded_segment_output)
        return quality

    def _for_segments(self, segments: list[str], op: Callable[[str, str], None], title: str, unit: str) -> None:
        with logging_redirect_tqdm(), \
             tqdm(desc=title, unit=unit, total=len(segments), **generic_utils.get_tqdm_defaults()) as pbar, \
             files_utils.ScopedDirectory(os.path.join(self.working_dir, "segments")) as wd_dir, \
             ThreadPoolExecutor() as executor:
            def worker(file_path):
                op(wd_dir, file_path)
                pbar.update(1)

            for segment in segments:
                #executor.submit(worker, segment)
                worker(segment)

    def _final_transcode(self, input_file: str, crf: int) -> None:
        """Perform the final transcoding with the best CRF using the determined extra_params."""
        _, basename, ext = files_utils.split_path(input_file)

        # As of now ffmpeg does not support rmvb outputs and copying cook audio codec
        overwrite_input = True
        audio_codec = ["-c:a", "copy"]
        if ext == "rmvb":
            ext = "mp4"
            audio_codec = ["-c:a", "libopus", "-b:a", "192k"]
            overwrite_input = False

        self.logger.info(f"Starting final transcoding with CRF: {crf}")
        temp_file = os.path.join(self.working_dir, f"{basename}.temp.{ext}")
        self._transcode_video(input_file, temp_file, crf, "veryslow", audio_codec = audio_codec, output_params = ["-vsync", "passthrough"], show_progress=True)

        original_size = os.path.getsize(input_file)
        final_size = os.path.getsize(temp_file)
        size_reduction = (final_size / original_size) * 100

        # Measure SSIM again after final transcoding
        final_quality = self._calculate_quality(input_file, temp_file)

        if final_quality is None:
            raise ValueError("Could not determine SSIM value.")

        try:
            if final_quality < self.target_ssim:
                self.logger.warning(
                    f"Final CRF: {crf}, SSIM: {final_quality}. "
                    f"Final transcode resulted in lower SSIM than requested: {final_quality} < {self.target_ssim}"
                )
                raise ValueError()

            if final_size > original_size:
                self.logger.warning(
                    f"Final CRF: {crf}, SSIM: {final_quality}. "
                    f"Encoded file is larger than the original. Keeping the original file."
                )
                raise ValueError()


            process_utils.start_process("exiftool", ["-overwrite_original", "-TagsFromFile", input_file, "-all:all>all:all", temp_file])

            if overwrite_input:
                try:
                    logging.debug(f"Replacing {input_file} with {temp_file}")
                    os.replace(temp_file, input_file)

                except OSError:
                    logging.debug(f"Replacing {input_file} with {temp_file} (second attempt)")
                    shutil.move(temp_file, input_file)
            else:
                final_output_file = os.path.join(dir, f"{basename}.{ext}")
                logging.debug(f"Renaming {temp_file} to {final_output_file}")
                shutil.move(temp_file, final_output_file)
                logging.debug(f"Removing {input_file}")
                os.remove(input_file)

            self.logger.info(
                f"Final CRF: {crf}, SSIM: {final_quality}, "
                f"encoded Size: {final_size} bytes, "
                f"size reduced by: {original_size - final_size} bytes "
                f"({size_reduction:.2f}% of original size)"
            )

        except ValueError:
            logging.error(f"Error occured, removing temporary file {temp_file}")
            os.remove(temp_file)



    def find_optimal_crf(self, input_file: str, allow_segments: bool = True) -> int | None:
        """Find the optimal CRF using bisection."""
        original_size = os.path.getsize(input_file)

        duration = video_utils.get_video_duration(input_file)
        if not duration:
            return None

        # convert to seconds
        duration /= 1000

        with files_utils.ScopedDirectory(os.path.join(self.working_dir, "opt_crf")) as wd_dir:
            segment_files = []
            if allow_segments and duration > 30:
                self.logger.info(f"Picking segments from {input_file}")
                segments = self._select_scenes(input_file)
                if len(segments) < 2:
                    segments = self._select_segments(input_file)

                segment_files = self._extract_segments(
                    input_file, segments, wd_dir)

                self.logger.info(
                    f"Starting CRF bisection for {input_file} "
                    f"with veryfast preset using {len(segment_files)} segments")
            else:
                segment_files = [input_file]
                self.logger.info(f"Starting CRF bisection for {input_file} with veryfast preset using whole file")

            def evaluate_crf(mid_crf):
                self._check_for_stop()
                qualities = []

                def get_quality(wd_dir, segment_file):
                    quality = self._transcode_segment_and_compare(wd_dir, segment_file, mid_crf)
                    if quality:
                        qualities.append(quality)

                self._for_segments(segment_files, get_quality, "SSIM calculation", "scene")

                avg_quality = sum(qualities) / len(qualities) if qualities else 0
                self.logger.info(f"CRF: {mid_crf}, Average Quality (SSIM): {avg_quality}")

                return avg_quality

            top_quality = evaluate_crf(0)
            if top_quality < 0.9975:
                raise RuntimeError(f"Sanity check failed: top SSIM value: {top_quality} < 0.9975")

            if top_quality < self.target_ssim:
                raise RuntimeError(f"Top SSIM value: {top_quality} < requested SSIM: {self.target_ssim}")

            crf_min, crf_max = 0, 51
            best_crf, best_quality = self._bisection_search(evaluate_crf, min_value = crf_min, max_value = crf_max, target_condition = lambda avg_quality: avg_quality >= self.target_ssim)

            if best_crf is not None and best_quality is not None:
                self.logger.info(f"Finished CRF bisection. Optimal CRF: {best_crf} with quality: {best_quality}")
            else:
                self.logger.warning(f"Finished CRF bisection. Could not find CRF matching desired quality ({self.target_ssim}).")
            return best_crf


    def transcode(self, directory: str) -> None:
        self.logger.info(f"Starting video transcoding with {self.codec}. Target SSIM: {self.target_ssim}")
        video_files = self._find_video_files(directory)

        for file in video_files:
            self._check_for_stop()
            self.logger.info(f"Processing {file}")
            best_crf = self.find_optimal_crf(file)
            if best_crf is not None and self.live_run:
                # increase crf by one as veryslow preset will be used, so result should be above requested quality anyway
                self._final_transcode(file, best_crf + 1)
            elif not self.live_run:
                self.logger.info(f"Dry run. Skipping final transcoding step.")

            self.logger.info(f"Finished processing {file}")

        self.logger.info("Video processing completed")


class TranscodeTool(Tool):
    @override
    def setup_parser(self, parser: argparse.ArgumentParser) -> None:
        def valid_ssim_value(value):
            try:
                fvalue = float(value)
                if 0 <= fvalue <= 1:
                    return fvalue
                else:
                    raise argparse.ArgumentTypeError(f"SSIM value must be between 0 and 1. Got {value}")
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid SSIM value: {value}")

        parser.add_argument("--ssim", "-s",
                            type=valid_ssim_value,
                            default=0.98,
                            help='Requested SSIM value (video quality). Valid values are between 0 and 1.')
        parser.add_argument('videos_path',
                            nargs=1,
                            help='Path with videos to transcode.')


    @override
    def analyze(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str) -> None:
        pass

    @override
    def perform(self, args: argparse.Namespace, no_dry_run: bool, logger: logging.Logger, working_dir: str) -> None:
        transcoder = Transcoder(working_dir = working_dir, logger = logger, live_run = no_dry_run, target_ssim = args.ssim)
        transcoder.transcode(args.videos_path[0])
