
import argparse
import logging
import os
import random
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from . import utils

class Transcoder(utils.InterruptibleProcess):
    def __init__(self, live_run: bool = False, target_ssim: float = 0.98):
        super().__init__()
        self.live_run = live_run
        self.target_ssim = target_ssim


    def _find_video_files(self, directory):
        """Find video files with specified extensions."""
        video_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if utils.is_video(file):
                    video_files.append(os.path.join(root, file))
        return video_files


    def _validate_ffmpeg_result(self, result: utils.ProcessResult):
        if result.returncode != 0:
            raise RuntimeError(result.stderr)


    def _select_random_segments(self, total_length, num_segments=5, segment_length=5):
        if total_length <= 0 or num_segments <= 0 or segment_length <= 0:
            raise ValueError("Total length, number of segments, and segment length must all be positive.")
        if segment_length > total_length:
            raise ValueError("Segment length cannot exceed total length.")
        if num_segments * segment_length > total_length:
            raise ValueError("Total segments cannot fit within the total length.")

        step = (total_length - segment_length) / (num_segments - 1) if num_segments > 1 else 0
        return [(round(i * step), segment_length) for i in range(num_segments)]


    def _calculate_quality(self, original, transcoded):
        """Calculate SSIM between original and transcoded video."""
        args = [
            "-i", original, "-i", transcoded,
            "-lavfi", "ssim", "-f", "null", "-"
        ]

        result = utils.start_process("ffmpeg", args)
        ssim_line = [line for line in result.stderr.decode("utf-8").splitlines() if "All:" in line]

        if ssim_line:
            # Extract the SSIM value immediately after "All:"
            ssim_value = ssim_line[-1].split("All:")[1].split()[0]
            try:
                return float(ssim_value)
            except ValueError:
                logging.error(f"Failed to parse SSIM value: {ssim_value}")
                return None

        return None


    def _transcode_video(self, input_file, output_file, crf, preset, input_params=[], output_params=[]):
        """Encode video with a given CRF, preset, and extra parameters."""
        args = [
            "-v", "error", "-stats", "-nostdin",
            *input_params,
            "-i", input_file,
            "-c:v", "libx265",
            "-crf", str(crf),
            "-preset", preset,
            "-profile:v", "main10",
            "-c:a", "copy",
            *output_params,
            output_file
        ]

        result = utils.start_process("ffmpeg", args)
        self._validate_ffmpeg_result(result)


    def _extract_segment(self, video_file, start_time, segment_length, output_file):
        """ Extract video segment. Video is transcoded with lossless quality to rebuild damaged or troublesome videos """
        self._transcode_video(
            video_file,
            output_file,
            crf = 0,
            preset = "veryfast",
            input_params = ["-ss", str(start_time), "-t", str(segment_length)],
            output_params = ["-an"]            # remove audio - some codecs may cause issues with proper extraction
        )


    def _extract_scenes(self, video_file, output_dir, segment_duration=5):
        """
        Extracts video segments around detected scene changes, merging nearby timestamps.

        Parameters:
            video_file (str): Path to the input video file.
            output_dir (str): Directory where the extracted video segments will be saved.
            segment_duration (int): Minimum duration (in seconds) of each segment.

        Returns:
            list: Full paths to the generated video files.
        """

        # FFmpeg command to detect scene changes and log timestamps
        args = [
            "-i", video_file,
            "-vf", "select='gt(scene,0.6)',showinfo",
            "-vsync", "vfr", "-f", "null", "/dev/null"
        ]

        result = utils.start_process("ffmpeg", args)

            # Parse timestamps from the ffmpeg output
        timestamps = []
        showinfo_output = result.stderr.decode("utf-8")
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

        # Merge overlapping segments
        merged_segments = []
        for start, end in sorted(segments):
            if not merged_segments or start > merged_segments[-1][1]:  # No overlap
                merged_segments.append((start, end))
            else:  # Overlap detected, merge
                merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], end))

        # Extract and save the merged segments
        output_files = []
        _, filename, ext = utils.split_path(video_file)

        for i, (start, end) in enumerate(merged_segments):
            self._check_for_stop()
            output_file = os.path.join(output_dir, f"{filename}.frag{i}.{ext}")
            self._extract_segment(video_file, start, end - start, output_file)
            output_files.append(output_file)

        return output_files


    def _extract_segments(self, video_file, output_dir, segment_duration=5):
        segment_files = []

        duration = utils.get_video_duration(video_file)
        num_segments = max(3, min(10, int(duration // 30)))
        segments = self._select_random_segments(duration, num_segments)

        _, filename, ext = utils.split_path(video_file)

        for segment, (start, length) in enumerate(segments):
            self._check_for_stop()
            segment_output = os.path.join(output_dir, f"{filename}_frag{segment}.{ext}")
            self._extract_segment(video_file, start, length, segment_output)
            segment_files.append(segment_output)

        return segment_files


    def _bisection_search(self, eval_func, min_value, max_value, target_condition):
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


    def _transcode_segment_and_compare(self, wd_dir: str, segment_file: str, crf: int) -> float or None:
        _, filename, ext = utils.split_path(segment_file)

        transcoded_segment_output = os.path.join(wd_dir, f"{filename}.transcoded.{ext}")

        self._transcode_video(segment_file, transcoded_segment_output, crf, "veryfast")

        quality = self._calculate_quality(segment_file, transcoded_segment_output)
        return quality


    def _for_segments(self, segments, op):
        with tempfile.TemporaryDirectory() as wd_dir, ThreadPoolExecutor() as executor:
            def worker(file_path):
                op(wd_dir, file_path)

            for segment in segments:
                executor.submit(worker, segment)


    def _final_transcode(self, input_file, crf, extra_params):
        """Perform the final transcoding with the best CRF using the determined extra_params."""
        _, basename, ext = utils.split_path(input_file)

        logging.info(f"Starting final transcoding with CRF: {crf} and extra params: {extra_params}")
        final_output_file = f"{basename}.temp.{ext}"
        self._transcode_video(input_file, final_output_file, crf, "veryslow", extra_params)

        original_size = os.path.getsize(input_file)
        final_size = os.path.getsize(final_output_file)
        size_reduction = (final_size / original_size) * 100

        # Measure SSIM again after final transcoding
        final_quality = self._calculate_quality(input_file, final_output_file)

        if final_size < original_size:
            utils.start_process("exiftool", ["-overwrite_original", "-TagsFromFile", input_file, "-all:all>all:all", final_output_file])
            os.rename(final_output_file, input_file)
            logging.info(
                f"Final CRF: {crf}, Final Encoding SSIM: {final_quality}, Encoded Size: {final_size} bytes, "
                f"Size reduced by: {original_size - final_size} bytes "
                f"({size_reduction:.2f}% of original size)"
            )
        else:
            os.remove(final_output_file)
            logging.warning(
                f"Final CRF: {crf}, Final Encoding SSIM: {final_quality}, "
                f"Encoded file is larger than the original. Keeping the original file."
            )


    def find_optimal_crf(self, input_file, allow_segments=True):
        """Find the optimal CRF using bisection."""
        original_size = os.path.getsize(input_file)

        duration = utils.get_video_duration(input_file)
        if not duration:
            return None

        with tempfile.TemporaryDirectory() as wd_dir:
            segment_files = []
            if allow_segments and duration > 30:
                logging.info(f"Picking segments from {input_file}")
                segment_files = self._extract_scenes(input_file, wd_dir)
                if len(segment_files) < 2:
                    segment_files = self._extract_segments(input_file, wd_dir)

                logging.info(f"Starting CRF bisection for {input_file} with veryfast preset using {len(segment_files)} segments")
            else:
                segment_files = [input_file]
                logging.info(f"Starting CRF bisection for {input_file} with veryfast preset using whole file")

            def evaluate_crf(mid_crf):
                self._check_for_stop()
                qualities = []

                def get_quality(wd_dir, segment_file):
                    quality = self._transcode_segment_and_compare(wd_dir, segment_file, mid_crf)
                    if quality:
                        qualities.append(quality)

                self._for_segments(segment_files, get_quality)

                avg_quality = sum(qualities) / len(qualities) if qualities else 0
                logging.info(f"CRF: {mid_crf}, Average Quality (SSIM): {avg_quality}")

                return avg_quality

            top_quality = evaluate_crf(0)
            if top_quality < 0.9975:
                raise RuntimeError(f"Sanity check failed: top SSIM value: {top_quality} < 0.998")

            if top_quality < self.target_ssim:
                raise RuntimeError(f"Top SSIM value: {top_quality} < requested SSIM: {self.target_ssim}")

            crf_min, crf_max = 0, 51
            best_crf, best_quality = self._bisection_search(evaluate_crf, min_value = crf_min, max_value = crf_max, target_condition=lambda avg_quality: avg_quality >= self.target_ssim)

            if best_crf is not None and best_quality is not None:
                logging.info(f"Finished CRF bisection. Optimal CRF: {best_crf} with quality: {best_quality}")
            else:
                logging.warning(f"Finished CRF bisection. Could not find CRF matching desired quality ({self.target_ssim}).")
            return best_crf


    def transcode(self, directory: str):
        logging.info(f"Starting video processing. Target SSIM: {self.target_ssim}")
        video_files = self._find_video_files(directory)

        for file in video_files:
            self._check_for_stop()
            logging.info(f"Processing {file}")
            best_crf = self.find_optimal_crf(file)
            if best_crf is not None and self.live_run:
                # increase crf by one as veryslow preset will be used, so result should be above requested quality anyway
                self._final_transcode(file, best_crf + 1, [])
            logging.info(f"Finished processing {file}")

        logging.info("Video processing completed")


def setup_parser(parser: argparse.ArgumentParser):
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
                        help='Requested SSIM value (video quality). Valid values are between 0 and 1. Default is 0.98.')
    parser.add_argument('videos_path',
                        nargs=1,
                        help='Path with videos to transcode.')


def run(args):
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    transcoder = Transcoder(live_run = args.no_dry_run, target_ssim = args.ssim)
    transcoder.transcode(args.videos_path[0])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    try:
        run(sys.argv[1:])
    except RuntimeError as e:
        logging.error(f"Unexpected error occurred: {e}. Terminating")
        exit(1)
