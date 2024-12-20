
import os
import logging
import random
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import utils


def find_video_files(directory):
    """Find video files with specified extensions."""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if utils.is_video(file):
                video_files.append(os.path.join(root, file))
    return video_files


def get_video_duration(video_file):
    """Get the duration of a video in seconds."""
    result = utils.start_process("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_file])

    try:
        return float(result.stdout.strip())
    except ValueError:
        logging.error(f"Failed to get duration for {video_file}")
        return None


def validate_ffmpeg_result(result: utils.ProcessResult):
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def select_random_segments(total_length, num_segments=5, segment_length=5):
    if total_length <= 0 or num_segments <= 0 or segment_length <= 0:
        raise ValueError("Total length, number of segments, and segment length must all be positive.")
    if segment_length > total_length:
        raise ValueError("Segment length cannot exceed total length.")
    if num_segments * segment_length > total_length:
        raise ValueError("Total segments cannot fit within the total length.")

    step = (total_length - segment_length) / (num_segments - 1) if num_segments > 1 else 0
    return [(round(i * step), segment_length) for i in range(num_segments)]


def calculate_quality(original, encoded):
    """Calculate SSIM between original and encoded video."""
    args = [
        "-i", original, "-i", encoded,
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


def encode_video(input_file, output_file, crf, preset, input_params=[], output_params=[]):
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
    validate_ffmpeg_result(result)


def extract_segment(video_file, start_time, segment_length, output_file):
    """ Extract video segment. Video is reencoded with lossless quality to rebuild damaged or troublesome videos """
    encode_video(video_file,
                 output_file,
                 crf = 0,
                 preset = "veryfast",
                 input_params = ["-ss", str(start_time), "-t", str(segment_length)],
                 output_params = ["-an"]            # remove audio - some codecs may cause issues with proper extraction
    )


def extract_scenes(video_file, output_dir, segment_duration=5):
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
        output_file = os.path.join(output_dir, f"{filename}.frag{i}.{ext}")
        extract_segment(video_file, start, end - start, output_file)
        output_files.append(output_file)

    return output_files


def extract_segments(video_file, output_dir, segment_duration=5):
    segment_files = []

    duration = get_video_duration(video_file)
    num_segments = max(3, min(10, int(duration // 30)))
    segments = select_random_segments(duration, num_segments)

    _, filename, ext = utils.split_path(video_file)

    for segment, (start, length) in enumerate(segments):
        segment_output = os.path.join(output_dir, f"{filename}_frag{segment}.{ext}")
        extract_segment(video_file, start, length, segment_output)
        segment_files.append(segment_output)

    return segment_files


def bisection_search(eval_func, min_value, max_value, target_condition):
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


def _encode_segment_and_compare(wd_dir: str, segment_file: str, crf: int) -> float or None:
    _, filename, ext = utils.split_path(segment_file)

    encoded_segment_output = os.path.join(wd_dir, f"{filename}.enc.{ext}")

    encode_video(segment_file, encoded_segment_output, crf, "veryfast")

    quality = calculate_quality(segment_file, encoded_segment_output)
    return quality


def _for_segments(segments, op):
    with tempfile.TemporaryDirectory() as wd_dir, ThreadPoolExecutor() as executor:
        def worker(file_path):
            op(wd_dir, file_path)

        for segment in segments:
            executor.submit(worker, segment)


def find_optimal_crf(input_file, requested_quality=0.98, allow_segments=True):
    """Find the optimal CRF using bisection."""
    original_size = os.path.getsize(input_file)

    duration = get_video_duration(input_file)
    if not duration:
        return None

    with tempfile.TemporaryDirectory() as wd_dir:
        segment_files = []
        if allow_segments and duration > 30:
            logging.info(f"Picking segments from {input_file}")
            segment_files = extract_scenes(input_file, wd_dir)
            if len(segment_files) < 2:
                segment_files = extract_segments(input_file, wd_dir)

            logging.info(f"Starting CRF bisection for {input_file} with veryfast preset using {len(segment_files)} segments")
        else:
            segment_files = [input_file]
            logging.info(f"Starting CRF bisection for {input_file} with veryfast preset using whole file")

        def evaluate_crf(mid_crf):
            qualities = []

            def get_quality(wd_dir, segment_file):
                quality = _encode_segment_and_compare(wd_dir, segment_file, mid_crf)
                if quality:
                    qualities.append(quality)

            _for_segments(segment_files, get_quality)

            avg_quality = sum(qualities) / len(qualities) if qualities else 0
            logging.info(f"CRF: {mid_crf}, Average Quality (SSIM): {avg_quality}")

            return avg_quality

        top_quality = evaluate_crf(0)
        if top_quality < 0.9975:
            raise ValueError(f"Sanity check failed: top SSIM value: {top_quality} < 0.998")

        crf_min, crf_max = 0, 51
        best_crf, best_quality = bisection_search(evaluate_crf, min_value = crf_min, max_value = crf_max, target_condition=lambda avg_quality: avg_quality >= requested_quality)

        if best_crf is not None and best_quality is not None:
            logging.info(f"Finished CRF bisection. Optimal CRF: {best_crf} with quality: {best_quality}")
        else:
            logging.warning(f"Finished CRF bisection. Could not find CRF matching desired quality ({requested_quality}).")
        return best_crf


def final_encode(input_file, crf, extra_params):
    """Perform the final encoding with the best CRF using the determined extra_params."""
    _, basename, ext = utils.split_path(input_file)

    logging.info(f"Starting final encoding with CRF: {crf} and extra params: {extra_params}")
    final_output_file = f"{basename}.temp.{ext}"
    encode_video(input_file, final_output_file, crf, "veryslow", extra_params)

    original_size = os.path.getsize(input_file)
    final_size = os.path.getsize(final_output_file)
    size_reduction = (final_size / original_size) * 100

    # Measure SSIM again after final encoding
    final_quality = calculate_quality(input_file, final_output_file)

    if final_size < original_size:
        subprocess.run(["exiftool", "-overwrite_original", "-TagsFromFile", input_file, "-all:all>all:all", final_output_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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


def main(directory):
    logging.info("Starting video processing")
    video_files = find_video_files(directory)

    for file in video_files:
        logging.info(f"Processing {file}")
        best_crf = find_optimal_crf(file)
        if best_crf is not None and False:
            # increase crf by one as veryslow preset will be used, so result should be above 0.98 quality anyway
            final_encode(file, best_crf + 1, [])
        logging.info(f"Finished processing {file}")

    logging.info("Video processing completed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encode_videos.py /path/to/directory")
        sys.exit(1)

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    directory = sys.argv[1]
    main(directory)
