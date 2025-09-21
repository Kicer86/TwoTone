
import argparse
import logging
import math
import os
import re

from overrides import override

from .tool import Tool
from .utils import video_utils, process_utils, files_utils


def extract_scenes(video_path, output_dir, format: str, scale: float):
    """
    Extracts all video frames, names them based on their timestamp, and groups them into scene subdirectories.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory where scene frame folders will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get scene change timestamps
    scene_changes = [float(sc) for sc in video_utils.detect_scene_changes(video_path)]
    scene_changes.append(math.inf)

    # Extract all frames while capturing PTS times
    temp_folder = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_folder, exist_ok=True)

    ascale = 100/scale
    output_pattern = os.path.join(temp_folder, f"frame_%06d.{format}")
    args = [
        "-i", video_path,
        "-frame_pts", "true",
        "-vsync", "0",
        "-q:v", "2",
        "-vf", f"showinfo,scale=iw/{ascale}:ih/{ascale}",
        output_pattern
    ]

    result = process_utils.start_process("ffmpeg", args = args)

    # Parse PTS times from stderr
    frame_pts_map = {}  # Maps sequential frame numbers to PTS timestamps
    frame_pattern = re.compile(r"n: *(\d+).*pts_time:([\d.]+)")

    for line in result.stderr.splitlines():
        match = frame_pattern.search(line)
        if match:
            frame_number = int(match.group(1))
            pts_time = float(match.group(2))
            frame_pts_map[frame_number] = pts_time

    scene_index = 0
    created_scenes = set()

    # Process frames: rename and move in the same loop
    frame_files = sorted(os.listdir(temp_folder))
    for frame_number, frame_file in enumerate(frame_files):
        if frame_number in frame_pts_map:
            timestamp = frame_pts_map[frame_number]

            if round(timestamp * 1000) >= scene_changes[scene_index]:
                scene_index += 1

            scene_dir = os.path.join(output_dir, f"scene_{scene_index}")

            if scene_index not in created_scenes:
                os.makedirs(scene_dir, exist_ok = False)
                created_scenes.add(scene_index)

            _, _, ext = files_utils.split_path(frame_file)

            new_name = f"frame_{timestamp:.3f}.{ext}"

            old_path = os.path.join(temp_folder, frame_file)
            new_path = os.path.join(scene_dir, new_name)

            # Rename and move the file in one step
            os.rename(old_path, new_path)
        else:
            print("Error")

    # Cleanup temp folder
    os.rmdir(temp_folder)


class UtilitiesTool(Tool):
    @override
    def setup_parser(self, parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(dest="subtool", help="Available subtools:")

        scenes_extractor = subparsers.add_parser(
            "scenes",
            help = "extract scenes from videos"
        )
        scenes_extractor.add_argument('video_path',
                                      nargs=1,
                                      help='Path to video file')
        scenes_extractor.add_argument("--output", "-o",
                                      required=True,
                                      help='Output directory')
        scenes_extractor.add_argument("--format", "-f",
                                      default = "jpeg",
                                      help = "File format for frames. Pass file extension like tiff, jpeg, png. jpeg is default")
        scenes_extractor.add_argument("--scale", "-s",
                                      default = 100,
                                      help = "Frames scale in %%. Default is 100")

    @override
    def analyze(self, args, logger: logging.Logger, working_dir: str):
        pass

    @override
    def perform(self, args, analysis, no_dry_run: bool, logger: logging.Logger, working_dir: str):
        if args.subtool == "scenes":
            extract_scenes(video_path = args.video_path[0], output_dir = args.output, format = args.format, scale = float(args.scale))
        else:
            logging.error(f"Error: Unknown subtool {args.subtool}")
