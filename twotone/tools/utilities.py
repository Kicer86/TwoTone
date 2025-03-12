
import argparse
import logging
import os
import re

from overrides import override

from .tool import Tool
from .utils2 import video, process, files


def extract_scenes(video_path, output_folder):
    """
    Extracts all video frames, names them based on their timestamp, and groups them into scene subdirectories.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory where scene frame folders will be stored.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get scene change timestamps
    scene_changes = video.detect_scene_changes(video_path)
    scene_changes.append(None)  # Ensure last segment is included

    # Extract all frames while capturing PTS times
    temp_folder = os.path.join(output_folder, "temp_frames")
    os.makedirs(temp_folder, exist_ok=True)

    output_pattern = os.path.join(temp_folder, "frame_%06d.png")
    args = [
        "-i", video_path,
        "-frame_pts", "true",
        "-vsync", "0",
        "-q:v", "2",  # High-quality PNG output
        "-vf", "showinfo",  # Enable frame metadata output
        output_pattern
    ]

    result = process.start_process("ffmpeg", args = args)

    # Parse PTS times from stderr
    frame_pts_map = {}  # Maps sequential frame numbers to PTS timestamps
    frame_pattern = re.compile(r"n: *(\d+).*pts_time:([\d.]+)")

    for line in result.stderr.splitlines():
        match = frame_pattern.search(line)
        if match:
            frame_number = int(match.group(1))
            pts_time = float(match.group(2))
            frame_pts_map[frame_number] = pts_time

    # Prepare initial scene directory
    scene_index = 0
    scene_folder = os.path.join(output_folder, f"scene_{scene_index}")
    os.makedirs(scene_folder, exist_ok=True)

    # Process frames: rename and move in the same loop
    for frame_number, frame_file in enumerate(sorted(os.listdir(temp_folder))):
        if frame_number in frame_pts_map:
            timestamp = frame_pts_map[frame_number]
            new_name = f"frame_{timestamp:.3f}.png"

            old_path = os.path.join(temp_folder, frame_file)
            new_path = os.path.join(scene_folder, new_name)

            # If we've reached a scene change, update scene directory
            if scene_changes[scene_index] is not None and timestamp >= scene_changes[scene_index]:
                scene_index += 1
                scene_folder = os.path.join(output_folder, f"scene_{scene_index}")
                os.makedirs(scene_folder, exist_ok=True)
                new_path = os.path.join(scene_folder, new_name)

            # Rename and move the file in one step
            os.rename(old_path, new_path)

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

    @override
    def run(self, args, no_dry_run: bool, logger: logging.Logger):
        if args.subtool == "scenes":
            extract_scenes(video_path = args.video_path[0], output = args.output)
        else:
            logger.error(f"Error: Unknown subtool {args.subtool}")
            sys.exit(1)
