
import argparse
import logging
import os
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from .tool import Tool
from ..tools.utils import process_utils, video_utils

"""
    Tool for fixing MKV files by removing audio/subtitle streams that are much longer or shorter than the main video stream.
    This tool now operates on MKV containers only and uses mkvmerge for all manipulations.
"""
class VideoFixerTool(Tool):
    analysis_results: Optional[Dict[str, List[Dict[str, Any]]]]

    def __init__(self) -> None:
        super().__init__()
        self.analysis_results = {}

    def setup_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", nargs='+', help="Input MKV file(s) or directories to fix")

    def analyze(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str) -> None:
        logger.info("Analyzing MKV files for stream length anomalies...")

        # Ensure required tools are available
        process_utils.ensure_tools_exist(["mkvmerge"], logger)

        # Gather MKV files from input paths
        input_files: List[str] = []

        for inp in args.input:
            if os.path.isdir(inp):
                input_files.extend(video_utils.collect_video_files(inp, interruptible=type('Dummy', (), {'_check_for_stop': lambda s: None})()))
            elif video_utils.is_video(inp):
                input_files.append(inp)

        # Filter to MKV only
        mkv_files: List[str] = []
        for f in input_files:
            if os.path.splitext(f)[1].lower() == ".mkv":
                mkv_files.append(f)
            else:
                logger.info(f"Skipping non-MKV input: {f}")

        self.analysis_results = {}

        for video_path in tqdm(mkv_files, desc="Analyzing files", unit="file"):
            # Only MKV files are processed
            ext = os.path.splitext(video_path)[1].lower()
            if ext != ".mkv":
                logger.info(f"Skipping non-MKV input: {video_path}")
                continue

            mkv_info: Dict[str, Any] = video_utils.get_video_data_mkvmerge(video_path, enrich=True)
            streams_by_type: Dict[str, List[Dict[str, Any]]] = mkv_info.get("tracks", {})

            video_streams: List[Dict[str, Any]] = streams_by_type.get('video', [])

            if not video_streams:
                logger.warning(f"No video stream found in {video_path}")
                continue

            video_duration: Optional[int] = video_streams[0].get('length')
            logger.info(f"File: {video_path}, Video duration: {video_duration/1000:.2f}s")
            flagged: List[Dict[str, Any]] = []

            for stype in ('audio', 'subtitle'):
                for s in streams_by_type.get(stype, []):
                    s_duration: Optional[int] = s.get('length')
                    if s_duration is None or video_duration is None:
                        continue

                    logger.info(f"  Checking {stype} stream #{s.get('tid')} \'{s.get('language') or '-'}\' duration: {s_duration/1000:.2f}s")

                    diff: float = abs(s_duration - video_duration)
                    rel_diff: float = diff / video_duration if video_duration else 0
                    if rel_diff > 0.1:
                        flagged.append({'type': stype, 'tid': s.get('tid'), 'duration': s_duration})
                        logger.warning(f"{stype} stream #{s.get('tid')} \'{s.get('language') or '-'}\' has abnormal duration. Will be removed. (duration: {s_duration/1000:.2f}s, diff: {rel_diff*100:.1f}%)")

            self.analysis_results[video_path] = flagged

            if not flagged:
                logger.info("No problematic streams detected.")
            else:
                logger.info(f"Flagged {len(flagged)} stream(s) for removal.")


    def perform(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str) -> None:
        logger.info("Fixing MKV files in place...")

        # Ensure required tools are available
        process_utils.ensure_tools_exist(["mkvmerge"], logger)

        if not self.analysis_results:
            logger.error("No analysis results found. Run analyze first.")
            return

        for video_path, flagged in tqdm(self.analysis_results.items(), desc="Fixing files", unit="file"):
            if not flagged:
                logger.info(f"{video_path}: No streams to remove.")
                continue

            ext = os.path.splitext(video_path)[1].lower()
            if ext != ".mkv":
                logger.info(f"Skipping non-MKV input: {video_path}")
                continue

            mkv_info: Dict[str, Any] = video_utils.get_video_data_mkvmerge(video_path, enrich=True)
            tracks: Dict[str, List[Dict[str, Any]]] = mkv_info.get("tracks", {})

            # Compute per-type keep lists based on mkvmerge track IDs
            flagged_set = set((f['type'], f['tid']) for f in flagged)
            keep_ids: Dict[str, List[int]] = {"video": [], "audio": [], "subtitle": []}
            original_counts: Dict[str, int] = {}

            for stype in ("video", "audio", "subtitle"):
                st_list = tracks.get(stype, [])
                original_counts[stype] = len(st_list)
                for s in st_list:
                    tid = s.get("tid")
                    if (stype, tid) not in flagged_set:
                        keep_ids[stype].append(tid)

            # Assemble mkvmerge command
            base, _ = os.path.splitext(video_path)
            tmp_path: str = f"{base}.twotone_temp.mkv"

            cmd: List[str] = ["mkvmerge", "-o", tmp_path]

            # If all tracks of a given type were removed, disable that type; otherwise restrict to keep list
            # Only restrict audio/subtitle; video is kept as-is unless explicitly empty (shouldn't happen)
            if original_counts.get("audio", 0) > 0:
                if len(keep_ids["audio"]) == 0:
                    cmd.extend(["--no-audio"])
                else:
                    cmd.extend(["--audio-tracks", ",".join(str(i) for i in keep_ids["audio"])])

            if original_counts.get("subtitle", 0) > 0:
                if len(keep_ids["subtitle"]) == 0:
                    cmd.extend(["--no-subtitles"])
                else:
                    cmd.extend(["--subtitle-tracks", ",".join(str(i) for i in keep_ids["subtitle"])])

            # Input file (options above apply to this single input)
            cmd.append(video_path)

            result = process_utils.start_process(cmd[0], cmd[1:])
            if result.returncode != 0:
                logger.error(f"mkvmerge failed for {video_path}: {result.stderr}")
                continue

            # Replace original in place
            os.replace(tmp_path, video_path)
            logger.info(f"Fixed {video_path}: removed {len(flagged)} stream(s)")
