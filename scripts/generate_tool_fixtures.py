#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_ROOT = Path("manual_fixtures")


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_placeholder(path: Path) -> None:
    _ensure_dir(path.parent)
    if not path.exists():
        path.write_bytes(b"")


def _generate_video(
    path: Path,
    *,
    duration: float = 1.0,
    size: str = "320x240",
    fps: int = 25,
    vcodec: str = "libx264",
    acodec: str = "aac",
    color: str = "blue",
) -> bool:
    if not _ffmpeg_available():
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:d={duration}",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-r",
        str(fps),
        "-c:v",
        vcodec,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        acodec,
        str(path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _ensure_video(path: Path, *, profile: str, color: str) -> None:
    _ensure_dir(path.parent)
    if path.exists():
        return

    if profile == "mp4":
        ok = _generate_video(path, vcodec="libx264", acodec="aac", color=color)
    elif profile == "mkv":
        ok = _generate_video(path, vcodec="libx264", acodec="aac", color=color)
    elif profile == "avi":
        ok = _generate_video(path, vcodec="mpeg4", acodec="mp3", color=color)
    elif profile == "rmvb":
        ok = False
    else:
        ok = False

    if not ok:
        _write_placeholder(path)


def _generate_concatenate(root: Path) -> None:
    base = root / "concatenate"
    ok_dir = base / "ok"
    warn_dir = base / "warnings"
    ignored_dir = base / "ignored"

    _ensure_dir(ok_dir)
    _ensure_dir(warn_dir)
    _ensure_dir(ignored_dir)

    # OK cases
    mp4_group = [
        ok_dir / "Movie One cd1.mp4",
        ok_dir / "Movie One cd2.mp4",
    ]
    for idx, path in enumerate(mp4_group):
        _ensure_video(path, profile="mp4", color="blue" if idx == 0 else "red")

    avi_group = [
        ok_dir / "Another-Movie-cd1.avi",
        ok_dir / "Another-Movie-cd2.avi",
    ]
    for idx, path in enumerate(avi_group):
        _ensure_video(path, profile="avi", color="green" if idx == 0 else "yellow")

    mkv_group = [
        ok_dir / "Trilogy cd1.mkv",
        ok_dir / "Trilogy cd2.mkv",
        ok_dir / "Trilogy cd3.mkv",
    ]
    for idx, path in enumerate(mkv_group):
        _ensure_video(path, profile="mkv", color="purple" if idx == 0 else "cyan")

    case_group = [
        ok_dir / "Case Movie CD1.mp4",
        ok_dir / "Case Movie CD2.mp4",
    ]
    for idx, path in enumerate(case_group):
        _ensure_video(path, profile="mp4", color="orange" if idx == 0 else "pink")

    folder_group_dir = ok_dir / "FolderMovie"
    folder_group = [
        folder_group_dir / "cd1.mp4",
        folder_group_dir / "cd2.mp4",
    ]
    for idx, path in enumerate(folder_group):
        _ensure_video(path, profile="mp4", color="white" if idx == 0 else "black")

    rmvb_group = [
        ok_dir / "Rmvb Movie cd1.rmvb",
        ok_dir / "Rmvb Movie cd2.rmvb",
    ]
    for path in rmvb_group:
        _ensure_video(path, profile="rmvb", color="gray")

    # Warning cases
    _ensure_video(warn_dir / "Single cd1.mp4", profile="mp4", color="blue")
    _ensure_video(warn_dir / "Gap cd1.mp4", profile="mp4", color="blue")
    _ensure_video(warn_dir / "Gap cd3.mp4", profile="mp4", color="blue")

    # Ignored cases
    _ensure_video(ignored_dir / "moviecd1.mp4", profile="mp4", color="blue")
    _write_placeholder(ignored_dir / "README.txt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fixture files for manual tool inspection.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Root directory where fixtures will be created (default: manual_fixtures).",
    )
    parser.add_argument(
        "--tool",
        default="concatenate",
        choices=["concatenate"],
        help="Tool fixtures to generate.",
    )

    args = parser.parse_args()
    root = Path(args.root)
    _ensure_dir(root)

    if args.tool == "concatenate":
        _generate_concatenate(root)

    print(f"Fixtures generated under: {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
