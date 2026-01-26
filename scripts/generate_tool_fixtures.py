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


def _write_subtitle(path: Path, text: str = "Hello world") -> None:
    _ensure_dir(path.parent)
    if path.exists():
        return
    payload = f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n\n"
    path.write_text(payload, encoding="utf-8")


def _format_srt_time(ms: int) -> str:
    hours = ms // 3_600_000
    ms -= hours * 3_600_000
    minutes = ms // 60_000
    ms -= minutes * 60_000
    seconds = ms // 1_000
    ms -= seconds * 1_000
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


def _write_srt(path: Path, entries: list[tuple[int, int, str]]) -> None:
    _ensure_dir(path.parent)
    if path.exists():
        return
    lines: list[str] = []
    for idx, (start_ms, end_ms, text) in enumerate(entries, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_srt_time(start_ms)} --> {_format_srt_time(end_ms)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _mux_subtitle(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    *,
    audio_title: str | None = None,
    subtitle_language: str | None = None,
) -> bool:
    if not _ffmpeg_available():
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(subtitle_path),
        "-map",
        "0",
        "-map",
        "1",
        "-c",
        "copy",
        "-c:s",
        "srt",
    ]
    if audio_title:
        cmd.extend(["-metadata:s:a:0", f"title={audio_title}"])
        cmd.extend(["-metadata:s:a:0", "language=und"])
    if subtitle_language:
        cmd.extend(["-metadata:s:s:0", f"language={subtitle_language}"])
    cmd.append(str(output_path))

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _remux_with_audio_title(video_path: Path, output_path: Path, audio_title: str) -> bool:
    if not _ffmpeg_available():
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0",
        "-c",
        "copy",
        "-metadata:s:a:0",
        f"title={audio_title}",
        "-metadata:s:a:0",
        "language=und",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


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


def _ensure_video(
    path: Path,
    *,
    profile: str,
    color: str,
    duration: float = 1.0,
    size: str = "320x240",
) -> None:
    _ensure_dir(path.parent)
    if path.exists():
        return

    if profile == "mp4":
        ok = _generate_video(path, vcodec="libx264", acodec="aac", color=color, duration=duration, size=size)
    elif profile == "mkv":
        ok = _generate_video(path, vcodec="libx264", acodec="aac", color=color, duration=duration, size=size)
    elif profile == "avi":
        ok = _generate_video(path, vcodec="mpeg4", acodec="mp3", color=color, duration=duration, size=size)
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


def _generate_merge(root: Path) -> None:
    base = root / "merge"
    ok_dir = base / "ok"
    warnings_dir = base / "warnings"

    single_dir = ok_dir / "single"
    single_subs = single_dir / "subtitles"
    multi_dir = ok_dir / "multi"
    dup_dir = warnings_dir / "duplicate_names"

    _ensure_dir(single_subs)
    _ensure_dir(multi_dir)
    _ensure_dir(dup_dir)

    # Single video with subtitles in the same dir and a subdir.
    single_video = single_dir / "Lone Movie.mp4"
    _ensure_video(single_video, profile="mp4", color="blue")
    _write_subtitle(single_dir / "Lone Movie.en.srt", text="Hello world")
    _write_subtitle(single_subs / "Lone Movie.pl.srt", text="Witaj swiecie")

    # Multiple videos matched by prefix in one directory.
    ep1 = multi_dir / "Series 01.mkv"
    ep2 = multi_dir / "Series 02.mkv"
    _ensure_video(ep1, profile="mkv", color="green")
    _ensure_video(ep2, profile="mkv", color="yellow")
    _write_subtitle(multi_dir / "Series 01.en.srt", text="Episode one")
    _write_subtitle(multi_dir / "Series 01.pl.srt", text="Odcinek pierwszy")
    _write_subtitle(multi_dir / "Series 02.en.srt", text="Episode two")

    # Warning case: duplicate video base names (different extensions).
    _ensure_video(dup_dir / "SameName.mp4", profile="mp4", color="red")
    _ensure_video(dup_dir / "SameName.mkv", profile="mkv", color="purple")


def _generate_subtitles_fixer(root: Path) -> None:
    base = root / "subtitles_fixer"
    ok_dir = base / "ok"
    broken_dir = base / "broken"
    no_subs_dir = base / "no_subs"

    _ensure_dir(ok_dir)
    _ensure_dir(broken_dir)
    _ensure_dir(no_subs_dir)

    # OK: subtitle ends before video end.
    ok_video = ok_dir / "ok_with_subs.mkv"
    ok_source = ok_dir / "ok_with_subs.source.mp4"
    ok_srt = ok_dir / "ok_with_subs.srt"
    if not ok_video.exists():
        _ensure_video(ok_source, profile="mp4", color="green")
        _write_srt(ok_srt, [(0, 800, "Hello world")])
        if not _mux_subtitle(ok_source, ok_srt, ok_video):
            _write_placeholder(ok_video)

    # Broken: subtitle track longer than video.
    broken_video = broken_dir / "broken_with_subs.mkv"
    broken_source = broken_dir / "broken_with_subs.source.mp4"
    broken_srt = broken_dir / "broken_with_subs.srt"
    if not broken_video.exists():
        _ensure_video(broken_source, profile="mp4", color="red", duration=1.0)
        _write_srt(broken_srt, [(0, 4_000, "Too long subtitle")])
        if not _mux_subtitle(broken_source, broken_srt, broken_video):
            _write_placeholder(broken_video)

    # No subtitles at all.
    _ensure_video(no_subs_dir / "no_subs.mkv", profile="mkv", color="blue")


def _generate_transcode(root: Path) -> None:
    base = root / "transcode"
    ok_dir = base / "ok"
    nested_dir = ok_dir / "nested"

    _ensure_dir(ok_dir)
    _ensure_dir(nested_dir)

    _ensure_video(ok_dir / "Sample One.mp4", profile="mp4", color="cyan")
    _ensure_video(ok_dir / "Sample Two.mkv", profile="mkv", color="orange")
    _ensure_video(nested_dir / "Nested Sample.avi", profile="avi", color="purple")


def _generate_language_fixer(root: Path) -> None:
    base = root / "language_fixer"
    missing_dir = base / "missing"
    audio_dir = base / "audio_only"

    _ensure_dir(missing_dir)
    _ensure_dir(audio_dir)

    # Missing subtitle language (subtitle language set to und).
    missing_source = missing_dir / "Missing Subs.source.mp4"
    missing_srt = missing_dir / "Missing Subs.srt"
    missing_output = missing_dir / "Missing Subs.mkv"
    if not missing_output.exists():
        _ensure_video(missing_source, profile="mp4", color="blue")
        _write_srt(
            missing_srt,
            [
                (0, 900, "Witaj swiecie"),
                (1200, 2300, "To jest test"),
            ],
        )
        if not _mux_subtitle(
            missing_source,
            missing_srt,
            missing_output,
            subtitle_language="und",
        ):
            _write_placeholder(missing_output)

    # Missing audio language but with a track name hint.
    audio_source = audio_dir / "Audio Missing.source.mp4"
    audio_output = audio_dir / "Audio Missing.mkv"
    if not audio_output.exists():
        _ensure_video(audio_source, profile="mp4", color="green")
        if not _remux_with_audio_title(audio_source, audio_output, "English"):
            _write_placeholder(audio_output)


def _generate_melt(root: Path) -> None:
    base = root / "melt"
    movie_dir = base / "movie"
    series_a = base / "series" / "set_a"
    series_b = base / "series" / "set_b"
    mismatch_dir = base / "mismatch"

    _ensure_dir(movie_dir)
    _ensure_dir(series_a)
    _ensure_dir(series_b)
    _ensure_dir(mismatch_dir)

    # Single-dir duplicates (use as --input <movie_dir>).
    _ensure_video(movie_dir / "Movie A 1080p.mkv", profile="mkv", color="blue", size="640x360")
    _ensure_video(movie_dir / "Movie A 720p.mkv", profile="mkv", color="red", size="320x240")

    # Multi-dir duplicates (use as --input <series_a> --input <series_b>).
    _ensure_video(series_a / "Episode 01.mkv", profile="mkv", color="green")
    _ensure_video(series_a / "Episode 02.mkv", profile="mkv", color="yellow")
    _ensure_video(series_b / "Episode 01.mkv", profile="mkv", color="purple")
    _ensure_video(series_b / "Episode 02.mkv", profile="mkv", color="cyan")

    # Length mismatch case to test allow-length-mismatch behavior.
    _ensure_video(mismatch_dir / "Mismatch A.mkv", profile="mkv", color="orange", duration=1.0)
    _ensure_video(mismatch_dir / "Mismatch B.mkv", profile="mkv", color="pink", duration=2.0)


def _generate_utilities(root: Path) -> None:
    base = root / "utilities"
    scenes_dir = base / "scenes"
    nested_dir = scenes_dir / "nested"

    _ensure_dir(scenes_dir)
    _ensure_dir(nested_dir)

    _ensure_video(scenes_dir / "Scenes Sample.mp4", profile="mp4", color="blue")
    _ensure_video(nested_dir / "Nested Scenes Sample.mkv", profile="mkv", color="yellow")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fixture files for manual tool inspection.")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Root directory where fixtures will be created (default: manual_fixtures).",
    )
    parser.add_argument(
        "--tool",
        default="all",
        choices=["all", "concatenate", "merge", "subtitles_fixer", "transcode", "language_fixer", "melt", "utilities"],
        help="Tool fixtures to generate (default: all).",
    )

    args = parser.parse_args()
    root = Path(args.root)
    _ensure_dir(root)

    if args.tool in {"all", "concatenate"}:
        _generate_concatenate(root)
    if args.tool in {"all", "merge"}:
        _generate_merge(root)
    if args.tool in {"all", "subtitles_fixer"}:
        _generate_subtitles_fixer(root)
    if args.tool in {"all", "transcode"}:
        _generate_transcode(root)
    if args.tool in {"all", "language_fixer"}:
        _generate_language_fixer(root)
    if args.tool in {"all", "melt"}:
        _generate_melt(root)
    if args.tool in {"all", "utilities"}:
        _generate_utilities(root)

    print(f"Fixtures generated under: {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
