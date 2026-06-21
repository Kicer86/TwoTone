#!/usr/bin/env python3
"""Probe AAC encoder-delay priming as the *active* (CI) ffmpeg build reports and decodes it.

Why this exists
---------------
The audio-alignment failures on ubuntu-26.04 (ffmpeg 8.0.1) come from AAC encoder-delay
priming (1024 samples / ~21.3 ms).  Local ffmpeg n8.1.2 and macOS *absorb* it; the
ubuntu CI build *exposes* it.  Our priming handling keys on the container-reported
``initial_padding`` field, and the suspicion is that this field is reported differently
across builds — which would make the gate a silent no-op on the exposing build.

This script settles that empirically.  For each variant it prints, to stdout (so it is
readable straight from the CI log, no artifact download), two independent views:

* what the container *says* — ``start_time`` / ``start_pts`` / ``initial_padding`` /
  ``codec_delay`` as raw ffprobe reports them, and as our ``video_utils`` wrapper parses
  them; and
* what the decoder *does* — leading silence and first-beep centre after decoding the
  stream to PCM with the active ffmpeg.

Run it on the absorbing build (macOS / local) and the exposing build (ubuntu-26.04) and
diff the rows: if ``initial_padding`` differs, the gate is build-dependent; if only the
decoded leading silence differs, the priming is exposed regardless of what the field says.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
TESTS_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TESTS_DIR))

import numpy as np  # noqa: E402

from common import run_ffmpeg  # noqa: E402
from melt.test_audio_alignment import AudioAlignmentTest, VARIANT_BY_NAME  # noqa: E402
from twotone.tools.utils import video_utils  # noqa: E402


# mkv-AAC asO (v08/v11) is the failing class; the rest are references across containers
# and across the asR/asO axis so the priming signal stands out against a clean baseline.
DEFAULT_VARIANTS = [
    "v00_asR_vsR_aeR_veR",  # mkv, asR, speed 1.0   — clean reference (no offset)
    "v08_asO_vsR_aeR_veR",  # mkv, asO, speed 1.0   — THE failing class
    "v11_asO_vsR_aeT_veT",  # mkv, asO, speed 1.0   — failing class, trimmed ends
    "v09_asO_vsR_aeR_veT",  # mp4, asO, speed 1.0   — fixed by the FLAC flow
    "v10_asO_vsR_aeT_veR",  # mov, asO, speed 1.02  — fixed by the FLAC flow
]


def _raw_ffprobe(path: str) -> dict[str, Any]:
    """Container-reported fields straight from ffprobe, bypassing our wrapper's parsing."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,start_time,start_pts,initial_padding,codec_delay,"
        "nb_frames,nb_read_frames,sample_rate,time_base",
        "-of", "json", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        data = json.loads(out.stdout or "{}")
        streams = data.get("streams") or [{}]
        return streams[0]
    except Exception as err:  # noqa: BLE001 - diagnostics must never abort
        return {"error": repr(err)}


def _wrapper_fields(path: str) -> dict[str, Any]:
    """The same fields as our app sees them, through ``video_utils.get_video_full_info``."""
    info = video_utils.get_video_full_info(path)
    audio = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {}
    )
    return {
        "codec_name": audio.get("codec_name"),
        "start_time": audio.get("start_time"),
        "start_pts": audio.get("start_pts"),
        "initial_padding": audio.get("initial_padding"),
        "codec_delay": audio.get("codec_delay"),
    }


def _decode_landing(path: str, sample_rate: int, work_dir: str) -> dict[str, Any]:
    """Decode the first audio stream with the active ffmpeg and measure where content lands."""
    wav_path = os.path.join(work_dir, "probe.wav")
    try:
        run_ffmpeg(
            [
                "-y", "-i", path, "-map", "0:a:0",
                "-ac", "1", "-ar", str(sample_rate),
                "-c:a", "pcm_s16le", wav_path,
            ],
            expected_path=wav_path,
        )
    except Exception as err:  # noqa: BLE001
        return {"decode_error": repr(err)}

    _sr, samples = AudioAlignmentTest._read_mono_wav(wav_path)
    centers = AudioAlignmentTest._detect_beep_centers(wav_path)
    loud = np.flatnonzero(np.abs(samples) > 0.02)
    return {
        "total_samples": int(samples.size),
        "leading_silence_ms": (
            round(float(loud[0]) / sample_rate * 1000, 3) if loud.size else None
        ),
        "first_beep_center_s": round(centers[0], 6) if centers else None,
        "beep_count": len(centers),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    args = parser.parse_args()

    AudioAlignmentTest.setUpClass()
    sample_rate = AudioAlignmentTest.SAMPLE_RATE

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="priming_probe_") as work_dir:
        for name in args.variants:
            spec = VARIANT_BY_NAME.get(name)
            if spec is None:
                print(f"!! unknown variant: {name}", flush=True)
                continue
            path = AudioAlignmentTest.variant_paths[name]
            raw = _raw_ffprobe(path)
            wrap = _wrapper_fields(path)
            land = _decode_landing(path, sample_rate, work_dir)
            rows.append({
                "variant": name,
                "container": spec.extension,
                "speed": spec.speed,
                "raw": raw,
                "wrapper": wrap,
                "decode": land,
            })

    # Compact, grep-friendly table — readable straight from the CI log.
    print("\n=== PRIMING PROBE (active ffmpeg build) ===", flush=True)
    header = (
        f"{'variant':<22} {'cont':<4} {'spd':<5} "
        f"{'raw.ipad':<9} {'raw.cdelay':<11} {'raw.start_t':<12} "
        f"{'wrap.ipad':<10} {'lead_sil_ms':<12} {'beep0_s':<9}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        raw, wrap, dec = row["raw"], row["wrapper"], row["decode"]
        print(
            f"{row['variant']:<22} {row['container']:<4} {_fmt(row['speed']):<5} "
            f"{_fmt(raw.get('initial_padding')):<9} {_fmt(raw.get('codec_delay')):<11} "
            f"{_fmt(raw.get('start_time')):<12} "
            f"{_fmt(wrap.get('initial_padding')):<10} "
            f"{_fmt(dec.get('leading_silence_ms')):<12} {_fmt(dec.get('first_beep_center_s')):<9}",
            flush=True,
        )

    # Full JSON too, for the record (single block, still cheap to read).
    print("\n=== PRIMING PROBE JSON ===", flush=True)
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
