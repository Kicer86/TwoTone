# Audio Alignment Debug Guide

Use `tests/melt/debug_audio_alignment.py` to trace one failing
`AudioAlignmentTest.test_audio_alignment_after_melt` case in two environments.
The default pair is the first GitHub Actions failure:

```bash
python tests/melt/debug_audio_alignment.py run \
  --lhs v00_asR_vsR_aeR_veR \
  --rhs v01_asR_vsR_aeR_veT \
  --output-dir /tmp/twotone-audio-debug \
  --overwrite
```

The command writes:

- `debug_report.json`: structured data for comparison.
- `twotone-debug.log`: full debug logging from `melt`.
- `work/` and `melt_output/`: preserved files for manual inspection.

The report captures:

- Python/package/tool versions and tool paths.
- Canonical, lhs, rhs, and output media hashes plus `ffprobe` and `mkvmerge`
  metadata.
- The full `MeltAnalyzer` plan.
- PairMatcher scene changes, frame probes, initial key-frame matches,
  constant-offset result, final mapping relation, and final matched timepoints.
- Audio patching decisions, sync offset calculations, prepared stream entries,
  and final `mkvmerge` arguments.
- Every external command run through `process_utils.start_process`, including
  normalized `ffmpeg`, `ffprobe`, `mkvmerge`, and `mkvextract` arguments.
- Extracted beep centers for both output audio tracks and their offsets.

Run the same command locally and in GitHub Actions, then compare the two
reports:

```bash
python tests/melt/debug_audio_alignment.py compare \
  /path/to/local/debug_report.json \
  /path/to/github/debug_report.json \
  --output /tmp/twotone-audio-debug-comparison.json
```

Interpretation:

- If input SHA-256 hashes differ, the synthetic fixtures were generated
  differently before `melt` started.
- If PairMatcher events differ, frame probing or matching diverged before audio
  patching.
- If the final mapping and `melt_run` command signatures are identical but the
  output beep centers differ, the divergence is in external tool output for the
  same parameters.
- `--cache-mode fresh` is the default and avoids stale local `MeltCache` data.
  Use `--cache-mode test` only when you want to match the normal pytest class
  cache behavior exactly.

## GitHub Actions

The `.github/workflows/audio-alignment-debug.yml` workflow runs the default
diagnostic pair on pushes to non-`master` branches when relevant project files
change.  It also supports manual `workflow_dispatch` runs with custom `lhs` and
`rhs` variant names.

After the workflow finishes, download the `audio-alignment-debug-<run-id>`
artifact.  It contains the same `debug_report.json`, `twotone-debug.log`,
`work/`, and `melt_output/` files as a local run.

Example comparison after downloading the artifact:

```bash
python tests/melt/debug_audio_alignment.py compare \
  /tmp/twotone-audio-debug-local/debug_report.json \
  /tmp/downloaded-audio-alignment-debug/debug_report.json
```
