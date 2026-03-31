## Tools for Batch Video File Manipulations

### Overview

**TwoTone** is a versatile tool with various subtools for batch video file manipulations.

### Installation

```bash
pip install -e .
```

This installs TwoTone in editable mode: the `twotone` command appears on PATH, and any code changes take effect immediately without reinstalling.

### Running TwoTone

```bash
twotone <global options> <tool-name> <tool-specific options>
```

### Shell Autocompletion (bash)

```bash
twotone --install-completion
```

Open a new terminal. Tab-completion now works:

```
twotone <TAB>           → concatenate, language_fix, melt, merge, ...
twotone --<TAB>         → --verbose, --quiet, --no-dry-run, ...
twotone merge --<TAB>   → merge-specific arguments
```

To remove: `twotone --uninstall-completion`

### Getting Help

To see a list of global options, available tools, and their descriptions:

```bash
twotone --help
```

To get help for a specific tool:

```bash
twotone <tool-name> --help
```

### Important Notes

Dry Run Mode: all tools run in a dry run mode by default (no files are being modified). Use the -r or --no-dry-run global option to perform actual operation. In live run mode most tools remove their input files after successful processing.<br/>
It is safe to stop execution with ctrl+c. All tools handle proper signal and will stop as soon as possible.<br/>
Working Directory: use -w or --working-dir to specify where temporary and working files are created. By default the directory is taken from the operating system's user data location for the application.<br/>

Data Safety: Always back up your data before using any tool, as source files may be deleted during processing.

### Available Tools
#### Merge Video Files with Subtitles into MKV Files

The merge tool searches for movie and subtitle files and merges them into a single MKV file.

By default, subtitles are added without a language label. You can specify a language with the --language option.

For a full list of options:

```bash
twotone merge --help
```

#### Fix Missing Track Languages

The language_fix tool scans MKV files and fills missing language metadata for subtitle tracks by extracting them and running language detection. It can also infer audio track language from the track name (use --audio to enable; heuristic may be inaccurate).

```bash
twotone language_fix --help
```

#### Automatic Video transcoding

The transcode tool transcodes videos using the x265 codec.

It takes a video directory as an input and determines the optimal CRF for each video by comparing the original and encoded versions.
The script aims to achieve a quality level where SSIM ≈ 0.98 (by default).

For a full list of options:

```bash
twotone transcode --help
```

#### Movies concatenation

The concatenate tool looks for movie files which seem to be split into a few files (like CD1, CD2 etc) and glues them into one file.

```bash
twotone concatenate --help
```

#### Combine Duplicate Videos (melt)

The melt tool scans for duplicate video files and creates a single output using the best quality segments from each copy. Duplicates can be provided manually or taken from a Jellyfin server. Input files are kept intact by default.

```bash
twotone melt --help
```

#### Miscellaneous utilities

The utilities tool groups smaller helpers. Currently it provides the *scenes* subtool for extracting frames from a video and saving them into per-scene folders.

```bash
twotone utilities scenes --help
```
