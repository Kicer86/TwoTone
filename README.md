## Tools for Batch Video File Manipulations

### Overview

**TwoTone** is a versatile tool with various subtools for batch video file manipulations.

### Running TwoTone

To run TwoTone, use the following command:

```bash
python -m twotone <global options> <tool-name> <tool-specific options>
```

### Getting Help

To see a list of global options, available tools, and their descriptions:

```bash
python -m twotone --help
```

To get help for a specific tool:

```bash
python -m twotone <tool-name> --help
```

### Important Notes

Dry Run Mode: all tools run in a dry run mode by default (no files are being modified). Use the -r or --no-dry-run global option to perform actual operation.<br/>
It is safe to stop execution with ctrl+c. All tools handle proper signal and will stop as soon as possible.<br/>

Data Safety: Always back up your data before using any tool, as source files may be deleted during processing.

### Available Tools
#### Merge Video Files with Subtitles into MKV Files

The merge tool searches for movie and subtitle files and merges them into a single MKV file.

By default, subtitles are added without a language label. You can specify a language with the --language option.

For a full list of options:

```bash
python -m twotone merge --help
```

#### Automatic Video transcoding

The transcode tool transcodes videos using the x265 codec.

It takes a video directory as an input and determines the optimal CRF for each video by comparing the original and encoded versions.
The script aims to achieve a quality level where SSIM ≈ 0.98 (by default).

For a full list of options:

```bash
python -m twotone transcode --help
```
