
import argparse
import logging
import os
import sys
import shutil

from overrides import override

from .tools import          \
    concatenate,            \
    melt,                   \
    merge,                  \
    subtitles_fixer,        \
    transcode,              \
    utilities

from .tools.utils import generic_utils

TOOLS = {
    "concatenate": (concatenate.ConcatenateTool(), "Concatenate multifile movies into one file"),
    "melt": (melt.MeltTool(), "[EXPERIMENTAL] Find same video files and combine them into one containg best of all copies."),
    "merge": (merge.MergeTool(), "Merge video files with corresponding subtitles into one MKV file"),
    "subtitles_fix": (subtitles_fixer.FixerTool(), "Fixes some specific issues with subtitles. Do not use until you are sure it will help for your problems."),
    "transcode": (transcode.TranscodeTool(), "Transcode videos from provided directory preserving quality."),
    "utilities": (utilities.UtilitiesTool(), "Various smaller tools"),
}


class CustomParserFormatter(argparse.HelpFormatter):
    @override
    def _split_lines(self, text: str, width: int) -> list[str]:
        return text.splitlines()

    @override
    def _get_help_string(self, action: argparse.Action) -> str:
        help_str = action.help or ""
        if '%(default)' not in help_str:
            if action.default is not argparse.SUPPRESS and action.default is not None:
                help_str += f' (default: {action.default})'
        return help_str

def execute(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog = 'twotone',
        description='Videos manipulation toolkit. '
                    'By default all tools do nothing but showing what would be done.\n'
                    'Use --no-dry-run option to perform actual operation.\n'
                    'Please mind that ALL source files will be modified in place, so consider making a backup.\n'
                    'It is safe to stop any tool with ctrl+c - it will quit '
                    'gracefully in a while.',
        formatter_class=CustomParserFormatter
    )

    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--quiet",
                        action="store_true",
                        help="Disable all output")
    parser.add_argument("--no-dry-run", "-r",
                        action='store_true',
                        default=False,
                        help='Perform actual operation.')
    parser.add_argument(
        "--working-dir",
        "-w",
        default=generic_utils.get_twotone_working_dir(),
        help="Directory for temporary files",
    )
    subparsers = parser.add_subparsers(dest="tool", help="Available tools:")

    for tool_name, (tool, desc) in TOOLS.items():
        tool_parser = subparsers.add_parser(
            tool_name,
            help=desc,
            formatter_class=CustomParserFormatter
        )
        tool.setup_parser(tool_parser)

    args = parser.parse_args(args = argv)

    if args.tool is None:
        parser.print_help()
        sys.exit(1)

    logger = logging.getLogger("TwoTone")

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.quiet:
        logger.setLevel(logging.CRITICAL)

    if args.tool in TOOLS:
        tool, _ = TOOLS[args.tool]
        base_wd = args.working_dir
        pid_wd = os.path.join(base_wd, str(os.getpid()))
        tool_wd = os.path.join(pid_wd, args.tool)
        os.makedirs(tool_wd, exist_ok=True)
        try:
            tool.run(
                args,
                no_dry_run=args.no_dry_run,
                logger=logger.getChild(args.tool),
                working_dir=tool_wd,
            )
        finally:
            shutil.rmtree(pid_wd, ignore_errors=True)
    else:
        logging.error(f"Error: Unknown tool {args.tool}")
        sys.exit(1)


class CustomLoggerFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_txt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_txt + reset,
        logging.INFO: grey + format_txt + reset,
        logging.WARNING: yellow + format_txt + reset,
        logging.ERROR: red + format_txt + reset,
        logging.CRITICAL: bold_red + format_txt + reset
    }

    @override
    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def main() -> None:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomLoggerFormatter())

    logging.basicConfig(level=logging.INFO, handlers=[console_handler])

    try:
        execute(sys.argv[1:])
    except RuntimeError as e:
        logging.error(f"Error occurred: {e}. Terminating")
    except ValueError as e:
        print(f"error: {e}")

if __name__ == '__main__':
    main()
