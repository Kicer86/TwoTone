
import argparse
import logging
import sys

from overrides import override

from .tools import          \
    concatenate,            \
    melt,                   \
    merge,                  \
    subtitles_fixer,        \
    transcode

TOOLS = {
    "concatenate": (concatenate.ConcatenateTool(), "Concatenate multifile movies into one file"),
    "melt": (melt.MeltTool(), "[Not ready yet] Find same video files and combine them into one containg best of all copies."),
    "merge": (merge.MergeTool(), "Merge video files with corresponding subtitles into one MKV file"),
    "subtitles_fix": (subtitles_fixer.FixerTool(), "Fixes some specific issues with subtitles. Do not use until you are sure it will help for your problems."),
    "transcode": (transcode.TranscodeTool(), "Transcode videos from provided directory preserving quality."),
}


class CustomParserFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        return text.splitlines()

def execute(argv):
    parser = argparse.ArgumentParser(
        prog = 'twotone',
        description='Videos manipulation toolkit. '
                    'By default all tools do nothing but showing what would be done. '
                    'Use --no-dry-run option to perform actual operation. '
                    'Please mind that ALL source files will be modified, so consider making a backup. '
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
        tool.run(args, logger.getChild(args.tool))
    else:
        logging.error(f"Error: Unknown tool {args.tool}")
        sys.exit(1)


class CustomLoggerFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    @override
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def main():
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomLoggerFormatter())

    logging.basicConfig(level=logging.INFO, handlers=[console_handler])

    try:
        execute(sys.argv[1:])
    except RuntimeError as e:
        logging.error(f"Error occurred: {e}. Terminating")

if __name__ == '__main__':
    main()
