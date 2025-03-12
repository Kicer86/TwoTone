
import argparse
import logging
import sys

from .tools import          \
    concatenate,            \
    melt,                   \
    merge,                  \
    subtitles_fixer,        \
    transcode,              \
    utilities

TOOLS = {
    "concatenate": (concatenate.setup_parser, concatenate.run, "Concatenate multifile movies into one file"),
    "melt": (melt.setup_parser, melt.run, "[Not ready yet] Find same video files and combine them into one containg best of all copies."),
    "merge": (merge.setup_parser, merge.run, "Merge video files with corresponding subtitles into one MKV file"),
    "subtitles_fix": (subtitles_fixer.setup_parser, subtitles_fixer.run, "Fixes some specific issues with subtitles. Do not use until you are sure it will help for your problems."),
    "transcode": (transcode.setup_parser, transcode.run, "Transcode videos from provided directory preserving quality."),
    "utilities": (utilities.setup_parser, utilities.run, "Various smaller tools"),
}


class CustomFormatter(argparse.HelpFormatter):
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
        formatter_class=CustomFormatter
    )

    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--no-dry-run", "-r",
                        action='store_true',
                        default=False,
                        help='Perform actual operation.')
    subparsers = parser.add_subparsers(dest="tool", help="Available tools:")

    for tool_name, (setup_parser, _, desc) in TOOLS.items():
        tool_parser = subparsers.add_parser(
            tool_name,
            help=desc,
            formatter_class=CustomFormatter
        )
        setup_parser(tool_parser)

    args = parser.parse_args(args = argv)

    if args.tool is None:
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.tool in TOOLS:
        tool = TOOLS[args.tool][1]
        tool(args)

    else:
        logging.error(f"Error: Unknown tool {args.tool}")
        sys.exit(1)


def main():
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    try:
        execute(sys.argv[1:])
    except RuntimeError as e:
        logging.error(f"Error occurred: {e}. Terminating")


if __name__ == '__main__':
    main()
