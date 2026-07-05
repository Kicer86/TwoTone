
import argparse
import logging
import os
import sys
import shutil

from importlib import metadata

import argcomplete
from overrides import override
from tqdm.contrib.logging import logging_redirect_tqdm

from .tools import          \
    concatenate,            \
    language_fixer,         \
    melt,                   \
    merge,                  \
    subtitles_fixer,        \
    transcode,              \
    utilities

from .tools.utils import files_utils, generic_utils, process_utils

TOOLS = {
    "concatenate": (concatenate.ConcatenateTool(), "Concatenate multifile movies into one file"),
    "language_fix": (language_fixer.LanguageFixerTool(), "Detect missing track languages and update MKV metadata."),
    "melt": (melt.MeltTool(), "Find same video files and combine them into one containg best of all copies."),
    "merge": (merge.MergeTool(), "Merge video files with corresponding subtitles into one MKV file"),
    "subtitles_fix": (subtitles_fixer.FixerTool(), "Fixes some specific issues with subtitles. Do not use until you are sure it will help for your problems."),
    "transcode": (transcode.TranscodeTool(), "Transcode videos from provided directory preserving quality."),
    "utilities": (utilities.UtilitiesTool(), "Various smaller tools"),
}


def _runtime_version_report() -> str:
    try:
        version = metadata.version("twotone")
    except metadata.PackageNotFoundError:
        version = "unknown"

    source_dir = os.path.dirname(os.path.abspath(__file__))
    launcher = os.path.abspath(os.path.expanduser(sys.argv[0]))
    lines = [
        f"TwoTone {version}",
        f"Launcher: {launcher}",
        f"Source: {source_dir}",
        f"Python: {sys.executable}",
    ]

    if shutil.which("git"):
        result = process_utils.start_process(
            "git",
            ["describe", "--always", "--dirty", "--long"],
            cwd=source_dir,
        )
        revision = result.stdout.strip()
        if result.returncode == 0 and revision:
            lines.append(f"Git: {revision}")

    return "\n".join(lines)

def _get_completion_dir() -> str:
    data_dir = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return os.path.join(data_dir, "bash-completion", "completions")


def _install_completion() -> None:
    completion_dir = _get_completion_dir()
    os.makedirs(completion_dir, exist_ok=True)
    dest = os.path.join(completion_dir, "twotone")

    script = argcomplete.shellcode(["twotone"], shell="bash")

    with open(dest, "w") as f:
        f.write(script)
    print(f"Completion installed: {dest}")
    print("Open a new terminal for it to take effect.")


def _uninstall_completion() -> None:
    dest = os.path.join(_get_completion_dir(), "twotone")
    if os.path.exists(dest):
        os.remove(dest)
        print(f"Completion removed: {dest}")
    else:
        print("Completion was not installed.")


def _plan_item_count(plan: object) -> int | None:
    items = getattr(plan, "items", None)
    if items is None:
        return None
    try:
        return len(items)
    except TypeError:
        return None


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
    parser.add_argument("--interactive", "-i",
                        action="store_true",
                        help="Show analysis report and ask whether to perform.")
    parser.add_argument(
        "--working-dir",
        "-w",
        default=None,
        help="Directory for temporary files.\n"
             f"When omitted, twotone works in a per-run subdirectory of {generic_utils.get_twotone_working_dir()}\n"
             "and removes leftovers of crashed runs from there.\n"
             "When provided, the directory is used exactly as given: no per-run subdirectory\n"
             "is created and no other run's data is scanned or removed.",
    )
    parser.add_argument(
        "--keep-wd",
        action="store_true",
        help="Do not delete anything from the working directory.\n"
             "Keeps all intermediate files (frames, audio segments, debug dumps) for inspection.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and runtime source details, then exit.",
    )
    parser.add_argument(
        "--install-completion",
        action="store_true",
        help="Install bash tab-completion and exit.",
    )
    parser.add_argument(
        "--uninstall-completion",
        action="store_true",
        help="Remove bash tab-completion and exit.",
    )
    subparsers = parser.add_subparsers(dest="tool", help="Available tools:")

    for tool_name, (tool, desc) in TOOLS.items():
        tool_parser = subparsers.add_parser(
            tool_name,
            help=desc,
            formatter_class=CustomParserFormatter
        )
        tool.setup_parser(tool_parser)

    argcomplete.autocomplete(parser)
    args = parser.parse_args(args = argv)

    if args.version:
        print(_runtime_version_report())
        return
    if args.install_completion:
        _install_completion()
        return
    if args.uninstall_completion:
        _uninstall_completion()
        return

    if args.tool is None:
        parser.print_help()
        sys.exit(1)
    if args.no_dry_run and args.interactive:
        parser.error("Use either --no-dry-run or --interactive, not both.")

    logger = logging.getLogger("TwoTone")

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.quiet:
        logger.setLevel(logging.CRITICAL)

    if args.tool in TOOLS:
        tool, _ = TOOLS[args.tool]

        with logging_redirect_tqdm(), \
             files_utils.open_workspace(
                 args.working_dir,
                 generic_utils.get_twotone_working_dir(),
                 keep=args.keep_wd,
                 logger=logger,
             ) as workspace:
            tool_logger = logger.getChild(args.tool)
            required_tools = sorted(tool.required_tools())
            if required_tools:
                process_utils.ensure_tools_exist(required_tools, tool_logger)
            plan = tool.analyze(
                args,
                logger=tool_logger,
                workspace=workspace,
            )

            if args.no_dry_run:
                plan.render(tool_logger)
                if plan.is_empty():
                    tool_logger.info("Nothing to do.")
                else:
                    tool.perform(
                        args,
                        logger=tool_logger,
                        workspace=workspace,
                        plan=plan,
                    )
            elif args.interactive:
                plan.render(tool_logger)
                if plan.is_empty():
                    tool_logger.info("Analysis complete: nothing to do.")
                    tool_logger.info("Skipping perform.")
                else:
                    plan_count = _plan_item_count(plan)
                    if plan_count is None:
                        tool_logger.info("Analysis complete: ready to perform.")
                    else:
                        tool_logger.info("Analysis complete: %d item(s) ready.", plan_count)
                    try:
                        answer = input("Proceed with perform? [y/N]: ").strip().lower()
                    except EOFError:
                        answer = ""
                    if answer in {"y", "yes"}:
                        tool_logger.info("User confirmed. Starting perform.")
                        tool.perform(
                            args,
                            logger=tool_logger,
                            workspace=workspace,
                            plan=plan,
                        )
                    else:
                        tool_logger.info("User aborted. Skipping perform.")
            else:
                if plan.is_empty():
                    plan.render(tool_logger)
                    tool_logger.info("Analysis complete: nothing to do.")
                    if args.no_dry_run:
                        tool_logger.info("Skipping perform.")
                    else:
                        tool_logger.info("Dry run mode: analyze completed, skipping perform.")
                elif args.no_dry_run:
                    plan_count = _plan_item_count(plan)
                    if plan_count is None:
                        tool_logger.info("Analysis complete: starting perform.")
                    else:
                        tool_logger.info(
                            "Analysis complete: %d item(s) to process. Starting perform.",
                            plan_count,
                        )
                    tool.perform(
                        args,
                        logger=tool_logger,
                        workspace=workspace,
                        plan=plan,
                    )
                else:
                    plan.render(tool_logger)
                    tool_logger.info("Dry run mode: analyze completed, skipping perform.")
    else:
        logger.error(f"Error: Unknown tool {args.tool}")
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
    logger = logging.getLogger("TwoTone")

    try:
        execute(sys.argv[1:])
    except RuntimeError as e:
        logger.error(f"Error occurred: {e}. Terminating")
    except ValueError as e:
        print(f"error: {e}")

if __name__ == '__main__':
    main()
