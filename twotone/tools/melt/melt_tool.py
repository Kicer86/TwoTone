import argparse
import logging
import os
import re

from overrides import override

from ..tool import EmptyPlan, Plan, Tool
from ..utils import generic_utils
from .duplicates_source import DuplicatesSource
from .jellyfin import JellyfinSource
from .static_source import StaticSource
from .melt_analyzer import MeltAnalyzer
from .melt_common import DEFAULT_TOLERANCE_MS, _ensure_working_dir, _split_path_fix
from .melt_performer import MeltPerformer
from .melt_plan import MeltPlan


class RequireJellyfinServer(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, "jellyfin_server", None) is None:
            parser.error(f"{option_string} requires --jellyfin-server to be specified")
        setattr(namespace, self.dest, values)


class MeltTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.parser: argparse.ArgumentParser | None = None

    @override
    def setup_parser(self, parser: argparse.ArgumentParser):
        self.parser = parser

        jellyfin_group = parser.add_argument_group("Jellyfin source")
        jellyfin_group.add_argument('--jellyfin-server',
                                    help='URL to the Jellyfin server which will be used as a source of video files duplicates')
        jellyfin_group.add_argument('--jellyfin-token',
                                    action=RequireJellyfinServer,
                                    help='Access token (http://server:8096/web/#/dashboard/keys)')
        jellyfin_group.add_argument('--jellyfin-path-fix',
                                    action=RequireJellyfinServer,
                                    help='Specify a replacement pattern for file paths to ensure "melt" can access Jellyfin video files.\n\n'
                                         '"Melt" requires direct access to video files. If Jellyfin is not running on the same machine as "melt",\n'
                                         'you need to set up network access to Jellyfin’s video storage and specify how paths should be resolved.\n\n'
                                         'For example, suppose Jellyfin runs on a Linux machine where the video library is stored at "/srv/videos" (a shared directory).\n'
                                         'If "melt" is running on another Linux machine that accesses this directory remotely at "/mnt/shared_videos,"\n'
                                         'you need to map "/srv/videos" (Jellyfin’s path) to "/mnt/shared_videos" (the path accessible on the machine running "melt").\n\n'
                                         'In this case, use: --jellyfin-path-fix \\"/srv/videos\\",\\"/mnt/shared_videos\\" to define the replacement pattern.' \
                                         'Please mind that \\ to preserve \" are crucial')

        manual_group = parser.add_argument_group("Manual input source")
        manual_group.add_argument('-t', '--title',
                                  help='Video (movie or series when directory is provided as an input) title.')
        manual_group.add_argument('-i', '--input', dest='input_files', action='append',
                                  help='Add an input video file or directory with video files (can be specified multiple times).\n'
                                       'path can be followed with a comma and some additional parameters:\n'
                                       'audio_lang:XXX       - information about audio language (like eng, de or pl).\n'
                                       'audio_prod_lang:XXX - original/production audio language.\n\n'
                                       'Example of usage:\n'
                                       '--input some/path/file.mp4,audio_lang:jp --input some/path/file.mp4,audio_lang:eng\n\n'
                                       'If files are provided with this option, all of them are treated as duplicates of given title.\n'
                                       'If directoriess are provided, a \'series\' mode is being used and melt will list and sort files from each dir, and corresponding '
                                       'files from provided directories will be grouped as duplicates.\n'
                                       'If only one directory is provided as input, all files found inside will be treated as duplicates of the title.\n'
                                       'No other scenarios and combinations of inputs are supported.')

        # global options
        parser.add_argument('-o', '--output-dir',
                            help="Directory for output files",
                            required = True)

        parser.add_argument('--allow-length-mismatch', action='store_true',
                            help='[EXPERIMENTAL] Continue processing even if input video lengths differ.\n'
                                 'This may require additional processing that can consume significant time and disk space.')

        parser.add_argument('--allow-language-guessing', action='store_true',
                            help='If audio language is not provided in file metadata, try find language codes (like EN or DE) in file names')

    @override
    def analyze(self, args, logger: logging.Logger, working_dir: str) -> Plan:
        interruption = generic_utils.InterruptibleProcess()
        data_source: DuplicatesSource | None = None
        parser = self.parser
        if parser is None:
            raise RuntimeError("Parser not initialized. Call setup_parser before analyze.")

        # Build data source based on arguments
        if args.jellyfin_server:
            path_fix_list = _split_path_fix(args.jellyfin_path_fix) if args.jellyfin_path_fix else None

            if path_fix_list and len(path_fix_list) != 2:
                parser.error(f"Invalid content for --jellyfin-path-fix argument. Got: {path_fix_list}")

            path_fix: tuple[str, str] | None = None
            if path_fix_list:
                path_fix = (path_fix_list[0], path_fix_list[1])

            data_source = JellyfinSource(interruption = interruption,
                                         url = args.jellyfin_server,
                                         token = args.jellyfin_token,
                                         path_fix = path_fix,
                                         logger = logger.getChild("JellyfinSource"))
        elif args.input_files:
            title = args.title
            input_entries = args.input_files

            if not title:
                parser.error("Missing required option: --title")

            src = StaticSource(interruption=interruption)

            for input in input_entries:
                # split by ',' but respect ""
                input_split = re.findall(r'(?:[^,"]|"(?:\\"|[^"])*")+', input)
                path = input_split[0]

                if not os.path.exists(path):
                    raise ValueError(f"Path {path} does not exist")

                audio_lang = ""
                audio_prod_lang = ""

                if len(input_split) > 1:
                    for extra_arg in input_split[1:]:
                        if extra_arg[:11] == "audio_lang:":
                            audio_lang = extra_arg[11:]
                        if extra_arg[:15] == "audio_prod_lang:":
                            audio_prod_lang = extra_arg[15:]

                src.add_entry(title, path)

                if audio_lang:
                    src.add_metadata(path, "audio_lang", audio_lang)
                if audio_prod_lang:
                    src.add_metadata(path, "audio_prod_lang", audio_prod_lang)

            data_source = src

        if not data_source:
            logger.info("No input source specified. Nothing to analyze.")
            return EmptyPlan()

        logger.debug("Collecting duplicates for analysis")
        duplicates_raw = data_source.collect_duplicates()
        duplicates = {title: list(files) for title, files in duplicates_raw.items()}

        analysis_wd = _ensure_working_dir(working_dir)
        analyzer = MeltAnalyzer(
            logger,
            data_source,
            analysis_wd,
            args.allow_language_guessing,
            args.allow_length_mismatch,
            DEFAULT_TOLERANCE_MS,
        )
        all_entries = [path for entries in duplicates.values() for path in entries]
        if path_fix:
            analyzer.base_path = path_fix[1]
        elif all_entries:
            try:
                analyzer.base_path = os.path.commonpath(all_entries)
            except ValueError:
                analyzer.base_path = None
        analysis = analyzer.analyze_duplicates(duplicates)
        return MeltPlan(
            items=analysis,
            output_dir=args.output_dir,
        )

    @override
    def perform(self, args, logger: logging.Logger, working_dir: str, plan: Plan) -> None:
        _ = args
        if plan.is_empty():
            logger.info("No analysis results, nothing to melt.")
            return

        if not isinstance(plan, MeltPlan):
            logger.info("Unsupported plan type, nothing to melt.")
            return

        interruption = generic_utils.InterruptibleProcess()
        performer = MeltPerformer(
            logger,
            interruption,
            working_dir,
            plan.output_dir,
            DEFAULT_TOLERANCE_MS,
        )
        performer.process_duplicates(plan.items)
