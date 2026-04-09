import argparse
import logging
import os

from overrides import override

from ..tool import EmptyPlan, Plan, Tool
from ..utils import generic_utils
from .duplicates_source import DuplicatesSource
from .jellyfin import JellyfinSource
from .static_source import StaticSource
from .melt_analyzer import MeltAnalyzer
from .melt_cache import MeltCache
from .melt_common import DEFAULT_TOLERANCE_MS, _ensure_working_dir, _split_path_fix
from .melt_performer import MeltPerformer
from .melt_plan import MeltPlan


class RequireJellyfinServer(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, "jellyfin_server", None) is None:
            parser.error(f"{option_string} requires --jellyfin-server to be specified")
        setattr(namespace, self.dest, values)


class InputAction(argparse.Action):
    """Appends a new input entry dict to namespace.input_entries."""
    def __call__(self, parser, namespace, values, option_string=None):
        entries = getattr(namespace, self.dest, None) or []
        entries.append({'path': values})
        setattr(namespace, self.dest, entries)


class PerInputMetadataAction(argparse.Action):
    """Attaches a metadata value to the most recently added -i/--input entry."""
    def __call__(self, parser, namespace, values, option_string=None):
        entries = getattr(namespace, 'input_entries', None)
        if not entries:
            parser.error(f"{option_string} must be specified after -i/--input")
        entries[-1][self.dest] = values


class PerInputFlagAction(argparse.Action):
    """Attaches a boolean metadata flag to the most recently added -i/--input entry."""
    def __call__(self, parser, namespace, values, option_string=None):
        entries = getattr(namespace, 'input_entries', None)
        if not entries:
            parser.error(f"{option_string} must be specified after -i/--input")
        entries[-1][self.dest] = True


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
        manual_group.add_argument('-i', '--input', dest='input_entries', action=InputAction,
                                  help='Add an input video file or directory with video files (can be specified multiple times).\n'
                                       'Per-input metadata can be specified with --audio-lang, --audio-prod-lang, --subtitle-lang\n'
                                       'flags placed after each -i.\n\n'
                                       'Example of usage:\n'
                                       '-i some/path/file.mp4 --audio-lang jp -i some/path/file2.mp4 --audio-lang eng\n\n'
                                       'If files are provided with this option, all of them are treated as duplicates of given title.\n'
                                       'If directories are provided, a \'series\' mode is being used and melt will list and sort files from each dir, and corresponding '
                                       'files from provided directories will be grouped as duplicates.\n'
                                       'If only one directory is provided as input, all files found inside will be treated as duplicates of the title.\n'
                                       'No other scenarios and combinations of inputs are supported.')
        manual_group.add_argument('--audio-lang', dest='audio_lang', action=PerInputMetadataAction,
                                  default=argparse.SUPPRESS,
                                  help='Audio language for the preceding -i input (e.g., eng, de, pl).\n'
                                       'Can be specified after each -i to set different languages per input.')
        manual_group.add_argument('--audio-prod-lang', dest='audio_prod_lang', action=PerInputMetadataAction,
                                  default=argparse.SUPPRESS,
                                  help='Original/production audio language for the preceding -i input.')
        manual_group.add_argument('--subtitle-lang', dest='subtitle_lang', action=PerInputMetadataAction,
                                  default=argparse.SUPPRESS,
                                  help='Subtitle language for the preceding -i input.\n'
                                       'Can be specified after each -i to set different languages per input.')
        manual_group.add_argument('--force-all-streams', dest='force_all_streams', action=PerInputFlagAction,
                                  nargs=0,
                                  default=argparse.SUPPRESS,
                                  help='Force all audio and subtitle streams from the preceding -i input to be kept,\n'
                                       'even if their language is unknown. Streams from non-forced inputs are\n'
                                       'only used when forced inputs do not already cover the same language.')

        # global options
        parser.add_argument('-o', '--output-dir',
                            help="Directory for output files",
                            required = True)

        parser.add_argument('--allow-length-mismatch', action='store_true',
                            help='[EXPERIMENTAL] Continue processing even if input video lengths differ.\n'
                                 'This may require additional processing that can consume significant time and disk space.')

        parser.add_argument('--fill-audio-gaps', action='store_true',
                            help='When audio comes from a file with different length, fill head/tail gaps\n'
                                 'with audio from the base video file. By default, gaps are filled with silence\n'
                                 'and the audio is shifted/trimmed without re-encoding when possible.')

        parser.add_argument('--tolerance', type=int, default=DEFAULT_TOLERANCE_MS,
                            help='Maximum allowed duration difference (in ms) between input files\n'
                                 'before alignment is triggered. Files within this tolerance are treated\n'
                                 'as having equal length. Default: %(default)d ms.')

        parser.add_argument('--cache-dir',
                            help='Directory for caching expensive per-video operations (scene detection, '
                                 'frame probing, frame extraction). Speeds up repeated runs on the same input files. '
                                 'Cache is invalidated automatically when the input file changes.')

    @override
    def analyze(self, args, logger: logging.Logger, working_dir: str) -> Plan:
        interruption = generic_utils.InterruptibleProcess()
        data_source: DuplicatesSource | None = None
        parser = self.parser
        if parser is None:
            raise RuntimeError("Parser not initialized. Call setup_parser before analyze.")

        # Build data source based on arguments
        path_fix: tuple[str, str] | None = None
        if args.jellyfin_server:
            path_fix_list = _split_path_fix(args.jellyfin_path_fix) if args.jellyfin_path_fix else None

            if path_fix_list and len(path_fix_list) != 2:
                parser.error(f"Invalid content for --jellyfin-path-fix argument. Got: {path_fix_list}")

            if path_fix_list:
                path_fix = (path_fix_list[0], path_fix_list[1])

            data_source = JellyfinSource(interruption = interruption,
                                         url = args.jellyfin_server,
                                         token = args.jellyfin_token,
                                         path_fix = path_fix,
                                         logger = logger.getChild("JellyfinSource"))
        elif args.input_entries:
            title = args.title
            input_entries = args.input_entries

            if not title:
                parser.error("Missing required option: --title")

            src = StaticSource(interruption=interruption)

            for entry in input_entries:
                path = entry['path']

                if not os.path.exists(path):
                    raise ValueError(f"Path {path} does not exist")

                src.add_entry(title, path)

                for key in ('audio_lang', 'audio_prod_lang', 'subtitle_lang', 'force_all_streams'):
                    value = entry.get(key)
                    if value is not None:
                        src.add_metadata(path, key, value)

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
            args.allow_length_mismatch,
            args.tolerance,
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
        if not isinstance(plan, MeltPlan):
            raise TypeError(f"Expected MeltPlan, got {type(plan).__name__}")

        interruption = generic_utils.InterruptibleProcess()
        cache = MeltCache(args.cache_dir, logger.getChild("cache")) if args.cache_dir else None
        performer = MeltPerformer(
            logger,
            interruption,
            working_dir,
            plan.output_dir,
            args.tolerance,
            cache=cache,
            fill_audio_gaps=args.fill_audio_gaps,
        )
        performer.process_duplicates(plan.items)
