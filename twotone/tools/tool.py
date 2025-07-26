
import argparse
import logging

class Tool:
    def setup_parser(self, parser: argparse.ArgumentParser):
        pass

    def run(self, args: argparse.Namespace, no_dry_run: bool,
            logger: logging.Logger, working_dir: str):
        pass
