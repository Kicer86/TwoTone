
import argparse
import logging
from typing import Any

class Tool:
    def setup_parser(self, parser: argparse.ArgumentParser):
        pass

    def analyze(self, args: argparse.Namespace,
                logger: logging.Logger, working_dir: str) -> Any:
        pass

    def perform(self, args: argparse.Namespace, analysis: Any, no_dry_run: bool,
                logger: logging.Logger, working_dir: str):
        pass
