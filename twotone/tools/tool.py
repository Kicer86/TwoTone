
import argparse
import logging

class Tool:
    def setup_parser(self, parser: argparse.ArgumentParser):
        pass

    def run(self, args, no_dry_run: bool, logger: logging.Logger):
        pass
