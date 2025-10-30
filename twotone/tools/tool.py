
import argparse
import logging

class Tool:
    def setup_parser(self, parser: argparse.ArgumentParser):
        pass

    def analyze(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str):
        pass

    def perform(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str):
        pass
