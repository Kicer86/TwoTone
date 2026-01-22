
import argparse
import logging
from typing import Protocol, runtime_checkable


@runtime_checkable
class Plan(Protocol):
    def is_empty(self) -> bool:
        ...

    def render(self, logger: logging.Logger) -> None:
        ...


class EmptyPlan:
    def is_empty(self) -> bool:
        return True

    def render(self, logger: logging.Logger) -> None:
        return None

class Tool:
    def setup_parser(self, parser: argparse.ArgumentParser):
        pass

    def analyze(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str) -> Plan:
        pass

    def perform(self, args: argparse.Namespace, logger: logging.Logger, working_dir: str, plan: Plan) -> None:
        pass
