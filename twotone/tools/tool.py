
import argparse
import logging

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from twotone.tools.utils import files_utils, requirements_utils


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


class Tool(ABC):
    @abstractmethod
    def setup_parser(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def required_tools(self) -> set[str]:
        return requirements_utils.collect_required_tools(self.analyze, self.perform)

    @abstractmethod
    def analyze(self, args: argparse.Namespace, logger: logging.Logger, working_dir: files_utils.Workspace) -> Plan:
        raise NotImplementedError

    @abstractmethod
    def perform(self, args: argparse.Namespace, logger: logging.Logger, working_dir: files_utils.Workspace, plan: Plan) -> None:
        raise NotImplementedError
