
import argparse
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

from twotone.tools.utils import files_utils, generic_utils, media_analysis, requirements_utils


@runtime_checkable
class Plan(Protocol):
    def is_empty(self) -> bool:
        ...

    def render(self, logger: logging.Logger) -> None:
        ...

    def input_files(self) -> Iterable[str]:
        """Return the source files this analyzed plan will read."""
        ...


class EmptyPlan:
    def is_empty(self) -> bool:
        return True

    def render(self, logger: logging.Logger) -> None:
        return None

    def input_files(self) -> Iterable[str]:
        return ()


@dataclass(frozen=True)
class ToolRuntimeContext:
    """Run-scoped services shared across a tool's analysis and execution."""

    workspace: files_utils.Workspace
    interruption: generic_utils.InterruptibleProcess
    media_analysis: media_analysis.MediaAnalysisSession


class Tool(ABC):
    @abstractmethod
    def setup_parser(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def required_tools(self) -> set[str]:
        return requirements_utils.collect_required_tools(self.analyze, self.perform)

    def media_analysis_requests(
        self,
        plan: Plan,
    ) -> Iterable[media_analysis.MediaAnalysisRequest]:
        """Return media data that must be ready before performing the plan."""
        return ()

    @abstractmethod
    def analyze(self, args: argparse.Namespace, logger: logging.Logger, context: ToolRuntimeContext) -> Plan:
        raise NotImplementedError

    @abstractmethod
    def perform(self, args: argparse.Namespace, logger: logging.Logger, context: ToolRuntimeContext, plan: Plan) -> None:
        raise NotImplementedError
