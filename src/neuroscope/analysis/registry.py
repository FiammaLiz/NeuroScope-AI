from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True)
class AnalysisTool:
    name: str
    description: str
    required_metadata: tuple[str, ...]
    function: Callable[..., Any]


class AnalysisRegistry:
    def __init__(self):
        self._tools: dict[str, AnalysisTool] = {}

    def register(self, tool: AnalysisTool):
        if tool.name in self._tools:
            raise ValueError(f"Analysis tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AnalysisTool:
        return self._tools[name]

    def available(self) -> list[AnalysisTool]:
        return list(self._tools.values())
