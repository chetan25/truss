from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from truss.errors import ToolOutOfScope


@dataclass
class McpCall:
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class McpManifest:
    allowed_tools: list[str]

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


class McpInterceptor:
    def __init__(self, manifest: McpManifest) -> None:
        self._manifest = manifest

    def check(self, call: McpCall) -> None:
        if not self._manifest.is_allowed(call.tool_name):
            raise ToolOutOfScope(f"{call.tool_name} denied by manifest")

    def wrap(self, call: McpCall, fn: Callable[[McpCall], Any]) -> Any:
        self.check(call)
        return fn(call)
