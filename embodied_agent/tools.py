from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .contracts import ContextToolRequest, ContextToolResult, ContextToolSpec
from .interfaces import ContextTool


@dataclass
class ContextToolRuntime:
    """Registry for local or remote tools used while building context."""

    _tools: dict[str, ContextTool] = field(default_factory=dict)

    def register(self, tool: ContextTool) -> None:
        tool_id = tool.spec.tool_id
        if tool_id in self._tools:
            raise ValueError(f"context tool already registered: {tool_id}")
        self._tools[tool_id] = tool

    def specs(self) -> tuple[ContextToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def invoke(
        self,
        tool_id: str,
        inputs: Mapping[str, Any],
        *,
        episode_id: str | None = None,
        step_id: int | None = None,
    ) -> ContextToolResult:
        try:
            tool = self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"unknown context tool: {tool_id}") from exc
        return tool.invoke(
            ContextToolRequest(
                tool_id=tool_id,
                inputs=dict(inputs),
                episode_id=episode_id,
                step_id=step_id,
            )
        )

    def reset(self) -> None:
        for tool in self._tools.values():
            tool.reset()

    def close(self) -> None:
        for tool in self._tools.values():
            tool.close()


@dataclass
class CallableContextTool:
    spec: ContextToolSpec
    function: Callable[[ContextToolRequest], ContextToolResult | Any]
    reset_function: Callable[[], None] | None = None
    close_function: Callable[[], None] | None = None

    def invoke(self, request: ContextToolRequest) -> ContextToolResult:
        result = self.function(request)
        return result if isinstance(result, ContextToolResult) else ContextToolResult(result)

    def reset(self) -> None:
        if self.reset_function is not None:
            self.reset_function()

    def close(self) -> None:
        if self.close_function is not None:
            self.close_function()


@dataclass
class RemoteContextTool:
    spec: ContextToolSpec
    request_function: Callable[[dict[str, Any]], Mapping[str, Any]]
    reset_function: Callable[[], None] | None = None
    close_function: Callable[[], None] | None = None

    def invoke(self, request: ContextToolRequest) -> ContextToolResult:
        response = self.request_function(
            {
                "tool_id": request.tool_id,
                "inputs": dict(request.inputs),
                "episode_id": request.episode_id,
                "step_id": request.step_id,
            }
        )
        return ContextToolResult(
            data=response.get("data"),
            confidence=response.get("confidence"),
            metadata=response.get("metadata") or {},
        )

    def reset(self) -> None:
        if self.reset_function is not None:
            self.reset_function()

    def close(self) -> None:
        if self.close_function is not None:
            self.close_function()
