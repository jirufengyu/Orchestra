from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .contracts import (
    AgentContext,
    CapabilityFeedback,
    CapabilityOutput,
    CapabilityRequest,
    CapabilitySpec,
)
from .interfaces import Capability


@dataclass
class CapabilityRuntime:
    _capabilities: dict[str, Capability] = field(default_factory=dict)
    _active_id: str | None = None

    def register(self, capability: Capability) -> None:
        capability_id = capability.spec.capability_id
        if capability_id in self._capabilities:
            raise ValueError(f"capability already registered: {capability_id}")
        self._capabilities[capability_id] = capability

    def specs(self) -> tuple[CapabilitySpec, ...]:
        return tuple(item.spec for item in self._capabilities.values())

    def invoke(self, request: CapabilityRequest, context: AgentContext) -> CapabilityOutput:
        try:
            capability = self._capabilities[request.capability_id]
        except KeyError as exc:
            raise KeyError(f"unknown capability: {request.capability_id}") from exc
        if self._active_id is not None and self._active_id != request.capability_id:
            self._capabilities[self._active_id].cancel("switched capability")
        self._active_id = request.capability_id
        return capability.invoke(request, context)

    def observe(self, context: AgentContext) -> None:
        if self._active_id is not None:
            self._capabilities[self._active_id].observe(context)

    def reset(self) -> None:
        for capability in self._capabilities.values():
            capability.reset()
        self._active_id = None

    def cancel(self, reason: str) -> None:
        if self._active_id is not None:
            self._capabilities[self._active_id].cancel(reason)
        self._active_id = None


@dataclass
class CallableCapability:
    spec: CapabilitySpec
    function: Callable[[CapabilityRequest, AgentContext], CapabilityOutput]
    observe_function: Callable[[AgentContext], None] | None = None
    reset_function: Callable[[], None] | None = None
    cancel_function: Callable[[str], None] | None = None

    def invoke(self, request: CapabilityRequest, context: AgentContext) -> CapabilityOutput:
        return self.function(request, context)

    def observe(self, context: AgentContext) -> None:
        if self.observe_function is not None:
            self.observe_function(context)

    def reset(self) -> None:
        if self.reset_function is not None:
            self.reset_function()

    def cancel(self, reason: str) -> None:
        if self.cancel_function is not None:
            self.cancel_function(reason)


@dataclass
class RemoteCapability:
    """Transport-neutral adapter for an external capability service."""

    spec: CapabilitySpec
    request_function: Callable[[dict[str, Any]], dict[str, Any]]
    observe_function: Callable[[dict[str, Any]], None] | None = None
    reset_function: Callable[[], None] | None = None
    cancel_function: Callable[[str], None] | None = None

    def invoke(self, request: CapabilityRequest, context: AgentContext) -> CapabilityOutput:
        response = self.request_function(
            {
                "capability_id": request.capability_id,
                "instruction": request.goal.instruction,
                "goal_id": request.goal.goal_id,
                "entity_bindings": dict(request.goal.entity_bindings),
                "parameters": dict(request.parameters),
                "prompt": request.prompt,
                "observation": context.observation,
                "context": context.fragments,
                "memory": context.memory,
            }
        )
        feedback_data = response.get("feedback", {})
        feedback = (
            feedback_data
            if isinstance(feedback_data, CapabilityFeedback)
            else CapabilityFeedback(**feedback_data)
        )
        return CapabilityOutput(actions=response.get("actions", ()), feedback=feedback)

    def observe(self, context: AgentContext) -> None:
        if self.observe_function is not None:
            self.observe_function(
                {
                    "observation": context.observation,
                    "context": context.fragments,
                    "memory": context.memory,
                }
            )

    def reset(self) -> None:
        if self.reset_function is not None:
            self.reset_function()

    def cancel(self, reason: str) -> None:
        if self.cancel_function is not None:
            self.cancel_function(reason)
