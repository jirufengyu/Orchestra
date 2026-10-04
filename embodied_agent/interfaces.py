from __future__ import annotations

from typing import Any, Protocol, Sequence

from .contracts import (
    AgentContext,
    AgentDecision,
    CapabilityOutput,
    CapabilityRequest,
    CapabilitySpec,
    CapabilityFeedback,
    ContextToolRequest,
    ContextToolResult,
    ContextToolSpec,
    EvaluatorSpec,
    Goal,
    Observation,
    TaskTransition,
)


class Embodiment(Protocol):
    def reset(self) -> Observation: ...

    def observe(self) -> Observation: ...

    def act(self, action: Any) -> Observation: ...

    def stop(self) -> None: ...


class TaskProtocol(Protocol):
    @property
    def done(self) -> bool: ...

    def start(self, observation: Observation) -> None: ...

    def current_goal(self) -> Goal: ...

    def update(self, observation: Observation, feedback: Any) -> TaskTransition: ...


class TaskPlanner(Protocol):
    def plan(self, instruction: str, *, mode: str = "auto") -> Sequence[Goal]: ...


class Capability(Protocol):
    @property
    def spec(self) -> CapabilitySpec: ...

    def invoke(self, request: CapabilityRequest, context: AgentContext) -> CapabilityOutput: ...

    def observe(self, context: AgentContext) -> None: ...

    def reset(self) -> None: ...

    def cancel(self, reason: str) -> None: ...


class ContextTool(Protocol):
    @property
    def spec(self) -> ContextToolSpec: ...

    def invoke(self, request: ContextToolRequest) -> ContextToolResult: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class Evaluator(Protocol):
    @property
    def spec(self) -> EvaluatorSpec: ...

    def evaluate(self, context: AgentContext) -> CapabilityFeedback: ...

    def observe(self, context: AgentContext) -> None: ...

    def reset(self) -> None: ...

    def cancel(self, reason: str) -> None: ...


class Executive(Protocol):
    def decide(
        self,
        context: AgentContext,
        capabilities: Sequence[CapabilitySpec],
    ) -> AgentDecision: ...
