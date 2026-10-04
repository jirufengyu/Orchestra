from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Observation:
    images: Mapping[str, Any]
    robot_state: Any
    timestamp: float = field(default_factory=monotonic)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Goal:
    goal_id: str
    instruction: str
    entity_bindings: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextFragment:
    name: str
    data: Any
    confidence: float | None = None


@dataclass(frozen=True)
class ContextToolSpec:
    tool_id: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextToolRequest:
    tool_id: str
    inputs: Mapping[str, Any]
    episode_id: str | None = None
    step_id: int | None = None


@dataclass(frozen=True)
class ContextToolResult:
    data: Any
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentContext:
    observation: Observation
    goal: Goal
    memory: Mapping[str, Any]
    fragments: Mapping[str, ContextFragment]


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityRequest:
    capability_id: str
    goal: Goal
    parameters: Mapping[str, Any] = field(default_factory=dict)
    prompt: str | None = None


class CapabilityStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CapabilityFeedback:
    status: CapabilityStatus = CapabilityStatus.RUNNING
    progress: float | None = None
    reason_code: str | None = None
    message: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorSpec:
    evaluator_id: str
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityOutput:
    actions: Sequence[Any]
    feedback: CapabilityFeedback = field(default_factory=CapabilityFeedback)


@dataclass(frozen=True)
class AgentDecision:
    request: CapabilityRequest
    reasoning: str = ""


@dataclass(frozen=True)
class TaskTransition:
    subtask_changed: bool = False
    done: bool = False
    success: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class AgentRunResult:
    success: bool
    cycles: int
    reason: str | None
    memory: Mapping[str, Any]


@dataclass(frozen=True)
class AgentStepResult:
    context: AgentContext
    output: CapabilityOutput
    transition: TaskTransition = field(default_factory=TaskTransition)
