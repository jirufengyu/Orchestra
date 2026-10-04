from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .contracts import AgentContext, ContextFragment, Goal, Observation
from .memory import MemorySystem
from .tools import ContextToolRuntime


class ContextProvider(Protocol):
    @property
    def name(self) -> str: ...

    def provide(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySystem,
    ) -> ContextFragment | None: ...


@dataclass
class ContextSystem:
    providers: list[ContextProvider] = field(default_factory=list)

    def register(self, provider: ContextProvider) -> None:
        if any(item.name == provider.name for item in self.providers):
            raise ValueError(f"context provider already registered: {provider.name}")
        self.providers.append(provider)

    def reset(self) -> None:
        for provider in self.providers:
            reset = getattr(provider, "reset", None)
            if reset is not None:
                reset()

    def close(self) -> None:
        for provider in self.providers:
            close = getattr(provider, "close", None)
            if close is not None:
                close()

    def build(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySystem,
    ) -> AgentContext:
        fragments: dict[str, ContextFragment] = {}
        for provider in self.providers:
            provide_with_context = getattr(provider, "provide_with_context", None)
            if provide_with_context is not None:
                fragment = provide_with_context(observation, goal, memory, fragments)
            else:
                fragment = provider.provide(observation, goal, memory)
            if fragment is not None:
                fragments[fragment.name] = fragment
        memory.record_fragments(fragments)
        return AgentContext(
            observation=observation,
            goal=goal,
            memory=memory.snapshot(),
            fragments=fragments,
        )


@dataclass
class CallableContextProvider:
    name: str
    function: Callable[[Observation, Goal, MemorySystem], ContextFragment | None]

    def provide(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySystem,
    ) -> ContextFragment | None:
        return self.function(observation, goal, memory)


@dataclass
class ToolBackedContextProvider:
    """Generic bridge from one context tool to a semantic context fragment."""

    name: str
    tool_id: str
    tools: ContextToolRuntime
    input_function: Callable[[Observation, Goal, MemorySystem], Mapping[str, Any]]

    def provide(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySystem,
    ) -> ContextFragment:
        result = self.tools.invoke(
            self.tool_id,
            self.input_function(observation, goal, memory),
            episode_id=memory.episode_id,
            step_id=memory.step_id,
        )
        return ContextFragment(
            name=self.name,
            data=result.data,
            confidence=result.confidence,
        )


@dataclass
class VisualGroundingProvider:
    grounding_function: Callable[[Observation, Goal], object]
    name: str = "visual_grounding"

    def provide(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySystem,
    ) -> ContextFragment:
        del memory
        return ContextFragment(
            name=self.name,
            data=self.grounding_function(observation, goal),
        )


@dataclass
class TemporalContextProvider:
    name: str = "temporal_context"

    def provide(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySystem,
    ) -> ContextFragment:
        del observation, goal
        return ContextFragment(
            name=self.name,
            data={
                "observations": tuple(memory.observations),
                "feedback": tuple(memory.feedback),
            },
        )
