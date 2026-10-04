from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque
from uuid import uuid4

from .contracts import CapabilityFeedback, ContextFragment, Goal, Observation


@dataclass
class MemoryEvent:
    event_type: str
    payload: Any


@dataclass
class MemorySystem:
    max_observations: int = 8
    max_feedback: int = 16
    episode_id: str | None = None
    episode_index: int = 0
    subtask_index: int = 0
    step_id: int = 0
    current_goal: Goal | None = None
    observations: Deque[Observation] = field(init=False)
    fragments: Deque[dict[str, ContextFragment]] = field(init=False)
    feedback: Deque[CapabilityFeedback] = field(init=False)
    events: list[MemoryEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.observations = deque(maxlen=self.max_observations)
        self.fragments = deque(maxlen=self.max_observations)
        self.feedback = deque(maxlen=self.max_feedback)

    def begin_episode(self, episode_id: str | None = None) -> None:
        self.episode_index += 1
        self.episode_id = episode_id or str(uuid4())
        self.subtask_index = 0
        self.step_id = 0
        self.current_goal = None
        self.observations.clear()
        self.fragments.clear()
        self.feedback.clear()
        self.events.clear()
        self.events.append(MemoryEvent("episode_started", self.episode_id))

    def begin_subtask(self, goal: Goal) -> None:
        if self.current_goal is not None:
            self.subtask_index += 1
        self.current_goal = goal
        self.step_id = 0
        self.observations.clear()
        self.fragments.clear()
        self.feedback.clear()
        self.events.append(MemoryEvent("subtask_started", goal))

    def record_observation(self, observation: Observation) -> None:
        self.observations.append(observation)

    def advance_step(self) -> None:
        self.step_id += 1

    def record_fragments(self, fragments: dict[str, ContextFragment]) -> None:
        self.fragments.append(fragments)

    def record_feedback(self, feedback: CapabilityFeedback) -> None:
        self.feedback.append(feedback)
        self.events.append(MemoryEvent("capability_feedback", feedback))

    def snapshot(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "subtask_index": self.subtask_index,
            "step_id": self.step_id,
            "current_goal": self.current_goal,
            "observation_history": tuple(self.observations),
            "context_history": tuple(self.fragments),
            "feedback_history": tuple(self.feedback),
            "events": tuple(self.events),
        }
