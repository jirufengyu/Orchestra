from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from .contracts import CapabilityFeedback, Goal, Observation, TaskTransition
from .executive import StructuredModel
from .prompts import TASK_AGENT_SYSTEM_PROMPT


AtomicGoalResolver = Callable[[str, int], Goal]


@dataclass
class LlmTaskPlanner:
    """Decompose tasks with an LLM. Short mode skips planning, not grounding."""

    model: StructuredModel
    atomic_goal_resolver: AtomicGoalResolver | None = None
    skill_context: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None
    system_prompt: str = TASK_AGENT_SYSTEM_PROMPT

    def plan(self, instruction: str, *, mode: str = "auto") -> tuple[Goal, ...]:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("task instruction cannot be empty")
        if mode not in {"auto", "short", "long"}:
            raise ValueError(f"unknown task mode: {mode!r}")
        if mode == "short":
            return (self._goal(instruction, 0),)
        skills = self.skill_context() if callable(self.skill_context) else (self.skill_context or {})
        complete = getattr(self.model, "complete_json_array", self.model.complete_json)
        payload = complete(
            self.system_prompt,
            json.dumps(
                {
                    **dict(skills),
                    "instruction": instruction,
                    "mode": mode,
                    "registered_action_policy_capabilities": skills.get(
                        "registered_action_policy_capabilities", skills.get("available_skills", [])
                    ),
                },
                ensure_ascii=False,
            ),
        )
        raw_subtasks = (
            payload if isinstance(payload, list)
            else payload.get("subtasks") if isinstance(payload, Mapping)
            else None
        )
        if not isinstance(raw_subtasks, list) or not raw_subtasks:
            raise ValueError("planner LLM must return a non-empty subtasks list")
        goals = []
        for index, item in enumerate(raw_subtasks):
            if isinstance(item, str):
                atomic_instruction = item.strip()
            elif isinstance(item, Mapping):
                atomic_instruction = str(item.get("subtask", item.get("instruction", ""))).strip()
            else:
                raise ValueError("planner subtask must be a string or object")
            if not atomic_instruction:
                raise ValueError(f"planner subtask {index} is missing instruction")
            goal = self._goal(atomic_instruction, index)
            if isinstance(item, Mapping) and "activate object" in item:
                entities = item["activate object"]
                if not isinstance(entities, list) or any(
                    not isinstance(entity, str) or not entity.strip() for entity in entities
                ):
                    raise ValueError(f"planner subtask {index} requires a list of entity descriptions")
                goal = replace(goal, metadata={**goal.metadata, "activate object": list(entities)})
            goals.append(goal)
        return tuple(goals)

    def _goal(self, instruction: str, index: int) -> Goal:
        if self.atomic_goal_resolver is None:
            return Goal(f"subtask:{index}", instruction)
        return self.atomic_goal_resolver(instruction, index)


DirectOrPlannedTaskPlanner = LlmTaskPlanner


def evaluator_reports_done(feedback: Any) -> bool:
    if not isinstance(feedback, CapabilityFeedback):
        return False
    evaluations = feedback.data.get("evaluations", {})
    if not isinstance(evaluations, Mapping):
        return False
    for evaluation in evaluations.values():
        if isinstance(evaluation, CapabilityFeedback):
            if bool(evaluation.data.get("subtask_done", False)):
                return True
        elif isinstance(evaluation, Mapping):
            if bool(evaluation.get("data", {}).get("subtask_done", False)):
                return True
    return False


@dataclass
class PlanTaskProtocol:
    planner: LlmTaskPlanner
    completion_predicate: Callable[[Any], bool] = evaluator_reports_done
    _goals: tuple[Goal, ...] = field(default=(), init=False)
    _index: int = field(default=0, init=False)
    _done: bool = field(default=False, init=False)
    _instruction: str | None = field(default=None, init=False)
    _mode: str = field(default="auto", init=False)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def plan_index(self) -> int:
        return self._index

    @property
    def plan_length(self) -> int:
        return len(self._goals)

    @property
    def root_instruction(self) -> str | None:
        return self._instruction

    def configure(self, instruction: str, *, mode: str = "auto") -> None:
        self._goals = self.planner.plan(instruction, mode=mode)
        self._instruction = instruction.strip()
        self._mode = mode
        self._index = 0
        self._done = False

    def start(self, observation: Observation) -> None:
        del observation
        if not self._goals:
            raise RuntimeError("PlanTaskProtocol.configure must be called before start")
        self._index = 0
        self._done = False

    def current_goal(self) -> Goal:
        if not self._goals:
            raise RuntimeError("task plan is not configured")
        return self._goals[self._index]

    def update(self, observation: Observation, feedback: Any) -> TaskTransition:
        del observation
        if not self.completion_predicate(feedback):
            return TaskTransition()
        if self._index + 1 < len(self._goals):
            self._index += 1
            return TaskTransition(
                subtask_changed=True,
                reason="progress_evaluator_completed_subtask",
            )
        self._done = True
        return TaskTransition(
            done=True,
            success=True,
            reason="plan_completed",
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_instruction": self._instruction,
            "task_mode": self._mode,
            "plan_index": self._index,
            "plan_length": len(self._goals),
            "current_instruction": None if not self._goals else self.current_goal().instruction,
            "task_done": self._done,
            "subtasks": [goal.instruction for goal in self._goals],
        }
