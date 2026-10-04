from __future__ import annotations

import concurrent.futures
import dataclasses
from dataclasses import dataclass, field

from .capabilities import CapabilityRuntime
from .context import ContextSystem
from .contracts import (
    AgentContext,
    AgentStepResult,
    CapabilityOutput,
    CapabilityRequest,
    Observation,
    TaskTransition,
)
from .evaluators import EvaluatorRuntime
from .interfaces import Executive, TaskProtocol
from .memory import MemorySystem


def invoke_with_evaluators(
    capabilities: CapabilityRuntime,
    evaluators: EvaluatorRuntime,
    request: CapabilityRequest,
    context: AgentContext,
) -> CapabilityOutput:
    if not evaluators:
        return capabilities.invoke(request, context)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        action_future = pool.submit(capabilities.invoke, request, context)
        evaluation_future = pool.submit(evaluators.evaluate, context)
        output = action_future.result()
        evaluations = evaluation_future.result()
    progress = output.feedback.progress
    for feedback in evaluations.values():
        if feedback.progress is not None:
            progress = feedback.progress
            break
    merged_data = dict(output.feedback.data)
    merged_data["evaluations"] = evaluations
    return dataclasses.replace(
        output,
        feedback=dataclasses.replace(
            output.feedback,
            progress=progress,
            data=merged_data,
        ),
    )


@dataclass
class AgentRuntime:
    """Step-oriented core used by active loops and remote gateways."""

    task: TaskProtocol
    executive: Executive
    capabilities: CapabilityRuntime
    context_system: ContextSystem = field(default_factory=ContextSystem)
    memory: MemorySystem = field(default_factory=MemorySystem)
    evaluators: EvaluatorRuntime = field(default_factory=EvaluatorRuntime)
    _started: bool = field(default=False, init=False)

    def begin_episode(
        self,
        observation: Observation,
        episode_id: str | None = None,
    ) -> None:
        self.reset_components()
        self.memory.begin_episode(episode_id)
        self.task.start(observation)
        self.memory.begin_subtask(self.task.current_goal())
        self.memory.record_observation(observation)
        self._started = True

    def begin_task(self, observation: Observation) -> None:
        if not self._started:
            self.begin_episode(observation)
            return
        self.reset_components()
        self.task.start(observation)
        self.memory.begin_subtask(self.task.current_goal())
        self.memory.record_observation(observation)

    def infer(
        self,
        observation: Observation,
        *,
        step_id: int | None = None,
        update_task: bool = True,
    ) -> AgentStepResult:
        self._require_started()
        if step_id is not None:
            self.memory.step_id = int(step_id)
        self.memory.record_observation(observation)
        goal = self.task.current_goal()
        context = self.context_system.build(observation, goal, self.memory)
        decision = self.executive.decide(context, self.capabilities.specs())
        output = invoke_with_evaluators(
            self.capabilities,
            self.evaluators,
            decision.request,
            context,
        )
        self.memory.record_feedback(output.feedback)
        transition = (
            self.update_task(observation, output.feedback)
            if update_task
            else TaskTransition()
        )
        return AgentStepResult(context, output, transition)

    def observe(
        self,
        observation: Observation,
        *,
        step_id: int | None = None,
        enrich_context: bool = False,
    ) -> AgentContext:
        self._require_started()
        if step_id is not None:
            self.memory.step_id = int(step_id)
        self.memory.record_observation(observation)
        goal = self.task.current_goal()
        if enrich_context:
            context = self.context_system.build(observation, goal, self.memory)
        else:
            context = AgentContext(
                observation=observation,
                goal=goal,
                memory=self.memory.snapshot(),
                fragments={},
            )
        self.capabilities.observe(context)
        self.evaluators.observe(context)
        return context

    def update_task(self, observation: Observation, feedback) -> TaskTransition:
        transition = self.task.update(observation, feedback)
        if transition.subtask_changed and not transition.done:
            self.reset_components()
            self.memory.begin_subtask(self.task.current_goal())
        return transition

    def reset_components(self) -> None:
        self.capabilities.reset()
        self.evaluators.reset()
        self.context_system.reset()

    def close(self) -> None:
        self.capabilities.cancel("agent runtime closed")
        self.evaluators.cancel("agent runtime closed")
        self.context_system.close()
        self._started = False

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("agent runtime has not started an episode")
