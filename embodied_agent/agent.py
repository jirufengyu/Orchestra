from __future__ import annotations

from dataclasses import dataclass, field

from .capabilities import CapabilityRuntime
from .context import ContextSystem
from .contracts import AgentRunResult, Observation, TaskTransition
from .interfaces import Embodiment, Executive, TaskProtocol
from .evaluators import EvaluatorRuntime
from .memory import MemorySystem
from .runtime import invoke_with_evaluators


@dataclass
class EmbodiedAgent:
    embodiment: Embodiment
    task: TaskProtocol
    executive: Executive
    capabilities: CapabilityRuntime
    context_system: ContextSystem = field(default_factory=ContextSystem)
    memory: MemorySystem = field(default_factory=MemorySystem)
    evaluators: EvaluatorRuntime = field(default_factory=EvaluatorRuntime)
    max_cycles: int = 10_000

    def run(self, episode_id: str | None = None) -> AgentRunResult:
        cycles = 0
        last_transition = TaskTransition()
        observation: Observation | None = None
        self.memory.begin_episode(episode_id)
        self.context_system.reset()
        self.evaluators.reset()

        try:
            observation = self.embodiment.reset()
            self.task.start(observation)
            self.memory.begin_subtask(self.task.current_goal())
            self.memory.record_observation(observation)
            context = None

            while not self.task.done and cycles < self.max_cycles:
                cycles += 1
                goal = self.task.current_goal()
                if context is None:
                    context = self.context_system.build(
                        observation=observation,
                        goal=goal,
                        memory=self.memory,
                    )
                decision = self.executive.decide(context, self.capabilities.specs())
                output = self._invoke_with_evaluators(decision.request, context)
                self.memory.record_feedback(output.feedback)

                if output.actions:
                    for action in output.actions:
                        observation = self.embodiment.act(action)
                        self.memory.advance_step()
                        last_transition = self.task.update(observation, output.feedback)
                        if last_transition.subtask_changed or last_transition.done:
                            break
                        self.memory.record_observation(observation)
                        context = self.context_system.build(
                            observation=observation,
                            goal=goal,
                            memory=self.memory,
                        )
                        self.capabilities.observe(context)
                        self.evaluators.observe(context)
                else:
                    observation = self.embodiment.observe()
                    last_transition = self.task.update(observation, output.feedback)
                    self.memory.record_observation(observation)
                    context = self.context_system.build(
                        observation=observation,
                        goal=goal,
                        memory=self.memory,
                    )
                    self.capabilities.observe(context)
                    self.evaluators.observe(context)

                if last_transition.subtask_changed and not last_transition.done:
                    self.capabilities.reset()
                    self.evaluators.reset()
                    self.context_system.reset()
                    self.memory.begin_subtask(self.task.current_goal())
                    observation = self.embodiment.observe()
                    self.memory.record_observation(observation)
                    context = None

                if last_transition.done:
                    break

            if cycles >= self.max_cycles and not last_transition.done:
                self.capabilities.cancel("agent cycle limit reached")
                self.evaluators.cancel("agent cycle limit reached")
                last_transition = TaskTransition(
                    done=True,
                    success=False,
                    reason="agent_cycle_limit",
                )

            return AgentRunResult(
                success=last_transition.success,
                cycles=cycles,
                reason=last_transition.reason,
                memory=self.memory.snapshot(),
            )
        except Exception:
            self.capabilities.cancel("agent exception")
            self.evaluators.cancel("agent exception")
            raise
        finally:
            self.embodiment.stop()
            self.context_system.close()

    def _invoke_with_evaluators(self, request, context):
        return invoke_with_evaluators(
            self.capabilities,
            self.evaluators,
            request,
            context,
        )
