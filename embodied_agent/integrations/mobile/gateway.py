from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ...contracts import AgentContext, CapabilityFeedback, Observation
from ...planning import PlanTaskProtocol
from ...runtime import AgentRuntime
from .debug import GatewayDebugRecorder
from .pi05 import Pi05ContextMapper

logger = logging.getLogger(__name__)


@dataclass
class MobileInferenceGateway:
    """Thin transport adapter over the reusable AgentRuntime."""

    runtime: AgentRuntime
    task: PlanTaskProtocol
    mapper: Pi05ContextMapper = field(default_factory=Pi05ContextMapper)
    debug: GatewayDebugRecorder | None = None
    _episode_id: str | None = field(default=None, init=False)
    _pending_episode_id: str | None = field(default=None, init=False)
    _task_instruction: str | None = field(default=None, init=False)
    _task_mode: str = field(default="auto", init=False)
    _last_step: int = field(default=-1, init=False)

    def handle_infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        observation = self._observation(request)
        self._ensure_task(request, observation)
        step_id = int(request.get("step_id", self._last_step + 1))
        if step_id < self._last_step:
            raise ValueError("step_id 不能倒退")
        context_started = time.perf_counter()
        result = self.runtime.infer(observation, step_id=step_id, update_task=True)
        context_ms = (time.perf_counter() - context_started) * 1000.0
        self._last_step = step_id
        evaluations = result.output.feedback.data.get("evaluations", {})
        progress_feedback = next(iter(evaluations.values()), None)
        subtask_done = result.transition.subtask_changed or result.transition.done
        policy_actions = np.asarray(result.output.actions)
        actions = policy_actions[:0] if subtask_done else policy_actions
        response = {
            "actions": actions,
            "action_type": "qpos",
            "progress": (
                progress_feedback.progress
                if isinstance(progress_feedback, CapabilityFeedback)
                else None
            ),
            "subtask_done": subtask_done,
            "task_done": result.transition.done,
            **self.task.snapshot(),
            "evaluations": evaluations,
            "annotation": result.context.fragments.get("visual_grounding"),
            "timing": {
                "context_ms": context_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0,
            },
        }
        self._record_debug(observation, result.context, response)
        return response

    def handle_observe(self, request: Mapping[str, Any]) -> dict[str, Any]:
        observation = self._observation(request)
        self._ensure_task(request, observation)
        step_id = int(request.get("step_id", self._last_step + 1))
        if step_id < self._last_step:
            raise ValueError("step_id 不能倒退")
        self.runtime.observe(observation, step_id=step_id, enrich_context=False)
        self._last_step = step_id
        return {"ok": True, "step_id": step_id, **self.task.snapshot()}

    def reset_episode(self, episode_id: str | None = None) -> None:
        self.runtime.reset_components()
        self._episode_id = None
        self._pending_episode_id = episode_id
        self._task_instruction = None
        self._last_step = -1
        if self.debug is not None:
            self.debug.record_reset(episode_id)

    def pause(self) -> None:
        """Pause keeps model and grounding memory for VR intervention."""

    def resume(self) -> None:
        """The next fresh observation re-synchronizes policy inference."""

    def close(self) -> None:
        try:
            if self.debug is not None:
                self.debug.close()
        finally:
            self.runtime.close()

    def _record_debug(
        self,
        observation: Observation,
        context: AgentContext,
        response: Mapping[str, Any],
    ) -> None:
        if self.debug is None:
            return
        try:
            action_payload = self.mapper.map(context, require_grounding=True)
        except Exception:
            logger.exception("debug dump could not map PI05 action context")
            action_payload = {
                "prompt": context.goal.instruction,
                "episode_id": context.memory.get("episode_id"),
                "step_id": context.memory.get("step_id"),
                "images": dict(observation.images),
            }
        try:
            self.debug.record_infer(
                observation=observation,
                context=context,
                response=response,
                action_payload=action_payload,
            )
        except Exception:
            logger.exception("debug dump enqueue failed")

    @staticmethod
    def _observation(request: Mapping[str, Any]) -> Observation:
        return Observation(
            images=dict(request["images"]),
            robot_state=np.asarray(request["state"], dtype=np.float32),
            metadata=dict(request.get("metadata") or {}),
        )

    def _ensure_task(
        self,
        request: Mapping[str, Any],
        observation: Observation,
    ) -> None:
        episode_id = str(
            request.get("episode_id")
            or self._pending_episode_id
            or "default"
        )
        instruction = str(
            request.get("task_instruction")
            or request.get("instruction")
            or ""
        ).strip()
        if not instruction:
            raise ValueError("instruction/task_instruction 不能为空")
        mode = str(request.get("task_mode", "auto"))
        if self._episode_id != episode_id:
            self.task.configure(instruction, mode=mode)
            self.runtime.begin_episode(observation, episode_id)
            self._episode_id = episode_id
            self._pending_episode_id = None
            self._task_instruction = instruction
            self._task_mode = mode
            self._last_step = -1
        elif instruction != self._task_instruction or mode != self._task_mode:
            self.task.configure(instruction, mode=mode)
            self.runtime.begin_task(observation)
            self._task_instruction = instruction
            self._task_mode = mode
            self._last_step = -1
