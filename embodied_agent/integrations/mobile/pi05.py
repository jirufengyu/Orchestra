from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ...contracts import (
    AgentContext,
    CapabilityFeedback,
    CapabilityOutput,
    CapabilityRequest,
    CapabilitySpec,
    CapabilityStatus,
    EvaluatorSpec,
)


@dataclass(frozen=True)
class Pi05ContextMapper:
    camera_names: Sequence[str] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    state_dim: int = 16
    seg_representation: str = "mask"

    def map(
        self,
        context: AgentContext,
        *,
        require_grounding: bool = True,
    ) -> dict[str, Any]:
        state = np.asarray(context.observation.robot_state, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,):
            raise ValueError(f"PI05 state 应为 {self.state_dim} 维，实际 {state.shape}")
        images = {
            name: self._chw(context.observation.images[name])
            for name in self.camera_names
            if name in context.observation.images
        }
        if "cam_high" not in images:
            raise ValueError("PI05 context 缺少 cam_high")
        result = {
            "state": state,
            "images": images,
            "prompt": context.goal.instruction,
            "episode_id": context.memory.get("episode_id"),
            "episode_index": context.memory.get("episode_index"),
            "step_id": context.memory.get("step_id"),
            "frame_index": context.memory.get("step_id"),
        }
        grounding = context.fragments.get("visual_grounding")
        if grounding is None:
            if require_grounding:
                raise ValueError("PI05 context 缺少 visual_grounding")
            return result
        if not isinstance(grounding.data, Mapping):
            raise ValueError("visual_grounding.data 必须是 mapping")
        seg_mask = np.asarray(grounding.data.get("seg_mask"), dtype=np.uint8)
        actors_meta = grounding.data.get("actors_frame_meta")
        if seg_mask.ndim != 2 or not isinstance(actors_meta, Mapping) or not actors_meta:
            raise ValueError("visual_grounding 缺少有效 seg_mask/actors_frame_meta")
        return {
            **result,
            "seg_cam_high": seg_mask,
            "actors_frame_meta": dict(actors_meta),
            "seg_representation": self.seg_representation,
        }

    @staticmethod
    def _chw(image: Any) -> np.ndarray:
        value = np.asarray(image)
        if value.ndim != 3:
            raise ValueError(f"camera image 应为三维数组，实际 {value.shape}")
        if value.shape[-1] == 3:
            value = np.transpose(value, (2, 0, 1))
        if value.shape[0] != 3:
            raise ValueError(f"camera image 应有三个通道，实际 {value.shape}")
        if np.issubdtype(value.dtype, np.floating):
            if value.max(initial=0) <= 1.0:
                value = value * 255
            value = np.clip(value, 0, 255).astype(np.uint8)
        return value


def _client_append_memory(client: Any, observation: Mapping[str, Any]) -> None:
    append = getattr(client, "append_memory", None)
    if callable(append):
        append(observation)
        return
    infer = getattr(client, "infer", None)
    if callable(infer):
        infer({"__openpi_rpc__": "append_memory", "observation": dict(observation)})
        return
    raise AttributeError(f"{type(client).__name__} 缺少 append_memory")


def _is_closed_connection(exc: BaseException) -> bool:
    name = type(exc).__name__
    if "ConnectionClosed" in name:
        return True
    message = str(exc).lower()
    return "connection closed" in message or "internal server error" in message


def _client_reset_memory(client: Any) -> None:
    try:
        reset_memory = getattr(client, "reset_memory", None)
        if callable(reset_memory):
            reset_memory()
            return
        reset = getattr(client, "reset", None)
        if callable(reset):
            reset()
            return
    except Exception as exc:
        if _is_closed_connection(exc):
            return
        raise
    raise AttributeError(f"{type(client).__name__} 缺少 reset_memory")


@dataclass
class Pi05ActionCapability:
    client: Any
    action_horizon: int = 30
    mapper: Pi05ContextMapper = field(default_factory=Pi05ContextMapper)
    capability_id: str = "pi05_mobile"
    _last_memory_key: tuple[Any, Any] | None = field(default=None, init=False)

    @property
    def spec(self) -> CapabilitySpec:
        return CapabilitySpec(
            self.capability_id,
            "Execute the current mobile dual-arm goal with PI05.",
            {"prompt": "string"},
            {"kind": "vla"},
        )

    def invoke(self, request: CapabilityRequest, context: AgentContext) -> CapabilityOutput:
        del request
        observation = self.mapper.map(context)
        key = self._memory_key(context)
        if key != self._last_memory_key:
            _client_append_memory(self.client, observation)
            self._last_memory_key = key
        result = self.client.infer({**observation, "skip_memory_append": True})
        actions = np.asarray(result["actions"])
        if actions.ndim != 2 or actions.shape[1] != self.mapper.state_dim:
            raise ValueError(f"PI05 actions shape 无效: {actions.shape}")
        return CapabilityOutput(
            actions=tuple(actions[: self.action_horizon]),
            feedback=CapabilityFeedback(
                CapabilityStatus.RUNNING,
                data={"policy_timing": result.get("server_timing", {})},
            ),
        )

    def observe(self, context: AgentContext) -> None:
        _client_append_memory(self.client, self.mapper.map(context, require_grounding=False))
        self._last_memory_key = self._memory_key(context)

    def reset(self) -> None:
        self._last_memory_key = None
        _client_reset_memory(self.client)

    def cancel(self, reason: str) -> None:
        del reason
        self.reset()

    @staticmethod
    def _memory_key(context: AgentContext) -> tuple[Any, Any]:
        return context.memory.get("episode_id"), context.memory.get("step_id")


@dataclass
class Pi05ProgressEvaluator:
    client: Any
    mapper: Pi05ContextMapper = field(default_factory=Pi05ContextMapper)
    done_threshold: float = 0.6
    progress_complete_threshold: float = 0.90
    done_count_threshold: int = 1
    evaluator_id: str = "pi05_progress"
    _done_count: int = field(default=0, init=False)
    _last_memory_key: tuple[Any, Any] | None = field(default=None, init=False)

    @property
    def spec(self) -> EvaluatorSpec:
        return EvaluatorSpec(
            self.evaluator_id,
            "Estimate task progress from the same context consumed by the action policy.",
        )

    def evaluate(self, context: AgentContext) -> CapabilityFeedback:
        observation = self.mapper.map(context)
        key = self._memory_key(context)
        if key != self._last_memory_key:
            _client_append_memory(self.client, observation)
            self._last_memory_key = key
        metrics = self.client.infer_progress({**observation, "skip_memory_append": True})
        if not metrics.get("progress_available"):
            self._done_count = 0
            return CapabilityFeedback(
                CapabilityStatus.FAILED,
                reason_code="progress_unavailable",
                data=dict(metrics),
            )
        if "progress_percent" in metrics:
            progress = float(metrics["progress_percent"]) / 100.0
        else:
            progress = float(metrics.get("progress_value", -1.0)) + 1.0
        done_hit = (
            float(metrics.get("done_prob", 0.0)) >= self.done_threshold
            or progress >= self.progress_complete_threshold
        )
        self._done_count = self._done_count + 1 if done_hit else 0
        done = self._done_count >= self.done_count_threshold
        return CapabilityFeedback(
            CapabilityStatus.SUCCEEDED if done else CapabilityStatus.RUNNING,
            progress=progress,
            reason_code="subtask_done" if done else None,
            data={**dict(metrics), "subtask_done": done, "done_count": self._done_count},
        )

    def observe(self, context: AgentContext) -> None:
        _client_append_memory(self.client, self.mapper.map(context, require_grounding=False))
        self._last_memory_key = self._memory_key(context)

    def reset(self) -> None:
        self._done_count = 0
        self._last_memory_key = None
        _client_reset_memory(self.client)

    def cancel(self, reason: str) -> None:
        del reason
        self.reset()

    @staticmethod
    def _memory_key(context: AgentContext) -> tuple[Any, Any]:
        return context.memory.get("episode_id"), context.memory.get("step_id")
