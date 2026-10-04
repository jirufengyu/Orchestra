from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ...contracts import (
    AgentContext,
    CapabilityFeedback,
    CapabilityOutput,
    CapabilityRequest,
    CapabilitySpec,
    CapabilityStatus,
    ContextFragment,
    Goal,
    Observation,
    TaskTransition,
)


class RMBenchObservationAdapter:
    def __init__(
        self,
        camera_names: Sequence[str] = ("head_camera", "right_camera", "left_camera"),
    ):
        self.camera_names = tuple(camera_names)

    def __call__(self, raw: Mapping[str, Any]) -> Observation:
        camera_data = raw["observation"]
        return Observation(
            images={name: camera_data[name]["rgb"] for name in self.camera_names},
            robot_state=raw["joint_action"]["vector"],
            metadata={"rmbench_raw": raw},
        )


@dataclass
class RMBenchEmbodiment:
    """Duck-typed adapter; importing this module does not import RMBench."""

    env: Any
    setup_episode: Callable[[Any], None] | None = None
    observation_adapter: Callable[[Mapping[str, Any]], Observation] = field(
        default_factory=RMBenchObservationAdapter
    )
    close_on_stop: bool = False

    def reset(self) -> Observation:
        if self.setup_episode is not None:
            self.setup_episode(self.env)
        return self.observe()

    def observe(self) -> Observation:
        return self.observation_adapter(self.env.get_obs())

    def act(self, action: Any) -> Observation:
        self.env.take_action(action)
        return self.observe()

    def stop(self) -> None:
        if self.close_on_stop and hasattr(self.env, "close_env"):
            self.env.close_env()


@dataclass
class RMBenchTaskProtocol:
    env: Any
    mode: str = "continuous"
    success_verifier: Callable[[Any], bool] | None = None
    _done: bool = False
    _goal: Goal | None = None

    @property
    def done(self) -> bool:
        return self._done

    def start(self, observation: Observation) -> None:
        del observation
        self._done = False
        self._goal = self._read_goal()

    def current_goal(self) -> Goal:
        if self._goal is None:
            raise RuntimeError("task protocol has not started")
        return self._goal

    def update(
        self,
        observation: Observation,
        feedback: CapabilityFeedback,
    ) -> TaskTransition:
        del observation
        if feedback.status in {CapabilityStatus.FAILED, CapabilityStatus.CANCELLED}:
            return TaskTransition(reason=feedback.reason_code or feedback.status.value)

        if getattr(self.env, "subtask_just_succeeded", False):
            self.env.on_subtask_succeeded()
            self.env.reset_arms_home()
            if self.env.advance_subtask():
                self._goal = self._read_goal()
                return TaskTransition(subtask_changed=True)
            self._done = True
            success = self._verify_success()
            return TaskTransition(
                done=True,
                success=success,
                reason=getattr(self.env, "end_reason", None),
            )

        if int(getattr(self.env, "take_action_cnt", 0)) >= int(
            getattr(self.env, "step_lim", 2**31 - 1)
        ):
            if hasattr(self.env, "mark_subtask_failed"):
                self.env.mark_subtask_failed(reason="timeout")
            self._done = True
            return TaskTransition(done=True, success=False, reason="timeout")

        return TaskTransition()

    def _read_goal(self) -> Goal:
        state = getattr(self.env, "_atomic_orchestrator_state", None)
        subtask_index = int(
            getattr(state, "subtask_index", getattr(self.env, "subtask_index", 0))
        )
        instruction = str(self.env.get_instruction())
        return Goal(
            goal_id=f"{self.mode}:{subtask_index}",
            instruction=instruction,
            metadata={"mode": self.mode, "subtask_index": subtask_index},
        )

    def _verify_success(self) -> bool:
        if self.success_verifier is not None:
            return bool(self.success_verifier(self.env))
        if self.mode == "continuous":
            max_subtasks = int(getattr(self.env, "max_subtasks", 0))
            success_length = int(getattr(self.env, "success_length", 0))
            end_reason = getattr(self.env, "end_reason", None)
            return (max_subtasks > 0 and success_length >= max_subtasks) or end_reason in {
                "reached_max",
                "completed_script",
            }
        state = getattr(self.env, "_atomic_orchestrator_state", None)
        if state is not None and hasattr(state, "eval_success"):
            return bool(state.eval_success)
        return bool(self.env.check_success())


@dataclass
class RMBenchGroundingProvider:
    extract_seg_mask: Callable[[Mapping[str, Any]], Any]
    extract_frame_meta: Callable[[Mapping[str, Any]], Any]
    name: str = "visual_grounding"

    def provide(self, observation: Observation, goal: Goal, memory: Any) -> ContextFragment:
        del goal, memory
        raw = observation.metadata["rmbench_raw"]
        return ContextFragment(
            name=self.name,
            data={
                "seg_mask": self.extract_seg_mask(raw),
                "actors_frame_meta": self.extract_frame_meta(raw),
            },
        )


@dataclass
class Pi05Capability:
    """Adapter around the existing PI0 client without importing RMBench or openpi."""

    client: Any
    pi0_step: int
    camera_names: Sequence[str] = ("head_camera", "right_camera", "left_camera")
    capability_id: str = "pi05_vla"

    @property
    def spec(self) -> CapabilitySpec:
        return CapabilitySpec(
            capability_id=self.capability_id,
            description="Execute the current manipulation goal with a pi05 VLA policy.",
            input_schema={"prompt": "string"},
            metadata={"kind": "vla"},
        )

    def invoke(
        self,
        request: CapabilityRequest,
        context: AgentContext,
    ) -> CapabilityOutput:
        instruction = request.prompt or request.goal.instruction
        self.client.set_language(instruction)
        self._push_context(context)
        uses_memory = bool(getattr(self.client, "use_short_horizon_memory", False))
        synced = bool(getattr(self.client, "_short_horizon_memory_synced", False))
        if uses_memory and not synced:
            self.client.append_memory()
        actions = self.client.get_action(
            skip_memory_append=uses_memory and bool(
                getattr(self.client, "_short_horizon_memory_synced", False)
            )
        )
        return CapabilityOutput(
            actions=tuple(actions[: self.pi0_step]),
            feedback=CapabilityFeedback(status=CapabilityStatus.RUNNING),
        )

    def observe(self, context: AgentContext) -> None:
        self._push_context(context)
        if bool(getattr(self.client, "use_short_horizon_memory", False)):
            self.client.append_memory()

    def _push_context(self, context: AgentContext) -> None:
        grounding = context.fragments.get("visual_grounding")
        grounding_data = grounding.data if grounding is not None else {}
        episode_index = int(context.memory.get("episode_index", 0))
        subtask_index = int(context.memory.get("subtask_index", 0))
        episode_id = episode_index * 1000 + subtask_index
        step_id = int(context.memory.get("step_id", 0))
        self.client.update_observation_window(
            [context.observation.images[name] for name in self.camera_names],
            context.observation.robot_state,
            seg_mask=grounding_data.get("seg_mask"),
            actors_frame_meta=grounding_data.get("actors_frame_meta"),
            episode_id=episode_id,
            step_id=step_id,
        )

    def reset(self) -> None:
        self.client.reset_obsrvationwindows()

    def cancel(self, reason: str) -> None:
        del reason
        self.client.reset_obsrvationwindows()
