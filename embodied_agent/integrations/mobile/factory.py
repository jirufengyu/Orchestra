from __future__ import annotations

from pathlib import Path
from typing import Any

from ...capabilities import CapabilityRuntime
from ...context import ContextSystem
from ...contracts import Goal
from ...evaluators import EvaluatorRuntime
from ...executive import FixedExecutive
from ...planning import LlmTaskPlanner, PlanTaskProtocol
from ...runtime import AgentRuntime
from ...tools import ContextToolRuntime
from .debug import GatewayDebugRecorder
from .gateway import MobileInferenceGateway
from .grounding import (
    EntityGroundingProvider,
    EntityResolverTool,
    MobileVisualGroundingProvider,
    MolmoPointTool,
    SamOnlineTool,
)
from .pi05 import Pi05ActionCapability, Pi05ProgressEvaluator
from .registry import RegistryEntityResolver, skill_catalog


def build_pi05_mobile_gateway(
    *,
    action_client: Any,
    progress_client: Any,
    molmo_url: str,
    sam_url: str,
    entity_model: Any,
    action_horizon: int = 30,
    done_threshold: float = 0.6,
    done_count_threshold: int = 1,
    debug_dir: str | Path | None = None,
) -> MobileInferenceGateway:
    entity_resolver = RegistryEntityResolver(model=entity_model)

    def resolve_atomic_goal(instruction: str, index: int) -> Goal:
        selection, canonical = entity_resolver.canonicalize(instruction)
        return Goal(
            goal_id=f"{selection.task_id}:{index}",
            instruction=canonical,
            entity_bindings={
                slot: str(instance_id) for slot, instance_id in selection.slots.items()
            },
            metadata={"task_id": selection.task_id},
        )

    tools = ContextToolRuntime()
    tools.register(EntityResolverTool(entity_resolver))
    tools.register(MolmoPointTool(molmo_url))
    tools.register(SamOnlineTool(sam_url))
    context_system = ContextSystem(
        [
            EntityGroundingProvider(tools),
            MobileVisualGroundingProvider(tools),
        ]
    )
    policy = Pi05ActionCapability(action_client, action_horizon=action_horizon)
    progress = Pi05ProgressEvaluator(
        progress_client,
        done_threshold=done_threshold,
        done_count_threshold=done_count_threshold,
    )
    evaluators = EvaluatorRuntime()
    evaluators.register(progress)
    capabilities = CapabilityRuntime()
    capabilities.register(policy)
    task = PlanTaskProtocol(
        LlmTaskPlanner(
            model=entity_model,
            atomic_goal_resolver=resolve_atomic_goal,
            skill_context={"available_skills": skill_catalog()},
        )
    )
    runtime = AgentRuntime(
        task=task,
        executive=FixedExecutive(policy.spec.capability_id),
        capabilities=capabilities,
        context_system=context_system,
        evaluators=evaluators,
    )
    debug = None
    if debug_dir is not None:
        debug = GatewayDebugRecorder(debug_dir)
    return MobileInferenceGateway(
        runtime=runtime,
        task=task,
        mapper=policy.mapper,
        debug=debug,
    )
