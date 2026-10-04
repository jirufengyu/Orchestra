from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import (
    AgentContext,
    AgentDecision,
    CapabilityRequest,
    CapabilitySpec,
)


class StructuredModel(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> Mapping[str, Any] | list[Any]: ...


@dataclass
class FixedExecutive:
    capability_id: str
    parameter_function: Callable[[AgentContext], Mapping[str, Any]] = field(
        default=lambda context: {}
    )

    def decide(
        self,
        context: AgentContext,
        capabilities: Sequence[CapabilitySpec],
    ) -> AgentDecision:
        available = {item.capability_id for item in capabilities}
        if self.capability_id not in available:
            raise ValueError(f"capability is not available: {self.capability_id}")
        return AgentDecision(
            request=CapabilityRequest(
                capability_id=self.capability_id,
                goal=context.goal,
                parameters=self.parameter_function(context),
                prompt=context.goal.instruction,
            )
        )


@dataclass
class LlmExecutive:
    model: StructuredModel
    system_prompt: str = (
        "You are the executive of an embodied agent. Select exactly one available "
        "capability for the current goal. Return JSON with capability_id, parameters, "
        "prompt, and reasoning. Do not invent capability IDs."
    )

    def decide(
        self,
        context: AgentContext,
        capabilities: Sequence[CapabilitySpec],
    ) -> AgentDecision:
        capability_data = [
            {
                "capability_id": item.capability_id,
                "description": item.description,
                "input_schema": item.input_schema,
            }
            for item in capabilities
        ]
        fragment_summary = {
            name: _compact(fragment.data)
            for name, fragment in context.fragments.items()
        }
        user_prompt = json.dumps(
            {
                "goal": {
                    "goal_id": context.goal.goal_id,
                    "instruction": context.goal.instruction,
                    "entity_bindings": dict(context.goal.entity_bindings),
                    "metadata": dict(context.goal.metadata),
                },
                "perception_context": fragment_summary,
                "latest_feedback": _compact(context.memory.get("feedback_history", ())[-3:]),
                "available_capabilities": capability_data,
            },
            ensure_ascii=False,
            default=str,
        )
        result = dict(self.model.complete_json(self.system_prompt, user_prompt))
        capability_id = str(result.get("capability_id", "")).strip()
        available = {item.capability_id for item in capabilities}
        if capability_id not in available:
            raise ValueError(
                f"LLM selected unavailable capability {capability_id!r}; "
                f"available={sorted(available)}"
            )
        parameters = result.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise TypeError("LLM parameters must be a JSON object")
        return AgentDecision(
            request=CapabilityRequest(
                capability_id=capability_id,
                goal=context.goal,
                parameters=dict(parameters),
                prompt=str(result.get("prompt") or context.goal.instruction),
            ),
            reasoning=str(result.get("reasoning", "")),
        )


class OpenAIChatModel:
    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        api_key: str = "EMPTY",
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install embodied-agent[openai] to use OpenAIChatModel") from exc
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = model

    def complete_json(self, system_prompt: str, user_prompt: str) -> Mapping[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")

    def complete_json_array(self, system_prompt: str, user_prompt: str) -> list[Any]:
        # JSON-object mode excludes the top-level array specified in Figure 6.
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        result = json.loads(response.choices[0].message.content or "[]")
        if not isinstance(result, list):
            raise ValueError("task agent must return a JSON array")
        return result


def _compact(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _compact(item, limit=limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(item, limit=limit) for item in value[-8:]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."
