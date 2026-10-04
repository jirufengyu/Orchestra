from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .contracts import AgentContext, CapabilityFeedback, EvaluatorSpec
from .interfaces import Evaluator


@dataclass
class EvaluatorRuntime:
    """Runs read-only sidecars against the same immutable AgentContext."""

    _evaluators: dict[str, Evaluator] = field(default_factory=dict)

    def register(self, evaluator: Evaluator) -> None:
        evaluator_id = evaluator.spec.evaluator_id
        if evaluator_id in self._evaluators:
            raise ValueError(f"evaluator already registered: {evaluator_id}")
        self._evaluators[evaluator_id] = evaluator

    def specs(self) -> tuple[EvaluatorSpec, ...]:
        return tuple(item.spec for item in self._evaluators.values())

    def evaluate(self, context: AgentContext) -> dict[str, CapabilityFeedback]:
        return {
            evaluator_id: evaluator.evaluate(context)
            for evaluator_id, evaluator in self._evaluators.items()
        }

    def observe(self, context: AgentContext) -> None:
        for evaluator in self._evaluators.values():
            evaluator.observe(context)

    def reset(self) -> None:
        for evaluator in self._evaluators.values():
            evaluator.reset()

    def cancel(self, reason: str) -> None:
        for evaluator in self._evaluators.values():
            evaluator.cancel(reason)

    def __bool__(self) -> bool:
        return bool(self._evaluators)


@dataclass
class CallableEvaluator:
    spec: EvaluatorSpec
    function: Callable[[AgentContext], CapabilityFeedback]
    observe_function: Callable[[AgentContext], None] | None = None
    reset_function: Callable[[], None] | None = None
    cancel_function: Callable[[str], None] | None = None

    def evaluate(self, context: AgentContext) -> CapabilityFeedback:
        return self.function(context)

    def observe(self, context: AgentContext) -> None:
        if self.observe_function is not None:
            self.observe_function(context)

    def reset(self) -> None:
        if self.reset_function is not None:
            self.reset_function()

    def cancel(self, reason: str) -> None:
        if self.cancel_function is not None:
            self.cancel_function(reason)
