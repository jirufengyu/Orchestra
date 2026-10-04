from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from string import Formatter
from typing import Any, Mapping

from ...executive import StructuredModel


_GENERIC_TEMPLATE_TOKENS = frozenset(
    {"place", "the", "a", "an", "on", "in", "inside", "at", "to"}
)


@dataclass(frozen=True)
class TaskRegistryEntry:
    task_id: str
    instruction_template: str
    slots: Mapping[str, tuple[int, ...]]
    instances: Mapping[int, str]
    extra_templates: tuple[str, ...] = ()

    @property
    def sentence_templates(self) -> tuple[str, ...]:
        return (self.instruction_template, *self.extra_templates)


MOBILE_TASK_REGISTRY: dict[str, TaskRegistryEntry] = {
    "breakfast_preparation": TaskRegistryEntry(
        "breakfast_preparation",
        "Place the {A} on the {B}",
        {"A": (1, 2, 3, 4, 5, 6), "B": (1, 2, 3, 4, 5, 6)},
        {
            1: "brown hexagonal plate",
            2: "silver fork with a brown handle",
            3: "small white rectangular object",
            4: "brown loaf of bread",
            5: "white bottle with a blue label",
            6: "white bowl",
        },
        extra_templates=("Place the {A} at the upper right",),
    ),
    "number_ordering": TaskRegistryEntry(
        "number_ordering",
        "Place {A} on the {B}",
        {"A": (1, 2, 3), "B": (4, 5, 6)},
        {
            1: "number 1 red sign",
            2: "number 2 red sign",
            3: "number 3 red sign",
            4: "left blue paper",
            5: "middle blue paper",
            6: "right blue paper",
        },
    ),
    "place_fruit_bowl": TaskRegistryEntry(
        "place_fruit_bowl",
        "Place {A} on the {B}",
        {"A": (1, 2, 3), "B": (4, 5, 6)},
        {
            1: "red apple",
            2: "green apple",
            3: "orange",
            4: "left bowl",
            5: "middle bowl",
            6: "right bowl",
        },
    ),
    "sorting_object": TaskRegistryEntry(
        "sorting_object",
        "Place {A} inside the {B}",
        {"A": (2, 3, 4, 5, 6), "B": (1,)},
        {
            1: "plastic container in the center",
            2: "white blue box",
            3: "white green box",
            4: "orange can",
            5: "plastic bottle with a green band",
            6: "snack bag with yellow edge",
        },
    ),
}


def template_slot_names(template: str) -> tuple[str, ...]:
    return tuple(
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    )


def template_distinctive_tokens(template: str) -> tuple[str, ...]:
    literals = " ".join(literal for literal, _, _, _ in Formatter().parse(template))
    return tuple(
        token
        for token in literals.casefold().split()
        if token not in _GENERIC_TEMPLATE_TOKENS
    )


def mentioned_instances(
    entry: TaskRegistryEntry,
    text: str,
) -> list[int]:
    spans: list[tuple[int, int, int]] = []
    for instance_id, name in entry.instances.items():
        needle = name.casefold()
        start = 0
        while True:
            index = text.find(needle, start)
            if index < 0:
                break
            spans.append((index, index + len(needle), instance_id))
            start = index + 1
    spans.sort(key=lambda item: item[1] - item[0], reverse=True)
    kept: list[tuple[int, int, int]] = []
    for start, end, instance_id in spans:
        if any(not (end <= other_start or start >= other_end) for other_start, other_end, _ in kept):
            continue
        kept.append((start, end, instance_id))
    kept.sort()
    return [instance_id for _, _, instance_id in kept]


def format_skill_instruction(
    entry: TaskRegistryEntry,
    slots: Mapping[str, int],
    template: str | None = None,
) -> str:
    chosen = template or entry.instruction_template
    values = {
        slot: entry.instances[int(slots[slot])]
        for slot in template_slot_names(chosen)
    }
    return chosen.format(**values)


def enumerate_skill_instructions(entry: TaskRegistryEntry) -> tuple[str, ...]:
    instructions: list[str] = []
    for template in entry.sentence_templates:
        slot_names = template_slot_names(template)
        for combo in product(*(entry.slots[slot] for slot in slot_names)):
            if len(set(combo)) != len(combo):
                continue
            instructions.append(
                format_skill_instruction(
                    entry,
                    dict(zip(slot_names, combo, strict=True)),
                    template=template,
                )
            )
    return tuple(instructions)


def skill_catalog(
    registry: Mapping[str, TaskRegistryEntry] = MOBILE_TASK_REGISTRY,
) -> list[dict[str, Any]]:
    """Sentence templates the planner may choose, then fill with instance names."""
    return [
        {
            "task_id": task_id,
            "sentence_templates": list(entry.sentence_templates),
            "instruction_template": entry.instruction_template,
            "instances": {str(instance_id): name for instance_id, name in entry.instances.items()},
            "slots": {slot: list(ids) for slot, ids in entry.slots.items()},
        }
        for task_id, entry in registry.items()
    ]


@dataclass(frozen=True)
class EntitySelection:
    task_id: str
    slots: Mapping[str, int]
    instances: Mapping[int, str]
    instruction_template: str | None = None

    @property
    def actors_frame_meta(self) -> dict[str, list[int]]:
        return {name: [instance_id] for instance_id, name in self.instances.items()}

    @property
    def instance_prompts(self) -> list[dict[str, Any]]:
        return [
            {
                "instance_id": instance_id,
                "label": name,
                "prompt": f"Point to the {name}.",
            }
            for instance_id, name in self.instances.items()
        ]


class RegistryEntityResolver:
    """Resolve free-form instructions to registry-constrained stable ids."""

    def __init__(
        self,
        model: StructuredModel | None = None,
        registry: Mapping[str, TaskRegistryEntry] = MOBILE_TASK_REGISTRY,
    ):
        self.model = model
        self.registry = dict(registry)

    def resolve(self, instruction: str) -> EntitySelection:
        exact = self._resolve_by_names(instruction)
        if exact is not None:
            return exact
        if self.model is None:
            raise ValueError(
                f"instruction 无法由注册表名字匹配确定，且未配置 LLM: {instruction!r}"
            )
        payload = self.model.complete_json(
            "Map the instruction to the supplied task registry. Return JSON only with "
            "task_id, template, and slots. template must be one of that task's "
            "sentence_templates. Only fill slots that appear in the chosen template. "
            "Every slot value must be an integer instance id from its candidates.",
            self._registry_prompt(instruction),
        )
        return self._validate_selection(payload, instruction=instruction)

    def canonicalize(self, instruction: str) -> tuple[EntitySelection, str]:
        selection = self.resolve(instruction)
        entry = self.registry[selection.task_id]
        template = selection.instruction_template or entry.instruction_template
        return selection, format_skill_instruction(
            entry,
            selection.slots,
            template=template,
        )

    def _resolve_by_names(self, instruction: str) -> EntitySelection | None:
        text = instruction.casefold()
        matches: list[EntitySelection] = []
        for task_id, entry in self.registry.items():
            ranked: list[tuple[int, int, str, dict[str, int]]] = []
            for template in entry.sentence_templates:
                bound = self._bind_template_slots(entry, template, text)
                if bound is None:
                    continue
                ranked.append(
                    (
                        len(bound),
                        len(template_distinctive_tokens(template)),
                        template,
                        bound,
                    )
                )
            if not ranked:
                continue
            ranked.sort(reverse=True)
            if len(ranked) > 1 and ranked[0][:2] == ranked[1][:2]:
                continue
            _, _, template, slots = ranked[0]
            selected = {
                instance_id: entry.instances[instance_id] for instance_id in slots.values()
            }
            matches.append(
                EntitySelection(
                    task_id,
                    slots,
                    selected,
                    instruction_template=template,
                )
            )
        return matches[0] if len(matches) == 1 else None

    def _bind_template_slots(
        self,
        entry: TaskRegistryEntry,
        template: str,
        text: str,
    ) -> dict[str, int] | None:
        distinctive = template_distinctive_tokens(template)
        if distinctive and not all(token in text for token in distinctive):
            return None
        slot_names = template_slot_names(template)
        mentions = mentioned_instances(entry, text)
        if len(mentions) != len(slot_names):
            return None
        slots: dict[str, int] = {}
        used: set[int] = set()
        for slot, instance_id in zip(slot_names, mentions, strict=True):
            if instance_id not in entry.slots[slot] or instance_id in used:
                return None
            slots[slot] = instance_id
            used.add(instance_id)
        return slots

    def _choose_template(
        self,
        entry: TaskRegistryEntry,
        instruction: str,
        provided_slots: Mapping[str, int],
    ) -> str:
        text = instruction.casefold()
        ranked: list[tuple[int, int, str]] = []
        for template in entry.sentence_templates:
            names = template_slot_names(template)
            if any(name not in provided_slots for name in names):
                continue
            distinctive = template_distinctive_tokens(template)
            if distinctive and not all(token in text for token in distinctive):
                continue
            ranked.append((len(names), len(distinctive), template))
        if not ranked:
            for template in entry.sentence_templates:
                names = template_slot_names(template)
                if all(name in provided_slots for name in names):
                    ranked.append((len(names), 0, template))
        if not ranked:
            return entry.instruction_template
        ranked.sort(reverse=True)
        return ranked[0][2]

    def _registry_prompt(self, instruction: str) -> str:
        tasks = {
            task_id: {
                "sentence_templates": list(entry.sentence_templates),
                "template": entry.instruction_template,
                "slots": {slot: list(ids) for slot, ids in entry.slots.items()},
                "instances": dict(entry.instances),
            }
            for task_id, entry in self.registry.items()
        }
        return json.dumps({"instruction": instruction, "registry": tasks}, ensure_ascii=False)

    def _validate_selection(
        self,
        payload: Mapping[str, Any],
        instruction: str = "",
    ) -> EntitySelection:
        task_id = str(payload.get("task_id", ""))
        if task_id not in self.registry:
            raise ValueError(f"LLM 返回未知 task_id: {task_id!r}")
        entry = self.registry[task_id]
        raw_slots = payload.get("slots")
        if not isinstance(raw_slots, Mapping):
            raise ValueError("LLM 返回缺少 slots 对象")
        provided: dict[str, int] = {}
        for slot, raw_value in raw_slots.items():
            instance_id = int(raw_value)
            candidates = entry.slots.get(str(slot))
            if candidates is None or instance_id not in candidates:
                raise ValueError(
                    f"LLM 返回非法 instance id: {task_id}.{slot}={instance_id}, "
                    f"候选={list(candidates or ())}"
                )
            provided[str(slot)] = instance_id
        template = str(payload.get("template") or payload.get("instruction_template") or "")
        if template not in entry.sentence_templates:
            template = self._choose_template(entry, instruction, provided)
        slots: dict[str, int] = {}
        for slot in template_slot_names(template):
            if slot not in provided:
                raise ValueError(
                    f"LLM 返回缺少 template 所需 slot: {task_id}.{slot} template={template!r}"
                )
            slots[slot] = provided[slot]
        selected = {instance_id: entry.instances[instance_id] for instance_id in slots.values()}
        return EntitySelection(task_id, slots, selected, instruction_template=template)
