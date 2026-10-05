"""Mixture-of-Experts (MoE) Action Expert for PI0.

This module implements a task-conditioned MoE architecture for the action expert,
where different tasks are routed to different expert models. This supports:

1. Hard routing by task_id (each task uses a dedicated expert)
2. Incremental learning (add new experts without modifying existing ones)
3. Capability preservation (freeze existing experts when training new ones)
4. Optional shared expert (maintains base generalization across all tasks)

Architecture:
    ┌──────────────────────────────────┐
    │        PaliGemma (VLM)           │  ← Shared across all tasks
    │   (image + language encoder)     │
    └────────────┬─────────────────────┘
                 │
    ┌────────────┼─────────────────────┐
    │      MoE Action Expert           │
    │                                  │
    │  task_id → Router → Expert_i     │
    │                                  │
    │  ┌─────────┐ ┌─────────┐        │
    │  │Expert_0 │ │Expert_1 │ ...     │  ← Task-specific experts
    │  │(Task A) │ │(Task B) │         │
    │  └─────────┘ └─────────┘         │
    │                                  │
    │  ┌─────────────────────┐         │
    │  │  Shared Expert      │ (opt)   │  ← Always active (default/fallback)
    │  └─────────────────────┘         │
    └──────────────────────────────────┘

Usage:
    # 1. Create MoE expert during model initialization
    moe_expert = MoEActionExpert(
        expert_config=gemma_config,
        task_names=["pick_place", "pour_water", "open_drawer"],
        use_shared_expert=True,
    )

    # 2. Route to a specific task expert
    expert = moe_expert.get_expert_for_task("pick_place")

    # 3. Add a new task (incremental learning)
    moe_expert.add_expert("new_task", init_from="shared")  # init from shared expert
    moe_expert.freeze_all_except("new_task")  # only train the new expert

    # 4. Get expert info
    print(moe_expert.get_expert_info())
"""

import copy
import logging
from typing import Optional

import torch
from torch import nn
from transformers import GemmaForCausalLM


class MoEActionExpert(nn.Module):
    """Mixture-of-Experts Action Expert with task-based hard routing.

    Each task is assigned a dedicated Gemma expert model. An optional shared expert
    serves as default/fallback and as weight initialization source for new task experts.

    Args:
        expert_config: HuggingFace GemmaConfig for creating expert models.
        task_names: Initial list of task names.
        use_shared_expert: If True, maintain a shared expert alongside task experts.
    """

    def __init__(
        self,
        expert_config,
        task_names: list[str],
        use_shared_expert: bool = True,
    ):
        super().__init__()
        self.expert_config = expert_config
        self.task_names = list(task_names)
        self.use_shared_expert = use_shared_expert

        # ── Shared expert (default/fallback, also serves as init source) ──
        if use_shared_expert:
            self.shared_expert = GemmaForCausalLM(config=expert_config)
            self.shared_expert.model.embed_tokens = None
        else:
            self.shared_expert = None

        # ── Task-specific experts ──
        self.task_experts = nn.ModuleDict()
        for name in task_names:
            expert = GemmaForCausalLM(config=expert_config)
            expert.model.embed_tokens = None
            self.task_experts[self._safe_key(name)] = expert

        # Task name → safe key mapping
        self._task_key_map: dict[str, str] = {
            name: self._safe_key(name) for name in task_names
        }

        logging.info(
            f"Initialized MoEActionExpert with {len(task_names)} task experts "
            f"(shared_expert={use_shared_expert}): {task_names}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Key helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_key(name: str) -> str:
        """Convert a task name to a valid nn.ModuleDict key."""
        return name.replace(".", "_dot_").replace("/", "_sl_").replace(" ", "_sp_")

    @property
    def model(self):
        """Compatibility property – returns the underlying GemmaModel of the
        default expert so that code referencing ``expert.model.layers`` etc.
        keeps working when no specific task_id is provided.
        """
        if self.shared_expert is not None:
            return self.shared_expert.model
        first_key = next(iter(self.task_experts))
        return self.task_experts[first_key].model

    # ──────────────────────────────────────────────────────────────────────
    # Routing
    # ──────────────────────────────────────────────────────────────────────

    def get_expert_for_task(self, task_id) -> GemmaForCausalLM:
        """Return the ``GemmaForCausalLM`` for *task_id*.

        Args:
            task_id: One of:
                - ``str``: task name
                - ``int``: index into ``self.task_names``
                - ``None``: returns the shared expert (or the first task expert)
        """
        if task_id is None:
            if self.shared_expert is not None:
                return self.shared_expert
            return next(iter(self.task_experts.values()))

        if isinstance(task_id, int):
            if task_id >= len(self.task_names):
                raise ValueError(
                    f"task_id {task_id} out of range (max {len(self.task_names) - 1})"
                )
            name = self.task_names[task_id]
        elif isinstance(task_id, str):
            name = task_id
        else:
            raise TypeError(f"task_id must be str | int | None, got {type(task_id)}")

        safe_key = self._task_key_map.get(name)
        if safe_key is None or safe_key not in self.task_experts:
            raise ValueError(
                f"No expert for task '{name}'. Available: {self.task_names}"
            )
        return self.task_experts[safe_key]

    def get_active_expert_model(self, task_id):
        """Return the underlying ``GemmaModel`` (not ``GemmaForCausalLM``)
        for the requested task.  This is the object whose ``.layers`` are
        used in the layer-wise interleaved computation.
        """
        return self.get_expert_for_task(task_id).model

    # ──────────────────────────────────────────────────────────────────────
    # Incremental-learning helpers
    # ──────────────────────────────────────────────────────────────────────

    def add_expert(
        self,
        task_name: str,
        init_from: Optional[str] = None,
    ) -> None:
        """Add a new task expert.

        Args:
            task_name: Name of the new task.
            init_from:
                - ``None`` – random initialisation
                - ``"shared"`` – deep-copy from the shared expert
                - ``<existing_task_name>`` – deep-copy from that task's expert
        """
        safe_key = self._safe_key(task_name)
        if safe_key in self.task_experts:
            raise ValueError(f"Expert for task '{task_name}' already exists")

        if init_from is not None:
            if init_from == "shared":
                if self.shared_expert is None:
                    raise ValueError("No shared expert to initialise from")
                source = self.shared_expert
            elif init_from in self._task_key_map:
                source_key = self._task_key_map[init_from]
                source = self.task_experts[source_key]
            else:
                raise ValueError(
                    f"Cannot init from '{init_from}'. "
                    f"Available: ['shared'] + {self.task_names}"
                )
            new_expert = copy.deepcopy(source)
            logging.info(f"Initialised expert for '{task_name}' from '{init_from}'")
        else:
            new_expert = GemmaForCausalLM(config=self.expert_config)
            new_expert.model.embed_tokens = None
            logging.info(f"Initialised expert for '{task_name}' with random weights")

        self.task_experts[safe_key] = new_expert
        self.task_names.append(task_name)
        self._task_key_map[task_name] = safe_key

    def freeze_all_except(self, task_name: Optional[str] = None) -> None:
        """Freeze every expert *except* the one for *task_name*.

        This is the primary mechanism for **incremental learning**: call
        ``freeze_all_except("new_task")`` so that only the new task's expert
        receives gradient updates.

        Args:
            task_name: The task whose expert should remain trainable.
                       Pass ``None`` to freeze everything (eval mode).
        """
        if self.shared_expert is not None:
            for param in self.shared_expert.parameters():
                param.requires_grad = False
            logging.info("Froze shared expert")

        for name in self.task_names:
            safe_key = self._task_key_map[name]
            expert = self.task_experts[safe_key]
            should_freeze = name != task_name
            for param in expert.parameters():
                param.requires_grad = not should_freeze
            status = "frozen" if should_freeze else "trainable"
            logging.info(f"Expert '{name}': {status}")

    def unfreeze_all(self) -> None:
        """Unfreeze every expert (for joint/full fine-tuning)."""
        if self.shared_expert is not None:
            for param in self.shared_expert.parameters():
                param.requires_grad = True
        for expert in self.task_experts.values():
            for param in expert.parameters():
                param.requires_grad = True
        logging.info("Unfroze all experts")

    # ──────────────────────────────────────────────────────────────────────
    # Inspection
    # ──────────────────────────────────────────────────────────────────────

    def get_expert_info(self) -> dict:
        """Return a dict describing every expert (param counts, trainable status, …)."""
        info: dict = {
            "num_task_experts": len(self.task_names),
            "task_names": list(self.task_names),
            "use_shared_expert": self.use_shared_expert,
        }
        if self.shared_expert is not None:
            total = sum(p.numel() for p in self.shared_expert.parameters())
            trainable = sum(
                p.numel() for p in self.shared_expert.parameters() if p.requires_grad
            )
            info["shared_expert"] = {
                "total_params": total,
                "trainable_params": trainable,
            }
        for name in self.task_names:
            safe_key = self._task_key_map[name]
            expert = self.task_experts[safe_key]
            total = sum(p.numel() for p in expert.parameters())
            trainable = sum(
                p.numel() for p in expert.parameters() if p.requires_grad
            )
            info[f"expert_{name}"] = {
                "total_params": total,
                "trainable_params": trainable,
            }
        return info
