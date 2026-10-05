"""Standalone progress/done evaluator without an action expert."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch import nn

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.memory_video_siglip as _memory_video_siglip
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS
from openpi.models_pytorch.progress_head_pytorch import ProgressEvaluatorHead


class TaskProgressEvaluator(nn.Module):
    """Predict task progress and completion from vision, language, and recent frames."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_progress_bins = int(getattr(config, "num_progress_bins", 11))
        self.progress_loss_weight = float(getattr(config, "progress_loss_weight", 1.0))
        self.subtask_done_loss_weight = float(getattr(config, "subtask_done_loss_weight", 1.0))
        self.done_focal_gamma = float(getattr(config, "done_focal_gamma", 2.0))
        done_focal_alpha = getattr(config, "done_focal_alpha", 0.75)
        self.done_focal_alpha = float(done_focal_alpha) if done_focal_alpha is not None else None
        self.use_short_horizon_memory = bool(getattr(config, "use_short_horizon_memory", False))
        self.use_head_only_vision = bool(getattr(config, "use_head_only_vision", False))

        backbone_config = _gemma.get_config(config.paligemma_variant)
        self.backbone = PaliGemmaWithExpertModel(
            backbone_config,
            backbone_config,
            use_adarms=[False, False],
            use_action_expert=False,
            precision=config.dtype,
        )
        # Query and backbone widths are identical, so no learned suffix bridge is needed.
        self.backbone.suffix_in_proj = nn.Identity()
        self.backbone.suffix_out_proj = nn.Identity()
        if self.backbone.gemma_expert is not None:
            raise RuntimeError("TaskProgressEvaluator must not create an action expert.")

        self.progress_query = nn.Parameter(torch.randn(1, 1, backbone_config.width) * 0.02)
        self.progress_evaluator = ProgressEvaluatorHead(
            backbone_config.width,
            num_progress_bins=self.num_progress_bins,
            dropout=float(getattr(config, "progress_head_dropout", 0.1)),
            use_mlp=bool(getattr(config, "progress_head_use_mlp", True)),
        )

        if self.use_short_horizon_memory:
            vision_width = int(self.backbone.paligemma.config.vision_config.hidden_size)
            self.register_buffer(
                "memory_temporal_pe",
                _memory_video_siglip.build_sinusoidal_temporal_pe(
                    int(getattr(config, "memory_num_frames", 6)),
                    vision_width,
                ),
                persistent=False,
            )
            self.memory_temporal_interval = int(getattr(config, "memory_temporal_interval", 4))
            self.memory_drop_past_after_layer = getattr(config, "memory_drop_past_after_layer", None)

        torch.set_float32_matmul_precision("high")

    @property
    def has_action_expert(self) -> bool:
        return self.backbone.gemma_expert is not None

    def gradient_checkpointing_enable(self) -> None:
        self.backbone.paligemma.language_model.gradient_checkpointing = True
        self.backbone.paligemma.vision_tower.gradient_checkpointing = True

    @staticmethod
    def _prepare_attention_masks_4d(att_2d_masks: Tensor) -> Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    @staticmethod
    def _replace_current_memory_frame(memory_frames: Tensor, current_image: Tensor) -> Tensor:
        """Put the seg-conditioned current image into the temporal window."""
        if memory_frames.ndim != 5:
            raise ValueError(f"Expected memory frames [B,K,C,H,W], got {memory_frames.shape}")
        if current_image.ndim != 4:
            raise ValueError(f"Expected current image [B,C,H,W], got {current_image.shape}")
        if memory_frames.shape[2:] != current_image.shape[1:]:
            raise ValueError(
                "Current image and memory frame shapes differ: "
                f"{current_image.shape[1:]} vs {memory_frames.shape[2:]}"
            )
        frames = memory_frames.clone()
        frames[:, -1] = current_image
        return frames

    def _preprocess_observation(self, observation, *, train: bool):
        image_keys = ("base_0_rgb",) if self.use_head_only_vision else IMAGE_KEYS
        return _preprocessing.preprocess_observation_pytorch(
            observation,
            train=train,
            image_keys=image_keys,
        )

    def _embed_image(
        self,
        image: Tensor,
        memory_frames: Tensor | None,
        memory_frame_mask: Tensor | None,
    ) -> Tensor:
        if self.use_short_horizon_memory and memory_frames is not None:
            frames = self._replace_current_memory_frame(memory_frames, image)
            return self.backbone.embed_image_with_memory(
                frames,
                memory_frame_mask,
                temporal_pe=self.memory_temporal_pe,
                temporal_interval=self.memory_temporal_interval,
                drop_past_after_layer=self.memory_drop_past_after_layer,
            )
        return self.backbone.embed_image(image)

    def embed_prefix(self, observation) -> tuple[Tensor, Tensor, Tensor]:
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        att_masks: list[bool] = []
        memory_frames_by_key = observation.memory_frames if self.use_short_horizon_memory else None

        for key, image in observation.images.items():
            memory_frames = None
            if isinstance(memory_frames_by_key, dict):
                memory_frames = memory_frames_by_key.get(key)
            elif memory_frames_by_key is not None and key == "base_0_rgb":
                memory_frames = memory_frames_by_key

            image_emb = self._embed_image(image, memory_frames, observation.memory_frame_mask)
            bsize, num_tokens = image_emb.shape[:2]
            embs.append(image_emb)
            pad_masks.append(observation.image_masks[key][:, None].expand(bsize, num_tokens))
            att_masks.extend([False] * num_tokens)

        language_emb = self.backbone.embed_language_tokens(observation.tokenized_prompt)
        language_emb = language_emb * math.sqrt(language_emb.shape[-1])
        embs.append(language_emb)
        pad_masks.append(observation.tokenized_prompt_mask)
        att_masks.extend([False] * language_emb.shape[1])

        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1).to(dtype=torch.bool)
        prefix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=prefix_embs.device)
        prefix_att_masks = prefix_att_masks[None, :].expand(prefix_embs.shape[0], -1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def encode(self, observation, *, train: bool | None = None) -> Tensor:
        if train is None:
            train = self.training
        processed = self._preprocess_observation(observation, train=train)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(processed)

        bsize = prefix_embs.shape[0]
        query = self.progress_query.expand(bsize, -1, -1).to(dtype=prefix_embs.dtype)
        query_pad_mask = torch.ones((bsize, 1), dtype=torch.bool, device=prefix_embs.device)
        query_att_mask = torch.ones((bsize, 1), dtype=torch.bool, device=prefix_embs.device)
        pad_masks = torch.cat([prefix_pad_masks, query_pad_mask], dim=1)
        att_masks = torch.cat([prefix_att_masks, query_att_mask], dim=1)
        attention_mask = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        outputs, _ = self.backbone(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[prefix_embs, query],
            use_cache=False,
            adarms_cond=[None, None],
        )
        return outputs[1][:, -1, :].to(dtype=torch.float32)

    def forward(self, observation, *, return_loss: bool = False) -> dict[str, Tensor]:
        hidden = self.encode(observation)
        if return_loss:
            if (
                observation.progress_bin is None
                or observation.subtask_done is None
                or observation.progress_mask is None
            ):
                raise ValueError("progress_bin, subtask_done, and progress_mask are required for evaluator training.")
            losses = self.progress_evaluator.loss(
                hidden,
                observation.progress_bin,
                observation.subtask_done,
                observation.progress_mask,
                done_focal_gamma=self.done_focal_gamma,
                done_focal_alpha=self.done_focal_alpha,
            )
            total_loss = (
                self.progress_loss_weight * losses["progress_loss"]
                + self.subtask_done_loss_weight * losses["done_loss"]
            )
            return {
                **losses,
                "total_loss": total_loss,
            }

        progress_logits, done_logit = self.progress_evaluator(hidden)
        return {
            "progress_logits": progress_logits,
            "done_logit": done_logit,
        }

    def compute_loss(self, observation) -> dict[str, Tensor]:
        return self.forward(observation, return_loss=True)

    @staticmethod
    def logits_to_progress_percent(logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        points = torch.linspace(0.0, 100.0, logits.shape[-1], device=logits.device, dtype=probs.dtype)
        return (probs * points[None, :]).sum(dim=-1)

    @torch.no_grad()
    def predict(self, observation) -> dict[str, Tensor]:
        outputs = self.forward(observation)
        return {
            **outputs,
            "progress_bin": torch.argmax(outputs["progress_logits"], dim=-1),
            "progress_percent": self.logits_to_progress_percent(outputs["progress_logits"]),
            "done_prob": torch.sigmoid(outputs["done_logit"]),
        }
