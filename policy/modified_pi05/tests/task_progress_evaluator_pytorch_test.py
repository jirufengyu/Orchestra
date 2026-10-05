from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from openpi.models import model as _model
from openpi.models_pytorch.task_progress_evaluator_pytorch import TaskProgressEvaluator


class _FakeBackbone(nn.Module):
    def __init__(self, vlm_config, _action_expert_config, **_kwargs):
        super().__init__()
        width = vlm_config.width
        self.gemma_expert = None
        self.suffix_in_proj = nn.Identity()
        self.suffix_out_proj = nn.Identity()
        self.image_proj = nn.Linear(3, width)
        self.language_embedding = nn.Embedding(32, width)
        self.seen_memory_frames: list[torch.Tensor] = []
        self.paligemma = SimpleNamespace(
            config=SimpleNamespace(vision_config=SimpleNamespace(hidden_size=width)),
            language_model=SimpleNamespace(gradient_checkpointing=False),
            vision_tower=SimpleNamespace(gradient_checkpointing=False),
        )

    def embed_image(self, image):
        pooled = image.mean(dim=(-2, -1))
        return self.image_proj(pooled)[:, None, :]

    def embed_image_with_memory(self, frames, _frame_mask, **_kwargs):
        self.seen_memory_frames.append(frames.detach().clone())
        return self.embed_image(frames[:, -1])

    def embed_language_tokens(self, tokens):
        return self.language_embedding(tokens)

    def forward(self, *, inputs_embeds, **_kwargs):
        prefix, suffix = inputs_embeds
        suffix_out = suffix + prefix.mean(dim=1, keepdim=True)
        return [prefix, suffix_out], None


def _config():
    return SimpleNamespace(
        paligemma_variant="dummy",
        dtype="float32",
        num_progress_bins=11,
        progress_loss_weight=1.0,
        subtask_done_loss_weight=1.0,
        done_focal_gamma=2.0,
        done_focal_alpha=0.75,
        progress_head_dropout=0.0,
        progress_head_use_mlp=False,
        use_short_horizon_memory=True,
        use_head_only_vision=False,
        memory_num_frames=3,
        memory_temporal_interval=1,
        memory_drop_past_after_layer=None,
    )


def _observation() -> _model.Observation:
    bsize = 2
    images = {
        "base_0_rgb": torch.full((bsize, 3, 8, 8), 0.75),
        "left_wrist_0_rgb": torch.full((bsize, 3, 8, 8), 0.25),
        "right_wrist_0_rgb": torch.full((bsize, 3, 8, 8), -0.25),
    }
    memory = {
        key: torch.zeros((bsize, 3, 3, 8, 8))
        for key in images
    }
    return _model.Observation(
        images=images,
        image_masks={key: torch.ones(bsize, dtype=torch.bool) for key in images},
        state=torch.zeros((bsize, 32)),
        memory_frames=memory,
        memory_frame_mask=torch.ones((bsize, 3), dtype=torch.bool),
        tokenized_prompt=torch.tensor([[1, 2, 3], [3, 2, 1]]),
        tokenized_prompt_mask=torch.ones((bsize, 3), dtype=torch.bool),
        progress_bin=torch.tensor([0, 10]),
        subtask_done=torch.tensor([0.0, 1.0]),
        progress_mask=torch.ones(bsize),
    )


def test_evaluator_has_no_action_expert_and_backpropagates_both_losses() -> None:
    with mock.patch(
        "openpi.models_pytorch.task_progress_evaluator_pytorch.PaliGemmaWithExpertModel",
        _FakeBackbone,
    ):
        evaluator = TaskProgressEvaluator(_config())

    evaluator.eval()
    losses = evaluator(_observation(), return_loss=True)
    losses["total_loss"].backward()

    assert not evaluator.has_action_expert
    assert not any("action" in name for name, _ in evaluator.named_parameters())
    assert evaluator.progress_evaluator.mlp is None
    assert losses["progress_logits"].shape == (2, 11)
    assert losses["done_logit"].shape == (2,)
    assert evaluator.progress_query.grad is not None
    assert len(evaluator.backbone.seen_memory_frames) == 3
    torch.testing.assert_close(
        evaluator.backbone.seen_memory_frames[0][:, -1],
        torch.full((2, 3, 224, 224), 0.75),
    )


def test_progress_logits_decode_to_zero_through_one_hundred_percent() -> None:
    logits = torch.full((3, 11), -100.0)
    logits[0, 0] = 100.0
    logits[1, 5] = 100.0
    logits[2, 10] = 100.0

    progress = TaskProgressEvaluator.logits_to_progress_percent(logits)

    torch.testing.assert_close(progress, torch.tensor([0.0, 50.0, 100.0]))
