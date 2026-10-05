"""Progress / subtask-done heads for joint pi05 training."""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812


class ProgressEvaluatorHead(nn.Module):
    """VF-style progress bins in [-1, 0] + binary subtask-done."""

    def __init__(
        self,
        hidden_dim: int,
        num_progress_bins: int = 101,
        dropout: float = 0.1,
        *,
        use_mlp: bool = True,
    ):
        super().__init__()
        self.num_progress_bins = num_progress_bins
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            if use_mlp
            else None
        )
        self.progress_head = nn.Linear(hidden_dim, num_progress_bins)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.dim() == 3:
            hidden = hidden.squeeze(1)
        h = self.norm(hidden)
        if self.mlp is not None:
            h = h + self.mlp(h)
        return self.progress_head(h), self.done_head(h).squeeze(-1)

    @staticmethod
    def logits_to_progress(logits: Tensor, num_bins: int) -> Tensor:
        probs = F.softmax(logits, dim=-1)
        bins = torch.linspace(-1.0, 0.0, num_bins, device=logits.device, dtype=probs.dtype)
        return (probs * bins.unsqueeze(0)).sum(dim=-1)

    def loss(
        self,
        hidden: Tensor,
        progress_bin: Tensor,
        subtask_done: Tensor,
        progress_mask: Tensor,
        *,
        done_focal_gamma: float = 0.0,
        done_focal_alpha: float | None = None,
    ) -> dict[str, Tensor]:
        progress_logits, done_logit = self.forward(hidden)
        mask = progress_mask.float()
        denom = torch.clamp(mask.sum(), min=1.0)

        progress_loss_all = F.cross_entropy(progress_logits, progress_bin.long(), reduction="none")
        progress_loss = (progress_loss_all * mask).sum() / denom

        done_target = subtask_done.float()
        done_bce = F.binary_cross_entropy_with_logits(done_logit, done_target, reduction="none")
        p_t = torch.exp(-done_bce)
        done_loss_all = (1.0 - p_t).pow(done_focal_gamma) * done_bce
        if done_focal_alpha is not None:
            alpha_t = done_focal_alpha * done_target + (1.0 - done_focal_alpha) * (1.0 - done_target)
            done_loss_all = alpha_t * done_loss_all
        done_loss = (done_loss_all * mask).sum() / denom

        return {
            "progress_loss": progress_loss,
            "done_loss": done_loss,
            "progress_logits": progress_logits,
            "done_logit": done_logit,
        }
