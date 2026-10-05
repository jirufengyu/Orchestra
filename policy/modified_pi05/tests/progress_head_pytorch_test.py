import torch

from openpi.models_pytorch.progress_head_pytorch import ProgressEvaluatorHead


def test_done_focal_loss_matches_manual_computation() -> None:
    head = ProgressEvaluatorHead(hidden_dim=4, num_progress_bins=3, dropout=0.0)
    hidden = torch.randn(4, 4)
    progress_bin = torch.tensor([0, 1, 2, 0])
    subtask_done = torch.tensor([0.0, 0.0, 1.0, 1.0])
    progress_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])

    losses = head.loss(
        hidden,
        progress_bin,
        subtask_done,
        progress_mask,
        done_focal_gamma=2.0,
        done_focal_alpha=0.75,
    )

    done_logit = losses["done_logit"]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(done_logit, subtask_done, reduction="none")
    p_t = torch.exp(-bce)
    alpha_t = 0.75 * subtask_done + 0.25 * (1.0 - subtask_done)
    expected = (alpha_t * (1.0 - p_t).pow(2.0) * bce * progress_mask).sum() / progress_mask.sum()

    torch.testing.assert_close(losses["done_loss"], expected)


def test_default_done_loss_remains_binary_cross_entropy() -> None:
    head = ProgressEvaluatorHead(hidden_dim=4, num_progress_bins=3, dropout=0.0)
    hidden = torch.randn(3, 4)
    progress_bin = torch.tensor([0, 1, 2])
    subtask_done = torch.tensor([0.0, 1.0, 0.0])
    progress_mask = torch.ones(3)

    losses = head.loss(hidden, progress_bin, subtask_done, progress_mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(losses["done_logit"], subtask_done)

    torch.testing.assert_close(losses["done_loss"], expected)
