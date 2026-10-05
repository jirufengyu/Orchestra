"""Train the standalone task progress/done evaluator with PyTorch DDP."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import os
import pathlib
import random
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import tqdm
import wandb

from openpi.models_pytorch.task_progress_evaluator_pytorch import TaskProgressEvaluator
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _setup_ddp() -> tuple[bool, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def _set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def _prepare_checkpoint_dir(config: _config.TrainConfig, *, is_main: bool) -> None:
    if is_main:
        if config.checkpoint_dir.exists() and config.overwrite and not config.resume:
            shutil.rmtree(config.checkpoint_dir)
        if (
            config.checkpoint_dir.exists()
            and not config.resume
            and not config.overwrite
            and any(config.checkpoint_dir.iterdir())
        ):
            raise FileExistsError(
                f"Checkpoint directory {config.checkpoint_dir} is not empty. "
                "Use --overwrite or --resume."
            )
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()


def _save_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: _config.TrainConfig,
    data_config,
    *,
    is_main: bool,
) -> None:
    if not is_main:
        return
    should_save = step > 0 and step % config.save_interval == 0
    should_save = should_save or step == config.num_train_steps
    if not should_save:
        return

    final_dir = config.checkpoint_dir / str(step)
    tmp_dir = config.checkpoint_dir / f"tmp_{step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    unwrapped = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.save_model(unwrapped, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(
        {
            "global_step": step,
            "config": dataclasses.asdict(config),
            "model_type": "task_progress_evaluator",
        },
        tmp_dir / "metadata.pt",
    )
    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_dir / "assets" / data_config.asset_id, data_config.norm_stats)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    logging.info("Saved evaluator checkpoint to %s", final_dir)


def _load_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: pathlib.Path,
    device: torch.device,
) -> int:
    steps = sorted(int(path.name) for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"No evaluator checkpoints found in {checkpoint_dir}")
    latest_dir = checkpoint_dir / str(steps[-1])
    unwrapped = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.load_model(unwrapped, latest_dir / "model.safetensors", device=str(device))
    optimizer.load_state_dict(
        torch.load(latest_dir / "optimizer.pt", map_location=device, weights_only=False)
    )
    metadata = torch.load(latest_dir / "metadata.pt", map_location="cpu", weights_only=False)
    step = int(metadata.get("global_step", steps[-1]))
    logging.info("Resumed evaluator from %s at step %d", latest_dir, step)
    return step


def _learning_rate(config: _config.TrainConfig, step: int) -> float:
    warmup = int(config.lr_schedule.warmup_steps)
    peak = float(config.lr_schedule.peak_lr)
    decay_steps = int(config.lr_schedule.decay_steps)
    end = float(config.lr_schedule.decay_lr)
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, decay_steps - warmup))
    return end + (peak - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _masked_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool)
    if not torch.any(valid):
        return torch.zeros((), device=logits.device)
    return (torch.argmax(logits[valid], dim=-1) == target[valid].long()).float().mean()


def train(config: _config.TrainConfig) -> None:
    use_ddp, local_rank, device = _setup_ddp()
    rank = dist.get_rank() if use_ddp else 0
    is_main = rank == 0
    _set_seed(config.seed, rank)
    _prepare_checkpoint_dir(config, is_main=is_main)

    if config.pytorch_weight_path is not None:
        raise ValueError(
            "This evaluator configuration is intentionally random-initialized. "
            "Use --resume to load an evaluator checkpoint."
        )

    if is_main:
        if config.wandb_enabled:
            if config.resume:
                run_id = (config.checkpoint_dir / "wandb_id.txt").read_text().strip()
                wandb.init(id=run_id, resume="must", project=config.project_name)
            else:
                wandb.init(
                    name=config.exp_name,
                    project=config.project_name,
                    config=dataclasses.asdict(config),
                )
                (config.checkpoint_dir / "wandb_id.txt").write_text(wandb.run.id)
        else:
            wandb.init(mode="disabled")

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    data_config = loader.data_config()

    object.__setattr__(config.model, "dtype", config.pytorch_training_precision)
    model = TaskProgressEvaluator(config.model).to(device)
    model.gradient_checkpointing_enable()
    if model.has_action_expert:
        raise RuntimeError("Standalone evaluator unexpectedly contains an action expert.")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    global_step = _load_checkpoint(model, optimizer, config.checkpoint_dir, device) if config.resume else 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation_steps = max(1, int(config.gradient_accumulation_steps))
    accumulation_index = 0
    metrics: list[dict[str, float]] = []
    last_log_time = time.time()
    progress_bar = tqdm.tqdm(
        total=config.num_train_steps,
        initial=global_step,
        desc="Progress evaluator",
        disable=not is_main,
    )

    while global_step < config.num_train_steps:
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step)
        for batch_observation, _actions in loader:
            if global_step >= config.num_train_steps:
                break
            observation = jax.tree.map(lambda value: value.to(device), batch_observation)
            lr = _learning_rate(config, global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            should_sync = (accumulation_index + 1) % accumulation_steps == 0
            sync_context = (
                model.no_sync()
                if use_ddp
                and isinstance(model, torch.nn.parallel.DistributedDataParallel)
                and not should_sync
                else contextlib.nullcontext()
            )
            with sync_context:
                losses = model(observation, return_loss=True)
                (losses["total_loss"] / accumulation_steps).backward()
            accumulation_index += 1
            if not should_sync:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.optimizer.clip_gradient_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_index = 0
            global_step += 1

            if is_main:
                mask = observation.progress_mask
                progress_acc = _masked_accuracy(losses["progress_logits"], observation.progress_bin, mask)
                done_pred = (torch.sigmoid(losses["done_logit"]) >= 0.5).to(observation.subtask_done.dtype)
                valid = mask.to(dtype=torch.bool)
                done_acc = (
                    (done_pred[valid] == observation.subtask_done[valid]).float().mean()
                    if torch.any(valid)
                    else torch.zeros((), device=device)
                )
                metrics.append(
                    {
                        "total_loss": float(losses["total_loss"].detach()),
                        "progress_loss": float(losses["progress_loss"].detach()),
                        "done_loss": float(losses["done_loss"].detach()),
                        "progress_accuracy": float(progress_acc.detach()),
                        "done_accuracy": float(done_acc.detach()),
                        "grad_norm": float(grad_norm),
                        "lr": lr,
                    }
                )

            if is_main and global_step % config.log_interval == 0:
                averaged = {
                    key: sum(item[key] for item in metrics) / len(metrics)
                    for key in metrics[0]
                }
                elapsed = time.time() - last_log_time
                logging.info(
                    "step=%d total=%.4f progress=%.4f done=%.4f progress_acc=%.3f "
                    "done_acc=%.3f lr=%.2e grad=%.2f time=%.1fs",
                    global_step,
                    averaged["total_loss"],
                    averaged["progress_loss"],
                    averaged["done_loss"],
                    averaged["progress_accuracy"],
                    averaged["done_accuracy"],
                    averaged["lr"],
                    averaged["grad_norm"],
                    elapsed,
                )
                if config.wandb_enabled:
                    wandb.log(averaged, step=global_step)
                metrics.clear()
                last_log_time = time.time()

            _save_checkpoint(
                model,
                optimizer,
                global_step,
                config,
                data_config,
                is_main=is_main,
            )
            if progress_bar is not None:
                progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    _setup_logging()
    train(_config.cli())


if __name__ == "__main__":
    main()
