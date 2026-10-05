"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_mobile_atomic_4task_short_horizon_memory_stride3 --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_mobile_atomic_4task_short_horizon_memory_stride3 --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import contextlib
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


class EMAModel:
    """Exponential Moving Average of model parameters.
    
    This class maintains a shadow copy of model parameters and updates them
    using exponential moving average. The EMA model can be placed on a different
    device to save memory on the training device.
    """
    
    def __init__(self, model, decay=0.999, device=None, update_after_step=0, update_every=1):
        """
        Args:
            model: The model to track with EMA
            decay: EMA decay rate (default: 0.999)
            device: Device to store EMA model (if None, uses same device as model)
            update_after_step: Start updating EMA after this many steps
            update_every: Update EMA every N steps
        """
        # Get the actual model (unwrap DDP if necessary)
        self.model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.num_updates = 0
        
        # Create EMA model as a deep copy
        self.ema_model = type(self.model)(self.model.config).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.eval()
        
        # Disable gradient computation for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
        logging.info(f"Initialized EMA model with decay={decay}, device={self.device}")
    
    @torch.no_grad()
    def update(self, step):
        """Update EMA parameters."""
        if step < self.update_after_step:
            return
        
        if step % self.update_every != 0:
            return
        
        self.num_updates += 1
        decay = self.decay
        
        # Get source model parameters (from training device)
        model_params = dict(self.model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())
        
        # Use CUDA streams for async transfer if on different devices
        use_async = (self.device != next(self.model.parameters()).device and 
                     self.device.type == 'cuda')
        
        if use_async:
            # Create non-blocking stream for async transfer
            stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(stream):
                for name, ema_param in ema_params.items():
                    if name in model_params:
                        model_param = model_params[name]
                        # Non-blocking async transfer
                        model_param_on_ema_device = model_param.data.to(
                            self.device, non_blocking=True
                        )
                        ema_param.data.mul_(decay).add_(
                            model_param_on_ema_device, alpha=1 - decay
                        )
            # Don't synchronize here - let it run async
        else:
            # Same device or CPU - direct update
            for name, ema_param in ema_params.items():
                if name in model_params:
                    model_param = model_params[name]
                    if self.device != model_param.device:
                        model_param_on_ema_device = model_param.data.to(self.device)
                        ema_param.data.mul_(decay).add_(
                            model_param_on_ema_device, alpha=1 - decay
                        )
                    else:
                        # Same device - no transfer needed
                        ema_param.data.mul_(decay).add_(
                            model_param.data, alpha=1 - decay
                        )
    
    def state_dict(self):
        """Return state dict of EMA model."""
        return {
            'ema_model': self.ema_model.state_dict(),
            'decay': self.decay,
            'num_updates': self.num_updates,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict into EMA model."""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.decay = state_dict.get('decay', self.decay)
        self.num_updates = state_dict.get('num_updates', 0)
        logging.info(f"Loaded EMA model state with {self.num_updates} updates")
    
    def get_model(self):
        """Get the EMA model."""
        return self.ema_model


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def get_ema_device(training_device, world_size):
    """Determine which device to use for EMA model.
    
    Strategy:
    - If training on CPU, use CPU for EMA
    - If single GPU training, check if there's another GPU available
    - If multi-GPU training, try to use a different GPU than the training device
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 1:
        # Only one GPU, have to share
        logging.info("Only 1 GPU available, EMA model will share the same GPU")
        return training_device
    
    # Multiple GPUs available
    if world_size >= num_gpus:
        # All GPUs are used for training, share with training device
        logging.info(f"All {num_gpus} GPUs used for training, EMA will share GPU")
        return training_device
    
    # Find an unused GPU for EMA
    training_gpu_id = training_device.index if training_device.type == "cuda" else 0
    
    # Try to use the next available GPU
    for gpu_id in range(num_gpus):
        if gpu_id >= world_size:
            ema_device = torch.device(f"cuda:{gpu_id}")
            logging.info(f"EMA model will use separate GPU: {ema_device}")
            return ema_device
    
    # Fallback to training device
    logging.info("No separate GPU available for EMA, sharing with training")
    return training_device


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, ema_model=None):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save EMA model if provided
        if ema_model is not None:
            logging.info(f"Saving EMA model at step {global_step}")
            ema_state = ema_model.state_dict()
            # Save EMA model weights using safetensors
            safetensors.torch.save_model(ema_model.get_model(), tmp_ckpt_dir / "ema_model.safetensors")
            # Save EMA metadata (decay, num_updates, etc.)
            ema_metadata = {
                'decay': ema_state['decay'],
                'num_updates': ema_state['num_updates'],
            }
            torch.save(ema_metadata, tmp_ckpt_dir / "ema_metadata.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, ema_model=None):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load EMA model if provided
        if ema_model is not None:
            ema_model_path = ckpt_dir / "ema_model.safetensors"
            ema_metadata_path = ckpt_dir / "ema_metadata.pt"
            
            if ema_model_path.exists() and ema_metadata_path.exists():
                logging.info("Loading EMA model state...")
                safetensors.torch.load_model(ema_model.get_model(), ema_model_path, device=str(ema_model.device))
                
                ema_metadata = torch.load(ema_metadata_path, map_location=ema_model.device, weights_only=False)
                ema_model.decay = ema_metadata['decay']
                ema_model.num_updates = ema_metadata['num_updates']
                del ema_metadata
                
                torch.cuda.empty_cache()
                gc.collect()
                logging.info(f"Loaded EMA model with {ema_model.num_updates} updates")
            else:
                logging.warning("EMA checkpoint not found, EMA model will be initialized from current model weights")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        missing_keys, unexpected_keys = safetensors.torch.load_model(
            model,
            model_path,
            strict=False,
            device=str(device),
        )
        allowed_missing_prefixes = (
            "keyframe_resampler.",
            "memory_cross_attn.",
            "progress_query",
            "progress_evaluator.",
            "ig_action_out_proj.",
        )
        disallowed_missing_keys = [
            key for key in missing_keys if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing_keys or unexpected_keys:
            raise RuntimeError(
                "Error(s) in loading fine-tune weights:\n"
                f"  Missing key(s): {disallowed_missing_keys}\n"
                f"  Unexpected key(s): {unexpected_keys}"
            )
        if missing_keys:
            logging.warning(
                "Initialized newly added parameters not found in fine-tune checkpoint: %s",
                missing_keys,
            )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=False,  # Enable for memory efficiency
            static_graph=False,#static_graph=world_size >= 8,  # Enable for 8+ GPUs
            broadcast_buffers=False,
        )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    # If action_expert_lr_multiplier is set, use different learning rates for different parts
    if config.action_expert_lr_multiplier is not None:
        # Get the actual model (unwrap DDP if necessary)
        actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        
        # Separate parameters into action expert and other parts
        action_expert_params = []
        other_params = []
        
        for name, param in actual_model.named_parameters():
            if "gemma_expert" in name:
                action_expert_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {
                "params": other_params,
                "lr": peak_lr,
                "name": "vlm_and_projections",
            },
            {
                "params": action_expert_params,
                "lr": peak_lr * config.action_expert_lr_multiplier,
                "name": "action_expert",
            },
        ]
        
        optim = torch.optim.AdamW(
            param_groups,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        
        if is_main:
            logging.info(
                f"Using different learning rates: VLM={peak_lr:.2e}, Action Expert={peak_lr * config.action_expert_lr_multiplier:.2e} (multiplier={config.action_expert_lr_multiplier})"
            )
            logging.info(f"VLM parameters: {len(other_params)}, Action Expert parameters: {len(action_expert_params)}")
    else:
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )

    # Initialize EMA model if enabled
    ema_model = None
    use_ema = getattr(config, 'use_ema', False)
    if use_ema and is_main:
        ema_decay = getattr(config, 'ema_decay', 0.9999)
        ema_update_after_step = getattr(config, 'ema_update_after_step', 0)
        ema_update_every = getattr(config, 'ema_update_every', 1)
        
        # Determine EMA device
        ema_device = get_ema_device(device, world_size)
        
        # Create EMA model
        ema_model = EMAModel(
            model=model,
            decay=ema_decay,
            device=ema_device,
            update_after_step=ema_update_after_step,
            update_every=ema_update_every,
        )
        
        logging.info(f"EMA enabled with decay={ema_decay}, update_after_step={ema_update_after_step}, update_every={ema_update_every}")

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, ema_model=ema_model)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    gradient_accumulation_steps = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"effective_optimizer_batch_size={config.batch_size * gradient_accumulation_steps}, "
            f"num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        if use_ema:
            logging.info(f"EMA enabled: decay={getattr(config, 'ema_decay', 0.9999)}, device={ema_model.device if ema_model else 'N/A'}")
        else:
            logging.info("EMA disabled")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    optim.zero_grad(set_to_none=True)
    accumulation_step = 0

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            if config.action_expert_lr_multiplier is not None:
                # Apply learning rate schedule to each parameter group with its own peak_lr
                for pg in optim.param_groups:
                    # Get the initial peak learning rate for this parameter group
                    if pg["name"] == "action_expert":
                        group_peak_lr = peak_lr * config.action_expert_lr_multiplier
                        group_end_lr = end_lr * config.action_expert_lr_multiplier
                    else:
                        group_peak_lr = peak_lr
                        group_end_lr = end_lr
                    
                    # Apply the same schedule but with group-specific learning rates
                    if global_step < warmup_steps:
                        init_lr = group_peak_lr / (warmup_steps + 1)
                        pg["lr"] = init_lr + (group_peak_lr - init_lr) * global_step / warmup_steps
                    else:
                        progress = min(1.0, (global_step - warmup_steps) / max(1, decay_steps - warmup_steps))
                        cos = 0.5 * (1 + np.cos(np.pi * progress))
                        pg["lr"] = group_end_lr + (group_peak_lr - group_end_lr) * cos
            else:
                for pg in optim.param_groups:
                    pg["lr"] = lr_schedule(global_step)

            should_sync = (accumulation_step + 1) % gradient_accumulation_steps == 0
            sync_context = (
                model.no_sync()
                if use_ddp
                and isinstance(model, torch.nn.parallel.DistributedDataParallel)
                and not should_sync
                else contextlib.nullcontext()
            )

            with sync_context:
                # Forward pass
                model_out = model(observation, actions, return_extra=True)
                if len(model_out) == 4:
                    losses, v_t, u_t, aux_losses = model_out
                else:
                    losses, v_t, u_t = model_out
                    aux_losses = None
                # Ensure losses is a tensor and handle different return types
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)

                model_loss = losses.mean()
                action_loss = torch.nn.functional.mse_loss(v_t, u_t, reduction="none").mean()
                
                # Compute DPO loss if enabled
                if config.use_dpo:
                    ### add contrastive dpo loss ###
                    a_pos = actions  # positive: real actions
                    a_neg = actions + config.dpo_noise_scale * torch.randn_like(actions)  # negative: noisy actions

                    B = actions.shape[0]
                    t = torch.rand(B, 1, 1, device=device)
                    eps = torch.randn_like(actions)

                    x_t_pos = t * eps + (1-t) * a_pos
                    x_t_neg = t * eps + (1-t) * a_neg

                    u_t_pos = eps - a_pos
                    u_t_neg = eps - a_neg

                    # Contrastive DPO: maximize margin between positive and negative
                    _, v_pos, *_ = model(observation, x_t_pos, return_extra=True)
                    _, v_neg, *_ = model(observation, x_t_neg, return_extra=True)

                    # Compute log probabilities (negative MSE)
                    logp_pos = -((v_pos - u_t_pos)**2).mean([1,2])
                    logp_neg = -((v_neg - u_t_neg)**2).mean([1,2])

                    # Contrastive loss: encourage logp_pos > logp_neg
                    beta = config.dpo_beta
                    dpo_loss = -torch.log(torch.sigmoid(beta * (logp_pos - logp_neg))).mean()
                    
                    # Backward pass
                    total_loss = model_loss + config.dpo_lambda * dpo_loss
                else:
                    dpo_loss = torch.tensor(0.0, device=device)
                    logp_pos = torch.tensor(0.0, device=device)
                    logp_neg = torch.tensor(0.0, device=device)
                    total_loss = model_loss
                
                # Scale the loss so accumulated gradients match a single large batch.
                (total_loss / gradient_accumulation_steps).backward()

            accumulation_step += 1
            if not should_sync:
                continue

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Update EMA model
            if ema_model is not None:
                ema_model.update(global_step)

            # Collect stats
            if is_main:
                info_dict = {
                    "loss": action_loss.item(),
                    "total_loss": total_loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }

                if aux_losses is not None:
                    for key in ("progress_loss", "progress_aux_loss"):
                        if key in aux_losses:
                            info_dict[key] = aux_losses[key].detach().mean().item()
                    if "done_loss" in aux_losses:
                        info_dict["progress_done_loss"] = aux_losses["done_loss"].detach().mean().item()
                
                # Add DPO metrics if enabled
                if config.use_dpo:
                    info_dict["dpo_loss"] = dpo_loss.item()
                    info_dict["logp_pos"] = logp_pos.mean().item()
                    info_dict["logp_neg"] = logp_neg.mean().item()
                
                # If using different learning rates, track them separately
                if config.action_expert_lr_multiplier is not None and len(optim.param_groups) > 1:
                    for pg in optim.param_groups:
                        info_dict[f"learning_rate_{pg['name']}"] = pg["lr"]
                
                infos.append(info_dict)

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_total_loss = sum(info["total_loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_progress_loss = None
                avg_progress_done_loss = None
                avg_progress_aux_loss = None
                if any("progress_loss" in info for info in infos):
                    vals = [info["progress_loss"] for info in infos if "progress_loss" in info]
                    avg_progress_loss = sum(vals) / len(vals)
                if any("progress_done_loss" in info for info in infos):
                    vals = [info["progress_done_loss"] for info in infos if "progress_done_loss" in info]
                    avg_progress_done_loss = sum(vals) / len(vals)
                if any("progress_aux_loss" in info for info in infos):
                    vals = [info["progress_aux_loss"] for info in infos if "progress_aux_loss" in info]
                    avg_progress_aux_loss = sum(vals) / len(vals)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                
                # Build log message
                if config.use_dpo:
                    avg_dpo_loss = sum(info["dpo_loss"] for info in infos) / len(infos)
                    avg_logp_pos = sum(info["logp_pos"] for info in infos) / len(infos)
                    avg_logp_neg = sum(info["logp_neg"] for info in infos) / len(infos)
                    log_msg = (
                        f"step={global_step} loss={avg_loss:.4f} total_loss={avg_total_loss:.4f} "
                        f"dpo_loss={avg_dpo_loss:.4f} logp_pos={avg_logp_pos:.4f} logp_neg={avg_logp_neg:.4f} "
                        f"lr={avg_lr:.2e}"
                    )
                else:
                    log_msg = (
                        f"step={global_step} loss={avg_loss:.4f} total_loss={avg_total_loss:.4f} lr={avg_lr:.2e}"
                    )
                if avg_progress_loss is not None:
                    log_msg += f" progress_loss={avg_progress_loss:.4f}"
                if avg_progress_done_loss is not None:
                    log_msg += f" progress_done_loss={avg_progress_done_loss:.4f}"
                if avg_progress_aux_loss is not None:
                    log_msg += f" progress_aux_loss={avg_progress_aux_loss:.4f}"
                
                if avg_grad_norm is not None:
                    log_msg += f" grad_norm={avg_grad_norm:.2f}"
                log_msg += f" time={elapsed:.1f}s"
                
                logging.info(log_msg)

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "total_loss": avg_total_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_progress_loss is not None:
                        log_payload["progress_loss"] = avg_progress_loss
                    if avg_progress_done_loss is not None:
                        log_payload["progress_done_loss"] = avg_progress_done_loss
                    if avg_progress_aux_loss is not None:
                        log_payload["progress_aux_loss"] = avg_progress_aux_loss
                    
                    # Add DPO metrics if enabled
                    if config.use_dpo:
                        log_payload["dpo_loss"] = avg_dpo_loss
                        log_payload["logp_pos"] = avg_logp_pos
                        log_payload["logp_neg"] = avg_logp_neg
                    
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    
                    # If using different learning rates, log them separately
                    if config.action_expert_lr_multiplier is not None and len(optim.param_groups) > 1:
                        for pg in optim.param_groups:
                            lr_key = f"learning_rate_{pg['name']}"
                            if lr_key in infos[0]:
                                avg_group_lr = sum(info[lr_key] for info in infos) / len(infos)
                                log_payload[lr_key] = avg_group_lr
                    
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config, ema_model=ema_model)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                postfix = {
                    "loss": f"{action_loss.item():.4f}",
                    "total": f"{total_loss.item():.4f}",
                    "lr": f"{optim.param_groups[0]['lr']:.2e}",
                    "step": global_step,
                }
                if aux_losses is not None:
                    if "progress_loss" in aux_losses:
                        postfix["progress"] = f"{aux_losses['progress_loss'].detach().mean().item():.4f}"
                    if "done_loss" in aux_losses:
                        postfix["progress_done"] = f"{aux_losses['done_loss'].detach().mean().item():.4f}"
                pbar.set_postfix(postfix)

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
