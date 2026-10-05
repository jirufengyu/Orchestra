# Mobile action policy and task progress evaluator

This OpenPI subset contains exactly two registered training configurations:

- `pi05_mobile_atomic_4task_short_horizon_memory_stride3`: action policy.
- `pi05_mobile_atomic_4task_progress_evaluator_300m_stride3`: independent progress/done evaluator.

Shared model classes and helper implementations are retained for compatibility;
no other training presets are registered.

## Installation

Use a separate Python 3.11 environment from the agent runtime. From this directory:

```bash
uv sync
uv run python scripts/install_transformers_patch.py
```

The bundled Transformers 4.53.2 modifications are required for the attention and
video-memory implementation. The installer patches only the active environment.
The LeRobot revision is pinned in `pyproject.toml`; its older dataset API is required.
The default Torch/JAX dependencies target CUDA. The first tokenizer use downloads
the public PaliGemma SentencePiece model (`gs://big_vision/paligemma_tokenizer.model`).
No private repository or original development checkout is required.

## Data and normalization

Datasets and trained weights are not bundled. Set `MOBILE_DATA_ROOT` to a directory
containing the following four local LeRobot datasets, in this order:

- `mobile_breakfast_preparation_seg`
- `mobile_number_ordering_seg`
- `mobile_place_fruit_bowl_seg`
- `mobile_sorting_object_seg`

Each dataset needs the LeRobot metadata, parquet data and referenced camera videos.
The input keys include `observation.images.cam_high`, `cam_left_wrist`,
`cam_right_wrist` (under the same image prefix), `observation.state`, `action`,
`observation.segmentation.cam_high`, `episode_index`, and `frame_index`.
`meta/episodes.jsonl` must also contain actor and subtask metadata used by
`seg_transforms.py` and `rmbench_progress_transforms.py` to construct active actor
conditioning and progress/done labels. Dataset indices and task prompts are supplied
by the data loader. The `anonymous/…` repo IDs are local identifiers, not hosted datasets.

```bash
export MOBILE_DATA_ROOT=/path/to/mobile_datasets
uv run python scripts/compute_norm_stats.py \
  --config-name pi05_mobile_atomic_4task_progress_evaluator_300m_stride3
uv run python scripts/train_progress_evaluator_pytorch.py \
  pi05_mobile_atomic_4task_progress_evaluator_300m_stride3 \
  --exp-name mobile_progress_evaluator
```

For multiple GPUs, launch the same training script with `torchrun --standalone
--nproc-per-node=N`. The original training settings are retained: batch size 20,
50,000 optimizer steps, checkpoint every 10,000 steps, random initialization,
300m Gemma variant, six frames with stride three, 11 progress bins, boundary margin
four, and progress plus focal completion losses. This is the backbone variant name,
not a claim about the complete model's parameter count. The original 16-dimensional
dual-arm layout is named `mobile` in this release. Tracking is disabled by default;
use `--wandb-enabled` to enable it explicitly.

## Serving

```bash
uv run python scripts/serve_progress_evaluator.py \
  --config pi05_mobile_atomic_4task_progress_evaluator_300m_stride3 \
  --checkpoint-dir /path/to/checkpoint/50000 \
  --port 8030
```

The server loads `model.safetensors` and exposes the OpenPI WebSocket policy RPC.
The inference adapter applies deterministic segmentation conditioning and maintains
recent camera history. It returns progress and completion predictions for use by
the agent's progress evaluator integration. State/action normalization is omitted
at inference because the evaluator consumes vision and language.
## Action policy training and serving

The action policy uses the same four datasets, shared normalization assets, 16-D
mobile dual-arm layout, and six-frame stride-three history. Its original settings
are preserved: `gemma_2b_lora` backbone, `gemma_300m_lora` action expert, 300 prompt
tokens, segmentation dropout 0.2, batch size 20, and 50,000 steps. It does not use
progress labels or a progress head.

For the original JAX checkpoint initialization and LoRA freeze filter:

```bash
uv run python scripts/train.py \
  pi05_mobile_atomic_4task_short_horizon_memory_stride3 \
  --exp-name mobile_action
```

This loads the public base parameters at
`gs://openpi-assets/checkpoints/pi05_base/params`. Run normalization computation
first as above; the two configs share its output directory.

The original PyTorch trainer is also included. Its initialization uses a separate
converted PyTorch checkpoint; it does not consume the JAX weight loader:

```bash
uv run python scripts/train_pytorch.py \
  pi05_mobile_atomic_4task_short_horizon_memory_stride3 \
  --exp-name mobile_action_pytorch \
  --pytorch-weight-path /path/to/pi05_base_pytorch
```

The PyTorch trainer follows its original parameter-training behavior; the JAX
LoRA freeze filter is not applied by that trainer.
Serve either checkpoint format with:

```bash
uv run python scripts/serve_policy.py \
  --policy.config pi05_mobile_atomic_4task_short_horizon_memory_stride3 \
  --policy.dir /path/to/action_checkpoint/50000 \
  --port 8020
```

The checkpoint must include its normalization assets. Serve the progress evaluator
on port 8030 with the separate command above.

## Validation and attribution

```bash
uv run pytest tests
```

Tests exercise progress/done losses, gradient propagation, temporal memory, and
release configuration constraints. They do not replace a full training run with
the four real datasets. Upstream OpenPI/Google/Hugging Face attribution and license
notices are preserved in `LICENSE`, `LICENSE_GEMMA.txt`, and source headers.
