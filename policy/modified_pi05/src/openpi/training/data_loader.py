from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class MultiLeRobotDatasetKeepFeatures(lerobot_dataset.MultiLeRobotDataset):
    """MultiLeRobotDataset that retains selected non-shared features.

    Upstream MultiLeRobotDataset drops any feature not present in *all* sub-datasets.
    This subclass keeps ``keep_features`` even when only some datasets provide them.
    Samples from datasets without those keys simply omit them.
    """

    def __init__(self, *args, keep_features: Sequence[str] = (), **kwargs):
        super().__init__(*args, **kwargs)
        keep = {str(k) for k in keep_features if k}
        if keep:
            retained = keep & set(self.disabled_features)
            if retained:
                logging.info(
                    "Keeping optional MultiLeRobot features that are not in all datasets: %s",
                    sorted(retained),
                )
            self.disabled_features -= keep


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def _validate_sampling_weights(weights: list[float], *, expected_len: int) -> torch.Tensor:
    """Validate sampling weights and return normalized probabilities.

    Weights can be any positive numbers; they are automatically normalized to sum to 1.
    """
    if len(weights) != expected_len:
        raise ValueError(f"lerobot_sampling_weights length ({len(weights)}) must match datasets ({expected_len})")
    w = torch.tensor(weights, dtype=torch.float64)
    if torch.any(~torch.isfinite(w)):
        raise ValueError(f"lerobot_sampling_weights must be finite, got {weights}")
    if torch.any(w < 0):
        raise ValueError(f"lerobot_sampling_weights must be non-negative, got {weights}")
    s = float(w.sum())
    if s <= 0:
        raise ValueError(f"lerobot_sampling_weights must sum to > 0, got {weights}")
    # Normalize to probabilities
    return (w / s).to(dtype=torch.float64)


class _MultiDatasetProbSampler(torch.utils.data.Sampler[int]):
    """Sample indices from a concatenated multi-dataset with per-dataset probabilities.

    This is designed for `lerobot_dataset.MultiLeRobotDataset` (or any concatenation-like dataset).
    We sample a dataset id according to provided probabilities, then uniformly sample an index within
    that dataset, and finally convert to a global index.
    """

    def __init__(
        self,
        dataset_lengths: list[int],
        *,
        probs: list[float],
        num_samples: int,
        seed: int = 0,
    ) -> None:
        if len(dataset_lengths) == 0:
            raise ValueError("dataset_lengths must be non-empty")
        if any(l <= 0 for l in dataset_lengths):
            raise ValueError(f"All dataset lengths must be > 0, got {dataset_lengths}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        self._prob = _validate_sampling_weights(probs, expected_len=len(dataset_lengths))

        self._lengths = dataset_lengths
        self._prefix = np.cumsum([0, *dataset_lengths]).tolist()
        self._num_samples = num_samples
        self._seed = seed
        self._epoch = 0

    def _draw_one(self, gen: torch.Generator) -> int:
        ds_idx = int(torch.multinomial(self._prob, 1, replacement=True, generator=gen).item())
        local = int(torch.randint(0, self._lengths[ds_idx], (1,), generator=gen).item())
        return self._prefix[ds_idx] + local

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self._seed + self._epoch)
        self._epoch += 1
        for _ in range(self._num_samples):
            yield self._draw_one(gen)

    def __len__(self) -> int:
        return self._num_samples


class _DistributedMultiDatasetProbSampler(torch.utils.data.Sampler[int]):
    """DDP-friendly sampler for multi-dataset probability sampling.

    Each rank gets a disjoint slice of a single global sampling stream by interleaving:
    rank r takes samples at positions r, r+num_replicas, ...
    """

    def __init__(
        self,
        dataset_lengths: list[int],
        *,
        probs: list[float],
        num_samples_total: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}")
        if not (0 <= rank < num_replicas):
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if len(dataset_lengths) == 0:
            raise ValueError("dataset_lengths must be non-empty")
        if any(l <= 0 for l in dataset_lengths):
            raise ValueError(f"All dataset lengths must be > 0, got {dataset_lengths}")
        if num_samples_total <= 0:
            raise ValueError(f"num_samples_total must be > 0, got {num_samples_total}")

        self._prob = _validate_sampling_weights(probs, expected_len=len(dataset_lengths))
        self._lengths = dataset_lengths
        self._prefix = np.cumsum([0, *dataset_lengths]).tolist()

        self._num_replicas = num_replicas
        self._rank = rank
        self._seed = seed
        self._epoch = 0
        self._drop_last = drop_last

        if drop_last:
            self._num_samples = num_samples_total // num_replicas
        else:
            self._num_samples = int(np.ceil(num_samples_total / num_replicas))
        if self._num_samples <= 0:
            raise ValueError(
                f"Per-rank num_samples computed as {self._num_samples}. "
                f"Check num_samples_total={num_samples_total} and world_size={num_replicas}."
            )

    def _draw_one(self, gen: torch.Generator) -> int:
        ds_idx = int(torch.multinomial(self._prob, 1, replacement=True, generator=gen).item())
        local = int(torch.randint(0, self._lengths[ds_idx], (1,), generator=gen).item())
        return self._prefix[ds_idx] + local

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self._seed + self._epoch)
        self._epoch += 1

        # Burn `rank` samples so each rank starts at its offset in the global stream.
        for _ in range(self._rank):
            _ = self._draw_one(gen)

        for _ in range(self._num_samples):
            yield self._draw_one(gen)
            # Burn samples for other ranks to keep streams disjoint.
            for _ in range(self._num_replicas - 1):
                _ = self._draw_one(gen)

    def __len__(self) -> int:
        return self._num_samples

class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _resolve_lerobot_dataset_root(data_config: _config.DataConfig, repo_ids: list[str]) -> str | None:
    """Infer a shared LeRobotDataset root from config-local dataset paths."""
    roots = tuple(str(root) for root in getattr(data_config, "lerobot_roots", ()) or ())
    if not roots:
        return None
    if len(roots) != len(repo_ids):
        raise ValueError(
            "lerobot_roots length must match repo_id length when local roots are provided. "
            f"Got {len(roots)} roots for {len(repo_ids)} repo ids."
        )

    path_roots = [pathlib.Path(root) for root in roots]
    parents: set[str] = set()
    for path_root, repo_id in zip(path_roots, repo_ids, strict=True):
        repo_parts = pathlib.Path(repo_id).parts
        if not repo_parts or path_root.parts[-len(repo_parts) :] != repo_parts:
            raise ValueError(
                "Each lerobot_roots entry must end with the matching repo_id path. "
                f"Got roots={roots} repo_ids={repo_ids}"
            )
        parents.add(str(path_root.parents[len(repo_parts) - 1]))
    if len(parents) != 1:
        raise ValueError(f"All lerobot_roots entries must share the same parent directory. Got {roots}")
    return next(iter(parents))


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # Support a single repo id (str) or multiple repo ids (list[str]).
    if isinstance(repo_id, str):
        # Allow simple comma-separated lists for convenience (e.g. CLI overrides).
        repo_ids = [r.strip() for r in repo_id.split(",") if r.strip()]
    elif isinstance(repo_id, list):
        repo_ids = repo_id
    else:
        raise TypeError(f"Unsupported repo_id type: {type(repo_id)}. Expected str or list[str].")

    if len(repo_ids) == 0:
        raise ValueError("Empty repo_id list. Cannot create dataset.")
    if any(r == "fake" for r in repo_ids):
        raise ValueError('repo_id list cannot contain "fake". Use repo_id="fake" for FakeDataset.')

    lerobot_root = _resolve_lerobot_dataset_root(data_config, repo_ids)

    # Load metadata to get fps and (optionally) task mapping.
    metas = [
        lerobot_dataset.LeRobotDatasetMetadata(
            repo_id,
            root=pathlib.Path(lerobot_root) / repo_id if lerobot_root is not None else None,
        )
        for repo_id in repo_ids
    ]
    fps = metas[0].fps
    mismatched = [(r, m.fps) for r, m in zip(repo_ids, metas, strict=True) if m.fps != fps]
    if mismatched:
        raise ValueError(
            "All LeRobot datasets must have the same fps for action sequencing. "
            f"Expected {fps=} from {repo_ids[0]!r}, but got mismatches: {mismatched}"
        )

    delta_timestamps = {key: [t / fps for t in range(action_horizon)] for key in data_config.action_sequence_keys}
    memory_num_frames = int(getattr(data_config, "memory_sequence_num_frames", 0))
    memory_frame_stride = int(getattr(data_config, "memory_sequence_frame_stride", 1))
    if memory_num_frames > 0:
        memory_offsets = [-(memory_num_frames - 1 - i) * memory_frame_stride / fps for i in range(memory_num_frames)]
        for key in getattr(data_config, "memory_sequence_keys", ()):
            delta_timestamps[key] = memory_offsets

    if len(repo_ids) == 1:
        dataset: Dataset = lerobot_dataset.LeRobotDataset(
            repo_ids[0],
            root=pathlib.Path(lerobot_root) / repo_ids[0] if lerobot_root is not None else None,
            delta_timestamps=delta_timestamps,
        )
    else:
        optional_features = tuple(getattr(data_config, "lerobot_optional_features", ()) or ())
        if optional_features:
            dataset = MultiLeRobotDatasetKeepFeatures(
                repo_ids,
                root=lerobot_root,
                delta_timestamps=delta_timestamps,
                keep_features=optional_features,
            )
        else:
            dataset = lerobot_dataset.MultiLeRobotDataset(
                repo_ids,
                root=lerobot_root,
                delta_timestamps=delta_timestamps,
            )

    if data_config.prompt_from_task:
        if len(repo_ids) == 1:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(metas[0].tasks)])
        else:
            tasks_by_dataset = {i: meta.tasks for i, meta in enumerate(metas)}
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTaskPerDataset(tasks_by_dataset)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            multi_hint = ""
            if isinstance(data_config.repo_id, list):
                multi_hint = (
                    " (检测到多 LeRobot dataset：请显式设置 `data.assets.asset_id`/`DataConfig.asset_id` "
                    "为一个共享目录名，或在调试时传 `skip_norm_stats=True`。)"
                )
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`. "
                f"{multi_hint}"
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            multi_hint = ""
            if isinstance(data_config.repo_id, list):
                multi_hint = (
                    " (检测到多 LeRobot dataset：请显式设置 `data.assets.asset_id`/`DataConfig.asset_id` "
                    "为一个共享目录名，或在调试时传 `skip_norm_stats=True`。)"
                )
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`. "
                f"{multi_hint}"
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset_raw = create_torch_dataset(data_config, action_horizon, model_config)

    # Optional: probability sampling across multiple LeRobot datasets.
    # NOTE: this sampler replaces DistributedSampler for DDP to preserve per-dataset probabilities.
    multi_sampler: torch.utils.data.Sampler | None = None
    if isinstance(dataset_raw, lerobot_dataset.MultiLeRobotDataset) and data_config.lerobot_sampling_weights is not None:
        probs = list(data_config.lerobot_sampling_weights)
        lengths = [int(d.num_frames) for d in dataset_raw._datasets]  # type: ignore[attr-defined]
        epoch_len = sum(lengths)
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            multi_sampler = _DistributedMultiDatasetProbSampler(
                lengths,
                probs=probs,
                num_samples_total=epoch_len,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                drop_last=True,
            )
        else:
            multi_sampler = _MultiDatasetProbSampler(
                lengths,
                probs=probs,
                num_samples=epoch_len,  # epoch length; wrapper will loop for infinite training
                seed=seed,
            )

    dataset = transform_dataset(dataset_raw, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            if multi_sampler is None:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=shuffle,
                    drop_last=True,
                )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    if sampler is None and multi_sampler is not None:
        sampler = multi_sampler
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


class EvalTorchDataLoader:
    """Evaluation-specific Torch data loader that iterates through the dataset exactly once."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        shuffle: bool = False,
        num_workers: int = 0,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """Create a PyTorch data loader for evaluation.

        Unlike TorchDataLoader, this loader iterates through the dataset exactly once
        and then stops (no infinite loop).

        Args:
            dataset: The dataset to load.
            local_batch_size: The batch size.
            shuffle: Whether to shuffle the data (usually False for evaluation).
            num_workers: The number of worker processes to use.
            seed: The seed to use for shuffling the data.
            drop_last: Whether to drop the last incomplete batch.
        """
        if len(dataset) < local_batch_size and drop_last:
            raise ValueError(f"Batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        self._dataset = dataset
        self._batch_size = local_batch_size

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=drop_last,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        return len(self._data_loader)

    def dataset_size(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self._dataset)

    def __iter__(self):
        """Iterate through the dataset exactly once."""
        for batch in self._data_loader:
            # Convert to torch tensors
            yield jax.tree.map(torch.as_tensor, batch)


class EvalDataLoaderImpl(DataLoader):
    """Evaluation-specific DataLoader wrapper that iterates exactly once."""

    def __init__(self, data_config: _config.DataConfig, data_loader: EvalTorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __len__(self) -> int:
        return len(self._data_loader)

    def dataset_size(self) -> int:
        return self._data_loader.dataset_size()

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]


def create_eval_data_loader(
    config: _config.TrainConfig,
    *,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
    drop_last: bool = False,
    skip_norm_stats: bool = False,
) -> EvalDataLoaderImpl:
    """Create a data loader specifically for evaluation.

    This loader iterates through the dataset exactly once and then stops.

    Args:
        config: Training config with data configuration.
        shuffle: Whether to shuffle the data (usually False for evaluation).
        num_workers: Number of worker processes.
        seed: Random seed.
        drop_last: Whether to drop the last incomplete batch.
        skip_norm_stats: Whether to skip data normalization.

    Returns:
        An EvalDataLoaderImpl that iterates exactly once.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = EvalTorchDataLoader(
        dataset,
        config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        seed=seed,
        drop_last=drop_last,
    )

    return EvalDataLoaderImpl(data_config, data_loader)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
