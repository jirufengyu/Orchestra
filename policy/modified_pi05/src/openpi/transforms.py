from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import pathlib
import re
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools
from PIL import Image

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item.get(k, None), self.structure)


@dataclasses.dataclass(frozen=True)
class LoadLeRobotKeyframes(DataTransformFn):
    """Adds fixed-size keyframe tensors from an extractor manifest to LeRobot samples."""

    manifest_path: str | pathlib.Path
    keyframe_source: Literal["episode", "task", "both"] = "episode"
    episode_image_key: str | None = None
    task_image_key: str | None = None
    max_episode_keyframes: int = 8
    max_task_keyframes: int = 0
    empty_image_shape: tuple[int, int, int] = (224, 224, 3)

    _manifest: dict[str, Any] | None = dataclasses.field(default=None, init=False, repr=False, compare=False)
    _manifest_dir: pathlib.Path | None = dataclasses.field(default=None, init=False, repr=False, compare=False)
    _episodes_by_index: dict[int, dict[str, Any]] | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )
    _episodes_by_task: dict[int, dict[str, Any]] | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.keyframe_source not in ("episode", "task", "both"):
            raise ValueError(f"keyframe_source must be 'episode', 'task', or 'both', got {self.keyframe_source!r}")

    def __call__(self, data: DataDict) -> DataDict:
        self._ensure_loaded()
        assert self._episodes_by_index is not None
        assert self._episodes_by_task is not None

        if "episode_index" not in data:
            raise ValueError('Cannot load LeRobot keyframes without "episode_index" in the sample.')

        episode_index = _as_int(data["episode_index"])
        episode = self._episodes_by_index.get(episode_index)
        if episode is None:
            raise ValueError(f"{episode_index=} not found in keyframe manifest {self.manifest_path!s}")

        if self.keyframe_source in ("episode", "both") and self.max_episode_keyframes > 0:
            frames, mask = self._load_keyframe_array(
                episode,
                image_key=self.episode_image_key,
                max_keyframes=self.max_episode_keyframes,
            )
            data["episode_keyframes"] = frames
            data["episode_keyframe_mask"] = mask

        if self.keyframe_source in ("task", "both") and self.max_task_keyframes > 0:
            task_episode = episode
            if "task_index" in data:
                task_episode = self._episodes_by_task.get(_as_int(data["task_index"]), episode)
            frames, mask = self._load_keyframe_array(
                task_episode,
                image_key=self.task_image_key or self.episode_image_key,
                max_keyframes=self.max_task_keyframes,
            )
            data["task_keyframes"] = frames
            data["task_keyframe_mask"] = mask

        return data

    def _ensure_loaded(self) -> None:
        if self._manifest is not None:
            return

        manifest_path = pathlib.Path(self.manifest_path).expanduser()
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        episodes = manifest.get("episodes", [])
        episodes_by_index = {int(episode["episode_index"]): episode for episode in episodes}
        episodes_by_task: dict[int, dict[str, Any]] = {}
        for episode in episodes:
            task_index = episode.get("task_index")
            if task_index is not None:
                episodes_by_task.setdefault(int(task_index), episode)

        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(self, "_manifest_dir", manifest_path.parent)
        object.__setattr__(self, "_episodes_by_index", episodes_by_index)
        object.__setattr__(self, "_episodes_by_task", episodes_by_task)

    def _load_keyframe_array(
        self,
        episode: dict[str, Any],
        *,
        image_key: str | None,
        max_keyframes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        records = self._keyframe_records(episode, image_key=image_key)
        selected = records[:max_keyframes]
        frames = [self._load_image(record["path"]) for record in selected]
        image_shape = frames[0].shape if frames else self.empty_image_shape

        out = np.zeros((max_keyframes, *image_shape), dtype=np.uint8)
        mask = np.zeros((max_keyframes,), dtype=np.bool_)
        for i, frame in enumerate(frames):
            out[i] = frame
            mask[i] = True
        return out, mask

    def _keyframe_records(self, episode: dict[str, Any], *, image_key: str | None) -> list[dict[str, Any]]:
        self._ensure_loaded()
        assert self._manifest is not None
        image_key = image_key or self._manifest.get("reference_image_key") or episode.get("image_key")
        views = episode.get("keyframes_by_image_key") or {}
        if image_key in views:
            return list(views[image_key].get("keyframes", []))
        if image_key == episode.get("image_key"):
            return list(episode.get("keyframes", []))
        raise ValueError(f"image key {image_key!r} not found for {episode.get('id', episode.get('episode_index'))}")

    def _load_image(self, path: str) -> np.ndarray:
        assert self._manifest_dir is not None
        image_path = pathlib.Path(path)
        if not image_path.is_absolute():
            image_path = self._manifest_dir / image_path
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        for key in ("episode_keyframes", "task_keyframes", "gist_keyframes"):
            if key in data and data[key] is not None:
                data[key] = image_tools.resize_with_pad(data[key], self.height, self.width)
        if "memory_frames" in data and isinstance(data["memory_frames"], dict):
            data["memory_frames"] = {
                k: image_tools.resize_with_pad(v, self.height, self.width)
                for k, v in data["memory_frames"].items()
            }
        elif "memory_frames" in data and data["memory_frames"] is not None:
            data["memory_frames"] = image_tools.resize_with_pad(data["memory_frames"], self.height, self.width)
        return data


@dataclasses.dataclass(frozen=True)
class PrependKeyframeCaption(DataTransformFn):
    """Prepends optional keyframe text summaries to the prompt before tokenization."""

    prefix: str = "Visual plan"

    def __call__(self, data: DataDict) -> DataDict:
        captions = []
        for key in ("task_plan_text", "keyframe_caption"):
            if (caption := data.pop(key, None)) is None:
                continue
            if not isinstance(caption, str):
                caption = caption.item()
            caption = str(caption).strip()
            if caption:
                captions.append(caption)
        if not captions:
            return data

        prompt = data.get("prompt", "")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        prompt = str(prompt)
        plan = " ".join(captions)
        data["prompt"] = f"{self.prefix}: {plan}\nInstruction: {prompt}"
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTaskPerDataset(DataTransformFn):
    """Extracts a prompt from LeRobot tasks for multi-dataset training.

    `lerobot_dataset.MultiLeRobotDataset` injects a `dataset_index` key in each sample.
    Different datasets may reuse the same `task_index` values, so we key the mapping by
    `(dataset_index, task_index)` to avoid collisions.
    """

    # Mapping: dataset_index -> {task_index -> prompt}
    tasks_by_dataset: dict[int, dict[int, str]]

    def __call__(self, data: DataDict) -> DataDict:
        if "dataset_index" not in data:
            raise ValueError('Cannot extract multi-dataset prompt without "dataset_index"')
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        dataset_index = int(data["dataset_index"])
        task_index = int(data["task_index"])
        tasks = self.tasks_by_dataset.get(dataset_index)
        if tasks is None:
            raise ValueError(f"{dataset_index=} not found in dataset task mapping: {list(self.tasks_by_dataset)}")
        if (prompt := tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found for {dataset_index=}: {tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def _as_int(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
