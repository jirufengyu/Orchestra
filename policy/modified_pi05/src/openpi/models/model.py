import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "episode_keyframes": (float32|uint8)[*b, k, h, w, 3],  # Optional RGB keyframes for this episode
#     "episode_keyframe_mask": bool[*b, k],  # True if episode keyframe is valid
#     "task_keyframes": (float32|uint8)[*b, k, h, w, 3],  # Optional RGB keyframes shared by the task
#     "task_keyframe_mask": bool[*b, k],  # True if task keyframe is valid
#     "memory_frames": {camera_key: (float32|uint8)[*b, k, h, w, 3]},  # Short-horizon RGB history per camera
#     "memory_frame_mask": bool[*b, k],  # True if memory frame is valid
#     "keyframe_caption": str | np.ndarray,  # Optional text summary to prepend to the prompt before tokenization
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Optional keyframe conditioning inputs. These are intentionally separate from
    # images so they can be compressed into a small number of prefix tokens instead
    # of expanding into full patch-token grids.
    episode_keyframes: at.Float[ArrayT, "*ep_b ep_k ep_h ep_w ep_c"] | None = None
    episode_keyframe_mask: at.Bool[ArrayT, "*ep_b ep_k"] | None = None
    task_keyframes: at.Float[ArrayT, "*task_b task_k task_h task_w task_c"] | None = None
    task_keyframe_mask: at.Bool[ArrayT, "*task_b task_k"] | None = None
    keyframe_caption: str | ArrayT | None = None

    # Short-horizon video memory (MEM-style), keyed like `images` (base + wrist cameras).
    memory_frames: dict[str, at.Float[ArrayT, "*mem_b mem_k mem_h mem_w mem_c"]] | None = None
    memory_frame_mask: at.Bool[ArrayT, "*mem_b mem_k"] | None = None

    # Long-horizon gist memory (HEVM): strided history compressed into prefix tokens.
    gist_keyframes: at.Float[ArrayT, "*gist_b gist_k gist_h gist_w gist_c"] | None = None
    gist_keyframe_mask: at.Bool[ArrayT, "*gist_b gist_k"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    return_target: at.Int[ArrayT, "*b n"] | None = None
    current_step: at.Int[ArrayT, "*b"] | None = None
    total_steps: at.Int[ArrayT, "*b"] | None = None
    success: at.Int[ArrayT, "*b"] | None = None

    # MoE task identifier – used to route to the correct action expert.
    # Can be an int tensor (task index) or left as None (use default expert).
    task_id: at.Int[ArrayT, "*b"] | None = None

    # RMBench subtask progress (joint training with pi05).
    progress_bin: at.Int[ArrayT, "*b"] | None = None
    subtask_done: at.Float[ArrayT, "*b"] | None = None
    progress_mask: at.Float[ArrayT, "*b"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        def _image_like_to_float(x):
            if isinstance(x, np.ndarray) and x.dtype == np.uint8:
                return x.astype(np.float32) / 255.0 * 2.0 - 1.0
            if isinstance(x, torch.Tensor) and x.dtype == torch.uint8:
                x = x.to(torch.float32) / 255.0 * 2.0 - 1.0
            if isinstance(x, torch.Tensor) and x.ndim == 5 and x.shape[-1] > 4 and x.shape[-3] <= 4:
                return x.permute(0, 1, 3, 4, 2)
            return x

        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        episode_keyframes = data.get("episode_keyframes")
        task_keyframes = data.get("task_keyframes")
        gist_keyframes = data.get("gist_keyframes")
        episode_keyframes = _image_like_to_float(episode_keyframes) if episode_keyframes is not None else None
        task_keyframes = _image_like_to_float(task_keyframes) if task_keyframes is not None else None
        gist_keyframes = _image_like_to_float(gist_keyframes) if gist_keyframes is not None else None
        memory_frames_raw = data.get("memory_frames")
        if isinstance(memory_frames_raw, dict):
            memory_frames = {
                key: _image_like_to_float(value) for key, value in memory_frames_raw.items() if value is not None
            }
            memory_frames = memory_frames or None
        else:
            memory_frames = _image_like_to_float(memory_frames_raw) if memory_frames_raw is not None else None

        episode_keyframe_mask = data.get("episode_keyframe_mask")
        task_keyframe_mask = data.get("task_keyframe_mask")
        gist_keyframe_mask = data.get("gist_keyframe_mask")
        memory_frame_mask = data.get("memory_frame_mask")
        if episode_keyframes is not None and episode_keyframe_mask is None:
            episode_keyframe_mask = np.ones(episode_keyframes.shape[:-3], dtype=np.bool_)
            if isinstance(episode_keyframes, torch.Tensor):
                episode_keyframe_mask = torch.ones(
                    episode_keyframes.shape[:-3], dtype=torch.bool, device=episode_keyframes.device
                )
        if task_keyframes is not None and task_keyframe_mask is None:
            task_keyframe_mask = np.ones(task_keyframes.shape[:-3], dtype=np.bool_)
            if isinstance(task_keyframes, torch.Tensor):
                task_keyframe_mask = torch.ones(
                    task_keyframes.shape[:-3], dtype=torch.bool, device=task_keyframes.device
                )
        if memory_frames is not None and memory_frame_mask is None:
            if isinstance(memory_frames, dict):
                first = next(iter(memory_frames.values()))
                batch_shape = first.shape[:-3]
            else:
                batch_shape = memory_frames.shape[:-3]
            memory_frame_mask = np.ones(batch_shape, dtype=np.bool_)
            if isinstance(first if isinstance(memory_frames, dict) else memory_frames, torch.Tensor):
                ref = first if isinstance(memory_frames, dict) else memory_frames
                memory_frame_mask = torch.ones(batch_shape, dtype=torch.bool, device=ref.device)
        if gist_keyframes is not None and gist_keyframe_mask is None:
            gist_keyframe_mask = np.ones(gist_keyframes.shape[:-3], dtype=np.bool_)
            if isinstance(gist_keyframes, torch.Tensor):
                gist_keyframe_mask = torch.ones(
                    gist_keyframes.shape[:-3], dtype=torch.bool, device=gist_keyframes.device
                )
        if "return_target" in data:
            return_target = data["return_target"][1]
        else:
            return_target = None
        if "current_step" in data:
            current_step = data["current_step"]
        else:
            current_step = None
        if "total_steps" in data:
            total_steps = data["total_steps"]
        else:
            total_steps = None
        if "success" in data:
            success = data["success"]
        else:
            success = None
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            episode_keyframes=episode_keyframes,
            episode_keyframe_mask=episode_keyframe_mask,
            task_keyframes=task_keyframes,
            task_keyframe_mask=task_keyframe_mask,
            keyframe_caption=data.get("keyframe_caption"),
            memory_frames=memory_frames,
            memory_frame_mask=memory_frame_mask,
            gist_keyframes=gist_keyframes,
            gist_keyframe_mask=gist_keyframe_mask,
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            return_target=return_target,
            current_step=current_step,
            total_steps=total_steps,
            success=success,
            task_id=data.get("task_id"),
            progress_bin=data.get("progress_bin"),
            subtask_done=data.get("subtask_done"),
            progress_mask=data.get("progress_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
