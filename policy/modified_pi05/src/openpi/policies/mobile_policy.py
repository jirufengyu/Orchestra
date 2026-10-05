import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

Mobile_ARM_DIM = 16
Mobile_MOBILE_ACTION_DIM = 19


def make_mobile_example() -> dict:
    """Creates a random input example for the  policy."""
    return {
        "state": np.ones((16,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def make_mobile_mobile_example() -> dict:
    """Creates a random input example for the Mobile mobile policy."""
    return {
        "state": np.ones((Mobile_MOBILE_ACTION_DIM,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class MobileInputs(transforms.DataTransformFn):
    """Inputs for the Mobile policy.

    Expected inputs:
    - images: dict[name, img] where img is [height, width, channel]. name must be in EXPECTED_CAMERAS.
    - state: [16]
    - actions: [action_horizon, 16]
    """

    # If true, this will convert the joint and gripper values from the standard Mobile space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }
        
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
    
@dataclasses.dataclass(frozen=True)
class MobileOutputs(transforms.DataTransformFn):
    """Outputs for the Mobile policy."""

    # If true, this will convert the joint and gripper values from the standard Mobile space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :Mobile_ARM_DIM])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
    

@dataclasses.dataclass(frozen=True)
class MobileInspireInputs(transforms.DataTransformFn):
    """Inputs for the Mobile policy.

    Expected inputs:
    - images: dict[name, img] where img is [height, width, channel]. name must be in EXPECTED_CAMERAS.
    - state: [26]
    - actions: [action_horizon, 26]
    """

    # If true, this will convert the joint and gripper values from the standard Mobile space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }
        
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
    
@dataclasses.dataclass(frozen=True)
class MobileInspireOutputs(transforms.DataTransformFn):
    """Outputs for the Mobile policy."""

    # If true, this will convert the joint and gripper values from the standard Mobile space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :26])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


@dataclasses.dataclass(frozen=True)
class MobileMobileInputs(transforms.DataTransformFn):
    """Inputs for the Mobile mobile policy.

    Expected inputs:
    - images: dict[name, img] where img is [height, width, channel]. name must be in EXPECTED_CAMERAS.
    - state: [19] = [left_arm (7), left_gripper (1), right_arm (7), right_gripper (1), base (3)]
    - actions: [action_horizon, 19]
    """

    adapt_to_pi: bool = True
    # If True, map optional cam_base into episode_keyframes (mask=False when missing).
    use_cam_base_as_episode_keyframe: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
        "cam_base",
    )

    def __call__(self, data: dict) -> dict:
        raw_images = dict(data["images"])
        cam_base = raw_images.pop("cam_base", None)
        # RepackTransform fills missing keys with None; drop them before decode.
        data = {
            **data,
            "images": {name: img for name, img in raw_images.items() if img is not None},
        }
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi, arm_dim=Mobile_ARM_DIM)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi, arm_dim=Mobile_ARM_DIM)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if self.use_cam_base_as_episode_keyframe:
            if cam_base is not None:
                keyframe = _convert_image(cam_base)
                inputs["episode_keyframes"] = keyframe[None, ...]
                inputs["episode_keyframe_mask"] = np.asarray([True], dtype=np.bool_)
            else:
                inputs["episode_keyframes"] = np.zeros((1, *base_image.shape), dtype=base_image.dtype)
                inputs["episode_keyframe_mask"] = np.asarray([False], dtype=np.bool_)
        else:
            for key in ("episode_keyframes", "episode_keyframe_mask", "task_keyframes", "task_keyframe_mask"):
                if key in data and data[key] is not None:
                    inputs[key] = data[key]

        return inputs

@dataclasses.dataclass(frozen=True)
class MobileMobileOutputs(transforms.DataTransformFn):
    """Outputs for the Mobile mobile policy."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :Mobile_MOBILE_ACTION_DIM])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi, arm_dim=Mobile_ARM_DIM)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _joint_flip_mask_mobile() -> np.ndarray:
    """Used to convert between aloha and pi joint angles for Mobile 7-DoF dual arms."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _convert_image(img) -> np.ndarray:
    img = np.asarray(img)
    # Convert to uint8 if using float images.
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    # Convert from [channel, height, width] to [height, width, channel].
    if img.shape[0] == 3:
        img = einops.rearrange(img, "c h w -> h w c")
    return img


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False, arm_dim: int = Mobile_ARM_DIM) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [7, 1, 7, 1] with optional trailing base dims for mobile robots.
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi, arm_dim=arm_dim)

    images = data["images"]
    images_dict = {name: _convert_image(img) for name, img in images.items() if img is not None}

    data["images"] = images_dict
    data["state"] = state
    return data

def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False, arm_dim: int = Mobile_ARM_DIM) -> np.ndarray:
    if adapt_to_pi:
        arm_state = np.asarray(state[..., :arm_dim])
        extra_state = state[..., arm_dim:]
        flip_mask = _joint_flip_mask_mobile() if arm_dim == Mobile_ARM_DIM else _joint_flip_mask()[:arm_dim]
        arm_state = flip_mask * arm_state
        arm_state[..., [6, 13]] = _gripper_to_angular(arm_state[..., [6, 13]])
        if extra_state.size > 0:
            return np.concatenate([arm_state, extra_state], axis=-1)
        return arm_state
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False, arm_dim: int = Mobile_ARM_DIM) -> np.ndarray:
    if adapt_to_pi:
        arm_actions = np.asarray(actions[..., :arm_dim])
        extra_actions = actions[..., arm_dim:]
        flip_mask = _joint_flip_mask_mobile() if arm_dim == Mobile_ARM_DIM else _joint_flip_mask()[:arm_dim]
        arm_actions = flip_mask * arm_actions
        arm_actions[..., [6, 13]] = _gripper_from_angular(arm_actions[..., [6, 13]])
        if extra_actions.size > 0:
            return np.concatenate([arm_actions, extra_actions], axis=-1)
        return arm_actions
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False, arm_dim: int = Mobile_ARM_DIM) -> np.ndarray:
    if adapt_to_pi:
        arm_actions = np.asarray(actions[..., :arm_dim])
        extra_actions = actions[..., arm_dim:]
        flip_mask = _joint_flip_mask_mobile() if arm_dim == Mobile_ARM_DIM else _joint_flip_mask()[:arm_dim]
        arm_actions = flip_mask * arm_actions
        arm_actions[..., [6, 13]] = _gripper_from_angular_inv(arm_actions[..., [6, 13]])
        if extra_actions.size > 0:
            return np.concatenate([arm_actions, extra_actions], axis=-1)
        return arm_actions
    return actions