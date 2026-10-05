from collections.abc import Sequence
import logging

import torch

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def _preprocess_keyframes(
    keyframes: torch.Tensor | None,
    keyframe_mask: torch.Tensor | None,
    *,
    image_resolution: tuple[int, int],
    state: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if keyframes is None:
        return None, None
    if keyframes.ndim != 5:
        raise ValueError(f"keyframes must have shape [B, K, H, W, C] or [B, K, C, H, W], got {keyframes.shape}")

    bsize, num_keyframes = keyframes.shape[:2]
    channels_last = keyframes.shape[-1] <= 4
    if channels_last:
        flat = keyframes.reshape(bsize * num_keyframes, *keyframes.shape[2:])
        if flat.shape[1:3] != image_resolution:
            flat = image_tools.resize_with_pad_torch(flat, *image_resolution)
        flat = flat.permute(0, 3, 1, 2)
    else:
        flat = keyframes.reshape(bsize * num_keyframes, *keyframes.shape[2:])
        if flat.shape[2:4] != image_resolution:
            flat = image_tools.resize_with_pad_torch(flat, *image_resolution)

    processed = flat.reshape(bsize, num_keyframes, *flat.shape[1:])
    if keyframe_mask is None:
        keyframe_mask = torch.ones((bsize, num_keyframes), dtype=torch.bool, device=state.device)
    else:
        keyframe_mask = keyframe_mask.to(device=state.device, dtype=torch.bool)
    return processed, keyframe_mask


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                height, width = image.shape[1:3]

                # Random crop and resize
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # Random crop
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # Resize back to original size
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                # Random rotation (small angles)
                # Use tensor operations instead of .item() for torch.compile compatibility
                angle = torch.rand(1, device=image.device) * 10 - 5  # Random angle between -5 and 5 degrees
                if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                    # Convert to radians
                    angle_rad = angle * torch.pi / 180.0

                    # Create rotation matrix
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # Apply rotation using grid_sample
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # Create meshgrid
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # Expand to batch dimension
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # Apply rotation transformation
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # Stack and reshape for grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # Color augmentations for all cameras
            # Random brightness
            # Use tensor operations instead of .item() for torch.compile compatibility
            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
            image = image * brightness_factor

            # Random contrast
            # Use tensor operations instead of .item() for torch.compile compatibility
            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            # Random saturation (convert to HSV, modify S, convert back)
            # For simplicity, we'll just apply a random scaling to the color channels
            # Use tensor operations instead of .item() for torch.compile compatibility
            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # Clamp values to [0, 1]
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    episode_keyframes, episode_keyframe_mask = _preprocess_keyframes(
        getattr(observation, "episode_keyframes", None),
        getattr(observation, "episode_keyframe_mask", None),
        image_resolution=image_resolution,
        state=observation.state,
    )
    task_keyframes, task_keyframe_mask = _preprocess_keyframes(
        getattr(observation, "task_keyframes", None),
        getattr(observation, "task_keyframe_mask", None),
        image_resolution=image_resolution,
        state=observation.state,
    )
    gist_keyframes, gist_keyframe_mask = _preprocess_keyframes(
        getattr(observation, "gist_keyframes", None),
        getattr(observation, "gist_keyframe_mask", None),
        image_resolution=image_resolution,
        state=observation.state,
    )
    raw_memory_frames = getattr(observation, "memory_frames", None)
    memory_frame_mask = getattr(observation, "memory_frame_mask", None)
    if isinstance(raw_memory_frames, dict):
        memory_frames: dict[str, torch.Tensor] | None = {}
        for key, frames in raw_memory_frames.items():
            processed, _ = _preprocess_keyframes(
                frames,
                memory_frame_mask,
                image_resolution=image_resolution,
                state=observation.state,
            )
            if processed is not None:
                memory_frames[key] = processed
        memory_frames = memory_frames or None
    else:
        memory_frames, memory_frame_mask = _preprocess_keyframes(
            raw_memory_frames,
            memory_frame_mask,
            image_resolution=image_resolution,
            state=observation.state,
        )

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        episode_keyframes=episode_keyframes,
        episode_keyframe_mask=episode_keyframe_mask,
        task_keyframes=task_keyframes,
        task_keyframe_mask=task_keyframe_mask,
        gist_keyframes=gist_keyframes,
        gist_keyframe_mask=gist_keyframe_mask,
        memory_frames=memory_frames,
        memory_frame_mask=memory_frame_mask,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        progress_bin=getattr(observation, "progress_bin", None),
        subtask_done=getattr(observation, "subtask_done", None),
        progress_mask=getattr(observation, "progress_mask", None),
    )
