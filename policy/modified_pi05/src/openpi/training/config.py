import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.seg_transforms as seg_transforms
import openpi.policies.rmbench_progress_transforms as rmbench_progress_transforms
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None

@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    # NOTE: Support multiple LeRobot datasets by passing a list of repo ids.
    # When multiple datasets are used, you typically must set `assets.asset_id`
    # explicitly (so norm stats can be found/saved under a single directory).
    repo_id: str | list[str] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    # Names of observation keys that should be loaded as a recent history sequence via LeRobot delta_timestamps.
    # These are typically image keys such as "observation.images.cam_high".
    memory_sequence_keys: Sequence[str] = ()
    memory_sequence_num_frames: int = 0
    memory_sequence_frame_stride: int = 1

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # When using multiple LeRobot datasets (repo_id is a list, or a comma-separated string),
    # this optional list controls sampling frequency across datasets during training.
    # - Length must match number of datasets.
    # - Values can be any non-negative numbers (they are automatically normalized to probabilities).
    #   e.g. [1, 3] means dataset2 is sampled 3x as often as dataset1.
    # - If None, sampling defaults to the concatenation behavior (roughly proportional to dataset sizes).
    lerobot_sampling_weights: Sequence[float] | None = None

    # Feature keys that MultiLeRobotDataset should keep even if they are not shared by all
    # sub-datasets (LeRobot otherwise drops non-intersection keys). Missing keys are simply
    # absent on those samples; downstream transforms should pad/mask as needed.
    # Example: ("observation.images.cam_base",)
    lerobot_optional_features: Sequence[str] = ()
    # Optional local LeRobot dataset roots aligned with repo_id entries. When set,
    # create_torch_dataset reads parquet/video directly from these paths instead of HF cache.
    lerobot_roots: Sequence[str] = ()

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""

@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        prompt_transforms: list[_transforms.DataTransformFn] = [_transforms.InjectDefaultPrompt(self.default_prompt)]
        if getattr(model_config, "use_keyframe_caption", False):
            prompt_transforms.append(_transforms.PrependKeyframeCaption())

        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        *prompt_transforms,
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        *prompt_transforms,
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        *prompt_transforms,
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )

@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str | list[str] = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        # `asset_id` must be a single directory name. For multi-dataset training, require
        # users to set it explicitly (or skip norm stats) to avoid ambiguous defaults.
        asset_id = self.assets.asset_id or (repo_id if isinstance(repo_id, str) else None)
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)

@dataclasses.dataclass
class DebugPrintPrompt(_transforms.DataTransformFn):
    max_prints: int = 5
    prefix: str = "[debug prompt]"
    _count: int = dataclasses.field(default=0, init=False, repr=False)

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if self._count >= self.max_prints:
            return data

        try:
            import os

            if os.environ.get("RANK", "0") != "0":
                return data
            try:
                from torch.utils.data import get_worker_info

                worker_info = get_worker_info()
                if worker_info is not None and worker_info.id != 0:
                    return data
            except Exception:
                pass
        except Exception:
            pass

        prompt = data.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
        print(f"{self.prefix} {self._count}: {prompt}", flush=True)
        self._count += 1
        return data

@dataclasses.dataclass(frozen=True)
class LeRobotAlohaSegDataConfig(DataConfigFactory):
    """Aloha data config with segmentation-conditioned training.

    Supports three seg condition types:
    - mask: overlay on RGB image (visual tokens)
    - bbox: embedded as text coordinates in prompt
    - point: embedded as text coordinates in prompt

    Training paradigm:
    - p_drop_seg: probability of dropping seg entirely (preserves original capability)
    - p_drop_image: probability of dropping base camera (forces seg reliance)
    - type_weights: sampling probabilities for (mask, bbox, point)
    """

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True
    # Use Mobile 7-DoF dual-arm action layout (16 dims) instead of 6-DoF Aloha (14 dims).
    use_mobile_action_space: bool = False

    # Seg condition parameters
    p_drop_seg: float = 0.2
    p_drop_image: float = 0.1
    p_keep_wrist_on_drop: float = 0.7
    seg_type_weights: tuple[float, float, float] = (0.34, 0.33, 0.33)
    overlay_style: str = "contour"
    overlay_alpha: float = 0.4
    overlay_color: tuple[int, int, int] = (0, 255, 0)
    coord_bins: int = 256
    # Default seg mode(s) at eval: mask | bbox | point | all | mask,bbox,point | mask,point, ...
    eval_seg_representation: str = "mask"

    # Paths to LeRobot dataset roots for loading actors_meta.
    # If empty, seg transform will use raw seg IDs without actor names.
    lerobot_roots: Sequence[str] = ()

    # Inject per-frame progress_bin / subtask_done for joint progress-head training.
    use_progress_labels: bool = False
    progress_num_bins: int = 101
    progress_boundary_margin: int = 8
    # For MN tasks: restrict seg conditioning to current subtask actors (active_seg_ids).
    use_subtask_seg: bool = False
    # For M1 tasks: union all subtask active actors in the episode for seg conditioning.
    use_episode_union_seg: bool = False
    # For M1 tasks: use fixed global instruction instead of per-frame task_index prompts.
    use_global_task: bool = False
    # For M1 tasks: inject episode-start full image as pi0.5 episode keyframe memory.
    use_anchor_memory: bool = False
    # For M1 tasks: inject recent base-camera frames as MEM-style short-horizon memory.
    use_short_horizon_memory: bool = False
    # HEVM: anchor (t=0) + gist (strided history) + recent (short-horizon) unified memory.
    use_episodic_memory: bool = False
    episodic_gist_stride: int = 10
    episodic_max_gist_frames: int = 16
    episodic_p_drop_anchor: float = 0.2
    episodic_p_drop_recent: float = 0.1
    episodic_p_drop_gist: float = 0.3
    memory_num_frames: int = 6
    memory_frame_stride: int = 1
    memory_image_keys: tuple[str, ...] = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    # If False, wrist cameras are zeroed and image_mask=False (Mem-0 style head-only vision).
    use_wrist_cameras: bool = True
    debug_prompt_prints: int = 0

    # Repack transforms (should include seg_cam_high, episode_index, frame_index).
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                        "seg_cam_high": "observation.segmentation.cam_high",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        from openpi.shared.rmbench_progress import resolve_rmbench_m1_global_task
        from openpi.shared.rmbench_progress import resolve_rmbench_m1_global_tasks_by_dataset

        lerobot_roots = tuple(self.lerobot_roots) or rmbench_progress_transforms.resolve_lerobot_roots(self.repo_id)
        global_prompt = (
            self.default_prompt or resolve_rmbench_m1_global_task(self.repo_id)
            if self.use_global_task
            else self.default_prompt
        )

        if self.use_episode_union_seg and self.use_subtask_seg:
            raise ValueError("use_episode_union_seg and use_subtask_seg are mutually exclusive.")

        # Pre-load actors metadata keyed by (dataset_index, episode_index)
        actors_meta_all = seg_transforms.load_all_actors_meta(lerobot_roots)
        need_active_meta = self.use_subtask_seg or self.use_episode_union_seg
        active_actors_all = (
            seg_transforms.load_all_active_actors_meta(lerobot_roots)
            if need_active_meta and lerobot_roots
            else None
        )
        episode_union_all = (
            seg_transforms.build_episode_union_active_seg_by_key(actors_meta_all, active_actors_all)
            if self.use_episode_union_seg and active_actors_all
            else None
        )
        anchor_images_all = (
            rmbench_progress_transforms.load_episode_anchor_images(lerobot_roots)
            if self.use_anchor_memory and lerobot_roots
            else None
        )
        memory_cam_keys = rmbench_progress_transforms.memory_cam_keys_from_lerobot_keys(self.memory_image_keys)
        seg_input_transforms: list[_transforms.DataTransformFn] = []
        if actors_meta_all:
            seg_input_transforms.append(
                seg_transforms.LoadActorsMeta(
                    actors_meta_all,
                    active_actors_by_key=active_actors_all,
                    episode_union_by_key=episode_union_all,
                    use_subtask_seg=self.use_subtask_seg,
                    use_episode_union_seg=self.use_episode_union_seg,
                )
            )

        if self.use_anchor_memory:
            if not anchor_images_all:
                raise ValueError("use_anchor_memory=True requires locally available RMBench LeRobot episode images.")
            seg_input_transforms.append(
                rmbench_progress_transforms.InjectRmbenchEpisodeAnchorKeyframes(
                    anchor_images_by_key=anchor_images_all,
                )
            )
            if not self.use_episodic_memory:
                seg_input_transforms.append(
                    rmbench_progress_transforms.EpisodicMemoryScopeDropout(
                        p_drop_anchor=self.episodic_p_drop_anchor,
                        p_drop_recent=0.0,
                        p_drop_gist=0.0,
                    )
                )

        if self.use_episodic_memory:
            if not lerobot_roots:
                raise ValueError("use_episodic_memory=True requires locally available RMBench LeRobot roots.")
            episodic_anchors = rmbench_progress_transforms.load_episodic_anchor_images_all_cams(lerobot_roots)
            seg_input_transforms.extend(
                [
                    rmbench_progress_transforms.InjectEpisodicAnchorKeyframes(
                        anchor_images_by_key=episodic_anchors,
                    ),
                    rmbench_progress_transforms.InjectEpisodicGistKeyframes(
                        lerobot_roots=tuple(lerobot_roots),
                        gist_stride=self.episodic_gist_stride,
                        max_gist_frames=self.episodic_max_gist_frames,
                    ),
                    rmbench_progress_transforms.EpisodicMemoryScopeDropout(
                        p_drop_anchor=self.episodic_p_drop_anchor,
                        p_drop_recent=self.episodic_p_drop_recent,
                        p_drop_gist=self.episodic_p_drop_gist,
                    ),
                    rmbench_progress_transforms.SplitRecentImageSequences(
                        image_keys=memory_cam_keys,
                        num_frames=self.memory_num_frames,
                        frame_stride=self.memory_frame_stride,
                    ),
                ]
            )
        elif self.use_short_horizon_memory:
            seg_input_transforms.append(
                rmbench_progress_transforms.SplitRecentImageSequences(
                    image_keys=memory_cam_keys,
                    num_frames=self.memory_num_frames,
                    frame_stride=self.memory_frame_stride,
                )
            )

        if self.use_global_task:
            if global_prompt is None:
                raise ValueError("use_global_task=True requires default_prompt or a known M1 repo_id.")
            if isinstance(self.repo_id, list):
                seg_input_transforms.append(
                    rmbench_progress_transforms.InjectRmbenchM1GlobalPromptPerDataset(
                        prompts_by_dataset=resolve_rmbench_m1_global_tasks_by_dataset(self.repo_id),
                    )
                )
            else:
                seg_input_transforms.append(
                    rmbench_progress_transforms.InjectRmbenchM1GlobalPrompt(prompt=global_prompt)
                )

        if self.use_progress_labels and lerobot_roots:
            progress_meta = rmbench_progress_transforms.load_episode_progress_meta(lerobot_roots)
            seg_input_transforms.append(
                rmbench_progress_transforms.InjectRmbenchProgressLabels(
                    episode_meta_by_key=progress_meta,
                    num_progress_bins=self.progress_num_bins,
                    boundary_margin=self.progress_boundary_margin,
                )
            )

        seg_input_transforms.append(
            seg_transforms.SegConditionTransform(
                p_drop_seg=self.p_drop_seg,
                p_drop_image=self.p_drop_image,
                p_keep_wrist_on_drop=self.p_keep_wrist_on_drop,
                type_weights=self.seg_type_weights,
                overlay_style=self.overlay_style,
                overlay_alpha=self.overlay_alpha,
                overlay_color=self.overlay_color,
                coord_bins=self.coord_bins,
                infer_representations=seg_transforms.parse_seg_representations(self.eval_seg_representation),
            )
        )

        data_transforms = _transforms.Group(
            inputs=[
                *seg_input_transforms,
                (
                    seg_transforms.MobileSegInputs(
                        adapt_to_pi=self.adapt_to_pi,
                        use_wrist_cameras=self.use_wrist_cameras,
                    )
                    if self.use_mobile_action_space
                    else seg_transforms.AlohaSegInputs(
                        adapt_to_pi=self.adapt_to_pi,
                        use_wrist_cameras=self.use_wrist_cameras,
                    )
                ),
            ],
            outputs=[
                (
                    seg_transforms.MobileSegOutputs(adapt_to_pi=self.adapt_to_pi)
                    if self.use_mobile_action_space
                    else seg_transforms.AlohaSegOutputs(adapt_to_pi=self.adapt_to_pi)
                )
            ],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = (
                _transforms.make_bool_mask(7, -1, 7, -1)
                if self.use_mobile_action_space
                else _transforms.make_bool_mask(6, -1, 6, -1)
            )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=global_prompt)(model_config)
        if self.debug_prompt_prints:
            model_transforms = _transforms.Group(
                inputs=(
                    DebugPrintPrompt(
                        max_prints=self.debug_prompt_prints,
                        prefix=f"[{self.repo_id} prompt]",
                    ),
                    *model_transforms.inputs,
                ),
                outputs=model_transforms.outputs,
            )

        base_config = dataclasses.replace(
            self.base_config or DataConfig(),
            prompt_from_task=False if self.use_global_task else (self.base_config or DataConfig()).prompt_from_task,
        )

        return dataclasses.replace(
            dataclasses.replace(self, base_config=base_config).create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            memory_sequence_keys=tuple(self.memory_image_keys)
            if (self.use_short_horizon_memory or self.use_episodic_memory)
            else (),
            memory_sequence_num_frames=self.memory_num_frames
            if (self.use_short_horizon_memory or self.use_episodic_memory)
            else 0,
            memory_sequence_frame_stride=self.memory_frame_stride,
            prompt_from_task=base_config.prompt_from_task,
            lerobot_roots=lerobot_roots,
        )

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    
    # EMA (Exponential Moving Average) settings
    # If True, enables EMA of model weights for better generalization
    use_ema: bool = False
    # EMA decay rate (typical values: 0.999-0.9999)
    ema_decay: float = 0.999
    # Start updating EMA after this many steps (allows initial training to stabilize)
    ema_update_after_step: int = 100
    # Update EMA every N steps (default: 1 means update every step)
    ema_update_every: int = 10
    
    # Learning rate multiplier for action expert (gemma_expert). If None, uses the same lr as VLM.
    # For example, set to 0.1 to use 10% of the VLM learning rate for action expert.
    # action_expert_lr_multiplier: float | None = None
    action_expert_lr_multiplier: float = 1.0

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./ckpts2"

    # DPO (Direct Preference Optimization) settings
    use_dpo: bool = False
    dpo_lambda: float = 0.1
    dpo_noise_scale: float = 0.1
    dpo_beta: float = 0.1

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of micro-batches to accumulate before each optimizer step.
    gradient_accumulation_steps: int = 1
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # If true, will train a Q-function instead of a value function.
    use_q_function: bool = False

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

_RMBENCH_SEG_REPACK_TRANSFORMS = _transforms.Group(
    inputs=[
        _transforms.RepackTransform(
            {
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
                "seg_cam_high": "observation.segmentation.cam_high",
                "episode_index": "episode_index",
                "frame_index": "frame_index",
                "dataset_index": "dataset_index",
            }
        )
    ]
)

_PI05_RMBENCH_PROGRESS_EVALUATOR_300M_MODEL = pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_300m",
    action_expert_variant="gemma_300m",
    max_token_len=300,
    use_short_horizon_memory=True,
    memory_num_frames=6,
    memory_frame_stride=3,
    memory_temporal_interval=4,
    memory_drop_past_after_layer=20,
    use_progress_head=True,
    num_progress_bins=11,
    progress_loss_weight=1.0,
    subtask_done_loss_weight=1.0,
    progress_head_dropout=0.1,
    progress_head_use_mlp=False,
    done_focal_gamma=2.0,
    done_focal_alpha=0.75,
    pytorch_compile_mode=None,
)

_DATASET_NAMES = (
    "mobile_breakfast_preparation_seg", "mobile_number_ordering_seg",
    "mobile_place_fruit_bowl_seg", "mobile_sorting_object_seg",
)
_DATA_ROOT = pathlib.Path(os.environ.get("MOBILE_DATA_ROOT", "./data"))
_CONFIGS = [TrainConfig(
    name="pi05_mobile_atomic_4task_progress_evaluator_300m_stride3",
    model=_PI05_RMBENCH_PROGRESS_EVALUATOR_300M_MODEL,
    data=LeRobotAlohaSegDataConfig(
        repo_id=[f"anonymous/{name}" for name in _DATASET_NAMES],
        assets=AssetsConfig(asset_id="pi05_mobile_atomic_4task_seg",
            assets_dir="./assets/pi05_mobile_atomic_4task_short_horizon_memory_stride3"),
        lerobot_roots=tuple(str(_DATA_ROOT / name) for name in _DATASET_NAMES),
        adapt_to_pi=False, use_mobile_action_space=True,
        p_drop_seg=0.0, p_drop_image=0.1, p_keep_wrist_on_drop=0.90,
        seg_type_weights=(0.34, 0.33, 0.33), overlay_style="contour",
        overlay_alpha=0.4, coord_bins=256, debug_prompt_prints=5,
        repack_transforms=_RMBENCH_SEG_REPACK_TRANSFORMS,
        base_config=DataConfig(prompt_from_task=True),
        use_progress_labels=True, use_subtask_seg=True,
        use_short_horizon_memory=True, memory_num_frames=6,
        memory_frame_stride=3, progress_num_bins=11, progress_boundary_margin=4,
    ),
    batch_size=20, num_train_steps=50000, save_interval=10000,
    weight_loader=weight_loaders.NoOpWeightLoader(), pytorch_weight_path=None,
    freeze_filter=_PI05_RMBENCH_PROGRESS_EVALUATOR_300M_MODEL.get_freeze_filter(),
    wandb_enabled=False,
)]
_ACTION_MODEL = pi0_config.Pi0Config(
    pi05=True, paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora", max_token_len=300,
    use_short_horizon_memory=True, memory_num_frames=6, memory_frame_stride=3,
    memory_temporal_interval=4, memory_drop_past_after_layer=20,
)
_CONFIGS.append(TrainConfig(
    name="pi05_mobile_atomic_4task_short_horizon_memory_stride3",
    model=_ACTION_MODEL,
    data=dataclasses.replace(
        _CONFIGS[0].data, p_drop_seg=0.2, use_progress_labels=False,
        progress_num_bins=101, progress_boundary_margin=8,
    ),
    batch_size=20, num_train_steps=50000, save_interval=10000,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"),
    freeze_filter=_ACTION_MODEL.get_freeze_filter(), wandb_enabled=False,
))

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
