import ast
from pathlib import Path

import pytest

from openpi.training import config

NAME = "pi05_mobile_atomic_4task_progress_evaluator_300m_stride3"


def test_only_requested_presets_are_registered():
    assert list(config._CONFIGS_DICT) == [NAME, "pi05_mobile_atomic_4task_short_horizon_memory_stride3"]
    with pytest.raises(ValueError, match="not found"):
        config.get_config("unreleased_experiment")
    c = config.get_config(NAME)
    assert (c.batch_size, c.num_train_steps, c.save_interval) == (20, 50000, 10000)
    assert c.model.paligemma_variant == "gemma_300m"
    assert c.model.use_progress_head and c.model.use_short_horizon_memory
    assert (c.model.memory_num_frames, c.model.memory_frame_stride) == (6, 3)
    assert c.model.num_progress_bins == c.data.progress_num_bins == 11
    assert c.data.progress_boundary_margin == 4
    assert c.data.use_subtask_seg and c.data.use_progress_labels
    assert c.data.use_mobile_action_space and not c.data.adapt_to_pi
    assert len(c.data.repo_id) == len(c.data.lerobot_roots) == 4
    assert c.pytorch_weight_path is None
    assert not c.wandb_enabled


def test_all_internal_imports_are_in_release():
    root = Path(__file__).resolve().parents[1]
    roots = {"openpi": root / "src", "openpi_client": root / "packages/openpi-client/src"}
    def exists(module):
        base = roots[module.split(".")[0]] / module.replace(".", "/")
        return base.with_suffix(".py").is_file() or base.is_dir()
    for path in [*root.glob("src/**/*.py"), *root.glob("scripts/*.py"), *root.glob("packages/**/*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
                # A namespace-package import must include its referenced modules.
                package = roots.get(node.module.split(".")[0])
                if package and (package / node.module.replace(".", "/")).is_dir():
                    modules += [node.module + "." + alias.name for alias in node.names]
            for module in modules:
                if module.split(".")[0] in roots:
                    assert exists(module), (path.relative_to(root), module)


def test_action_policy_settings():
    c = config.get_config("pi05_mobile_atomic_4task_short_horizon_memory_stride3")
    assert (c.batch_size, c.num_train_steps, c.save_interval) == (20, 50000, 10000)
    assert c.model.paligemma_variant == "gemma_2b_lora"
    assert c.model.action_expert_variant == "gemma_300m_lora"
    assert (c.model.memory_num_frames, c.model.memory_frame_stride) == (6, 3)
    assert not c.model.use_progress_head and not c.data.use_progress_labels
    assert c.data.use_subtask_seg and c.data.use_mobile_action_space
    assert c.data.p_drop_seg == 0.2
    assert c.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_base/params"
    assert not c.wandb_enabled
