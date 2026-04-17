import importlib.util

import pytest

from dual_arm_gpu_mpc.common.paths import project_root
from dual_arm_gpu_mpc.robotics.collision.curobo_loader import (
    build_robot_world_pair,
    load_robot_world_config,
    resolve_curobo_config_paths,
    update_world_obstacle_pose,
)


def _load_example_bootstrap():
    module_path = project_root() / "src" / "dual_arm_gpu_mpc" / "common" / "example_bootstrap.py"
    spec = importlib.util.spec_from_file_location("example_bootstrap", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_curobo_resource_resolution_uses_absolute_paths():
    robot_path, world_path = resolve_curobo_config_paths("dual_arm_model_real.yml", "dual_arm_collision_env.yml")

    assert robot_path.is_absolute()
    assert world_path.is_absolute()
    assert robot_path.exists()
    assert world_path.exists()
    assert project_root() in robot_path.parents
    assert project_root() in world_path.parents


def test_curobo_loader_resolves_legacy_robot_assets():
    config = load_robot_world_config(
        "dual_arm_model_real.yml",
        "ur_p2p_exp1_env.yml",
        collision_activation_distance=0.0,
        self_collision_activation_distance=0.0,
    )

    assert config is not None


def test_example_bootstrap_patches_legacy_curobo_loader_strings():
    example_bootstrap = _load_example_bootstrap()
    example_bootstrap.patch_legacy_curobo_load_from_config()

    from curobo.wrap.model.robot_world import RobotWorldConfig

    config = RobotWorldConfig.load_from_config(
        "dual_arm_model_real.yml",
        "ur_p2p_exp1_env.yml",
        collision_activation_distance=0.0,
        self_collision_activation_distance=0.0,
    )

    assert config is not None


def test_dynamic_world_config_supports_runtime_obstacle_pose_updates():
    tensor_args, _, robot_world, _ = build_robot_world_pair(
        "dual_arm_model_real.yml",
        "ur_dynamic_exp1_env.yml",
        collision_activation_distance=0.0,
        self_collision_activation_distance=0.0,
    )

    update_world_obstacle_pose(
        robot_world,
        "moving_obstacle",
        [0.32, 0.10, 0.38],
        tensor_args=tensor_args,
    )

    updated_pose = robot_world.world_model.world_model.get_obstacle("moving_obstacle").pose
    assert updated_pose[:3] == pytest.approx([0.32, 0.10, 0.38])
