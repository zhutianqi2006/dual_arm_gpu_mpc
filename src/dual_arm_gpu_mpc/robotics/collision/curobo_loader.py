from __future__ import annotations

from pathlib import Path

from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

from dual_arm_gpu_mpc.common.paths import resolve_robot_config, resolve_world_config


def resolve_curobo_config_paths(robot_config: str, world_config: str) -> tuple[Path, Path]:
    return resolve_robot_config(robot_config), resolve_world_config(world_config)


def load_robot_world_config(
    robot_config: str,
    world_config: str,
    *,
    tensor_args: TensorDeviceType | None = None,
    collision_activation_distance: float,
    self_collision_activation_distance: float,
) -> RobotWorldConfig:
    tensor_args = tensor_args or TensorDeviceType()
    robot_path, world_path = resolve_curobo_config_paths(robot_config, world_config)
    return RobotWorldConfig.load_from_config(
        str(robot_path),
        str(world_path),
        tensor_args=tensor_args,
        collision_activation_distance=collision_activation_distance,
        self_collision_activation_distance=self_collision_activation_distance,
    )


def build_robot_world_pair(
    robot_config: str,
    world_config: str,
    *,
    collision_activation_distance: float,
    self_collision_activation_distance: float,
) -> tuple[TensorDeviceType, RobotWorldConfig, RobotWorld, RobotWorld]:
    tensor_args = TensorDeviceType()
    config = load_robot_world_config(
        robot_config,
        world_config,
        tensor_args=tensor_args,
        collision_activation_distance=collision_activation_distance,
        self_collision_activation_distance=self_collision_activation_distance,
    )
    return tensor_args, config, RobotWorld(config), RobotWorld(config)

