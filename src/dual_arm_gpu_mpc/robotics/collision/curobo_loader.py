from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

from dual_arm_gpu_mpc.common.paths import (
    _resolve_existing,
    configs_dir,
    legacy_model_dir,
    project_root,
    resolve_robot_config,
    resolve_world_config,
)


def resolve_curobo_config_paths(robot_config: str, world_config: str) -> tuple[Path, Path]:
    return resolve_robot_config(robot_config), resolve_world_config(world_config)


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in curobo config {path}, got {type(loaded).__name__}")

    return loaded


def _resolve_repo_path(path_like: str | Path, *, base_dir: Path | None = None) -> str:
    path = Path(path_like)
    if path.is_absolute():
        return str(path)

    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(base_dir / path)
    if path.parts and path.parts[0] == "robot" and len(path.parts) > 1:
        candidates.append(legacy_model_dir() / Path(*path.parts[1:]))
    candidates.extend(
        [
            project_root() / path,
            configs_dir() / path,
            legacy_model_dir() / path,
        ]
    )
    return str(_resolve_existing(path_like, candidates))


def _normalize_robot_kinematics(kinematics: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    normalized = deepcopy(kinematics)
    for key in ("urdf_path", "usd_path", "usd_robot_root", "isaac_usd_path", "asset_root_path"):
        value = normalized.get(key)
        if isinstance(value, str) and value:
            normalized[key] = _resolve_repo_path(value, base_dir=base_dir)

    for key in ("collision_spheres", "extra_collision_spheres"):
        value = normalized.get(key)
        if isinstance(value, str) and value:
            normalized[key] = _resolve_repo_path(value, base_dir=base_dir)

    return normalized


def _load_robot_config(robot_path: Path) -> dict[str, Any]:
    config_data = _load_yaml_dict(robot_path)
    curobo_robot_config = config_data.get("curobo_robot_config")
    if isinstance(curobo_robot_config, str) and curobo_robot_config:
        robot_path = resolve_robot_config(curobo_robot_config)
        config_data = _load_yaml_dict(robot_path)

    robot_cfg = config_data.get("robot_cfg")
    if isinstance(robot_cfg, dict):
        config_data = deepcopy(config_data)
        normalized_robot_cfg = deepcopy(robot_cfg)
        kinematics = normalized_robot_cfg.get("kinematics")
        if isinstance(kinematics, dict):
            normalized_robot_cfg["kinematics"] = _normalize_robot_kinematics(kinematics, base_dir=robot_path.parent)
        config_data["robot_cfg"] = normalized_robot_cfg

    return config_data


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
        robot_config=_load_robot_config(robot_path),
        world_model=_load_yaml_dict(world_path),
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


def update_world_obstacle_pose(
    robot_world: RobotWorld,
    obstacle_name: str,
    position: list[float] | tuple[float, float, float],
    *,
    tensor_args: TensorDeviceType,
) -> None:
    if robot_world.world_model is None or not obstacle_name:
        return

    obstacle_pose = Pose.from_list(
        [float(position[0]), float(position[1]), float(position[2]), 1.0, 0.0, 0.0, 0.0],
        tensor_args,
    )
    robot_world.world_model.update_obstacle_pose(
        obstacle_name,
        obstacle_pose,
        update_cpu_reference=True,
    )
