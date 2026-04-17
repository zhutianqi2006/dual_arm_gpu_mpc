#!/usr/bin/env python
from pathlib import Path
import sys

_EXAMPLE_PROJECT_ROOT = next(
    (
        candidate
        for candidate in Path(__file__).resolve().parents
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir()
    ),
    None,
)
if _EXAMPLE_PROJECT_ROOT is None:
    raise RuntimeError("Unable to resolve dual_arm_gpu_mpc project root for example script.")
_EXAMPLE_PROJECT_ROOT_STR = str(_EXAMPLE_PROJECT_ROOT)
_EXAMPLE_SRC_ROOT_STR = str(_EXAMPLE_PROJECT_ROOT / "src")
if _EXAMPLE_PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _EXAMPLE_PROJECT_ROOT_STR)
if _EXAMPLE_SRC_ROOT_STR not in sys.path:
    sys.path.insert(0, _EXAMPLE_SRC_ROOT_STR)

from dual_arm_gpu_mpc.common.example_bootstrap import bootstrap_example_paths

bootstrap_example_paths(__file__)

import os
import time

import rclpy

from dual_arm_gpu_mpc.common.example_presets import LEGACY_UR_LOW_LEVEL_TARGETS
from dual_arm_gpu_mpc.config.loader import ConfigModule
from dual_arm_gpu_mpc.controllers.low_level.relative import RelativePoseLowLevelController


def main(args=None):
    os.environ["ROS_DOMAIN_ID"] = "16"
    rclpy.init(args=args)
    low_level_controller = None

    desire_abs_pose = LEGACY_UR_LOW_LEVEL_TARGETS["desire_abs_pose"]
    desire_rel_pose = LEGACY_UR_LOW_LEVEL_TARGETS["desire_rel_pose"]
    try:
        config_path = os.path.join(os.path.dirname(__file__), "ur3_and_ur3e.yaml")
        config = ConfigModule(config_path)
        low_level_controller = RelativePoseLowLevelController(config, desire_abs_pose, desire_rel_pose)

        while True:
            low_level_controller.play_once()
            time.sleep(low_level_controller.control_dt)
    except KeyboardInterrupt:
        pass
    finally:
        if low_level_controller is not None:
            low_level_controller.shutdown()
        elif rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
