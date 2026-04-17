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

import rclpy

from dual_arm_gpu_mpc.config.loader import ConfigModule
from dual_arm_gpu_mpc.controllers.high_level.mppi.kmeans_adpan import MPPIKmeansAdpAnModule


def main(args=None):
    os.environ["ROS_DOMAIN_ID"] = "16"
    rclpy.init(args=args)
    mppi_module = None

    desire_abs_pose = [-0.009809, -0.700866, -0.008828, 0.713171, -0.02773, 0.000088, -0.342689, -0.004537]
    desire_abs_position = [0.45, 0.0, 0.52]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.001626, -0.002018, 0.28023, 0.00204]
    desire_line_d = [0.0, 0.0, 0.0, 1.0]
    desire_quat_line_ref = [0.0, -0.9995, -0.026341, 0.017418]
    try:
        config_path = os.path.join(os.path.dirname(__file__), "ur3_and_ur3e.yaml")
        config = ConfigModule(config_path)
        mppi_module = MPPIKmeansAdpAnModule(
            config,
            desire_abs_pose,
            desire_abs_position,
            desire_rel_pose,
            desire_line_d,
            desire_quat_line_ref,
        )
        mppi_module.warm_up()

        while True:
            mppi_module.play_once()
    except KeyboardInterrupt:
        pass
    finally:
        if mppi_module is not None:
            mppi_module.shutdown()
        elif rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
