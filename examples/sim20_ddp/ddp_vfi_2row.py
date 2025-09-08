import os
import time
import numpy as np
import math
from math import pi
import threading
# DQ Robotics cpu
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH, DQ_SerialManipulatorMDH, DQ_CooperativeDualTaskSpace
# DQ Robotics used in cuda
from utils.config_module import ConfigModule
from utils.high_ros_module import HighROSModule
from utils.crocoddyl_ddp_vfi_module import CrocoddylDDPVFIModule
# 
import rclpy
import array

if __name__ == "__main__":
    # 示例参数（请根据实际修改）
    os.environ['ROS_DOMAIN_ID'] = '16'
    import rclpy
    import time
    time.sleep(2.0)
    rclpy.init(args=None)
    desire_abs_pose = [- 0.009809, - 0.700866, - 0.008828, 0.713171, - 0.002769, - 0.000221, - 0.318158, - 0.004193]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.001626, -0.002018, 0.28023, 0.00204]

    cfg_path = os.path.join(os.path.dirname(__file__), "ur3_and_ur3e_2row.yaml")
    cfg = ConfigModule(cfg_path)

    # 可选：障碍（球）
    obstacles = [
        {"center": np.array([0.5, 0.00, 0.25]), "radius": 0.04, "buffer": 0.01, "weight": 2e6, "color": [1, 0, 0, 0.35]},
        {"center": np.array([0.5, 0.06, 0.25]), "radius": 0.04, "buffer": 0.01, "weight": 2e6, "color": [1, 0, 0, 0.35]},
        {"center": np.array([0.5, -0.06, 0.25]), "radius": 0.04, "buffer": 0.01, "weight": 2e6, "color": [1, 0, 0, 0.35]},
        {"center": np.array([0.5, -0.12, 0.25]), "radius": 0.04, "buffer": 0.01, "weight": 2e6, "color": [1, 0, 0, 0.35]},
        {"center": np.array([0.5, 0.12, 0.25]), "radius": 0.04, "buffer": 0.01, "weight": 2e6, "color": [1, 0, 0, 0.35]},
        {"center": np.array([0.45, 0.00, 0.25]), "radius": 0.04, "buffer": 0.01, "weight": 2e6, "color": [1, 0, 0, 0.35]},
        {"center": np.array([0.45, 0.06, 0.25]), "radius": 0.04, "buffer": 0.01, "weight": 2e6, "color": [1, 0, 0, 0.35]},
        {"center": np.array([0.45, -0.06, 0.25]), "radius": 0.04, "buffer": 0.01, "weight": 2e6, "color": [1, 0, 0, 0.35]},
        {"center": np.array([0.45, -0.12, 0.25]), "radius": 0.04, "buffer": 0.01, "weight": 2e6, "color": [1, 0, 0, 0.35]},
        {"center": np.array([0.45, 0.12, 0.25]), "radius": 0.04, "buffer": 0.01, "weight": 2e6, "color": [1, 0, 0, 0.35]},
    ]

    ctrl = CrocoddylDDPVFIModule(cfg, desire_abs_pose, desire_rel_pose, obstacles)
    ctrl.warm_up()
    print("[CrocoddylDDPVFIModule] running. Press Ctrl+C to stop.")
    try:
        while True:
            ctrl.play_once()
            time.sleep(max(0.03, ctrl.ddp_dt))
    except KeyboardInterrupt:
        pass
