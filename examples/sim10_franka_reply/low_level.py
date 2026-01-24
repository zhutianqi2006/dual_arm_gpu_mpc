#!/usr/bin/env python
# system library
import os
import time
import numpy as np
import math
from math import pi
import threading
# curobo for collision detection
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH, DQ_CooperativeDualTaskSpace
# DQ Robotics used in cuda
from utils.config_module import ConfigModule
from utils.low_ros_module import LowROSModule
from utils.low_level_module import LowLevelModule
import rclpy
def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    # 步骤 3: 实例化DQ_SerialManipulatorDH
    desire_abs_pose = [-0.67977, -0.153082, 0.702202, -0.146254, 0.000243, -0.038812, -0.100193, -0.441557]
    desire_rel_pose = [-0.161469, 0.982784, -0.000002, 0.089794, 0.000098, -0.000579, 0.420438, 0.006521]
    
    config_path = os.path.join(os.path.dirname(__file__), 'two_franka_r9.yaml')
    config = ConfigModule(config_path)
    low_level_module = LowLevelModule(config, desire_abs_pose, desire_rel_pose)
    while True:
        low_level_module.play_once()


if __name__ == "__main__":
    main()