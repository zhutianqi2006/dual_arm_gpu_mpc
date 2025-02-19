#!/usr/bin/env python
# system library
import os
import time
import numpy as np
import math
from math import pi
import threading
import torch
# curobo for collision detection
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH,DQ_SerialManipulatorMDH, DQ_CooperativeDualTaskSpace
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
# DQ Robotics used in cuda
from dq_torch import rel_abs_pose_rel_jac
from utils.config_module import ConfigModule
from utils.low_level_module import LowLevelModule
# ros2 
import rclpy

def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    # 步骤 3: 实例化DQ_SerialManipulatorDH
    desire_abs_pose = [0.00085, 0.923642, -0.383209, -0.005971, -0.225078, 0.128528, 0.310921, -0.104754]
    desire_rel_pose = [9.63267947e-05,  7.07173586e-01, -7.07039957e-01, -9.63267808e-05, 3.51225857e-06, 2.47457994e-01, 2.47504763e-01, -9.62904111e-07]

    config_path = os.path.join(os.path.dirname(__file__), 'two_franka.yaml')
    config = ConfigModule(config_path)
    low_level_module = LowLevelModule(config, desire_abs_pose, desire_rel_pose)
    while True:
        low_level_module.play_once()


if __name__ == "__main__":
    main()