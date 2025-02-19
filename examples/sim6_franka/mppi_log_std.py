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
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
# DQ Robotics cpu
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH, DQ_SerialManipulatorMDH, DQ_CooperativeDualTaskSpace
# DQ Robotics used in cuda
from dq_torch import rel_abs_pose_rel_jac
from utils.config_module import ConfigModule
from utils.high_ros_module import HighROSModule
from utils.mppi_log_std_module import MPPILogStdModule
# 
import rclpy
import array


def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    # 步骤 3: 实例化DQ_SerialManipulatorDH
    desire_abs_pose = [0.00085, 0.923642, -0.383209, -0.005971, - 0.077555, 0.113491, 0.278521, - 0.330388]
    desire_abs_position = [0.40, 0.55, 0.60]
    desire_rel_pose = [9.63267947e-05,  7.07244290e-01, -7.06969239e-01, -3.67320509e-06, 3.03159877e-01,  1.23636280e-01,  1.23726146e-01, -8.79988859e-02]
    desire_line_d = [0,0,0,1]
    desire_quat_line_ref = [0, -0.011682, 0.003006, -0.999927]
    config_path = os.path.join(os.path.dirname(__file__), 'two_franka.yaml')
    config = ConfigModule(config_path)
    mppi_module = MPPILogStdModule(config, desire_abs_pose, desire_abs_position, desire_rel_pose, desire_line_d, desire_quat_line_ref)
    mppi_module.warm_up()
    while True:
        mppi_module.play_once()



if __name__ == "__main__":
    main()