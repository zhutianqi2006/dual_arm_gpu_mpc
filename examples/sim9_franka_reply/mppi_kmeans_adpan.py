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
from utils.mppi_kmeans_adpan_module import MPPIKmeansAdpAnModule
# 
import rclpy
import array


def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    # 步骤 3: 实例化DQ_SerialManipulatorDH
    desire_abs_pose = [-0.743697, 0.002919, 0.668503, 0.003245, 0.001775, -0.011929, 0.003932, -0.392559]
    desire_abs_position = [-0.507146, -0.006007, 0.599849]
    desire_rel_pose = [-0.161469, 0.982784, -0.000002, 0.089794, 0.000098, -0.000579, 0.420438, 0.006521]
    desire_line_d = [0, 0, 0, 1]
    desire_quat_line_ref = [0, -0.994308, 0.00868, 0.106191]
    config_path = os.path.join(os.path.dirname(__file__), 'two_franka_r9.yaml')
    config = ConfigModule(config_path)
    mppi_module =  MPPIKmeansAdpAnModule(config, desire_abs_pose, desire_abs_position, desire_rel_pose, desire_line_d, desire_quat_line_ref)
    mppi_module.warm_up()
    while True:
        mppi_module.play_once()


if __name__ == "__main__":
    main()