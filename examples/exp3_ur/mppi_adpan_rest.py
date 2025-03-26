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
from utils.mppi_adpan_module import MPPIAdpAnModule
# 
import rclpy
import array


def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    desire_abs_pose = [-0.0014448813045894147, -0.6833794594621869, 0.010435169865650795, 0.729987351899373, 0.06228083685414962, 0.07360905319880795, -0.17842947213147717, 0.07158322006278149]
    desire_abs_position = [0.3469, 0.20452, 0.151]
    # desire_abs_pose = [0.005744, - 0.683663, 0.020305, 0.729493,  0.06289854,  0.07243161, -0.17836156,  0.07235046]
    # desire_abs_position = [0.35, 0.20, 0.15]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.054285, - 0.000927, - 0.262089, - 0.003409]
    desire_line_d = [0,0,0,1]
    desire_quat_line_ref = [0,-0.9995,-0.026341,0.017418]
    config_path = os.path.join(os.path.dirname(__file__), 'ur3_and_ur3e_no_obstacle.yaml')
    config = ConfigModule(config_path)
    mppi_module = MPPIAdpAnModule(config, desire_abs_pose, desire_abs_position, desire_rel_pose, desire_line_d, desire_quat_line_ref)
    mppi_module.warm_up()
    while True:
        mppi_module.play_once()



if __name__ == "__main__":
    main()