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
import rclpy
import array
import datetime


def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    desire_abs_pose = [0.055365, - 0.588516, 0.051009, 0.804973, - 0.025382, - 0.005936, - 0.240929, 0.012673]
    # init abs pose
    # desire_abs_pose = [-0.011961, -0.701712, -0.008065, 0.712315, 0.122037, -0.002277, -0.196003, -0.002413]
    desire_abs_position = [0.45, 0.0, 0.35]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.054285, - 0.000927, - 0.262089, - 0.003409]
    desire_line_d = [0,0,0,1]
    desire_quat_line_ref = [0,-0.9995,-0.026341,0.017418]
    config_path = os.path.join(os.path.dirname(__file__), 'ur3_and_ur3e_man.yaml')
    config = ConfigModule(config_path)
    mppi_module = MPPILogStdModule(config, desire_abs_pose, desire_abs_position, desire_rel_pose, desire_line_d, desire_quat_line_ref)

    # record manipulability trace
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mppi_module.manip_log_path = os.path.join(os.path.dirname(__file__), "..", "..", "logs", f"manip_trace_{stamp}.npz")
    mppi_module.manip_log_flush_every = 50
    mppi_module.manip_log_print_every = 50

    # debug: print relative_jacobian[0] and its rank
    mppi_module.debug_print_rel_jacobian = True
    mppi_module.debug_print_rel_jacobian_every = 1

    # debug: print converted twist_jacobian[0] (6x12) and its rank
    mppi_module.debug_print_twist_jacobian = True
    mppi_module.debug_print_twist_jacobian_every = 10

    # debug: print singular values and Gram eigenvalues
    mppi_module.debug_print_twist_spectrum = True
    mppi_module.debug_print_twist_spectrum_every = 10
    mppi_module.debug_twist_gram_eig_last3_div = 5.0
    mppi_module.warm_up()
    while True:
        mppi_module.play_once()



if __name__ == "__main__":
    main()