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
    desire_abs_pose = [-0.011961693390801373, -0.701712136452997, -0.008064925867787953, 0.7123145038650708, 0.12203720037624906, -0.002276549048810388, -0.1960031380395545, -0.0024125035955698537]
    # init abs pose
    # desire_abs_pose = [-0.011961, -0.701712, -0.008065, 0.712315, 0.122037, -0.002277, -0.196003, -0.002413]
    desire_abs_position = [0.450595, 0.000028, 0.101239]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.054285, - 0.000927, - 0.262089, - 0.003409]
    desire_line_d = [0,0,0,1]
    desire_quat_line_ref = [0,-0.9995,-0.026341,0.017418]
    config_path = os.path.join(os.path.dirname(__file__), 'ur3_and_ur3e.yaml')
    config = ConfigModule(config_path)
    mppi_module = MPPIAdpAnModule(config, desire_abs_pose, desire_abs_position, desire_rel_pose, desire_line_d, desire_quat_line_ref)
    mppi_module.warm_up()
    while True:
        mppi_module.play_once()



if __name__ == "__main__":
    main()