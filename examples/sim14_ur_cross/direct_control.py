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
from utils.direct_conrtol_module import DirectConrtolModule
import rclpy
import time
def main(args=None):
    time.sleep(5)
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    # 步骤 3: 实例化DQ_SerialManipulatorDH
    desire_abs_pose = [0.055365, - 0.588516, 0.051009, 0.804973, - 0.025382, - 0.005936, - 0.240929, 0.012673]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.001626, - 0.002018, 0.28023, 0.00204]

    config_path = os.path.join(os.path.dirname(__file__), 'ur3_and_ur3e.yaml')
    config = ConfigModule(config_path)
    direct_control_module = DirectConrtolModule(config, desire_abs_pose, desire_rel_pose)
    while True:
        direct_control_module.play_once()


if __name__ == "__main__":
    main()