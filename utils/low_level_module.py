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
from utils.low_ros_module import LowROSModule
# ROS
import rclpy

import rclpy
class LowLevelModule():
    def __init__(self, config: ConfigModule, desire_abs_pose, desire_rel_pose):
        # init conrtol gain
        self.rel_gain = config.rel_gain
        self.abs_gain = config.abs_gain
        # init desire
        self.desire_rel_pose = DQ(desire_rel_pose)
        self.desire_rel_pose = self.desire_rel_pose.normalize()
        self.desire_abs_pose = DQ(desire_abs_pose)
        self.desire_abs_pose = self.desire_abs_pose.normalize()
        # robot1
        robot1_config_dh_mat = np.array(config.robot1_dh_mat)
        self.robot1_q_num = config.robot1_q_num
        self.robot1_dh_mat =  robot1_config_dh_mat.T
        self.robot1_base = DQ(config.robot1_base)
        self.robot1_base = self.robot1_base.normalize() 
        self.robot1_effector = DQ(config.robot1_effector)
        self.robot1_effector = self.robot1_effector.normalize()
        if(config.robot1_dh_type==1):
            self.cpu_robot1 = DQ_SerialManipulatorMDH(self.robot1_dh_mat)
        else:
            self.cpu_robot1 = DQ_SerialManipulatorDH(self.robot1_dh_mat)
        self.cpu_robot1.set_base_frame(self.robot1_base)
        self.cpu_robot1.set_reference_frame(self.robot1_base)
        self.cpu_robot1.set_effector(self.robot1_effector)
        # robot2
        robot2_config_dh_mat = np.array(config.robot2_dh_mat)
        self.robot2_q_num = config.robot2_q_num
        self.robot2_dh_mat =  robot2_config_dh_mat.T
        self.robot2_base = DQ(config.robot2_base) 
        self.robot2_base = self.robot2_base.normalize()
        self.robot2_effector = DQ(config.robot2_effector)
        self.robot2_effector = self.robot2_effector.normalize()
        if(config.robot2_dh_type==1):
            self.cpu_robot2 = DQ_SerialManipulatorMDH(self.robot2_dh_mat)
        else:
            self.cpu_robot2 = DQ_SerialManipulatorDH(self.robot2_dh_mat)
        self.cpu_robot2.set_base_frame(self.robot2_base)
        self.cpu_robot2.set_reference_frame(self.robot2_base)
        self.cpu_robot2.set_effector(self.robot2_effector)
        # robot1 and robot2
        self.cpu_dq_dual_arm_model = DQ_CooperativeDualTaskSpace(self.cpu_robot1, self.cpu_robot2)
        self.ros_module = LowROSModule(config)
        self.ros_thread = threading.Thread(target=self.ros_module.run)
        self.ros_thread.start()

    def play_once(self):
        self.update_joint_states()
        self.get_high_level_u()
        self.get_u()
        self.send_u()
        self.ros_module.publish_abs_error_data(vec8(self.desire_abs_pose), vec8(self.dual_arm_abs_feedback))
        self.ros_module.publish_current_abs_pose(vec8(self.dual_arm_abs_feedback))


    def update_joint_states(self):
        self.robot1_q, self.robot2_q = self.ros_module.read_joint_state()
        self.dual_arm_joint_pos = np.concatenate((self.robot1_q, self.robot2_q))

    def get_high_level_u(self):
        self.high_level_u = self.ros_module.read_high_level_u()
    
    def get_u(self):
        # relative control
        dual_arm_rel_feedback = vec8(self.cpu_dq_dual_arm_model.relative_pose(self.dual_arm_joint_pos))
        dual_arm_rel_error = vec8(self.desire_rel_pose) - dual_arm_rel_feedback
        dual_arm_rel_jacobian = self.cpu_dq_dual_arm_model.relative_pose_jacobian(self.dual_arm_joint_pos)
        dual_arm_rel_jacobian_roboust_inv = dual_arm_rel_jacobian.T @ np.linalg.pinv(np.matmul(dual_arm_rel_jacobian, dual_arm_rel_jacobian.T) + 0.0000001 * np.eye(8))
        dual_arm_rel_joint_vel = self.rel_gain * np.matmul(dual_arm_rel_jacobian_roboust_inv, (dual_arm_rel_error))
        self.dual_arm_abs_feedback = self.cpu_dq_dual_arm_model.absolute_pose(self.dual_arm_joint_pos)
        self.dual_arm_joint_vel = dual_arm_rel_joint_vel + np.matmul(np.eye(self.robot1_q_num+self.robot2_q_num)-dual_arm_rel_jacobian_roboust_inv@(dual_arm_rel_jacobian), self.high_level_u)

    def send_u(self):
        robot1_dq = list(self.dual_arm_joint_vel[:self.robot1_q_num])
        robot2_dq = list(self.dual_arm_joint_vel[self.robot1_q_num:])
        self.ros_module.write_u(robot1_dq, robot2_dq) 