from pathlib import Path
import sys

_EXAMPLE_PROJECT_ROOT = next(
    (
        candidate
        for candidate in Path(__file__).resolve().parents
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir()
    ),
    None,
)
if _EXAMPLE_PROJECT_ROOT is None:
    raise RuntimeError("Unable to resolve dual_arm_gpu_mpc project root for example script.")
_EXAMPLE_PROJECT_ROOT_STR = str(_EXAMPLE_PROJECT_ROOT)
_EXAMPLE_SRC_ROOT_STR = str(_EXAMPLE_PROJECT_ROOT / "src")
if _EXAMPLE_PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _EXAMPLE_PROJECT_ROOT_STR)
if _EXAMPLE_SRC_ROOT_STR not in sys.path:
    sys.path.insert(0, _EXAMPLE_SRC_ROOT_STR)

from dual_arm_gpu_mpc.common.example_bootstrap import bootstrap_example_paths

bootstrap_example_paths(__file__)
# Python standard lib
import os
import rtde_control
import rtde_receive
import numpy as np
# DQ Robotics
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH, DQ_CooperativeDualTaskSpace
from math import pi
import time
# ROS2
import rclpy
import rclpy.logging
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class DualArmDQController(Node):
    def __init__(self):
        super().__init__('dual_arm_dq_controller')
        self.setup_ros2()

    def setup_ros2(self):
        # init ros2 message for control
        self.ur3_vel_msg = JointState()
        self.ur3e_vel_msg = JointState()
        self.ur3_vel_msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ur3e_vel_msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                  'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ur3_vel_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ur3e_vel_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ur3_current_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ur3e_current_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ur3_last_quat = [0.0, 0.0, 0.0, 1.0]
        self.ur3e_last_quat = [0.0, 0.0, 0.0, 1.0]
        self.ur3_init_flag = False
        self.ur3e_init_flag = False
        self.refer_init_flag = False
        # init ros2 message for record
        self.record_data_msg = Float64MultiArray()

        self.publisher_ur3_velocity = self.create_publisher(
            JointState,
            'ur3_joint_command',
            1)
        self.publisher_ur3e_velocity = self.create_publisher(
            JointState,
            'ur3e_joint_command',
            1)
        self.timer = self.create_timer(0.008, self.velocity_publisher_callback)


    def velocity_publisher_callback(self):
        ur3_joint_vel = np.array([0.0]*6)
        ur3e_joint_vel = np.array([0.0]*6)
        self.ur3_vel_msg.velocity = ur3_joint_vel.tolist()
        self.ur3e_vel_msg.velocity = ur3e_joint_vel.tolist()
        self.publisher_ur3_velocity.publish(self.ur3_vel_msg)
        self.publisher_ur3e_velocity.publish(self.ur3e_vel_msg)

def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    dual_arm_model= DualArmDQController()
    rclpy.spin(dual_arm_model)

if __name__ == "__main__":
    main()

