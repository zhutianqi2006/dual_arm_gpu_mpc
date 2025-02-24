# Python standard lib
import os
import sys
import math
import pathlib
from threading import Lock
# pybullet to display
# import pybullet as pyb
# import pybullet_data
# import pyb_utils
# TIMESTEP = 1/60
# ROS2
import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

from utils.config_module import ConfigModule

class LowROSModule(Node):
    def __init__(self, config: ConfigModule):
        self.ros_node_name = config.ros_node_name
        super().__init__(self.ros_node_name)
        self.dt = 1.0/config.ros_update_rate
        # init robot 1
        self.robot1_q_num = config.robot1_q_num
        self.robot1_q = [0.0] * self.robot1_q_num
        self.robot1_q_name_list = config.robot1_q_name_list
        self.robot1_q_sub_topic = config.robot1_q_sub_topic
        self.robot1_dq_pub_topic = config.robot1_dq_pub_topic
        # init robot 2
        self.robot2_q_num = config.robot2_q_num
        self.robot2_q = [0.0] * self.robot2_q_num
        self.robot2_q_name_list = config.robot2_q_name_list
        self.robot2_q_sub_topic = config.robot2_q_sub_topic
        self.robot2_dq_pub_topic = config.robot2_dq_pub_topic
        # init high level u
        self.high_level_u_topic = config.high_level_u_topic
        self.high_level_u = [0.0] * (self.robot1_q_num + self.robot2_q_num)
        self.setup_ros2()
        # abs error data


    def read_joint_state(self):
        return self.robot1_q, self.robot2_q
    
    def read_high_level_u(self):
        return self.high_level_u
    
    def write_u(self, robot1_dq , robot2_dq):
        self.robot1_dq_msg.velocity = robot1_dq
        self.robot2_dq_msg.velocity = robot2_dq
        self.robot1_pub.publish(self.robot1_dq_msg)
        self.robot2_pub.publish(self.robot2_dq_msg)

    def publish_abs_error_data(self, desire_abs_pose, feedback_abs_pose):
        abs_error = desire_abs_pose - feedback_abs_pose
        for i in range(8):
            self.abs_error_data_msg.data[i] = abs_error[i]
        self.abs_error_data_pub.publish(self.abs_error_data_msg)
    
    def publish_deisre_abs_pose(self, desire_abs_pose):
        for i in range(8):
            self.desire_abs_pose_msg.data[i] = desire_abs_pose[i]
        self.desire_abs_pose_pub.publish(self.desire_abs_pose_msg)

    def publish_current_abs_pose(self, current_abs_pose):
        for i in range(8):
            self.current_abs_pose_msg.data[i] = current_abs_pose[i]
        self.current_abs_pose_pub.publish(self.current_abs_pose_msg)

    def setup_ros2(self):
        # init robot1 ros2 for control
        self.robot1_q_msg = JointState()
        self.robot1_dq_msg = JointState()
        self.robot1_q_msg.name = self.robot1_q_name_list
        self.robot1_dq_msg.name = self.robot1_q_name_list
        self.robot1_dq_msg.velocity = [0.0] * self.robot1_q_num
        self.robot1_sub = self.create_subscription(
            JointState,
            self.robot1_q_sub_topic,
            self.robot1_q_callback,
            0)
        self.robot1_pub = self.create_publisher(
            JointState,
            self.robot1_dq_pub_topic,
            1)
        # init robot2 ros2 for control
        self.robot2_q_msg = JointState()
        self.robot2_dq_msg = JointState()
        self.robot2_q_msg.name = self.robot2_q_name_list
        self.robot2_dq_msg.name = self.robot2_q_name_list
        self.robot2_dq_msg.velocity = [0.0] * self.robot2_q_num
        self.robot2_sub = self.create_subscription(
            JointState,
            self.robot2_q_sub_topic,
            self.robot2_q_callback,
            0)
        self.robot2_pub = self.create_publisher(
            JointState,
            self.robot2_dq_pub_topic,
            1)
        # high level u
        self.high_level_u_msg = Float64MultiArray()
        self.high_level_u_msg.data = [0.0] * (self.robot1_q_num + self.robot2_q_num)
        self.high_level_u_sub = self.create_subscription(
            Float64MultiArray,
            self.high_level_u_topic,
            self.high_level_u_callback,
            0)
        # abs error data
        self.abs_error_data_msg = Float64MultiArray()
        self.abs_error_data_msg.data = [0.0] * 8
        self.abs_error_data_pub = self.create_publisher(
            Float64MultiArray,
            'abs_error_data',
            1)
        # desire abs pose
        self.desire_abs_pose_msg = Float64MultiArray()
        self.desire_abs_pose_msg.data = [0.0] * 8
        self.desire_abs_pose_pub = self.create_publisher(
            Float64MultiArray,
            'desire_abs_pose',
            1)
        # current abs pose
        self.current_abs_pose_msg = Float64MultiArray()
        self.current_abs_pose_msg.data = [0.0] * 8
        self.current_abs_pose_pub = self.create_publisher(
            Float64MultiArray,
            'current_abs_pose',
            1)

        
    def high_level_u_callback(self, msg: Float64MultiArray):
        self.high_level_u = list(msg.data)

    def robot1_q_callback(self, msg: JointState):
        self.robot1_q = list(msg.position[:self.robot1_q_num])
        self.robot1_init_flag = True
        
    def robot2_q_callback(self, msg: JointState):
        self.robot2_q = list(msg.position[:self.robot2_q_num])
        self.robot2_init_flag = True
    
    def run(self):
        rclpy.spin(self)
        self.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=None)
    config_path = os.path.join(os.path.dirname(__file__), 'ur3_and_ur3e.yaml')
    config = ConfigModule(config_path)
    ros_module = LowROSModule(config)

