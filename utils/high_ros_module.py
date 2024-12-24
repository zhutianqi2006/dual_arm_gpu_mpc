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
class HighROSModule(Node):
    def __init__(self, config: ConfigModule):
        self.ros_node_name = config.ros_node_name
        super().__init__(self.ros_node_name)
        self.dt = 1.0/config.ros_update_rate
        # init robot 1
        self.robot1_q_num = config.robot1_q_num
        self.robot1_q = [0.0] * self.robot1_q_num
        self.robot1_q_name_list = config.robot1_q_name_list
        self.robot1_q_sub_topic = config.robot1_q_sub_topic
        # init robot 2
        self.robot2_q_num = config.robot2_q_num
        self.robot2_q = [0.0] * self.robot2_q_num
        self.robot2_q_name_list = config.robot2_q_name_list
        self.robot2_q_sub_topic = config.robot2_q_sub_topic
        # init high level u
        self.high_level_u_topic = config.high_level_u_topic
        self.obstacle_topic = "moving_obstacle"
        self.setup_ros2()

    def read_joint_state(self):
        return self.robot1_q, self.robot2_q
    
    def setup_ros2(self):
        # init robot1 ros2 for control
        self.robot1_q_msg = JointState()
        self.robot1_q_msg.name = self.robot1_q_name_list
        self.robot1_sub = self.create_subscription(
            JointState,
            self.robot1_q_sub_topic,
            self.robot1_q_callback,
            0)
        # init robot2 ros2 for control
        self.robot2_q_msg = JointState()
        self.robot2_q_msg.name = self.robot2_q_name_list
        self.robot2_sub = self.create_subscription(
            JointState,
            self.robot2_q_sub_topic,
            self.robot2_q_callback,
            0)
        # init high level u ros2 for control
        self.high_level_u_msg = Float64MultiArray()
        self.high_level_u_msg.data = [0.0] * (self.robot1_q_num + self.robot2_q_num)
        self.high_level_u_pub = self.create_publisher(
            Float64MultiArray,
            self.high_level_u_topic,
            1)
        # init obstacle ros2 for control
        self.obstacle_msg = Float64MultiArray()
        self.obstacle_msg.data = [0.0] * 30
        self.obstacle_pub = self.create_publisher(
            Float64MultiArray,
            self.obstacle_topic,
            1)
        
    def write_high_u(self, mppi_u0):
        for i in range(self.robot1_q_num+self.robot2_q_num):
            self.high_level_u_msg.data[i] = mppi_u0[i]
        self.high_level_u_pub.publish(self.high_level_u_msg)

    def write_obstacle(self, obstacle, obstacle_num:int):
        for i in range(3*obstacle_num):
            self.obstacle_msg.data[i] = obstacle[i]
        self.obstacle_pub.publish(self.obstacle_msg)

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

