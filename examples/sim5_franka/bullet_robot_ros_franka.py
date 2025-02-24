# Python standard lib
import os
import sys
import math
import pathlib
from threading import Lock
# pybullet to display
import pybullet as pyb
import pybullet_data
import pyb_utils
TIMESTEP = 1/125
# ROS2
import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class DualArmBulletModel(Node):
    def __init__(self, franka1_q, franka2_q,
                 dt: float = 0.01):
        super().__init__('dual_arm_model')
        self.dt = dt
        self.franka1_q = franka1_q
        self.franka2_q = franka2_q
        self.gui_id = pyb.connect(pyb.GUI)
        self.pyb_dual_robot, self.obstacles = self.pyb_load_environment(self.gui_id) 
        self.setup_ros2()

    def setup_ros2(self):
        # init robot1
        self.franka1_pos_msg = JointState()
        self.franka1_vel_msg = JointState()
        self.franka1_pos_msg.name = ['joint1', 'joint2', 'joint3',
                                     'joint4', 'joint5', 'joint6', 'joint7']
        self.franka1_vel_msg.name = ['joint1', 'joint2', 'joint3',
                                     'joint4', 'joint5', 'joint6', 'joint7']
        self.franka1_vel_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.franka1_current_joint_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.franka1_current_joint_pos = self.franka1_q
        self.publisher_franka1 = self.create_publisher(JointState,'franka1_joint_states', 1)
        self.subscription_franka1_velocity = self.create_subscription(
            JointState, 'franka1_joint_command', self.franka1_joint_vel_callback, 1)
        # init robot2
        self.franka2_pos_msg = JointState()
        self.franka2_vel_msg = JointState()
        self.franka2_pos_msg.name = ['joint1', 'joint2', 'joint3',
                                     'joint4', 'joint5', 'joint6', 'joint7']
        self.franka2_vel_msg.name = ['joint1', 'joint2', 'joint3',
                                     'joint4', 'joint5', 'joint6', 'joint7']
        self.franka2_vel_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.franka2_current_joint_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.franka2_current_joint_pos = self.franka2_q
        self.publisher_franka2 = self.create_publisher(JointState, 'franka2_joint_states', 1)
        self.subscription_franka2_velocity = self.create_subscription(
            JointState, 'franka2_joint_command', self.franka2_joint_vel_callback, 1)
        # setup ros2 publishers and subscribers
        self.timer = self.create_timer(self.dt, self.joint_pos_pub)
        # init subscriber for obstacle
        self.obstacle_sub = self.create_subscription(
            Float64MultiArray, 'moving_obstacle', self.obstacle_callback, 1)
        
    def obstacle_callback(self, msg:Float64MultiArray):
        cube4_id = self.obstacles["cube4"]
        pyb.resetBasePositionAndOrientation(cube4_id, msg.data[0:3], [1,0,0,0])

    def franka1_joint_vel_callback(self, msg:JointState):
        self.franka1_current_joint_vel = np.array(msg.velocity[:7])
        
    def franka2_joint_vel_callback(self, msg:JointState):
        self.franka2_current_joint_vel = np.array(msg.velocity[:7])

    def pyb_load_environment(self,client_id):
        pyb.setTimeStep(TIMESTEP, physicsClientId=client_id)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        # ground plane
        ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id)
        dual_arm_robot_id = pyb.loadURDF(
            "model/dual_panda_model/dual_panda_urdf.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id
        )
        dual_arm_robot = pyb_utils.Robot(dual_arm_robot_id, client_id=client_id)
        # some cubes for obstacles
        cube2_id = pyb.loadURDF(
            "model/plane/thine_plane.urdf", [0.50, 0.02, 0.58], useFixedBase=True, physicsClientId=client_id
        )
        cube3_id = pyb.loadURDF(
            "model/plane/thine_plane.urdf", [0.50, 0.02, 0.83], useFixedBase=True, physicsClientId=client_id
        )
        cube4_id = pyb.loadURDF(
            "model/plane/plane.urdf", [0.50, 0.02, 100.440], useFixedBase=True, physicsClientId=client_id
        )
        # store body indices in a dict with more convenient key names
        obstacles = {
            "ground": ground_id,
            "cube2": cube2_id,
            "cube3": cube3_id,
            "cube4": cube4_id,
        }
        pyb.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=50,
        cameraPitch=-40,
        cameraTargetPosition=[0, 0, 0.1]
        )
        return dual_arm_robot, obstacles
    
    def update_cube_position(obstacles, client_id, cube_name, new_position, new_orientation=[0, 0, 0, 1]):
        if cube_name in obstacles:
            cube_id = obstacles[cube_name]
            pyb.resetBasePositionAndOrientation(cube_id, new_position, new_orientation, physicsClientId=client_id)
        else:
            print(f"Cube {cube_name} 不存在。")

    def pyb_update_joint_state(self):
        self.pyb_dual_robot.reset_joint_configuration(self.dual_arm_joint_pos)
        
    def joint_pos_pub(self):
        # robot1
        self.franka1_current_joint_pos += self.dt*self.franka1_current_joint_vel
        self.franka1_pos_msg.position = self.franka1_current_joint_pos.tolist()
        self.publisher_franka1.publish(self.franka1_pos_msg)
        # robot2
        self.franka2_current_joint_pos += self.dt*self.franka2_current_joint_vel
        self.franka2_pos_msg.position = self.franka2_current_joint_pos.tolist()
        self.publisher_franka2.publish(self.franka2_pos_msg)
        # robot1 and robot2 gripper
        robot1_gripper_pos =np.array([0.01, 0.01])
        robot2_gripper_pos =np.array([0.01, 0.01])
        # concatenate joint positions
        self.dual_arm_joint_pos = np.concatenate((self.franka1_current_joint_pos, robot1_gripper_pos, self.franka2_current_joint_pos, robot2_gripper_pos))
        self.pyb_update_joint_state()

def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    franka1_q = np.array([0, -1.0471, 0, -2.6178,  1.5707,  1.5707, 0.7853])
    franka2_q = np.array([0, -1.0471, 0, -2.6178, -1.5707, 1.5707, 0.7853])
    dual_arm_model= DualArmBulletModel(franka1_q , franka2_q, 0.008)
    rclpy.spin(dual_arm_model)
  
if __name__ == "__main__":
    main()