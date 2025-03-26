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
TIMESTEP = 1/60
# ROS2
import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class DualArmBulletModel(Node):
    def __init__(self, init_ur3_q, init_ur3e_q,
                 dt: float = 0.01):
        super().__init__('dual_arm_model')
        self.dt = dt
        self.ur3_q = init_ur3_q
        self.ur3e_q = init_ur3e_q
        self.gui_id = pyb.connect(pyb.GUI)
        self.pyb_dual_robot, _ = self.pyb_load_environment(self.gui_id) 
        self.setup_ros2()

    def setup_ros2(self):
        # init ros2 message for control
        self.ur3_pos_msg = JointState()
        self.ur3e_pos_msg = JointState()
        self.ur3_vel_msg = JointState()
        self.ur3e_vel_msg = JointState()
        self.ur3_pos_msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ur3e_pos_msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                  'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ur3_vel_msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ur3e_vel_msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                  'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ur3_vel_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ur3e_vel_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ur3_current_joint_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ur3e_current_joint_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ur3_current_joint_pos = self.ur3_q
        self.ur3e_current_joint_pos = self.ur3e_q  
        # setup ros2 publishers and subscribers
        self.publisher_ur3 = self.create_publisher(
            JointState,
            'ur3_joint_states', 
            1)
        self.publisher_ur3e = self.create_publisher(
            JointState, 
            'ur3e_joint_states', 
            1)
        self.subscription_ur3_velocity = self.create_subscription(
            JointState, 'ur3_joint_command', self.ur3_joint_vel_callback, 1)
        self.subscription_ur3e_velocity = self.create_subscription(
            JointState, 'ur3e_joint_command', self.ur3e_joint_vel_callback, 1)
        
        self.timer = self.create_timer(self.dt, self.joint_pos_pub)

    def ur3_joint_vel_callback(self, msg:JointState):
        self.ur3_current_joint_vel = np.array(msg.velocity[:6])
        
    def ur3e_joint_vel_callback(self, msg:JointState):
        self.ur3e_current_joint_vel = np.array(msg.velocity[:6])

    def pyb_load_environment(self,client_id):
        pyb.setTimeStep(TIMESTEP, physicsClientId=client_id)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        # ground plane
        ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id)
        dual_arm_robot_id = pyb.loadURDF(
            "model/dual_arm_model/dual_arm_model.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id
        )
        dual_arm_robot = pyb_utils.Robot(dual_arm_robot_id, client_id=client_id)
        # some cubes for obstacles
        # store body indices in a dict with more convenient key names
        cube2_id = pyb.loadURDF(
            "model/plane/thine_plane.urdf", [0.46, 0.0, 0.006], useFixedBase=True, physicsClientId=client_id
        )
        cube3_id = pyb.loadURDF(
            "model/plane/thine_plane.urdf", [0.46, 0.0, 0.256], useFixedBase=True, physicsClientId=client_id
        )
        cube4_id = pyb.loadURDF(
            "model/plane/thine_plane.urdf", [0.46, 0.0, 0.506], useFixedBase=True, physicsClientId=client_id
        )
        # store body indices in a dict with more convenient key names
        obstacles = {
            "ground": ground_id,
            "cube2": cube2_id,
            "cube3": cube3_id,
            "cube4": cube4_id,
        }
        pyb.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=51,
        cameraPitch=-32,
        cameraTargetPosition=[-0.0, 0.0, 0.0]
        )

        return dual_arm_robot, obstacles
    
    def pyb_update_joint_state(self):
        self.pyb_dual_robot.reset_joint_configuration(self.dual_arm_joint_pos)
        
    def joint_pos_pub(self):
        self.ur3_current_joint_pos += self.dt*self.ur3_current_joint_vel
        self.ur3e_current_joint_pos += self.dt*self.ur3e_current_joint_vel
        self.ur3_pos_msg.position = self.ur3_current_joint_pos.tolist()
        self.ur3e_pos_msg.position = self.ur3e_current_joint_pos.tolist()
        self.publisher_ur3.publish(self.ur3_pos_msg)
        self.publisher_ur3e.publish(self.ur3e_pos_msg)
        self.dual_arm_joint_pos = np.concatenate((self.ur3_current_joint_pos, self.ur3e_current_joint_pos))
        self.pyb_update_joint_state()

def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    # ur3_q = np.array([-1.91668255, -2.30539877, -1.55328495, -1.11481983,  2.02716804, -0.35711939])
    # ur3e_q = np.array([1.90909815, -0.88395007,  1.61091215, -2.09752192, -2.02674181,  3.44990301])
    ur3_q = np.array([-1.8470081584056457, -2.7298507268179617, -0.6953932972144096, -1.508942496823497,  2.0236098037789576, -0.31532559669045146])
    ur3e_q = np.array([1.842840084853423, -0.48057750070854266,  0.8378998011418625, -1.7586738880406665, -2.056763439048601,  3.415677557660605])
    dual_arm_model= DualArmBulletModel(ur3_q, ur3e_q, 0.01)
    rclpy.spin(dual_arm_model)

if __name__ == "__main__":
    main()
