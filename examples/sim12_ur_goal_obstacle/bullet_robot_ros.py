# file: bullet_robot_ros.py
"""ROS2 + PyBullet dual-arm demo.

Adds a static red cube of 5 cm (0.05 m) side length at world position (0.45, 0.0, 0.45)
without changing existing functionality. The cube is registered in the `obstacles`
map as `"red_cube"` for potential future reference.
"""

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
        """Initialize ROS2 publishers/subscribers and message templates."""
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

    def ur3_joint_vel_callback(self, msg: JointState):
        self.ur3_current_joint_vel = np.array(msg.velocity[:6])

    def ur3e_joint_vel_callback(self, msg: JointState):
        self.ur3e_current_joint_vel = np.array(msg.velocity[:6])

    def pyb_load_environment(self, client_id):
        """Create world, load robot, and add a static red cube.

        Why the cube is static: it should behave like a visual marker/obstacle and
        not fall due to gravity or interact dynamically unless later required.
        """
        pyb.setTimeStep(TIMESTEP, physicsClientId=client_id)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id)
        dual_arm_robot_id = pyb.loadURDF(
            "model/dual_arm_model/dual_arm_model.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id
        )
        dual_arm_robot = pyb_utils.Robot(dual_arm_robot_id, client_id=client_id)

        obstacles = {
            "ground": ground_id
        }

        # --- Add a 5 cm red cube at (0.45, 0.0, 0.45) ---
        cube_size_m = 0.05
        half = cube_size_m / 2.0
        cube_pos = [0.45, 0.0, 0.45]
        vis = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=[half, half, half],
            rgbaColor=[1.0, 0.0, 0.0, 1.0],  # red
            physicsClientId=client_id,
        )
        col = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=[half, half, half],
            physicsClientId=client_id,
        )
        red_cube_id = pyb.createMultiBody(
            baseMass=0.0,  # fixed
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=cube_pos,
            physicsClientId=client_id,
        )
        obstacles["red_cube"] = red_cube_id
        # ------------------------------------------------

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
        self.ur3_current_joint_pos += self.dt * self.ur3_current_joint_vel
        self.ur3e_current_joint_pos += self.dt * self.ur3e_current_joint_vel
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
    dual_arm_model = DualArmBulletModel(ur3_q, ur3e_q, 0.01)
    rclpy.spin(dual_arm_model)


if __name__ == "__main__":
    main()
