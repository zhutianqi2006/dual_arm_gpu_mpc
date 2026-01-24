# file: bullet_robot_ros_franka.py

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
        cube5_id = self.obstacles["cube5"]
        pyb.resetBasePositionAndOrientation(cube4_id, msg.data[0:3], [1,0,0,0])
        pyb.resetBasePositionAndOrientation(cube5_id, msg.data[3:6], [1,0,0,0])

    def franka1_joint_vel_callback(self, msg:JointState):
        self.franka1_current_joint_vel = np.array(msg.velocity[:7])

    def franka2_joint_vel_callback(self, msg:JointState):
        self.franka2_current_joint_vel = np.array(msg.velocity[:7])

    def pyb_load_environment(self, client_id):
        pyb.setTimeStep(TIMESTEP, physicsClientId=client_id)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        # ground plane
        ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id)
        dual_arm_robot_id = pyb.loadURDF(
            "model/dual_panda_model/dual_panda_r9_urdf.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id
        )
        dual_arm_robot = pyb_utils.Robot(dual_arm_robot_id, client_id=client_id)
        # store body indices in a dict with more convenient key names
        obstacles = {
            "ground": ground_id
        }

        # === add 9 static spherical obstacles (exactly as provided) ===
        # why: strictly visualize given obstacles without touching control/ROS code
        sphere_defs = {
            "sphere1": {"pose": [-0.125, 0.0, 0.7],   "radius": 0.1},
            "sphere2": {"pose": [-0.125, 0.125, 0.7], "radius": 0.1},
            "sphere3": {"pose": [-0.125, -0.125, 0.7],"radius": 0.1},
            "sphere4": {"pose": [-0.125, 0.0, 0.8],   "radius": 0.1},
            "sphere5": {"pose": [-0.125, 0.125, 0.8], "radius": 0.1},
            "sphere6": {"pose": [-0.125, -0.125, 0.8],"radius": 0.1},
            "sphere10": {"pose": [-0.125, 0.0, 0.6],"radius": 0.1},
            "sphere11": {"pose": [-0.125, 0.125, 0.6],"radius": 0.1},
            "sphere12": {"pose": [-0.125, -0.125, 0.6],"radius": 0.1},
            "sphere13": {"pose": [-0.125, 0.0, 0.5],"radius": 0.1},
            "sphere14": {"pose": [-0.125, 0.125, 0.5],"radius": 0.1},
            "sphere15": {"pose": [-0.125, -0.125, 0.5],"radius": 0.1},
        }

        def _spawn_sphere(radius, position):
            col_id = pyb.createCollisionShape(pyb.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
            vis_id = pyb.createVisualShape(
                shapeType=pyb.GEOM_SPHERE,
                radius=radius,
                rgbaColor=[1,0,0,0.9],  # neutral gray
                physicsClientId=client_id,
            )
            body_id = pyb.createMultiBody(
                baseMass=0.0,  # static obstacle
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=position,
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=client_id,
            )
            return body_id

        for name, cfg in sphere_defs.items():
            body = _spawn_sphere(cfg["radius"], cfg["pose"])
            obstacles[name] = body

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
    franka1_q = np.array([1.387536, 1.3089969, -1.5707963, -0.61086523,  0.0,  2.5307273, 1.3089969])
    franka2_q = np.array([-1.40499, 1.3089969, 1.5707963, -0.61086523, 0.0, 2.5307273, -1.3089969])
    dual_arm_model= DualArmBulletModel(franka1_q , franka2_q, 0.008)
    rclpy.spin(dual_arm_model)


if __name__ == "__main__":
    main()
