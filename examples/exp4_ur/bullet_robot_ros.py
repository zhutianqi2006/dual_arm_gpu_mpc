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
from geometry_msgs.msg import PointStamped   # >>> 新增：订阅世界系点
import numpy as np

class DualArmBulletModel(Node):
    def __init__(self, init_ur3_q, init_ur3e_q,
                 dt: float = 0.01):
        super().__init__('dual_arm_model')
        self.dt = dt
        self.ur3_q = init_ur3_q
        self.ur3e_q = init_ur3e_q
        self.gui_id = pyb.connect(pyb.GUI)
        # >>> 修改：拿到 obstacles，保存 cube2 id
        self.pyb_dual_robot, self.obstacles = self.pyb_load_environment(self.gui_id)
        self.cube2_id = self.obstacles["cube2"]
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

        # >>> 新增：订阅世界系障碍物位置（PointStamped）
        self.object_topic = '/dock_position_world'  # 如需改名，这里改
        self.object_sub = self.create_subscription(
            PointStamped,
            self.object_topic,
            self.object_pos_callback,
            10
        )
        
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
        # cube2（障碍物，初始放个占位）
        cube2_id = pyb.loadURDF(
            "model/plane/dynamic.urdf", [0.42, 0.20, 0.38], useFixedBase=True, physicsClientId=client_id
        )
        obstacles = {
            "ground": ground_id,
            "cube2": cube2_id
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

    # >>> 新增：接收世界系障碍物位置并更新 cube2
    def object_pos_callback(self, msg: PointStamped):
        x = float(msg.point.x)
        y = float(msg.point.y)
        z = float(msg.point.z)
        # 如果需要做坐标轴映射/偏移，可在这里改，例如 pos = [x, y, z]
        pos = [x, y, z]
        try:
            pyb.resetBasePositionAndOrientation(
                self.cube2_id,
                pos,
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.gui_id
            )
        except Exception as e:
            self.get_logger().warn(f"update cube2 failed: {e}")

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
    ur3_q = np.array([-1.8470081584056457, -2.7298507268179617, -0.6953932972144096, -1.508942496823497,  2.0236098037789576, -0.31532559669045146])
    ur3e_q = np.array([1.842840084853423, -0.48057750070854266,  0.8378998011418625, -1.7586738880406665, -2.056763439048601,  3.415677557660605])
    dual_arm_model= DualArmBulletModel(ur3_q, ur3e_q, 0.01)
    rclpy.spin(dual_arm_model)

if __name__ == "__main__":
    main()
