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
# pybullet to display
import pybullet as pyb
import pybullet_data
TIMESTEP = 0.00001
# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
# ur rtde
import rtde_control
import rtde_receive

from dual_arm_gpu_mpc.common.paths import resolve_asset_path
from dual_arm_gpu_mpc.common.pybullet_robot import BulletJointResetRobot

class DualArmRealModel(Node):
    def __init__(self, dt: float = 0.01):
        super().__init__('dual_arm_model')
        self.dt = dt
        self.gui_id = pyb.connect(pyb.GUI)
        self.pyb_dual_robot, _ = self.pyb_load_environment(self.gui_id) 
        self.init_rtde() 
        self.setup_ros2()

    def init_rtde(self):
        self.ur3_robot_control = rtde_control.RTDEControlInterface("192.168.100.7")
        self.ur3_robot_receive = rtde_receive.RTDEReceiveInterface("192.168.100.7")
        self.ur3e_robot_control = rtde_control.RTDEControlInterface("192.168.100.9")
        self.ur3e_robot_receive = rtde_receive.RTDEReceiveInterface("192.168.100.9")

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
        self.ur3_current_joint_vel =np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ur3e_current_joint_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ur3_current_joint_pos = np.array(self.ur3_robot_receive.getActualQ())
        self.ur3e_current_joint_pos = np.array(self.ur3e_robot_receive.getActualQ())
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
        self.ur3_robot_control.speedJ(qd=self.ur3_current_joint_vel, acceleration=1, time=0.008)
       
        
    def ur3e_joint_vel_callback(self, msg:JointState):
        self.ur3e_current_joint_vel = np.array(msg.velocity[:6])
        self.ur3e_robot_control.speedJ(qd=self.ur3e_current_joint_vel, acceleration=1, time=0.008)

    def pyb_load_environment(self,client_id):
        pyb.setTimeStep(TIMESTEP, physicsClientId=client_id)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        # ground plane
        ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id)
        dual_arm_robot_id = pyb.loadURDF(
            str(resolve_asset_path("dual_arm_model/dual_arm_model.urdf")),
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id
        )
        dual_arm_robot = BulletJointResetRobot(dual_arm_robot_id, client_id=client_id, pybullet_module=pyb)
        # some cubes for obstacles
        # store body indices in a dict with more convenient key names
        cube2_id = pyb.loadURDF(
            str(resolve_asset_path("plane/dynamic.urdf")),
            [0.32, 0.22, 0.38],
            useFixedBase=True,
            physicsClientId=client_id
        )
        # store body indices in a dict with more convenient key names
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
        
    def joint_pos_pub(self):
        self.ur3_current_joint_pos = self.ur3_robot_receive.getActualQ()
        self.ur3e_current_joint_pos = self.ur3e_robot_receive.getActualQ()
        self.ur3_pos_msg.position = self.ur3_current_joint_pos
        self.ur3e_pos_msg.position = self.ur3e_current_joint_pos
        self.publisher_ur3.publish(self.ur3_pos_msg)
        self.publisher_ur3e.publish(self.ur3e_pos_msg)
        self.dual_arm_joint_pos = np.concatenate((self.ur3_current_joint_pos, self.ur3e_current_joint_pos))
        self.pyb_update_joint_state()

def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    dual_arm_model= DualArmRealModel(0.008)
    rclpy.spin(dual_arm_model)
    
if __name__ == "__main__":
    main()
