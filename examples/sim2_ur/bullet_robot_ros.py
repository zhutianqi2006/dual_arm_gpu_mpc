#!/usr/bin/env python
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

import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from dual_arm_gpu_mpc.common.paths import resolve_asset_path
from dual_arm_gpu_mpc.common.pybullet_robot import BulletJointResetRobot, load_pybullet_modules


TIMESTEP = 1 / 60


def _exit_after_interrupt(exit_code: int = 130):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


class DualArmBulletModel(Node):
    def __init__(self, init_ur3_q, init_ur3e_q, dt: float = 0.01):
        super().__init__("dual_arm_model")
        self.dt = dt
        self.ur3_q = init_ur3_q
        self.ur3e_q = init_ur3e_q
        self.pyb, self.pybullet_data = load_pybullet_modules()
        self.gui_id = self.pyb.connect(self.pyb.GUI)
        self.pyb_dual_robot, _ = self.pyb_load_environment(self.gui_id)
        self.setup_ros2()

    def setup_ros2(self):
        self.ur3_pos_msg = JointState()
        self.ur3e_pos_msg = JointState()
        self.ur3_vel_msg = JointState()
        self.ur3e_vel_msg = JointState()
        self.ur3_pos_msg.name = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.ur3e_pos_msg.name = list(self.ur3_pos_msg.name)
        self.ur3_vel_msg.name = list(self.ur3_pos_msg.name)
        self.ur3e_vel_msg.name = list(self.ur3_pos_msg.name)
        self.ur3_vel_msg.velocity = [0.0] * 6
        self.ur3e_vel_msg.velocity = [0.0] * 6
        self.ur3_current_joint_vel = np.zeros(6)
        self.ur3e_current_joint_vel = np.zeros(6)
        self.ur3_current_joint_pos = self.ur3_q
        self.ur3e_current_joint_pos = self.ur3e_q

        self.publisher_ur3 = self.create_publisher(JointState, "ur3_joint_states", 1)
        self.publisher_ur3e = self.create_publisher(JointState, "ur3e_joint_states", 1)
        self.create_subscription(JointState, "ur3_joint_command", self.ur3_joint_vel_callback, 1)
        self.create_subscription(JointState, "ur3e_joint_command", self.ur3e_joint_vel_callback, 1)
        self.timer = self.create_timer(self.dt, self.joint_pos_pub)

    def ur3_joint_vel_callback(self, msg: JointState):
        self.ur3_current_joint_vel = np.array(msg.velocity[:6])

    def ur3e_joint_vel_callback(self, msg: JointState):
        self.ur3e_current_joint_vel = np.array(msg.velocity[:6])

    def pyb_load_environment(self, client_id):
        pyb = self.pyb
        pybullet_data = self.pybullet_data
        pyb.setTimeStep(TIMESTEP, physicsClientId=client_id)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id)
        dual_arm_robot_id = pyb.loadURDF(
            str(resolve_asset_path("dual_arm_model/dual_arm_model.urdf")),
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id,
        )
        dual_arm_robot = BulletJointResetRobot(dual_arm_robot_id, client_id=client_id, pybullet_module=pyb)
        cube2_id = pyb.loadURDF(
            str(resolve_asset_path("plane/thine_plane.urdf")),
            [0.46, 0.0, 0.006],
            useFixedBase=True,
            physicsClientId=client_id,
        )
        cube3_id = pyb.loadURDF(
            str(resolve_asset_path("plane/thine_plane.urdf")),
            [0.46, 0.0, 0.256],
            useFixedBase=True,
            physicsClientId=client_id,
        )
        cube4_id = pyb.loadURDF(
            str(resolve_asset_path("plane/thine_plane.urdf")),
            [0.46, 0.0, 0.506],
            useFixedBase=True,
            physicsClientId=client_id,
        )
        pyb.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=51,
            cameraPitch=-32,
            cameraTargetPosition=[0.0, 0.0, 0.0],
        )
        return dual_arm_robot, {"ground": ground_id, "cube2": cube2_id, "cube3": cube3_id, "cube4": cube4_id}

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

    def close(self):
        if self.pyb.isConnected(self.gui_id):
            self.pyb.disconnect(self.gui_id)
        self.destroy_node()


def main(args=None):
    os.environ["ROS_DOMAIN_ID"] = "16"
    rclpy.init(args=args)
    dual_arm_model = None
    interrupted = False
    ur3_q = np.array([-1.91668255, -2.30539877, -1.55328495, -1.11481983, 2.02716804, -0.35711939])
    ur3e_q = np.array([1.90909815, -0.88395007, 1.61091215, -2.09752192, -2.02674181, 3.44990301])
    try:
        dual_arm_model = DualArmBulletModel(ur3_q, ur3e_q, 0.01)
        rclpy.spin(dual_arm_model)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if interrupted:
            if dual_arm_model is not None:
                dual_arm_model.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            _exit_after_interrupt()
        if dual_arm_model is not None:
            dual_arm_model.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
