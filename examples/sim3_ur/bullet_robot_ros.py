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
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

from dual_arm_gpu_mpc.common.moving_obstacle import PingPongObstacleTrajectory
from dual_arm_gpu_mpc.common.paths import resolve_asset_path
from dual_arm_gpu_mpc.common.pybullet_robot import BulletJointResetRobot, load_pybullet_modules


TIMESTEP = 1 / 60
MOVING_OBSTACLE_DIMS = [0.12, 0.12, 0.12]
MOVING_OBSTACLE_COLOR = [0.1, 0.6, 1.0, 0.85]
MOVING_OBSTACLE_START = [0.38, -1, 0.58]
MOVING_OBSTACLE_END = [0.38, 1, 0.58]
MOVING_OBSTACLE_SPEED = 0.08
MOVING_OBSTACLE_TOPIC = "/dock_position_world"
MOVING_OBSTACLE_NAME = "moving_obstacle"


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
        self.moving_obstacle_trajectory = PingPongObstacleTrajectory(
            start=MOVING_OBSTACLE_START,
            end=MOVING_OBSTACLE_END,
            speed=MOVING_OBSTACLE_SPEED,
        )
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
        self.dynamic_obstacle_pub = self.create_publisher(PointStamped, MOVING_OBSTACLE_TOPIC, 10)
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

        obstacle_half_extents = [dimension / 2.0 for dimension in MOVING_OBSTACLE_DIMS]
        obstacle_collision_shape = pyb.createCollisionShape(
            pyb.GEOM_BOX,
            halfExtents=obstacle_half_extents,
            physicsClientId=client_id,
        )
        obstacle_visual_shape = pyb.createVisualShape(
            pyb.GEOM_BOX,
            halfExtents=obstacle_half_extents,
            rgbaColor=MOVING_OBSTACLE_COLOR,
            physicsClientId=client_id,
        )
        self.moving_obstacle_id = pyb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=obstacle_collision_shape,
            baseVisualShapeIndex=obstacle_visual_shape,
            basePosition=self.moving_obstacle_trajectory.position,
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=client_id,
        )
        pyb.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=51,
            cameraPitch=-32,
            cameraTargetPosition=[0.0, 0.0, 0.0],
        )
        return dual_arm_robot, {"ground": ground_id, MOVING_OBSTACLE_NAME: self.moving_obstacle_id}

    def publish_dynamic_obstacle(self, obstacle_position: list[float]) -> None:
        obstacle_msg = PointStamped()
        obstacle_msg.header.stamp = self.get_clock().now().to_msg()
        obstacle_msg.header.frame_id = "world"
        obstacle_msg.point.x = float(obstacle_position[0])
        obstacle_msg.point.y = float(obstacle_position[1])
        obstacle_msg.point.z = float(obstacle_position[2])
        self.dynamic_obstacle_pub.publish(obstacle_msg)

    def update_dynamic_obstacle(self):
        obstacle_position = self.moving_obstacle_trajectory.step(self.dt)
        self.pyb.resetBasePositionAndOrientation(
            self.moving_obstacle_id,
            obstacle_position,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.gui_id,
        )
        self.publish_dynamic_obstacle(obstacle_position)

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
        self.update_dynamic_obstacle()

    def close(self):
        if self.pyb.isConnected(self.gui_id):
            self.pyb.disconnect(self.gui_id)
        self.destroy_node()


def main(args=None):
    os.environ["ROS_DOMAIN_ID"] = "16"
    rclpy.init(args=args)
    dual_arm_model = None
    interrupted = False
    ur3_q = np.array(
        [-1.8470081584056457, -2.7298507268179617, -0.6953932972144096, -1.508942496823497, 2.0236098037789576, -0.31532559669045146]
    )
    ur3e_q = np.array(
        [1.842840084853423, -0.48057750070854266, 0.8378998011418625, -1.7586738880406665, -2.056763439048601, 3.415677557660605]
    )
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
