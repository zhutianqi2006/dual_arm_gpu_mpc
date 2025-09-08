import os
import json
import time
import atexit
import pathlib
import datetime as _dt
from typing import List, Tuple

import numpy as np
import pybullet as pyb
import pybullet_data
import pyb_utils

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

TIMESTEP = 1.0 / 125.0


class TrajectoryRecorderFranka:
    """Buffer samples in memory; write once on exit to avoid jitter.
    Records:
      - time t (s)
      - franka1 joint pos (7)
      - franka2 joint pos (7)
      - obstacles poses (K, 7) as [x,y,z,qx,qy,qz,qw]
    """

    def __init__(
        self,
        out_dir: pathlib.Path,
        dt: float,
        obstacle_ids: List[int],
        obstacle_names: List[str],
        client_id: int,
    ) -> None:
        self.out_dir = out_dir
        self.dt = float(dt)
        self._client_id = client_id
        self._obs_ids = list(obstacle_ids)
        self._obs_names = list(obstacle_names)
        self.t_list: List[float] = []
        self.f1_list: List[np.ndarray] = []
        self.f2_list: List[np.ndarray] = []
        self.obs_list: List[np.ndarray] = []  # each (K,7)
        self._saved = False
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _grab_obstacles(self) -> np.ndarray:
        poses: List[List[float]] = []
        for bid in self._obs_ids:
            pos, orn = pyb.getBasePositionAndOrientation(bid, physicsClientId=self._client_id)
            poses.append([pos[0], pos[1], pos[2], orn[0], orn[1], orn[2], orn[3]])
        return np.asarray(poses, dtype=np.float64)

    def add(self, t: float, franka1: np.ndarray, franka2: np.ndarray) -> None:
        self.t_list.append(float(t))
        self.f1_list.append(np.asarray(franka1, dtype=np.float64))
        self.f2_list.append(np.asarray(franka2, dtype=np.float64))
        self.obs_list.append(self._grab_obstacles())

    def save(self, prefix: str = "franka_with_obs") -> pathlib.Path:
        if self._saved:
            return self._last_path  # type: ignore[attr-defined]
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"{prefix}.npz"
        t = np.asarray(self.t_list, dtype=np.float64)
        f1 = np.vstack(self.f1_list) if self.f1_list else np.zeros((0, 7))
        f2 = np.vstack(self.f2_list) if self.f2_list else np.zeros((0, 7))
        obs = np.stack(self.obs_list, axis=0) if self.obs_list else np.zeros((0, len(self._obs_ids), 7))
        meta = {
            "created_at": ts,
            "count": int(len(t)),
            "dt": self.dt,
            "schema": {
                "t": "(N,)",
                "franka1": "(N,7)",
                "franka2": "(N,7)",
                "obstacles": f"(N,{len(self._obs_ids)},7)",
                "obs_names": self._obs_names,
            },
        }
        np.savez_compressed(
            path,
            t=t,
            franka1=f1,
            franka2=f2,
            obstacles=obs,
            obs_names=np.asarray(self._obs_names),
            dt=self.dt,
            meta=json.dumps(meta, ensure_ascii=False),
        )
        self._saved = True
        self._last_path = path
        return path


class DualArmBulletModel(Node):
    def __init__(self, franka1_q: np.ndarray, franka2_q: np.ndarray, dt: float = 0.01, record_dir: str | None = None):
        super().__init__('dual_arm_model')
        self.dt = float(dt)
        self.t_elapsed = 0.0
        self.franka1_current_joint_pos = franka1_q.astype(np.float64)
        self.franka2_current_joint_pos = franka2_q.astype(np.float64)
        self.franka1_current_joint_vel = np.zeros(7, dtype=np.float64)
        self.franka2_current_joint_vel = np.zeros(7, dtype=np.float64)

        # PyBullet
        self.gui_id = pyb.connect(pyb.GUI)
        pyb.setTimeStep(TIMESTEP, physicsClientId=self.gui_id)
        self.pyb_dual_robot, self.obstacles = self.pyb_load_environment(self.gui_id)

        # Recorder
        out_dir = pathlib.Path(record_dir or './logs')
        self._obs_order = ['cube4', 'cube5']
        self.recorder = TrajectoryRecorderFranka(
            out_dir,
            self.dt,
            [self.obstacles[name] for name in self._obs_order],
            self._obs_order,
            self.gui_id,
        )
        atexit.register(self._save_and_cleanup)

        # ROS2
        self.setup_ros2()

        # Initialize one frame
        self._publish_and_render()
        self.recorder.add(self.t_elapsed, self.franka1_current_joint_pos, self.franka2_current_joint_pos)

    # ------------------ ROS2 ------------------
    def setup_ros2(self) -> None:
        # msgs
        self.franka1_pos_msg = JointState()
        self.franka2_pos_msg = JointState()
        self.franka1_vel_msg = JointState()
        self.franka2_vel_msg = JointState()
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.franka1_pos_msg.name = joint_names
        self.franka2_pos_msg.name = joint_names
        self.franka1_vel_msg.name = joint_names
        self.franka2_vel_msg.name = joint_names
        self.franka1_vel_msg.velocity = [0.0] * 7
        self.franka2_vel_msg.velocity = [0.0] * 7

        self.publisher_franka1 = self.create_publisher(JointState, 'franka1_joint_states', 1)
        self.publisher_franka2 = self.create_publisher(JointState, 'franka2_joint_states', 1)
        self.subscription_franka1_velocity = self.create_subscription(JointState, 'franka1_joint_command', self.franka1_joint_vel_callback, 1)
        self.subscription_franka2_velocity = self.create_subscription(JointState, 'franka2_joint_command', self.franka2_joint_vel_callback, 1)

        # moving obstacles: Float64MultiArray: [x4,y4,z4,x5,y5,z5]
        self.obstacle_sub = self.create_subscription(Float64MultiArray, 'moving_obstacle', self.obstacle_callback, 1)

        self.timer = self.create_timer(self.dt, self.joint_pos_pub)

    def franka1_joint_vel_callback(self, msg: JointState) -> None:
        self.franka1_current_joint_vel = np.asarray(msg.velocity[:7], dtype=np.float64)

    def franka2_joint_vel_callback(self, msg: JointState) -> None:
        self.franka2_current_joint_vel = np.asarray(msg.velocity[:7], dtype=np.float64)

    def obstacle_callback(self, msg: Float64MultiArray) -> None:
        data = list(msg.data)
        if len(data) >= 3:
            pyb.resetBasePositionAndOrientation(self.obstacles['cube4'], data[0:3], [0.0, 0.0, 0.0, 1.0], physicsClientId=self.gui_id)
        if len(data) >= 6:
            pyb.resetBasePositionAndOrientation(self.obstacles['cube5'], data[3:6], [0.0, 0.0, 0.0, 1.0], physicsClientId=self.gui_id)

    # ------------------ PyBullet ------------------
    def pyb_load_environment(self, client_id: int):
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id)
        dual_arm_robot_id = pyb.loadURDF(
            "model/dual_panda_model/dual_panda_urdf.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id,
        )
        dual_arm_robot = pyb_utils.Robot(dual_arm_robot_id, client_id=client_id)

        # obstacles (two plates)
        cube4_id = pyb.loadURDF("model/plane/plane.urdf", [0.50, 0.0, 0.85], useFixedBase=True, physicsClientId=client_id)
        cube5_id = pyb.loadURDF("model/plane/plane.urdf", [0.50, 0.0, 0.85], useFixedBase=True, physicsClientId=client_id)
        obstacles = {"ground": ground_id, "cube4": cube4_id, "cube5": cube5_id}

        pyb.resetDebugVisualizerCamera(
            cameraDistance=1.55,
            cameraYaw=50,
            cameraPitch=-40,
            cameraTargetPosition=[0.0, 0.0, 0.1],
        )
        return dual_arm_robot, obstacles

    def _publish_and_render(self) -> None:
        # publish states
        self.franka1_pos_msg.position = self.franka1_current_joint_pos.tolist()
        self.franka2_pos_msg.position = self.franka2_current_joint_pos.tolist()
        self.publisher_franka1.publish(self.franka1_pos_msg)
        self.publisher_franka2.publish(self.franka2_pos_msg)

        # visual-only grippers to match URDF DOF layout
        robot1_gripper_pos = np.array([0.01, 0.01], dtype=np.float64)
        robot2_gripper_pos = np.array([0.01, 0.01], dtype=np.float64)

        self.dual_arm_joint_pos = np.concatenate(
            (self.franka1_current_joint_pos, robot1_gripper_pos, self.franka2_current_joint_pos, robot2_gripper_pos)
        )
        self.pyb_dual_robot.reset_joint_configuration(self.dual_arm_joint_pos)

    # ------------------ Tick ------------------
    def joint_pos_pub(self) -> None:
        # integrate
        self.franka1_current_joint_pos = self.franka1_current_joint_pos + self.dt * self.franka1_current_joint_vel
        self.franka2_current_joint_pos = self.franka2_current_joint_pos + self.dt * self.franka2_current_joint_vel
        self.t_elapsed += self.dt

        # publish + render
        self._publish_and_render()

        # record
        self.recorder.add(self.t_elapsed, self.franka1_current_joint_pos, self.franka2_current_joint_pos)

    # ------------------ Cleanup ------------------
    def _save_and_cleanup(self) -> None:
        try:
            path = self.recorder.save("dy_h20")
            self.get_logger().info(f"Trajectory saved: {path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save trajectory: {e}")
        try:
            if pyb.isConnected(self.gui_id):
                pyb.disconnect(self.gui_id)
        except Exception:
            pass


def main(args=None):
    os.environ.setdefault('ROS_DOMAIN_ID', '16')
    rclpy.init(args=args)
    franka1_q = np.array([0.0, -1.0471, 0.0, -2.6178, 1.5707, 1.5707, 0.7853], dtype=np.float64)
    franka2_q = np.array([0.0, -1.0471, 0.0, -2.6178, -1.5707, 1.5707, 0.7853], dtype=np.float64)
    node = DualArmBulletModel(franka1_q, franka2_q, dt=0.008)
    rclpy.spin(node)


if __name__ == "__main__":
    main()


