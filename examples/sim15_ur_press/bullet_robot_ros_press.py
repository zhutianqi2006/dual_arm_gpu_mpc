# filepath: bullet_robot_ros_cross.py
"""
DualArm 仿真节点（改造版）：
- 保持原有 ROS2 发布/订阅与 PyBullet 可视化。
- 以固定 dt 采样保存双臂 12 关节位置到 .npz（含时间戳 t）。
- 退出（Ctrl+C）时自动落盘；也可通过参数 --record_path 指定保存目录。

文件格式：npz
  keys: t (N,), ur3 (N,6), ur3e (N,6), dt (float), meta (str JSON)
"""
from __future__ import annotations

# Python stdlib
import os
import sys
import math
import json
import time
import atexit
import pathlib
import datetime as _dt
from typing import Optional

# Third-party
import numpy as np
import pybullet as pyb
import pybullet_data
import pyb_utils

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

TIMESTEP = 1.0 / 60.0


class TrajectoryRecorder:
    """采样缓存 -> 程序结束一次性写盘（避免频繁 IO）。
    为什么：高频 timer 回调下写文件容易抖动、掉帧。
    """

    def __init__(self, out_dir: pathlib.Path, dt: float) -> None:
        self.out_dir = out_dir
        self.dt = float(dt)
        self.t_list: list[float] = []
        self.ur3_list: list[np.ndarray] = []
        self.ur3e_list: list[np.ndarray] = []
        self._start_t = time.time()
        self._saved = False
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, t: float, ur3: np.ndarray, ur3e: np.ndarray) -> None:
        self.t_list.append(float(t))
        self.ur3_list.append(np.asarray(ur3, dtype=np.float64))
        self.ur3e_list.append(np.asarray(ur3e, dtype=np.float64))

    def save(self, prefix: str = "dual_arm_traj") -> pathlib.Path:
        if self._saved:
            return self._last_path  # type: ignore[attr-defined]
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"{prefix}.npz"
        t = np.asarray(self.t_list, dtype=np.float64)
        ur3 = np.vstack(self.ur3_list) if self.ur3_list else np.zeros((0, 6))
        ur3e = np.vstack(self.ur3e_list) if self.ur3e_list else np.zeros((0, 6))
        meta = {
            "created_at": ts,
            "count": len(t),
            "dt": self.dt,
            "schema": {"t": "(N,)", "ur3": "(N,6)", "ur3e": "(N,6)"},
        }
        np.savez_compressed(path, t=t, ur3=ur3, ur3e=ur3e, dt=self.dt, meta=json.dumps(meta, ensure_ascii=False))
        self._saved = True
        self._last_path = path
        return path


class DualArmBulletModel(Node):
    def __init__(self, init_ur3_q: np.ndarray, init_ur3e_q: np.ndarray, dt: float = 0.01, record_dir: Optional[str] = None):
        super().__init__('dual_arm_model')
        self.dt = float(dt)
        self.t_elapsed = 0.0
        self.ur3_q = init_ur3_q.astype(np.float64)
        self.ur3e_q = init_ur3e_q.astype(np.float64)

        # PyBullet
        self.gui_id = pyb.connect(pyb.GUI)
        self.pyb_dual_robot, _ = self.pyb_load_environment(self.gui_id)

        # Recorder
        out_dir = pathlib.Path(record_dir or './logs')
        self.recorder = TrajectoryRecorder(out_dir, self.dt)
        atexit.register(self._save_and_cleanup)

        # ROS2
        self.setup_ros2()

        # 初始一帧
        self.ur3_current_joint_pos = self.ur3_q.copy()
        self.ur3e_current_joint_pos = self.ur3e_q.copy()
        self.dual_arm_joint_pos = np.concatenate((self.ur3_current_joint_pos, self.ur3e_current_joint_pos))
        self.pyb_update_joint_state()
        self.recorder.add(self.t_elapsed, self.ur3_current_joint_pos, self.ur3e_current_joint_pos)

    # ------------------ ROS2 ------------------
    def setup_ros2(self) -> None:
        self.ur3_pos_msg = JointState()
        self.ur3e_pos_msg = JointState()
        self.ur3_vel_msg = JointState()
        self.ur3e_vel_msg = JointState()
        names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ur3_pos_msg.name = names
        self.ur3e_pos_msg.name = names
        self.ur3_vel_msg.name = names
        self.ur3e_vel_msg.name = names
        self.ur3_vel_msg.velocity = [0.0] * 6
        self.ur3e_vel_msg.velocity = [0.0] * 6
        self.ur3_current_joint_vel = np.zeros(6, dtype=np.float64)
        self.ur3e_current_joint_vel = np.zeros(6, dtype=np.float64)

        self.publisher_ur3 = self.create_publisher(JointState, 'ur3_joint_states', 1)
        self.publisher_ur3e = self.create_publisher(JointState, 'ur3e_joint_states', 1)
        self.subscription_ur3_velocity = self.create_subscription(JointState, 'ur3_joint_command', self.ur3_joint_vel_callback, 1)
        self.subscription_ur3e_velocity = self.create_subscription(JointState, 'ur3e_joint_command', self.ur3e_joint_vel_callback, 1)
        self.timer = self.create_timer(self.dt, self.joint_pos_pub)

    def ur3_joint_vel_callback(self, msg: JointState) -> None:
        self.ur3_current_joint_vel = np.asarray(msg.velocity[:6], dtype=np.float64)

    def ur3e_joint_vel_callback(self, msg: JointState) -> None:
        self.ur3e_current_joint_vel = np.asarray(msg.velocity[:6], dtype=np.float64)

    # ------------------ PyBullet ------------------
    def pyb_load_environment(self, client_id):
        pyb.setTimeStep(TIMESTEP, physicsClientId=client_id)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        ground_id = pyb.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True, physicsClientId=client_id)
        dual_arm_robot_id = pyb.loadURDF(
            'model/dual_arm_model/dual_arm_notray_model.urdf', [0, 0, 0], useFixedBase=True, physicsClientId=client_id
        )
        dual_arm_robot = pyb_utils.Robot(dual_arm_robot_id, client_id=client_id)
        pyb.resetDebugVisualizerCamera(
            cameraDistance=1.0, cameraYaw=89, cameraPitch=-28, cameraTargetPosition=[-0.0, 0.0, 0.0]
        )
        obstacles = {"ground": ground_id}
        return dual_arm_robot, obstacles

    def pyb_update_joint_state(self) -> None:
        self.pyb_dual_robot.reset_joint_configuration(self.dual_arm_joint_pos)

    # ------------------ Tick ------------------
    def joint_pos_pub(self) -> None:
        # integrate
        self.ur3_current_joint_pos = self.ur3_current_joint_pos + self.dt * self.ur3_current_joint_vel
        self.ur3e_current_joint_pos = self.ur3e_current_joint_pos + self.dt * self.ur3e_current_joint_vel
        self.t_elapsed += self.dt

        # publish
        self.ur3_pos_msg.position = self.ur3_current_joint_pos.tolist()
        self.ur3e_pos_msg.position = self.ur3e_current_joint_pos.tolist()
        self.publisher_ur3.publish(self.ur3_pos_msg)
        self.publisher_ur3e.publish(self.ur3e_pos_msg)

        # render
        self.dual_arm_joint_pos = np.concatenate((self.ur3_current_joint_pos, self.ur3e_current_joint_pos))
        self.pyb_update_joint_state()

        # record
        self.recorder.add(self.t_elapsed, self.ur3_current_joint_pos, self.ur3e_current_joint_pos)

    # ------------------ Cleanup ------------------
    def _save_and_cleanup(self) -> None:
        try:
            path = self.recorder.save("press")
            self.get_logger().info(f"Trajectory saved: {path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save trajectory: {e}")
        try:
            if pyb.isConnected(self.gui_id):
                pyb.disconnect(self.gui_id)
        except Exception:
            pass


# ---------------------- 入口 ----------------------

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description='DualArm Bullet ROS Cross (with recording)')
    parser.add_argument('--dt', type=float, default=0.01, help='Timer dt (s)')
    parser.add_argument('--record_path', type=str, default='./logs', help='Directory to store .npz recording')
    args = parser.parse_args(argv)

    os.environ['ROS_DOMAIN_ID'] = os.environ.get('ROS_DOMAIN_ID', '16')

    rclpy.init(args=None)
    try:
        ur3_q = np.array([-1.8470081584056457, -2.7298507268179617, -0.6953932972144096,
                          -1.508942496823497,  2.0236098037789576, -0.31532559669045146])
        ur3e_q = np.array([1.842840084853423, -0.48057750070854266,  0.8378998011418625,
                           -1.7586738880406665, -2.056763439048601,  3.415677557660605])
        node = DualArmBulletModel(ur3_q, ur3e_q, dt=float(args.dt), record_dir=args.record_path)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node._save_and_cleanup()
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()


