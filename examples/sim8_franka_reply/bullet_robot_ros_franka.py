# file: bullet_robot_ros_franka.py
"""
Dual Franka pybullet + ROS2 模拟节点（带关节 & 障碍记录）

特性：
- 与原版一致地加载 dual_panda_r9_urdf.urdf 与静态球体障碍（可通过 topic 进行移动）。
- 定时发布 franka1/franka2 的 JointState，并根据接收的关节速度积分。
- 可选启用轨迹记录：时间戳、两侧 7 轴关节、所有障碍位姿（排除 ground）。
- 稳定保存为 .npz：{t, q1, q2, obs_names, obs_pos, obs_orn, dt, rate}。
- 障碍回调统一：支持 Float64MultiArray 长度为 3*k 或 7*k，依 self.obs_names 顺序更新前 k 个障碍。

用法（示例）：
  python3 bullet_robot_ros_franka.py --log --log_dir ./logs --log_rate 125
  # 运行后 Ctrl+C 退出，将自动保存到 logs/franka_run_YYYYmmdd_HHMMSS.npz

注意：
- 仅在 log_enabled 时才创建日志定时器并缓冲数据；否则不影响现有行为。
- 夹爪在回放中使用占位值 [0.01, 0.01]。
"""
from __future__ import annotations

# Python stdlib
import os
import sys
import math
import time
import json
import argparse
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple

# Third-party
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


class DualArmBulletModel(Node):
    def __init__(
        self,
        franka1_q: np.ndarray,
        franka2_q: np.ndarray,
        dt: float = 0.008,
        *,
        log_enabled: bool = False,
        log_dir: str = "./logs",
        log_rate_hz: float = 125.0,
        log_name_prefix: str = "franka_run",
    ) -> None:
        super().__init__('dual_arm_model')
        self.dt = float(dt)
        self.franka1_q = franka1_q.astype(float)
        self.franka2_q = franka2_q.astype(float)
        self.gui_id = pyb.connect(pyb.GUI)
        self.pyb_dual_robot, self.obstacles = self.pyb_load_environment(self.gui_id)

        # 固定顺序：排除 ground，其他名称按字典序排序，保证回放一致
        self.obs_names: List[str] = [k for k in sorted(self.obstacles.keys()) if k != 'ground']

        # ROS2 I/O
        self.setup_ros2()

        # 记录相关
        self.log_enabled = bool(log_enabled)
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_rate_hz = float(log_rate_hz)
        self.log_name_prefix = str(log_name_prefix)
        self._t0 = time.monotonic()

        self._log_t: List[float] = []
        self._log_q1: List[np.ndarray] = []
        self._log_q2: List[np.ndarray] = []
        # 动态二维：每帧 M 个障碍的 (pos, orn)
        self._log_obs_pos: List[np.ndarray] = []  # (M,3)
        self._log_obs_orn: List[np.ndarray] = []  # (M,4)

        # 定时器：发布关节与记录
        self.timer = self.create_timer(self.dt, self.joint_pos_pub)
        self._log_timer = None
        if self.log_enabled:
            period = 1.0 / max(1.0, self.log_rate_hz)
            self._log_timer = self.create_timer(period, self._log_tick)
            self.get_logger().info(
                f"[logger] enabled: rate={self.log_rate_hz}Hz, dir={self.log_dir}")

    # ===================== ROS2 setup & callbacks =====================
    def setup_ros2(self) -> None:
        # Franka 1
        self.franka1_pos_msg = JointState()
        self.franka1_vel_msg = JointState()
        self.franka1_pos_msg.name = [f'joint{i}' for i in range(1, 8)]
        self.franka1_vel_msg.name = [f'joint{i}' for i in range(1, 8)]
        self.franka1_vel_msg.velocity = [0.0] * 7
        self.franka1_current_joint_vel = np.zeros(7, dtype=float)
        self.franka1_current_joint_pos = self.franka1_q.copy()
        self.publisher_franka1 = self.create_publisher(JointState, 'franka1_joint_states', 1)
        self.subscription_franka1_velocity = self.create_subscription(
            JointState, 'franka1_joint_command', self.franka1_joint_vel_callback, 1)

        # Franka 2
        self.franka2_pos_msg = JointState()
        self.franka2_vel_msg = JointState()
        self.franka2_pos_msg.name = [f'joint{i}' for i in range(1, 8)]
        self.franka2_vel_msg.name = [f'joint{i}' for i in range(1, 8)]
        self.franka2_vel_msg.velocity = [0.0] * 7
        self.franka2_current_joint_vel = np.zeros(7, dtype=float)
        self.franka2_current_joint_pos = self.franka2_q.copy()
        self.publisher_franka2 = self.create_publisher(JointState, 'franka2_joint_states', 1)
        self.subscription_franka2_velocity = self.create_subscription(
            JointState, 'franka2_joint_command', self.franka2_joint_vel_callback, 1)

        # 障碍位姿更新（Float64MultiArray: [x,y,z,(qx,qy,qz,qw)?]*k，按 obs_names 顺序）
        self.obstacle_sub = self.create_subscription(
            Float64MultiArray, 'moving_obstacle', self.obstacle_callback, 1)

    def obstacle_callback(self, msg: Float64MultiArray) -> None:
        # why: 统一更新接口，避免对特定 cube 名称的硬编码
        data = list(msg.data)
        if not data:
            return
        m = len(self.obs_names)
        if len(data) % 3 == 0:
            k = min(m, len(data) // 3)
            for i in range(k):
                x, y, z = data[3 * i : 3 * i + 3]
                pyb.resetBasePositionAndOrientation(
                    self.obstacles[self.obs_names[i]], [x, y, z], [0, 0, 0, 1], physicsClientId=self.gui_id)
        elif len(data) % 7 == 0:
            k = min(m, len(data) // 7)
            for i in range(k):
                x, y, z, qx, qy, qz, qw = data[7 * i : 7 * i + 7]
                pyb.resetBasePositionAndOrientation(
                    self.obstacles[self.obs_names[i]], [x, y, z], [qx, qy, qz, qw], physicsClientId=self.gui_id)
        else:
            self.get_logger().warn(
                f"moving_obstacle payload len={len(data)} is not 3*k or 7*k; ignored")

    def franka1_joint_vel_callback(self, msg: JointState) -> None:
        self.franka1_current_joint_vel = np.array(msg.velocity[:7], dtype=float)

    def franka2_joint_vel_callback(self, msg: JointState) -> None:
        self.franka2_current_joint_vel = np.array(msg.velocity[:7], dtype=float)

    # ===================== PyBullet env =====================
    def pyb_load_environment(self, client_id: int):
        pyb.setTimeStep(TIMESTEP, physicsClientId=client_id)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

        ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id)
        dual_arm_robot_id = pyb.loadURDF(
            "model/dual_panda_model/dual_panda_r9_urdf.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id,
        )
        dual_arm_robot = pyb_utils.Robot(dual_arm_robot_id, client_id=client_id)

        obstacles: Dict[str, int] = {"ground": ground_id}

        # 9 个静态球体（可通过 topic 变为动态）
        sphere_defs = {
            "sphere1": {"pose": [-0.125, 0.0, 1.0],   "radius": 0.1},
            "sphere2": {"pose": [-0.125, 0.125, 1.0], "radius": 0.1},
            "sphere3": {"pose": [-0.125, -0.125, 1.0],"radius": 0.1},
            "sphere4": {"pose": [-0.125, 0.0, 0.7],   "radius": 0.1},
            "sphere5": {"pose": [-0.125, 0.125, 0.7], "radius": 0.1},
            "sphere6": {"pose": [-0.125, -0.125, 0.7],"radius": 0.1},
            "sphere7": {"pose": [0.35, 0.0, 0.6],     "radius": 0.1},
            "sphere8": {"pose": [0.35, 0.125, 0.6],   "radius": 0.1},
            "sphere9": {"pose": [0.35, -0.125, 0.6],  "radius": 0.1},
        }

        def _spawn_sphere(radius: float, position: List[float]) -> int:
            col_id = pyb.createCollisionShape(pyb.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
            vis_id = pyb.createVisualShape(
                shapeType=pyb.GEOM_SPHERE,
                radius=radius,
                rgbaColor=[1, 0, 0, 0.9],  # why: 强调障碍
                physicsClientId=client_id,
            )
            body_id = pyb.createMultiBody(
                baseMass=0.0,  # static by default
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
            cameraTargetPosition=[0, 0, 0.1],
        )
        return dual_arm_robot, obstacles

    # ===================== Joint publish & logging =====================
    def pyb_update_joint_state(self) -> None:
        # 7 + 2 + 7 + 2 = 18（夹爪用固定占位）
        gr1 = np.array([0.01, 0.01])
        gr2 = np.array([0.01, 0.01])
        q = np.concatenate((self.franka1_current_joint_pos, gr1, self.franka2_current_joint_pos, gr2))
        self.pyb_dual_robot.reset_joint_configuration(q)

    def joint_pos_pub(self) -> None:
        # 半隐式欧拉积分
        self.franka1_current_joint_pos = self.franka1_current_joint_pos + self.dt * self.franka1_current_joint_vel
        self.franka2_current_joint_pos = self.franka2_current_joint_pos + self.dt * self.franka2_current_joint_vel

        self.franka1_pos_msg.position = self.franka1_current_joint_pos.tolist()
        self.franka2_pos_msg.position = self.franka2_current_joint_pos.tolist()
        self.publisher_franka1.publish(self.franka1_pos_msg)
        self.publisher_franka2.publish(self.franka2_pos_msg)

        self.pyb_update_joint_state()

    def _log_tick(self) -> None:
        # why: 记录真实发生的状态与环境，供离线复现/分析
        t = time.monotonic() - self._t0
        q1 = self.franka1_current_joint_pos.copy()
        q2 = self.franka2_current_joint_pos.copy()

        # 读取障碍位姿（排除 ground）
        pos_list: List[List[float]] = []
        orn_list: List[List[float]] = []
        for name in self.obs_names:
            bid = self.obstacles[name]
            p, o = pyb.getBasePositionAndOrientation(bid, physicsClientId=self.gui_id)
            pos_list.append(list(p))
            orn_list.append(list(o))

        self._log_t.append(t)
        self._log_q1.append(q1)
        self._log_q2.append(q2)
        self._log_obs_pos.append(np.asarray(pos_list, dtype=float))
        self._log_obs_orn.append(np.asarray(orn_list, dtype=float))

    # ===================== Save logs =====================
    def save_logs(self, path: pathlib.Path | None = None) -> pathlib.Path | None:
        if not self.log_enabled or len(self._log_t) == 0:
            return None
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = path or (self.log_dir / f"munihei.npz")

        t = np.asarray(self._log_t, dtype=float)
        q1 = np.vstack(self._log_q1).astype(float)
        q2 = np.vstack(self._log_q2).astype(float)
        obs_pos = np.stack(self._log_obs_pos, axis=0).astype(float)  # (N,M,3)
        obs_orn = np.stack(self._log_obs_orn, axis=0).astype(float)  # (N,M,4)
        obs_names = np.array(self.obs_names)

        np.savez_compressed(
            out,
            t=t,
            q1=q1,
            q2=q2,
            obs_names=obs_names,
            obs_pos=obs_pos,
            obs_orn=obs_orn,
            dt=np.array(self.dt),
            rate=np.array(self.log_rate_hz),
        )
        self.get_logger().info(f"[logger] saved: {out}")
        return out


# ===================== CLI / main =====================

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Dual Franka with logging (PyBullet + ROS2)')
    p.add_argument('--dt', type=float, default=0.008)
    p.add_argument('--log', action='store_true', help='enable logging to .npz')
    p.add_argument('--log_dir', type=str, default='./logs')
    p.add_argument('--log_rate', type=float, default=125.0, help='logging rate in Hz')
    p.add_argument('--domain', type=str, default='16', help='ROS_DOMAIN_ID')
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)

    rclpy.init(args=None)

    # 初始姿态（与原版一致）
    franka1_q = np.array([1.387536, 1.3089969, -1.5707963, -0.61086523,  0.0,  2.5307273, 1.3089969])
    franka2_q = np.array([-1.40499, 1.3089969, 1.5707963, -0.61086523, 0.0, 2.5307273, -1.3089969])

    node = DualArmBulletModel(
        franka1_q,
        franka2_q,
        dt=float(args.dt),
        log_enabled=1,
        log_dir=str(args.log_dir),
        log_rate_hz=float(args.log_rate),
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        out = node.save_logs()
        if pyb.isConnected(node.gui_id):
            pyb.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


