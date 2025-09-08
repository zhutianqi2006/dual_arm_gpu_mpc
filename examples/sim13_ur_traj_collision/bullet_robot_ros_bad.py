# file: bullet_robot_ros.py
"""
在原有 bullet_robot_ros 的 ROS2 + PyBullet 可视化基础上，
新增将话题 `current_abs_position` 的位置(最后三项 xyz；第一项忽略)绘制为**小球**。

其余功能保持不变：话题名、计时器、关节状态发布/订阅、URDF 加载均不改动。

配置（可选，环境变量）：
- `BULLET_ROBOT_OBSTACLES`：障碍物 JSON 字符串或文件路径（仅显示）。
- `BULLET_CURRENT_ABS_RADIUS`：小球半径，浮点数，默认 0.02（米）。
- `BULLET_CURRENT_ABS_COLOR`：小球 RGBA，逗号分隔，如 "0,1,0,0.9"；默认绿色半透明。
- `BULLET_CURRENT_ABS_TRAIL`：是否绘制短暂轨迹线，"1" 开启，默认关闭。

注意
- 小球仅用于视觉显示（visual-only），不参与动力学/碰撞。
- 位置更新通过订阅 `current_abs_position` 并调用 `resetBasePositionAndOrientation` 实时刷新。
"""

# Python standard lib
import os
import sys
import json
import math
import pathlib
from threading import Lock
from typing import List, Tuple, Optional

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


# ---------------------- 工具函数：障碍配置 ----------------------

def _default_obstacles():
    """默认球形障碍(仅显示)。"""
    rgba = [1, 0, 0, 0.35]
    obs = [
        {"center": [0.5, 0.00, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.5, 0.06, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.5, -0.06, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.5, -0.12, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.5, 0.12, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.5, -0.18, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.5, 0.18, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.5, -0.24, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.5, 0.24, 0.25], "radius": 0.04, "color": rgba},

        {"center": [0.45, 0.00, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.45, 0.06, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.45, -0.06, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.45, -0.12, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.45, 0.12, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.45, -0.18, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.45, 0.18, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.45, -0.24, 0.25], "radius": 0.04, "color": rgba},
        {"center": [0.45, 0.24, 0.25], "radius": 0.04, "color": rgba},
    ]
    return obs


def _load_obstacles_from_env():
    """从环境变量读取障碍列表，失败时返回 None。支持 JSON 字符串或 JSON 文件路径。"""
    key = "BULLET_ROBOT_OBSTACLES"
    raw = os.environ.get(key)
    if not raw:
        return None
    try:
        if os.path.exists(raw):
            with open(raw, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(raw)
    except Exception as e:
        print(f"[bullet_robot_ros] 环境变量 {key} 解析失败：{e}. 使用默认障碍。")
        return None


# ---------------------- 主要模型类 ----------------------

class DualArmBulletModel(Node):
    def __init__(self, init_ur3_q, init_ur3e_q, dt: float = 0.01):
        super().__init__('dual_arm_model')
        self.dt = dt
        self.ur3_q = init_ur3_q
        self.ur3e_q = init_ur3e_q
        self.gui_id = pyb.connect(pyb.GUI)

        # 配置：障碍与小球
        self.obstacles_cfg = _load_obstacles_from_env() or _default_obstacles()
        self.abs_radius = float(os.environ.get('BULLET_CURRENT_ABS_RADIUS', '0.02'))
        self.abs_color = _parse_rgba(os.environ.get('BULLET_CURRENT_ABS_COLOR', '0,1,0,0.9'))
        self.abs_trail = os.environ.get('BULLET_CURRENT_ABS_TRAIL', '0') == '1'

        # 加载环境
        self.pyb_dual_robot, self.env_objs = self.pyb_load_environment(self.gui_id, self.obstacles_cfg)

        # current_abs_position 可视化体的句柄
        self.abs_marker_body_id: Optional[int] = None
        self.abs_marker_vis_id: Optional[int] = None
        self._abs_last_pos: Optional[List[float]] = None

        self.setup_ros2()

    # -------------------- ROS2 --------------------
    def setup_ros2(self):
        # 关节状态/速度话题
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
        self.ur3_vel_msg.velocity = [0.0] * 6
        self.ur3e_vel_msg.velocity = [0.0] * 6
        self.ur3_current_joint_vel = np.zeros(6)
        self.ur3e_current_joint_vel = np.zeros(6)
        self.ur3_current_joint_pos = self.ur3_q
        self.ur3e_current_joint_pos = self.ur3e_q

        # pub/sub
        self.publisher_ur3 = self.create_publisher(JointState, 'ur3_joint_states', 1)
        self.publisher_ur3e = self.create_publisher(JointState, 'ur3e_joint_states', 1)
        self.subscription_ur3_velocity = self.create_subscription(
            JointState, 'ur3_joint_command', self.ur3_joint_vel_callback, 1)
        self.subscription_ur3e_velocity = self.create_subscription(
            JointState, 'ur3e_joint_command', self.ur3e_joint_vel_callback, 1)

        # 订阅 current_abs_position（Float64MultiArray，4 项；[*, x, y, z]）
        self.subscription_current_abs_position = self.create_subscription(
            Float64MultiArray, 'current_abs_position', self.current_abs_position_callback, 1)

        # 定时器：刷新关节状态（也会触发小球的惯性/轨迹绘制）
        self.timer = self.create_timer(self.dt, self.joint_pos_pub)

    def ur3_joint_vel_callback(self, msg: JointState):
        self.ur3_current_joint_vel = np.array(msg.velocity[:6])

    def ur3e_joint_vel_callback(self, msg: JointState):
        self.ur3e_current_joint_vel = np.array(msg.velocity[:6])

    # -------------------- current_abs_position 小球 --------------------
    def current_abs_position_callback(self, msg: Float64MultiArray):
        data = list(msg.data)
        if len(data) < 4:
            return
        # 仅取后三项 xyz；第一项忽略
        pos = [float(data[1]), float(data[2]), float(data[3])]
        if self.abs_marker_body_id is None:
            self._create_abs_marker(pos)
        else:
            pyb.resetBasePositionAndOrientation(
                self.abs_marker_body_id,
                pos,
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.gui_id,
            )
        # 轨迹（短暂保留）
        if self.abs_trail and self._abs_last_pos is not None:
            try:
                pyb.addUserDebugLine(
                    self._abs_last_pos, pos, [0, 1, 0],
                    lifeTime=0.5, physicsClientId=self.gui_id
                )
            except Exception:
                pass
        self._abs_last_pos = pos

    def _create_abs_marker(self, position_xyz):
        """首次创建小球；仅视觉体。
        为了渲染稳定，使用单独的 multi-body 并在回调中 resetBasePosition。
        """
        try:
            vis = pyb.createVisualShape(
                pyb.GEOM_SPHERE,
                radius=self.abs_radius,
                rgbaColor=self.abs_color,
                physicsClientId=self.gui_id,
            )
            bid = pyb.createMultiBody(
                baseMass=0.0,  # 纯视觉
                baseVisualShapeIndex=vis,
                basePosition=position_xyz,
                physicsClientId=self.gui_id,
            )
            self.abs_marker_body_id = bid
            self.abs_marker_vis_id = vis
        except Exception as e:
            print(f"[bullet_robot_ros] 创建 current_abs_position 小球失败: {e}")

    # -------------------- PyBullet 环境 --------------------
    def _draw_spherical_obstacles(self, client_id, obstacles):
        created = []
        for o in obstacles or []:
            try:
                center = np.asarray(o.get("center", [0, 0, 0]), dtype=float).reshape(3).tolist()
                radius = float(o.get("radius", 0.05))
                color = o.get("color", [1, 0, 0, 0.35])
                vis = pyb.createVisualShape(
                    pyb.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client_id
                )
                bid = pyb.createMultiBody(
                    baseMass=0.0, baseVisualShapeIndex=vis, basePosition=center, physicsClientId=client_id
                )
                created.append((bid, vis))
            except Exception as e:
                print(f"[bullet_robot_ros] 绘制球形障碍失败: {e}")
        return created

    def pyb_load_environment(self, client_id, obstacles=None):
        pyb.setTimeStep(TIMESTEP, physicsClientId=client_id)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=client_id)
        dual_arm_robot_id = pyb.loadURDF(
            "model/dual_arm_model_without_plane/dual_arm_model.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id
        )
        dual_arm_robot = pyb_utils.Robot(dual_arm_robot_id, client_id=client_id)
        obstacle_bodies = self._draw_spherical_obstacles(client_id, obstacles)
        env_objs = {
            "ground": ground_id,
            "obstacles": obstacle_bodies,
        }
        pyb.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=51,
            cameraPitch=-32,
            cameraTargetPosition=[-0.0, 0.0, 0.0]
        )
        return dual_arm_robot, env_objs

    def pyb_update_joint_state(self):
        self.pyb_dual_robot.reset_joint_configuration(self.dual_arm_joint_pos)

    def joint_pos_pub(self):
        # 简单积分以演示：外部若无速度指令则保持静止
        self.ur3_current_joint_pos += self.dt * self.ur3_current_joint_vel
        self.ur3e_current_joint_pos += self.dt * self.ur3e_current_joint_vel
        self.ur3_pos_msg.position = self.ur3_current_joint_pos.tolist()
        self.ur3e_pos_msg.position = self.ur3e_current_joint_pos.tolist()
        self.publisher_ur3.publish(self.ur3_pos_msg)
        self.publisher_ur3e.publish(self.ur3e_pos_msg)
        self.dual_arm_joint_pos = np.concatenate((self.ur3_current_joint_pos, self.ur3e_current_joint_pos))
        self.pyb_update_joint_state()


# ---------------------- 辅助解析 ----------------------

def _parse_rgba(raw: str) -> List[float]:
    """解析 'r,g,b,a' 字符串为 4 项浮点；失败回退到绿色。"""
    try:
        parts = [float(x.strip()) for x in raw.split(',')]
        if len(parts) != 4:
            raise ValueError('require 4 components')
        return parts
    except Exception:
        return [0.0, 1.0, 0.0, 0.9]


# ---------------------- 入口 ----------------------

def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    # 默认初始位姿
    ur3_q = np.array([-1.8470081584056457, -2.7298507268179617, -0.6953932972144096,
                      -1.508942496823497,  2.0236098037789576, -0.31532559669045146])
    ur3e_q = np.array([1.842840084853423, -0.48057750070854266,  0.8378998011418625,
                       -1.7586738880406665, -2.056763439048601,  3.415677557660605])
    dual_arm_model = DualArmBulletModel(ur3_q, ur3e_q, 0.01)
    rclpy.spin(dual_arm_model)


if __name__ == "__main__":
    main()
