#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Apriltag 在相机光学坐标系下的位姿（来自 TF）转换到世界坐标系下的位置。
需求：只关心“位置”，不关心姿态。

假设：相机在世界坐标系中的位姿已知（常量），默认仅有 z=0.5 m 的高度，
且世界坐标系与相机光学坐标系平行（可选：也支持设置相机在世界系下的 RPY 角）。

运行示例：
  ros2 run <your_pkg> ros2_cam_to_world.py \
    --ros-args -p camera_frame:=camera_color_optical_frame \
               -p tag_frame:=dock_frame \
               -p world_frame:=world \
               -p camera_xyz:="[0.0, 0.0, 0.5]" \
               -p camera_rpy_deg:="[0.0, 0.0, 0.0]" \
               -p axes_aligned:=true

输出：发布 geometry_msgs/PointStamped 到 /dock_position_world （frame_id=world）
"""

import math
import os
from typing import List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener, TransformException


class CamToWorldNode(Node):
    def __init__(self):
        super().__init__('cam_to_world_node')

        # ----------------- 参数 -----------------
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('tag_frame', 'dock_frame')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('camera_xyz', [1.38, 0.0, 0.0])  # 相机在世界系的位置（米）
        self.declare_parameter('camera_rpy_deg', [-90.0, 0.0, 90.0])  # 相机在世界系的RPY角（度）
        self.declare_parameter('axes_aligned', False)  # 若为True，忽略RPY，默认世界轴与相机轴平行
        self.declare_parameter('rate_hz', 10.0)

        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.tag_frame = self.get_parameter('tag_frame').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.axes_aligned = self.get_parameter('axes_aligned').get_parameter_value().bool_value
        self.rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value

        cam_xyz_param = self.get_parameter('camera_xyz').get_parameter_value().double_array_value
        self.t_wc = np.array([cam_xyz_param[0], cam_xyz_param[1], cam_xyz_param[2]], dtype=float)

        cam_rpy_param = self.get_parameter('camera_rpy_deg').get_parameter_value().double_array_value
        self.rpy_deg = [cam_rpy_param[0], cam_rpy_param[1], cam_rpy_param[2]]

        # 计算世界->相机的旋转矩阵（这里我们实际需要的是 R_wc：相机相对于世界的旋转）
        if self.axes_aligned:
            self.R_wc = np.eye(3, dtype=float)
        else:
            roll = math.radians(self.rpy_deg[0])
            pitch = math.radians(self.rpy_deg[1])
            yaw = math.radians(self.rpy_deg[2])
            self.R_wc = CamToWorldNode.rpy_to_matrix(roll, pitch, yaw)

        # TF 监听器
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 发布器（只发布点即可）
        self.pub_point = self.create_publisher(PointStamped, '/dock_position_world', 10)

        # 定时查询 TF
        period = 1.0 / max(self.rate_hz, 1e-3)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f"camera_frame={self.camera_frame}, tag_frame={self.tag_frame}, world_frame={self.world_frame}\n"
            f"camera_xyz={self.t_wc.tolist()}, camera_rpy_deg={self.rpy_deg}, axes_aligned={self.axes_aligned}"
        )

    @staticmethod
    def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """按照 ROS 常用顺序 Z(yaw) * Y(pitch) * X(roll) 生成旋转矩阵。"""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]], dtype=float)
        Ry = np.array([[ cp, 0, sp],
                       [  0, 1,  0],
                       [-sp, 0, cp]], dtype=float)
        Rx = np.array([[1,  0,   0],
                       [0, cr, -sr],
                       [0, sr,  cr]], dtype=float)
        return Rz @ Ry @ Rx

    def on_timer(self):
        try:
            # 取得 将 tag_frame 表达到 camera_frame 的变换（source=tag, target=camera）
            # 这个 transform 的 translation 正是“标签原点在相机坐标系下的位置”
            tf_cam_from_tag = self.tf_buffer.lookup_transform(
                self.camera_frame,  # target
                self.tag_frame,     # source
                rclpy.time.Time())  # 最新

            t = tf_cam_from_tag.transform.translation
            p_cam = np.array([t.x, t.y, t.z], dtype=float)  # 物体在相机系下的位置

            # 世界系下的位置：p_w = R_wc * p_cam + t_wc
            p_w = self.R_wc @ p_cam + self.t_wc

            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.world_frame
            msg.point.x = float(p_w[0]+0.06)
            msg.point.y = float(p_w[1]-0.10)
            msg.point.z = float(p_w[2]+0.20)
            self.pub_point.publish(msg)

            self.get_logger().debug(
                f"p_cam={p_cam.tolist()} -> p_world={p_w.tolist()} (published /dock_position_world)"
            )

        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")


def main(args: List[str] = None):
    import os
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    node = CamToWorldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
