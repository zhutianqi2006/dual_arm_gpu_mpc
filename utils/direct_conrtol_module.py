#!/usr/bin/env python
# file: direct_conrtol_module.py
"""
Direct Jacobian-combination low-level controller for dual-arm DQ robotics.

- Initializes with desired **absolute pose** and **relative pose** (both as 8D DQ arrays).
- Computes joint velocities by stacking the relative and absolute Jacobians and solving a
  damped least-squares problem each tick:  u = J^T (J J^T + λI)^{-1} e.
- No upper-layer command is required; this module generates joint velocities directly.
- Publishes current absolute pose and errors via LowROSModule for monitoring.

Why damped LS and weighting:
- Damping improves numerical robustness near singularities.
- Per-task gains weight the rows so you can balance rel/abs tracking priorities.
"""
from __future__ import annotations

# system
import numpy as np
import threading
from typing import Tuple

# dqrobotics (CPU)
from dqrobotics import DQ, vec8, vec4
from dqrobotics.robot_modeling import (
    DQ_SerialManipulatorDH,
    DQ_SerialManipulatorMDH,
    DQ_CooperativeDualTaskSpace,
)

# app utils
from utils.config_module import ConfigModule
from utils.low_ros_module import LowROSModule

# ros
import rclpy


class DirectConrtolModule:  # NOTE: name kept as requested (typo preserved)
    """Direct control that stacks rel/abs tasks and solves one DLS at runtime."""

    def __init__(self, config: ConfigModule, desire_abs_pose, desire_rel_pose):
        # --- gains & limits
        self.rel_gain: float = getattr(config, "rel_gain", 1.0)
        self.abs_gain: float = getattr(config, "abs_gain", 1.0)
        self.dls_lambda: float = getattr(config, "dls_lambda", 1e-7)
        self.max_joint_speed: float = float(getattr(config, "low_level_max_vel", 0.3))

        # --- desired tasks (normalize for numerical stability)
        self.desire_rel_pose = DQ(desire_rel_pose).normalize()
        self.desire_abs_pose = DQ(desire_abs_pose).normalize()

        # --- robot 1 setup
        self.robot1_q_num: int = int(config.robot1_q_num)
        r1_dh_mat = np.array(config.robot1_dh_mat).T
        r1_base = DQ(config.robot1_base).normalize()
        r1_eff = DQ(config.robot1_effector).normalize()
        if int(getattr(config, "robot1_dh_type", 0)) == 1:
            self.cpu_robot1 = DQ_SerialManipulatorMDH(r1_dh_mat)
        else:
            self.cpu_robot1 = DQ_SerialManipulatorDH(r1_dh_mat)
        self.cpu_robot1.set_base_frame(r1_base)
        self.cpu_robot1.set_reference_frame(r1_base)
        self.cpu_robot1.set_effector(r1_eff)

        # --- robot 2 setup
        self.robot2_q_num: int = int(config.robot2_q_num)
        r2_dh_mat = np.array(config.robot2_dh_mat).T
        r2_base = DQ(config.robot2_base).normalize()
        r2_eff = DQ(config.robot2_effector).normalize()
        if int(getattr(config, "robot2_dh_type", 0)) == 1:
            self.cpu_robot2 = DQ_SerialManipulatorMDH(r2_dh_mat)
        else:
            self.cpu_robot2 = DQ_SerialManipulatorDH(r2_dh_mat)
        self.cpu_robot2.set_base_frame(r2_base)
        self.cpu_robot2.set_reference_frame(r2_base)
        self.cpu_robot2.set_effector(r2_eff)

        # --- cooperative dual-arm model (CPU)
        self.dual = DQ_CooperativeDualTaskSpace(self.cpu_robot1, self.cpu_robot2)

        # --- ROS I/O
        self.ros = LowROSModule(config)
        self.ros_thread = threading.Thread(target=self.ros.run, daemon=True)
        self.ros_thread.start()

        # state
        self.dual_arm_joint_pos = None  # type: ignore
        self.dual_arm_abs_feedback = None  # type: ignore
        self.dual_arm_joint_vel = None  # type: ignore

    # ----------------------------- lifecycle ---------------------------------
    def play_once(self) -> None:
        self.update_joint_states()
        self.compute_control()
        self.send_u()
        # telemetry
        self.ros.publish_abs_error_data(vec8(self.desire_abs_pose), vec8(self.dual_arm_abs_feedback))
        self.ros.publish_current_abs_pose(vec8(self.dual_arm_abs_feedback))
        self.ros.publish_current_abs_position(vec4(self.dual_arm_abs_feedback.translation()))

    # ------------------------------ internals --------------------------------
    def update_joint_states(self) -> None:
        r1_q, r2_q = self.ros.read_joint_state()
        self.dual_arm_joint_pos = np.concatenate((r1_q, r2_q))

    def compute_control(self) -> None:
        # feedback
        rel_fb = vec8(self.dual.relative_pose(self.dual_arm_joint_pos))
        abs_fb_dq = self.dual.absolute_pose(self.dual_arm_joint_pos)
        self.dual_arm_abs_feedback = abs_fb_dq
        abs_fb = vec8(abs_fb_dq)

        # task errors (8D each)
        e_rel = vec8(self.desire_rel_pose) - rel_fb
        e_abs = vec8(self.desire_abs_pose) - abs_fb

        # Jacobians (8 x (n1+n2))
        J_rel = self.dual.relative_pose_jacobian(self.dual_arm_joint_pos)
        J_abs = self.dual.absolute_pose_jacobian(self.dual_arm_joint_pos)

        # Row-weighting via gains (balance priorities)
        s_rel = self.rel_gain*0.1
        s_abs = self.abs_gain*0.1
        J = np.vstack((J_rel, J_abs))  # (16 x n)
        e = np.hstack((s_rel * e_rel, s_abs * e_abs))  # (16,)

        # Damped least-squares: J^T (JJ^T + λI)^{-1} e
        n_rows = J.shape[0]
        JJt = J @ J.T
        lamI = self.dls_lambda * np.eye(n_rows)
        J_pinv = J.T @ np.linalg.inv(JJt + lamI)
        u = J_pinv @ e

        # speed clamp (safety)
        u = np.clip(u, -self.max_joint_speed, self.max_joint_speed)
        self.dual_arm_joint_vel = u

    def send_u(self) -> None:
        n1 = self.robot1_q_num
        r1_dq = list(self.dual_arm_joint_vel[:n1])
        r2_dq = list(self.dual_arm_joint_vel[n1:])
        self.ros.write_u(r1_dq, r2_dq)

