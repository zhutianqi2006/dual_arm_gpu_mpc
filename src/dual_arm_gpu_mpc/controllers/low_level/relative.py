from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
from dqrobotics import DQ, vec4, vec8
from dqrobotics.robot_modeling import (
    DQ_CooperativeDualTaskSpace,
    DQ_SerialManipulatorDH,
    DQ_SerialManipulatorMDH,
)

from dual_arm_gpu_mpc.config.loader import ConfigModule
from dual_arm_gpu_mpc.ros.low import LowROSModule


class RelativePoseLowLevelController:
    def __init__(
        self,
        config: ConfigModule,
        desire_abs_pose: Sequence[float],
        desire_rel_pose: Sequence[float],
    ):
        self.rel_gain = float(config.rel_gain)
        self.abs_gain = float(config.abs_gain)
        self.control_dt = 1.0 / float(config.ros_update_rate)

        self.desire_rel_pose = DQ(desire_rel_pose).normalize()
        self.desire_abs_pose = DQ(desire_abs_pose).normalize()

        self.robot1_q_num = int(config.robot1_q_num)
        self.robot2_q_num = int(config.robot2_q_num)

        self.cpu_robot1 = _build_robot_model(
            config.robot1_dh_mat,
            config.robot1_base,
            config.robot1_effector,
            config.robot1_dh_type,
        )
        self.cpu_robot2 = _build_robot_model(
            config.robot2_dh_mat,
            config.robot2_base,
            config.robot2_effector,
            config.robot2_dh_type,
        )
        self.cpu_dq_dual_arm_model = DQ_CooperativeDualTaskSpace(self.cpu_robot1, self.cpu_robot2)

        self.ros_module = LowROSModule(config)
        self.ros_thread = threading.Thread(target=self.ros_module.run, daemon=True)
        self.ros_thread.start()

    def play_once(self):
        self.update_joint_states()
        self.read_high_level_u()
        self.compute_control()
        self.send_u()
        self.ros_module.publish_abs_error_data(vec8(self.desire_abs_pose), vec8(self.dual_arm_abs_feedback))
        self.ros_module.publish_current_abs_pose(vec8(self.dual_arm_abs_feedback))
        self.ros_module.publish_current_abs_position(vec4(self.dual_arm_abs_feedback.translation()))

    def update_joint_states(self):
        robot1_q, robot2_q = self.ros_module.read_joint_state()
        self.dual_arm_joint_pos = np.concatenate((robot1_q, robot2_q))

    def read_high_level_u(self):
        self.high_level_u = np.asarray(self.ros_module.read_high_level_u(), dtype=np.float64)

    def compute_control(self):
        dual_arm_rel_feedback = vec8(self.cpu_dq_dual_arm_model.relative_pose(self.dual_arm_joint_pos))
        dual_arm_rel_error = vec8(self.desire_rel_pose) - dual_arm_rel_feedback
        dual_arm_rel_jacobian = self.cpu_dq_dual_arm_model.relative_pose_jacobian(self.dual_arm_joint_pos)
        dual_arm_rel_jacobian_robust_inv = dual_arm_rel_jacobian.T @ np.linalg.pinv(
            np.matmul(dual_arm_rel_jacobian, dual_arm_rel_jacobian.T) + 1.0e-7 * np.eye(8)
        )
        dual_arm_rel_joint_vel = self.rel_gain * np.matmul(dual_arm_rel_jacobian_robust_inv, dual_arm_rel_error)
        self.dual_arm_abs_feedback = self.cpu_dq_dual_arm_model.absolute_pose(self.dual_arm_joint_pos)
        nullspace_projector = (
            np.eye(self.robot1_q_num + self.robot2_q_num)
            - dual_arm_rel_jacobian_robust_inv @ dual_arm_rel_jacobian
        )
        self.dual_arm_joint_vel = dual_arm_rel_joint_vel + np.matmul(nullspace_projector, self.high_level_u)

    def send_u(self):
        robot1_dq = list(self.dual_arm_joint_vel[: self.robot1_q_num])
        robot2_dq = list(self.dual_arm_joint_vel[self.robot1_q_num :])
        self.ros_module.write_u(robot1_dq, robot2_dq)

    def shutdown(self, join_timeout: float = 1.0):
        try:
            import rclpy
        except ImportError:
            return

        if rclpy.ok():
            rclpy.shutdown()

        if self.ros_thread.is_alive() and self.ros_thread is not threading.current_thread():
            self.ros_thread.join(timeout=join_timeout)


LowLevelModule = RelativePoseLowLevelController


def _build_robot_model(dh_mat, base, effector, dh_type: int):
    robot_dh_mat = np.array(dh_mat).T
    robot_base = DQ(base).normalize()
    robot_effector = DQ(effector).normalize()
    if int(dh_type) == 1:
        robot = DQ_SerialManipulatorMDH(robot_dh_mat)
    else:
        robot = DQ_SerialManipulatorDH(robot_dh_mat)
    robot.set_base_frame(robot_base)
    robot.set_reference_frame(robot_base)
    robot.set_effector(robot_effector)
    return robot
