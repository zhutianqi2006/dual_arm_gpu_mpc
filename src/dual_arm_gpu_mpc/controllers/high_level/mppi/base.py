#!/usr/bin/env python
import math
import threading
import time

import numpy as np
import torch
from dqrobotics import DQ, vec4, vec8
from dqrobotics.robot_modeling import (
    DQ_CooperativeDualTaskSpace,
    DQ_SerialManipulatorDH,
    DQ_SerialManipulatorMDH,
)

from dual_arm_gpu_mpc.config.loader import ConfigModule
from dual_arm_gpu_mpc.controllers.high_level.mppi.common_ops import moving_average_filter_tensor
from dual_arm_gpu_mpc.robotics.collision.curobo_loader import build_robot_world_pair, update_world_obstacle_pose


class DualArmMPPICore:
    def _init_common_state(
        self,
        config: ConfigModule,
        desire_abs_pose: torch.Tensor,
        desire_abs_position: torch.Tensor,
        desire_rel_pose: torch.Tensor,
        desire_line_d: torch.Tensor,
        desire_quat_line_ref: torch.Tensor,
    ):
        self.dtype = torch.float32
        self.device = "cuda:0"

        self.mppi_T = config.mppi_T
        self.mppi_dt = config.mppi_dt
        self.mppi_seed = config.mppi_seed
        self.batch_size = config.batch_size
        self.min_collision_distance = config.min_collision_distance
        self.min_self_collision_distance = config.min_self_collision_distance
        self.mean = config.mean
        self.std = config.std
        self.an_std = config.an_std
        self.log_std = config.log_std
        self.gamma = config.gamma
        self.batch_eps = 1e-8 * torch.eye(8, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1, 1)
        self.param_lambda = 0.5
        self.max_acc_abs_value = config.max_acc_abs_value
        self.warm_up_flag = False
        self.max_abs_tilt_angle = config.max_abs_tilt_angle

        self.collision_constraint_weight = config.collision_constraint_weight
        self.q_limit_constraint_weight = config.q_limit_constraint_weight
        self.q_vel_constraint_weight = config.q_vel_constraint_weight
        self.tilt_constraint_weight = config.tilt_constraint_weight
        self.abs_weight = config.abs_weight
        self.abs_position_weight = config.abs_position_weight
        self.terminal_abs_weight = config.terminal_abs_weight
        self.stagnation_weight = config.stagnation_weight
        self.terminal_abs_position_weight = config.terminal_abs_position_weight
        self.q_acc_weight = config.q_acc_weight
        self.q_vel_weight = config.q_vel_weight

        self.robot1_dh_mat = config.robot1_dh_mat
        self.robot1_base = config.robot1_base
        self.robot1_effector = config.robot1_effector
        self.robot1_q_num = config.robot1_q_num
        self.robot1_dh_type = config.robot1_dh_type

        self.robot2_dh_mat = config.robot2_dh_mat
        self.robot2_base = config.robot2_base
        self.robot2_effector = config.robot2_effector
        self.robot2_q_num = config.robot2_q_num
        self.robot2_dh_type = config.robot2_dh_type

        self.robot1_q_min = config.robot1_q_min
        self.robot1_q_max = config.robot1_q_max
        self.robot2_q_min = config.robot2_q_min
        self.robot2_q_max = config.robot2_q_max

        self.robot1_dq_min = config.robot1_dq_min
        self.robot1_dq_max = config.robot1_dq_max
        self.robot2_dq_min = config.robot2_dq_min
        self.robot2_dq_max = config.robot2_dq_max

        self.robot1_ddq_min = config.robot1_ddq_min
        self.robot1_ddq_max = config.robot1_ddq_max
        self.robot2_ddq_min = config.robot2_ddq_min
        self.robot2_ddq_max = config.robot2_ddq_max

        self.desire_abs_pose = desire_abs_pose
        self.desire_abs_position = desire_abs_position
        self.desire_rel_pose = desire_rel_pose
        self.desire_line_d = desire_line_d
        self.desire_quat_line_ref = desire_quat_line_ref
        self.high_rel_gain = config.high_rel_gain
        self.high_abs_gain = config.high_abs_gain
        self.c_abs_max = config.c_abs_max
        self.c_eta = config.c_eta
        self.c = 0

        self.init_cpu_dq_model()
        self.init_tensor()

        self.curobo_world_file = config.curobo_world_file
        self.curobo_robot_file = config.curobo_robot_file
        self.dynamic_obstacle_name = getattr(config, "dynamic_obstacle_name", "")
        self.init_collision_model()

        from dual_arm_gpu_mpc.ros.high import HighROSModule

        self.ros_module = HighROSModule(config)
        self.ros_thread = threading.Thread(target=self.ros_module.run)
        self.ros_thread.start()

    def _before_play_once(self):
        return None

    def _after_play_once(self):
        return None

    def _select_warm_up2_result(self, mppi_u0, mppi_u):
        return mppi_u0.cpu().numpy()

    def init_cpu_dq_model(self):
        self.cpu_desire_abs_pose = DQ(self.desire_abs_pose)
        self.cpu_desire_abs_pose = self.cpu_desire_abs_pose.normalize()
        self.cpu_desire_abs_pose = vec8(self.cpu_desire_abs_pose)
        self.cpu_desire_rel_pose = DQ(self.desire_rel_pose)
        self.cpu_desire_rel_pose = self.cpu_desire_rel_pose.normalize()
        self.cpu_desire_rel_pose = vec8(self.cpu_desire_rel_pose)
        self.cpu_desire_line_d = DQ(self.desire_line_d)
        self.cpu_desire_quat_line_ref = DQ(self.desire_quat_line_ref)

        robot1_config_dh_mat = np.array(self.robot1_dh_mat)
        self.cpu_robot1_dh_mat = robot1_config_dh_mat.T
        self.cpu_robot1_base = DQ(self.robot1_base)
        self.cpu_robot1_base = self.cpu_robot1_base.normalize()
        self.cpu_robot1_effector = DQ(self.robot1_effector)
        self.cpu_robot1_effector = self.cpu_robot1_effector.normalize()
        if self.robot1_dh_type == 1:
            self.cpu_robot1 = DQ_SerialManipulatorMDH(self.cpu_robot1_dh_mat)
        else:
            self.cpu_robot1 = DQ_SerialManipulatorDH(self.cpu_robot1_dh_mat)
        self.cpu_robot1.set_base_frame(self.cpu_robot1_base)
        self.cpu_robot1.set_reference_frame(self.cpu_robot1_base)
        self.cpu_robot1.set_effector(self.cpu_robot1_effector)

        robot2_config_dh_mat = np.array(self.robot2_dh_mat)
        self.cpu_robot2_dh_mat = robot2_config_dh_mat.T
        self.cpu_robot2_base = DQ(self.robot2_base)
        self.cpu_robot2_base = self.cpu_robot2_base.normalize()
        self.cpu_robot2_effector = DQ(self.robot2_effector)
        self.cpu_robot2_effector = self.cpu_robot2_effector.normalize()
        if self.robot2_dh_type == 1:
            self.cpu_robot2 = DQ_SerialManipulatorMDH(self.cpu_robot2_dh_mat)
        else:
            self.cpu_robot2 = DQ_SerialManipulatorDH(self.cpu_robot2_dh_mat)
        self.cpu_robot2.set_base_frame(self.cpu_robot2_base)
        self.cpu_robot2.set_reference_frame(self.cpu_robot2_base)
        self.cpu_robot2.set_effector(self.cpu_robot2_effector)

        self.cpu_dq_dual_arm_model = DQ_CooperativeDualTaskSpace(self.cpu_robot1, self.cpu_robot2)

    def init_ros(self):
        self.ros_thread = threading.Thread(target=self.ros_module.run)
        self.ros_thread.start()

    def update_joint_states(self):
        self.robot1_q, self.robot2_q = self.ros_module.read_joint_state()
        self.batch_fake_robot1_q = torch.tensor(self.robot1_q, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_fake_robot2_q = torch.tensor(self.robot2_q, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)

    def init_tensor(self):
        self.gpu_desire_abs_pose = torch.tensor(self.desire_abs_pose, device=self.device, dtype=self.dtype)
        self.gpu_desire_abs_position = torch.tensor(self.desire_abs_position, device=self.device, dtype=self.dtype)
        self.gpu_desire_rel_pose = torch.tensor(self.desire_rel_pose, device=self.device, dtype=self.dtype)
        self.batch_line_d = torch.tensor(self.desire_line_d, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_quat_line_ref = torch.tensor(self.desire_quat_line_ref, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )

        self.gpu_robot1_q_min = torch.tensor(self.robot1_q_min, device=self.device, dtype=self.dtype)
        self.gpu_robot1_q_max = torch.tensor(self.robot1_q_max, device=self.device, dtype=self.dtype)
        self.gpu_robot2_q_min = torch.tensor(self.robot2_q_min, device=self.device, dtype=self.dtype)
        self.gpu_robot2_q_max = torch.tensor(self.robot2_q_max, device=self.device, dtype=self.dtype)

        self.gpu_robot1_dq_min = torch.tensor(self.robot1_dq_min, device=self.device, dtype=self.dtype)
        self.gpu_robot1_dq_max = torch.tensor(self.robot1_dq_max, device=self.device, dtype=self.dtype)
        self.gpu_robot2_dq_min = torch.tensor(self.robot2_dq_min, device=self.device, dtype=self.dtype)
        self.gpu_robot2_dq_max = torch.tensor(self.robot2_dq_max, device=self.device, dtype=self.dtype)

        self.gpu_robot1_ddq_min = torch.tensor(self.robot1_ddq_min, device=self.device, dtype=self.dtype)
        self.gpu_robot1_ddq_max = torch.tensor(self.robot1_ddq_max, device=self.device, dtype=self.dtype)
        self.gpu_robot2_ddq_min = torch.tensor(self.robot2_ddq_min, device=self.device, dtype=self.dtype)
        self.gpu_robot2_ddq_max = torch.tensor(self.robot2_ddq_max, device=self.device, dtype=self.dtype)

        self.batch_desire_abs_position = self.gpu_desire_abs_position.repeat(self.batch_size, 1)
        self.batch_desire_abs_pose = self.gpu_desire_abs_pose.repeat(self.batch_size, 1)
        self.batch_desire_rel_pose = self.gpu_desire_rel_pose.repeat(self.batch_size, 1)

        self.gpu_robot1_dh_mat = torch.tensor(self.robot1_dh_mat, device=self.device, dtype=torch.float32)
        self.gpu_robot1_dh_mat = self.gpu_robot1_dh_mat.reshape(-1).contiguous()
        self.gpu_robot1_base = torch.tensor(self.robot1_base, device=self.device, dtype=self.dtype)
        self.batch_robot1_base = self.gpu_robot1_base.repeat(self.batch_size, 1)
        self.gpu_robot1_effector = torch.tensor(self.robot1_effector, device=self.device, dtype=self.dtype)
        self.batch_robot1_effector = self.gpu_robot1_effector.repeat(self.batch_size, 1)

        self.gpu_robot2_dh_mat = torch.tensor(self.robot2_dh_mat, device=self.device, dtype=torch.float32)
        self.gpu_robot2_dh_mat = self.gpu_robot2_dh_mat.reshape(-1).contiguous()
        self.gpu_robot2_base = torch.tensor(self.robot2_base, device=self.device, dtype=self.dtype)
        self.batch_robot2_base = self.gpu_robot2_base.repeat(self.batch_size, 1)
        self.gpu_robot2_effector = torch.tensor(self.robot2_effector, device=self.device, dtype=self.dtype)
        self.batch_robot2_effector = self.gpu_robot2_effector.repeat(self.batch_size, 1)

        total_q_num = self.robot1_q_num + self.robot2_q_num
        self.last_mppi_result = torch.zeros(self.mppi_T, total_q_num, device=self.device, dtype=self.dtype)
        self.current_mppi_result = torch.zeros(self.mppi_T, total_q_num, device=self.device, dtype=self.dtype)
        self.first_element_mppi_result = torch.zeros(total_q_num, device=self.device, dtype=self.dtype)
        self.batch_max_abs_tilt_angle = torch.tensor(self.max_abs_tilt_angle, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )

        self.batch_robot1_q_min = torch.tensor(self.robot1_q_min, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )
        self.batch_robot1_q_max = torch.tensor(self.robot1_q_max, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )
        self.batch_robot2_q_min = torch.tensor(self.robot2_q_min, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )
        self.batch_robot2_q_max = torch.tensor(self.robot2_q_max, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )

        self.batch_robot1_dq_min = torch.tensor(self.robot1_dq_min, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )
        self.batch_robot1_dq_max = torch.tensor(self.robot1_dq_max, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )
        self.batch_robot2_dq_min = torch.tensor(self.robot2_dq_min, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )
        self.batch_robot2_dq_max = torch.tensor(self.robot2_dq_max, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )

    def init_collision_model(self):
        self.tensor_args, self.curobo_config, self.curobo_fn, self.curobo_fn2 = build_robot_world_pair(
            self.curobo_robot_file,
            self.curobo_world_file,
            collision_activation_distance=self.min_collision_distance,
            self_collision_activation_distance=self.min_self_collision_distance,
        )

    def update_obstacle_velocity_estimate(self):
        pass

    def sync_dynamic_obstacle(self):
        if not self.dynamic_obstacle_name:
            return

        obstacle_position = getattr(self.ros_module, "dynamic_obstacle", None)
        if obstacle_position is None or len(obstacle_position) < 3:
            return

        update_world_obstacle_pose(
            self.curobo_fn,
            self.dynamic_obstacle_name,
            obstacle_position,
            tensor_args=self.tensor_args,
        )
        update_world_obstacle_pose(
            self.curobo_fn2,
            self.dynamic_obstacle_name,
            obstacle_position,
            tensor_args=self.tensor_args,
        )

    def get_collision_cost(self, weight: float):
        q = torch.cat((self.batch_fake_robot1_q, self.batch_fake_robot2_q), dim=1)
        q_mid = torch.cat(
            (
                (self.last_batch_fake_robot1_q + self.batch_fake_robot1_q) / 2.0,
                (self.last_batch_fake_robot2_q + self.batch_fake_robot2_q) / 2.0,
            ),
            dim=1,
        )
        d_world1, d_self1 = self.curobo_fn.get_world_self_collision_distance_from_joints(q)
        d_world2, d_self2 = self.curobo_fn.get_world_self_collision_distance_from_joints(q_mid)
        d_new = d_world1 + d_world2 + d_self1 + d_self2
        d_new[d_new != 0] = weight
        num_samples = d_new.size(0)
        return d_new.view(num_samples, 1)

    def moving_average_filter(self, xx: torch.Tensor, window_size: int) -> torch.Tensor:
        return moving_average_filter_tensor(xx, window_size)

    def mppi_result_modefied(self):
        dual_arm_joint_pos = np.concatenate((self.robot1_q, self.robot2_q))
        for i in range(self.mppi_T):
            dual_arm_rel_feedback = vec8(self.cpu_dq_dual_arm_model.relative_pose(dual_arm_joint_pos))
            dual_arm_rel_error = self.cpu_desire_rel_pose - dual_arm_rel_feedback
            dual_arm_rel_jacobian = self.cpu_dq_dual_arm_model.relative_pose_jacobian(dual_arm_joint_pos)
            dual_arm_rel_jacobian_roboust_inv = dual_arm_rel_jacobian.T @ np.linalg.pinv(
                np.matmul(dual_arm_rel_jacobian, dual_arm_rel_jacobian.T) + 0.0000001 * np.eye(8)
            )
            dual_arm_rel_joint_vel = self.high_rel_gain * np.matmul(dual_arm_rel_jacobian_roboust_inv, dual_arm_rel_error)
            current_mppi_vel = self.current_mppi_result[i, :].cpu().numpy()
            dual_arm_guide_joint_vel = np.matmul(
                np.eye(self.robot1_q_num + self.robot2_q_num) - dual_arm_rel_jacobian_roboust_inv @ (dual_arm_rel_jacobian),
                current_mppi_vel,
            )
            dual_arm_joint_vel = dual_arm_rel_joint_vel + dual_arm_guide_joint_vel
            self.current_mppi_result[i, :] = torch.tensor(dual_arm_joint_vel, device=self.device, dtype=self.dtype)
            dual_arm_joint_pos += self.mppi_dt * dual_arm_joint_vel

    def update_c(self, mppi_energy, p_energy):
        flag = False
        c_add = self.mppi_dt * (p_energy / mppi_energy - self.c_eta)
        self.c += c_add
        self.c = max(-0.05, min(self.c, 0.4))
        if self.c < 0:
            flag = True
        return flag

    def warm_up(self):
        for _ in range(10):
            self.update_joint_states()
            self.sync_dynamic_obstacle()
            mppi_u0, mppi_energy = self.mppi_worker()
            _, _ = self.mppi_worker2()
            p_u0, p_energy = self.traditional_control_result()
            mppi_u0 = mppi_u0.cpu().numpy()
            flag = self.update_c(mppi_energy, p_energy)
        self.last_mppi_result = torch.zeros(
            self.mppi_T, (self.robot1_q_num + self.robot2_q_num), device=self.device, dtype=self.dtype
        )
        self.current_mppi_result = torch.zeros(
            self.mppi_T, (self.robot1_q_num + self.robot2_q_num), device=self.device, dtype=self.dtype
        )
        self.first_element_mppi_result = torch.zeros(
            (self.robot1_q_num + self.robot2_q_num), device=self.device, dtype=self.dtype
        )
        self.c = 0.0
        self.start_time = time.time()

    def warm_up2(self):
        for _ in range(10):
            self.update_joint_states()
            self.sync_dynamic_obstacle()
            mppi_u0, mppi_energy = self.mppi_worker()
            mppi_u, _ = self.mppi_worker2()
            p_u0, p_energy = self.traditional_control_result()
            chosen_u = self._select_warm_up2_result(mppi_u0, mppi_u)
            flag = self.update_c(mppi_energy, p_energy)
        self.c = 0.0
        self.start_time = time.time()

    def play_once(self):
        self.update_joint_states()
        self.sync_dynamic_obstacle()
        self._before_play_once()
        _, mppi_energy = self.mppi_worker()
        mppi_u, _ = self.mppi_worker2()
        p_u0, p_energy = self.traditional_control_result()
        print("mppi_energy: ", mppi_energy)
        print("p_energy: ", p_energy)
        mppi_u0 = mppi_u[0].cpu().numpy()
        flag = self.update_c(mppi_energy, p_energy)
        print(self.c)
        if flag is True:
            u0 = p_u0
            self.last_mppi_result = torch.zeros(
                self.mppi_T, (self.robot1_q_num + self.robot2_q_num), device=self.device, dtype=self.dtype
            )
            self.current_mppi_result = torch.zeros(
                self.mppi_T, (self.robot1_q_num + self.robot2_q_num), device=self.device, dtype=self.dtype
            )
        else:
            u0 = mppi_u0
        u0 = u0.tolist()
        self.ros_module.write_high_u(u0)
        self._after_play_once()

    def shutdown(self, join_timeout: float = 1.0):
        try:
            import rclpy
        except ImportError:
            return

        if rclpy.ok():
            rclpy.shutdown()

        ros_thread = getattr(self, "ros_thread", None)
        if ros_thread is not None and ros_thread.is_alive() and ros_thread is not threading.current_thread():
            ros_thread.join(timeout=join_timeout)
