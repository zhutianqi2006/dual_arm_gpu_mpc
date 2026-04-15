#!/usr/bin/env python
import math

import numpy as np
import torch
from dqrobotics import DQ, vec4, vec8
from dq_torch import mppi_project_step, rel_abs_pose_rel_jac

from dual_arm_gpu_mpc.config.loader import ConfigModule
from dual_arm_gpu_mpc.controllers.high_level.mppi.base import DualArmMPPICore
from dual_arm_gpu_mpc.controllers.high_level.mppi.common_ops import (
    compute_weights,
    compute_weights_k,
    epsilon_generator,
    epsilon_generator_colored,
    epsilon_generator_log,
    get_acc_cost,
    get_abs_cost,
    get_all_dq_seq,
    get_all_q,
    get_current_vel,
    get_pos_constraint_cost,
    get_proj_qd,
    get_rel_jacobian_null,
    get_tilt_constraint_cost,
    get_vel_constraint_cost,
    get_vel_cost,
    update_fake_joint_pos,
    update_joint_position_with_limits,
)


class MPPIAdpAnModule(DualArmMPPICore):
    def __init__(
        self,
        config: ConfigModule,
        desire_abs_pose: torch.Tensor,
        desire_abs_position: torch.Tensor,
        desire_rel_pose: torch.Tensor,
        desire_line_d: torch.Tensor,
        desire_quat_line_ref: torch.Tensor,
    ):
        self._init_common_state(
            config,
            desire_abs_pose,
            desire_abs_position,
            desire_rel_pose,
            desire_line_d,
            desire_quat_line_ref,
        )

    def mppi_worker(self):
        batch_last_mppi_result = self.last_mppi_result.repeat(self.batch_size, 1, 1)
        robot1_acc_explore_seq, robot2_acc_explore_seq = epsilon_generator_colored(
            int(self.batch_size),
            self.robot1_q_num,
            self.robot2_q_num,
            self.mppi_T,
            self.mean,
            self.std,
            self.gamma,
            self.mppi_dt * self.max_acc_abs_value,
            self.mppi_seed,
        )
        batch_last_robot1_mppi_result = batch_last_mppi_result[:, :, : self.robot1_q_num]
        batch_last_robot2_mppi_result = batch_last_mppi_result[:, :, self.robot1_q_num :]
        batch_robot1_dq_seq = robot1_acc_explore_seq + batch_last_mppi_result[:, :, : self.robot1_q_num]
        batch_robot2_dq_seq = robot2_acc_explore_seq + batch_last_mppi_result[:, :, self.robot1_q_num :]
        self.last_batch_fake_robot1_q = self.batch_fake_robot1_q.clone()
        self.last_batch_fake_robot2_q = self.batch_fake_robot2_q.clone()
        self.first_batch_fake_robot1_q = self.batch_fake_robot1_q.clone()
        self.first_batch_fake_robot2_q = self.batch_fake_robot2_q.clone()
        self.stage_cost = torch.zeros(self.batch_size, 1, dtype=self.dtype, device=self.device)
        for i in range(self.mppi_T):
            if i == 0:
                rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(
                    self.gpu_robot1_dh_mat,
                    self.gpu_robot2_dh_mat,
                    self.batch_robot1_base,
                    self.batch_robot2_base,
                    self.batch_robot1_effector,
                    self.batch_robot2_effector,
                    self.batch_fake_robot1_q,
                    self.batch_fake_robot2_q,
                    self.batch_line_d,
                    self.batch_quat_line_ref,
                    self.robot1_q_num,
                    self.robot2_q_num,
                    self.robot1_dh_type,
                    self.robot2_dh_type,
                )
                bacth_rel_jacobian_null = get_rel_jacobian_null(
                    bacth_rel_jacobian, self.robot1_q_num, self.robot2_q_num, self.batch_size
                )

            batch_robot1_ith_dq, batch_robot2_ith_dq = get_current_vel(batch_robot1_dq_seq, batch_robot2_dq_seq, i)

            last_batch_robot1_mppi_proj, last_batch_robot2_mppi_proj = get_proj_qd(
                batch_last_robot1_mppi_result[:, i, :],
                batch_last_robot2_mppi_result[:, i, :],
                self.robot1_q_num,
                self.robot2_q_num,
                bacth_rel_jacobian_null,
            )
            self.last_batch_fake_robot1_q = self.batch_fake_robot1_q
            self.last_batch_fake_robot2_q = self.batch_fake_robot2_q
            self.batch_fake_robot1_q, self.batch_fake_robot2_q, batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq = (
                mppi_project_step(
                    bacth_rel_jacobian,
                    batch_robot1_ith_dq,
                    batch_robot2_ith_dq,
                    self.batch_fake_robot1_q,
                    self.batch_fake_robot2_q,
                    self.gpu_robot1_dq_min,
                    self.gpu_robot1_dq_max,
                    self.gpu_robot2_dq_min,
                    self.gpu_robot2_dq_max,
                    self.gpu_robot1_q_min,
                    self.gpu_robot1_q_max,
                    self.gpu_robot2_q_min,
                    self.gpu_robot2_q_max,
                    self.mppi_dt,
                )
            )
            robot1_acc_explore_seq[:, i, : self.robot1_q_num] = batch_robot1_ith_proj_dq - last_batch_robot1_mppi_proj
            robot2_acc_explore_seq[:, i, : self.robot2_q_num] = batch_robot2_ith_proj_dq - last_batch_robot2_mppi_proj
            robot1_acc_explore_seq[:, i, : self.robot1_q_num] = torch.clamp(
                robot1_acc_explore_seq[:, i, : self.robot1_q_num],
                self.mppi_dt * self.gpu_robot1_ddq_min,
                self.mppi_dt * self.gpu_robot1_ddq_max,
            )
            robot2_acc_explore_seq[:, i, : self.robot2_q_num] = torch.clamp(
                robot2_acc_explore_seq[:, i, : self.robot2_q_num],
                self.mppi_dt * self.gpu_robot2_ddq_min,
                self.mppi_dt * self.gpu_robot2_ddq_max,
            )

            rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(
                self.gpu_robot1_dh_mat,
                self.gpu_robot2_dh_mat,
                self.batch_robot1_base,
                self.batch_robot2_base,
                self.batch_robot1_effector,
                self.batch_robot2_effector,
                self.batch_fake_robot1_q,
                self.batch_fake_robot2_q,
                self.batch_line_d,
                self.batch_quat_line_ref,
                self.robot1_q_num,
                self.robot2_q_num,
                self.robot1_dh_type,
                self.robot2_dh_type,
            )
            bacth_rel_jacobian_null = get_rel_jacobian_null(
                bacth_rel_jacobian, self.robot1_q_num, self.robot2_q_num, self.batch_size
            )
            abs_cost = get_abs_cost(
                self.batch_desire_abs_pose,
                bacth_abs_pos,
                self.batch_desire_abs_position,
                batch_abs_position,
                self.abs_weight,
                self.abs_position_weight,
            )
            vel_cost = get_vel_cost(batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq, self.q_vel_weight)
            tilt_constraint_cost = get_tilt_constraint_cost(
                batch_angle, self.batch_max_abs_tilt_angle, self.tilt_constraint_weight
            )
            acc_cost = get_acc_cost(
                batch_robot1_ith_proj_dq,
                batch_robot2_ith_proj_dq,
                batch_last_mppi_result,
                self.robot1_q_num,
                self.robot2_q_num,
                i,
                self.q_acc_weight,
            )
            collision_cost = self.get_collision_cost(self.collision_constraint_weight)
            self.stage_cost += abs_cost + vel_cost + collision_cost + acc_cost + tilt_constraint_cost

        joint_change = torch.square(self.first_batch_fake_robot1_q - self.batch_fake_robot1_q).sum(
            dim=1, keepdim=True
        ) + torch.square(self.first_batch_fake_robot2_q - self.batch_fake_robot2_q).sum(dim=1, keepdim=True)
        min_joint_change = 0.001
        joint_change = torch.clamp(joint_change, min=min_joint_change)
        stagnation_cost = self.stagnation_weight * joint_change
        abs_terminal_cost = get_abs_cost(
            self.batch_desire_abs_pose,
            bacth_abs_pos,
            self.batch_desire_abs_position,
            batch_abs_position,
            self.terminal_abs_weight,
            self.terminal_abs_position_weight,
        )
        self.stage_cost += abs_terminal_cost + abs_terminal_cost / stagnation_cost
        min_energy = self.stage_cost.min()
        epsilon = get_all_dq_seq(robot1_acc_explore_seq, robot2_acc_explore_seq)
        w_epsilon = compute_weights(epsilon, self.stage_cost, self.batch_size, self.param_lambda)
        w_epsilon = self.moving_average_filter(w_epsilon, int(self.mppi_T))
        self.current_mppi_result = w_epsilon + self.last_mppi_result
        self.current_mppi_result = torch.clamp(
            self.current_mppi_result,
            torch.cat((self.gpu_robot1_dq_min, self.gpu_robot2_dq_min)),
            torch.cat((self.gpu_robot1_dq_max, self.gpu_robot2_dq_max)),
        )
        self.mppi_result_modefied()
        self.last_mppi_result = self.current_mppi_result
        return self.current_mppi_result, min_energy

    def mppi_worker2(self):
        self.robot1_q_temp = self.robot1_q
        self.robot2_q_temp = self.robot2_q
        self.batch_fake_robot1_q = torch.tensor(self.robot1_q_temp, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )
        self.batch_fake_robot2_q = torch.tensor(self.robot2_q_temp, device=self.device, dtype=self.dtype).repeat(
            self.batch_size, 1
        )
        batch_last_mppi_result = self.last_mppi_result.repeat(self.batch_size, 1, 1)
        robot1_acc_explore_seq, robot2_acc_explore_seq = epsilon_generator_log(
            int(self.batch_size),
            self.robot1_q_num,
            self.robot2_q_num,
            self.mppi_T,
            self.mean,
            self.an_std,
            self.log_std,
            self.mppi_dt * self.max_acc_abs_value,
            self.mppi_seed,
        )
        batch_last_robot1_mppi_result = batch_last_mppi_result[:, :, : self.robot1_q_num]
        batch_last_robot2_mppi_result = batch_last_mppi_result[:, :, self.robot1_q_num :]
        batch_robot1_dq_seq = robot1_acc_explore_seq + batch_last_mppi_result[:, :, : self.robot1_q_num]
        batch_robot2_dq_seq = robot2_acc_explore_seq + batch_last_mppi_result[:, :, self.robot1_q_num :]
        self.last_batch_fake_robot1_q = self.batch_fake_robot1_q.clone()
        self.last_batch_fake_robot2_q = self.batch_fake_robot2_q.clone()
        self.first_batch_fake_robot1_q = self.batch_fake_robot1_q.clone()
        self.first_batch_fake_robot2_q = self.batch_fake_robot2_q.clone()
        self.stage_cost = torch.zeros(self.batch_size, 1, dtype=self.dtype, device=self.device)
        for i in range(self.mppi_T):
            if i == 0:
                rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(
                    self.gpu_robot1_dh_mat,
                    self.gpu_robot2_dh_mat,
                    self.batch_robot1_base,
                    self.batch_robot2_base,
                    self.batch_robot1_effector,
                    self.batch_robot2_effector,
                    self.batch_fake_robot1_q,
                    self.batch_fake_robot2_q,
                    self.batch_line_d,
                    self.batch_quat_line_ref,
                    self.robot1_q_num,
                    self.robot2_q_num,
                    self.robot1_dh_type,
                    self.robot2_dh_type,
                )
                bacth_rel_jacobian_null = get_rel_jacobian_null(
                    bacth_rel_jacobian, self.robot1_q_num, self.robot2_q_num, self.batch_size
                )

            batch_robot1_ith_dq, batch_robot2_ith_dq = get_current_vel(batch_robot1_dq_seq, batch_robot2_dq_seq, i)

            last_batch_robot1_mppi_proj, last_batch_robot2_mppi_proj = get_proj_qd(
                batch_last_robot1_mppi_result[:, i, :],
                batch_last_robot2_mppi_result[:, i, :],
                self.robot1_q_num,
                self.robot2_q_num,
                bacth_rel_jacobian_null,
            )
            self.last_batch_fake_robot1_q = self.batch_fake_robot1_q
            self.last_batch_fake_robot2_q = self.batch_fake_robot2_q
            self.batch_fake_robot1_q, self.batch_fake_robot2_q, batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq = (
                mppi_project_step(
                    bacth_rel_jacobian,
                    batch_robot1_ith_dq,
                    batch_robot2_ith_dq,
                    self.batch_fake_robot1_q,
                    self.batch_fake_robot2_q,
                    self.gpu_robot1_dq_min,
                    self.gpu_robot1_dq_max,
                    self.gpu_robot2_dq_min,
                    self.gpu_robot2_dq_max,
                    self.gpu_robot1_q_min,
                    self.gpu_robot1_q_max,
                    self.gpu_robot2_q_min,
                    self.gpu_robot2_q_max,
                    self.mppi_dt,
                )
            )
            robot1_acc_explore_seq[:, i, : self.robot1_q_num] = batch_robot1_ith_proj_dq - last_batch_robot1_mppi_proj
            robot2_acc_explore_seq[:, i, : self.robot2_q_num] = batch_robot2_ith_proj_dq - last_batch_robot2_mppi_proj
            robot1_acc_explore_seq[:, i, : self.robot1_q_num] = torch.clamp(
                robot1_acc_explore_seq[:, i, : self.robot1_q_num],
                self.mppi_dt * self.gpu_robot1_ddq_min,
                self.mppi_dt * self.gpu_robot1_ddq_max,
            )
            robot2_acc_explore_seq[:, i, : self.robot2_q_num] = torch.clamp(
                robot2_acc_explore_seq[:, i, : self.robot2_q_num],
                self.mppi_dt * self.gpu_robot2_ddq_min,
                self.mppi_dt * self.gpu_robot2_ddq_max,
            )

            rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(
                self.gpu_robot1_dh_mat,
                self.gpu_robot2_dh_mat,
                self.batch_robot1_base,
                self.batch_robot2_base,
                self.batch_robot1_effector,
                self.batch_robot2_effector,
                self.batch_fake_robot1_q,
                self.batch_fake_robot2_q,
                self.batch_line_d,
                self.batch_quat_line_ref,
                self.robot1_q_num,
                self.robot2_q_num,
                self.robot1_dh_type,
                self.robot2_dh_type,
            )
            bacth_rel_jacobian_null = get_rel_jacobian_null(
                bacth_rel_jacobian, self.robot1_q_num, self.robot2_q_num, self.batch_size
            )
            abs_cost = get_abs_cost(
                self.batch_desire_abs_pose,
                bacth_abs_pos,
                self.batch_desire_abs_position,
                batch_abs_position,
                self.abs_weight,
                self.abs_position_weight,
            )
            vel_cost = get_vel_cost(batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq, self.q_vel_weight)
            tilt_constraint_cost = get_tilt_constraint_cost(
                batch_angle, self.batch_max_abs_tilt_angle, self.tilt_constraint_weight
            )
            acc_cost = get_acc_cost(
                batch_robot1_ith_proj_dq,
                batch_robot2_ith_proj_dq,
                batch_last_mppi_result,
                self.robot1_q_num,
                self.robot2_q_num,
                i,
                self.q_acc_weight,
            )
            collision_cost = self.get_collision_cost(self.collision_constraint_weight)
            self.stage_cost += abs_cost + vel_cost + collision_cost + acc_cost + tilt_constraint_cost
        joint_change = torch.square(self.first_batch_fake_robot1_q - self.batch_fake_robot1_q).sum(
            dim=1, keepdim=True
        ) + torch.square(self.first_batch_fake_robot2_q - self.batch_fake_robot2_q).sum(dim=1, keepdim=True)
        abs_terminal_cost = get_abs_cost(
            self.batch_desire_abs_pose,
            bacth_abs_pos,
            self.batch_desire_abs_position,
            batch_abs_position,
            self.terminal_abs_weight,
            self.terminal_abs_position_weight,
        )
        self.stage_cost += abs_terminal_cost + 0.01 * joint_change
        min_energy = self.stage_cost.min()
        epsilon = get_all_dq_seq(robot1_acc_explore_seq, robot2_acc_explore_seq)
        w_epsilon = compute_weights(epsilon, self.stage_cost, self.batch_size, self.param_lambda)
        w_epsilon = self.moving_average_filter(w_epsilon, int(self.mppi_T))
        self.current_mppi_result = w_epsilon + self.last_mppi_result
        self.current_mppi_result = torch.clamp(
            self.current_mppi_result,
            torch.cat((self.gpu_robot1_dq_min, self.gpu_robot2_dq_min)),
            torch.cat((self.gpu_robot1_dq_max, self.gpu_robot2_dq_max)),
        )
        self.mppi_result_modefied()
        self.last_mppi_result[:-1, :] = self.current_mppi_result[1:, :]
        self.last_mppi_result[-1, :] = self.current_mppi_result[-1, :]
        return self.current_mppi_result, min_energy

    def traditional_control_result(self):
        dual_arm_joint_pos = np.concatenate((self.robot1_q, self.robot2_q))
        energy = 0
        for i in range(self.mppi_T):
            dual_arm_abs_feedback = vec8(self.cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos))
            dual_arm_rel_feedback = vec8(self.cpu_dq_dual_arm_model.relative_pose(dual_arm_joint_pos))
            dual_arm_rel_error = self.cpu_desire_rel_pose - dual_arm_rel_feedback
            dual_arm_abs_error = self.cpu_desire_abs_pose - dual_arm_abs_feedback
            dual_arm_rel_jacobian = self.cpu_dq_dual_arm_model.relative_pose_jacobian(dual_arm_joint_pos)
            dual_arm_rel_jacobian_roboust_inv = dual_arm_rel_jacobian.T @ np.linalg.pinv(
                np.matmul(dual_arm_rel_jacobian, dual_arm_rel_jacobian.T) + 0.0000001 * np.eye(8)
            )
            dual_arm_abs_jacobian = self.cpu_dq_dual_arm_model.absolute_pose_jacobian(dual_arm_joint_pos)
            dual_arm_abs_feedback = vec8(self.cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos))
            dual_arm_abs_refer = vec8(DQ(self.desire_abs_pose).normalize())
            dual_arm_abs_error = dual_arm_abs_refer - dual_arm_abs_feedback
            dual_arm_abs_jacobian_roboust_inv = dual_arm_abs_jacobian.T @ np.linalg.pinv(
                np.matmul(dual_arm_abs_jacobian, dual_arm_abs_jacobian.T) + 0.0000001 * np.eye(8)
            )
            dual_arm_abs_joint_vel = self.high_abs_gain * np.matmul(dual_arm_abs_jacobian_roboust_inv, dual_arm_abs_error)
            dual_arm_joint_vel = np.matmul(
                np.eye(self.robot1_q_num + self.robot2_q_num) - dual_arm_rel_jacobian_roboust_inv @ (dual_arm_rel_jacobian),
                dual_arm_abs_joint_vel,
            )
            dual_arm_joint_vel = np.clip(dual_arm_joint_vel, -0.3, 0.3)
            if i == 0:
                dual_arm_return = dual_arm_joint_vel
            dual_arm_joint_pos += self.mppi_dt * dual_arm_joint_vel
            dual_arm_abs_feedback = self.cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos)
            abs_cost = self.abs_weight * np.linalg.norm(dual_arm_abs_refer - vec8(dual_arm_abs_feedback))
            abs_pose_p = dual_arm_abs_feedback.P()
            abs_pose_d = dual_arm_abs_feedback.D()
            abs_position = 2 * abs_pose_d * abs_pose_p.conj()
            current_l_quat = abs_pose_p * self.cpu_desire_line_d * abs_pose_p.conj()
            dot_val = vec4(current_l_quat).dot(vec4(self.cpu_desire_quat_line_ref))
            dot_val = max(min(dot_val, 1.0), -1.0)
            angle = 57.2958 * math.acos(dot_val)
            if abs(angle) > self.max_abs_tilt_angle:
                tilt_cost = 1 * self.tilt_constraint_weight
            else:
                tilt_cost = 0
            desire_abs_position = [0, self.desire_abs_position[0], self.desire_abs_position[1], self.desire_abs_position[2]]
            abs_position_cost = self.abs_position_weight * np.linalg.norm(desire_abs_position - vec4(abs_position))
            dual_arm_joint_pos_cuda = torch.from_numpy(dual_arm_joint_pos).view(
                1, (self.robot1_q_num + self.robot2_q_num)
            ).cuda().float()
            d_world, d_self = self.curobo_fn2.get_world_self_collision_distance_from_joints(dual_arm_joint_pos_cuda)
            d_new = d_world + d_self
            d_new[d_new != 0] = self.collision_constraint_weight
            energy += abs_cost + abs_position_cost + d_new + tilt_cost
        terminal_abs_cost = self.terminal_abs_weight * np.linalg.norm(dual_arm_abs_refer - vec8(dual_arm_abs_feedback))
        terminal_abs_position_cost = self.terminal_abs_position_weight * np.linalg.norm(desire_abs_position - vec4(abs_position))
        energy += terminal_abs_cost + terminal_abs_position_cost + tilt_cost
        return dual_arm_return, energy
