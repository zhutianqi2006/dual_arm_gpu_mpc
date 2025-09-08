
from __future__ import annotations

import math
import threading
from typing import Tuple, Optional

import numpy as np
import torch

# curobo for collision detection
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# DQ Robotics (CPU)
from dqrobotics import DQ, vec8
from dqrobotics.robot_modeling import (
    DQ_SerialManipulatorDH,
    DQ_SerialManipulatorMDH,
    DQ_CooperativeDualTaskSpace,
)

# DQ Robotics (CUDA)
from dq_torch import rel_abs_pose_rel_jac

# Project utilities
from utils.config_module import ConfigModule
from utils.high_ros_module import HighROSModule


class ICEMModule:
    """iCEM-based controller compatible with the MPPI/CEM module public API.

    仅在采样/分布更新与执行策略上与原 CEM 不同，其他代价、约束与接口保持一致。
    """

    # --------------------------- init ---------------------------
    def __init__(
        self,
        config: ConfigModule,
        desire_abs_pose: torch.Tensor,
        desire_abs_position: torch.Tensor,
        desire_rel_pose: torch.Tensor,
        desire_line_d: torch.Tensor,
        desire_quat_line_ref: torch.Tensor,
    ) -> None:
        # types/devices
        self.dtype = torch.float32
        self.device = "cuda:0"

        # horizon & sampling
        self.T = config.mppi_T
        self.dt = config.mppi_dt
        self.seed = config.mppi_seed
        self.batch_size = config.batch_size

        # constraints
        self.min_collision_distance = config.min_collision_distance
        self.min_self_collision_distance = config.min_self_collision_distance
        self.max_acc_abs_value = config.max_acc_abs_value
        self.max_abs_tilt_angle = config.max_abs_tilt_angle

        # weights
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

        # robots
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

        # limits
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

        # targets & aux
        self.desire_abs_pose = desire_abs_pose
        self.desire_abs_position = desire_abs_position
        self.desire_rel_pose = desire_rel_pose
        self.desire_line_d = desire_line_d
        self.desire_quat_line_ref = desire_quat_line_ref
        self.high_rel_gain = config.high_rel_gain
        self.high_abs_gain = config.high_abs_gain

        # curobo config
        self.curobo_world_file = config.curobo_world_file
        self.curobo_robot_file = config.curobo_robot_file

        # iCEM hyper-parameters (带默认值，若 config 未设置)
        self.cem_elite_frac = getattr(config, "cem_elite_frac", 0.03)   # 用于 top-k 选择
        self.cem_iters = getattr(config, "cem_iters", 5)
        self.cem_alpha = getattr(config, "cem_alpha", 0.8)              # 对精英分布的平滑
        self.init_std = getattr(config, "cem_init_std", getattr(config, "std", 0.3))
        self.decay = getattr(config, "cem_decay", 1.0)

        # iCEM-specific
        self.icem_beta = getattr(config, "icem_beta", 0.7)              # 每次迭代从精英附近重采样的比例
        self.icem_keep_frac = getattr(config, "icem_keep_frac", 0.3)    # 迭代内保留的精英比例
        self.icem_shift_frac = getattr(config, "icem_shift_frac", 0.3)  # 跨控制步时间移位的精英比例
        self.icem_elite_noise_scale = getattr(config, "icem_elite_noise_scale", 0.5)
        self.icem_include_mean_last_iter = getattr(config, "icem_include_mean_last_iter", True)

        # internal buffers
        self._init_cpu_dq_model()
        self._init_tensors()
        self._init_collision_model()

        # memory across control steps（iCEM: shift elites）
        self.prev_shifted_elites: torch.Tensor = torch.empty(0, self.T, self.total_q, device=self.device, dtype=self.dtype)

        # ROS
        self.ros_module = HighROSModule(config)
        self.ros_thread = threading.Thread(target=self.ros_module.run)
        self.ros_thread.start()

    # ------------------ init helpers ------------------
    def _init_cpu_dq_model(self) -> None:
        # targets
        self.cpu_desire_abs_pose = vec8(DQ(self.desire_abs_pose).normalize())
        self.cpu_desire_rel_pose = vec8(DQ(self.desire_rel_pose).normalize())
        self.cpu_desire_line_d = DQ(self.desire_line_d)
        self.cpu_desire_quat_line_ref = DQ(self.desire_quat_line_ref)

        # robot 1
        robot1_dh = np.array(self.robot1_dh_mat).T
        self.cpu_robot1_base = DQ(self.robot1_base).normalize()
        self.cpu_robot1_effector = DQ(self.robot1_effector).normalize()
        self.cpu_robot1 = (
            DQ_SerialManipulatorMDH(robot1_dh)
            if self.robot1_dh_type == 1
            else DQ_SerialManipulatorDH(robot1_dh)
        )
        self.cpu_robot1.set_base_frame(self.cpu_robot1_base)
        self.cpu_robot1.set_reference_frame(self.cpu_robot1_base)
        self.cpu_robot1.set_effector(self.cpu_robot1_effector)

        # robot 2
        robot2_dh = np.array(self.robot2_dh_mat).T
        self.cpu_robot2_base = DQ(self.robot2_base).normalize()
        self.cpu_robot2_effector = DQ(self.robot2_effector).normalize()
        self.cpu_robot2 = (
            DQ_SerialManipulatorMDH(robot2_dh)
            if self.robot2_dh_type == 1
            else DQ_SerialManipulatorDH(robot2_dh)
        )
        self.cpu_robot2.set_base_frame(self.cpu_robot2_base)
        self.cpu_robot2.set_reference_frame(self.cpu_robot2_base)
        self.cpu_robot2.set_effector(self.cpu_robot2_effector)

        # dual
        self.cpu_dual = DQ_CooperativeDualTaskSpace(self.cpu_robot1, self.cpu_robot2)

    def _init_tensors(self) -> None:
        dev, dt = self.device, self.dtype

        # targets (GPU)
        self.g_abs_pose = torch.tensor(self.desire_abs_pose, device=dev, dtype=dt)
        self.g_abs_position = torch.tensor(self.desire_abs_position, device=dev, dtype=dt)
        self.g_rel_pose = torch.tensor(self.desire_rel_pose, device=dev, dtype=dt)
        self.b_line_d = torch.tensor(self.desire_line_d, device=dev, dtype=dt).repeat(self.batch_size, 1)
        self.b_quat_line_ref = torch.tensor(self.desire_quat_line_ref, device=dev, dtype=dt).repeat(self.batch_size, 1)

        # DH/base/eff (GPU)
        self.g_r1_dh = torch.tensor(self.robot1_dh_mat, device=dev, dtype=torch.float32).reshape(-1).contiguous()
        self.g_r2_dh = torch.tensor(self.robot2_dh_mat, device=dev, dtype=torch.float32).reshape(-1).contiguous()
        self.g_r1_base = torch.tensor(self.robot1_base, device=dev, dtype=dt)
        self.g_r2_base = torch.tensor(self.robot2_base, device=dev, dtype=dt)
        self.g_r1_eff = torch.tensor(self.robot1_effector, device=dev, dtype=dt)
        self.g_r2_eff = torch.tensor(self.robot2_effector, device=dev, dtype=dt)

        # limits (GPU)
        self.g_r1_q_min = torch.tensor(self.robot1_q_min, device=dev, dtype=dt)
        self.g_r1_q_max = torch.tensor(self.robot1_q_max, device=dev, dtype=dt)
        self.g_r2_q_min = torch.tensor(self.robot2_q_min, device=dev, dtype=dt)
        self.g_r2_q_max = torch.tensor(self.robot2_q_max, device=dev, dtype=dt)

        self.g_r1_dq_min = torch.tensor(self.robot1_dq_min, device=dev, dtype=dt)
        self.g_r1_dq_max = torch.tensor(self.robot1_dq_max, device=dev, dtype=dt)
        self.g_r2_dq_min = torch.tensor(self.robot2_dq_min, device=dev, dtype=dt)
        self.g_r2_dq_max = torch.tensor(self.robot2_dq_max, device=dev, dtype=dt)

        self.g_r1_ddq_min = torch.tensor(self.robot1_ddq_min, device=dev, dtype=dt)
        self.g_r1_ddq_max = torch.tensor(self.robot1_ddq_max, device=dev, dtype=dt)
        self.g_r2_ddq_min = torch.tensor(self.robot2_ddq_min, device=dev, dtype=dt)
        self.g_r2_ddq_max = torch.tensor(self.robot2_ddq_max, device=dev, dtype=dt)

        # dist over action sequences (joint velocities)
        self.total_q = self.robot1_q_num + self.robot2_q_num
        self.action_mean = torch.zeros(self.T, self.total_q, device=dev, dtype=dt)
        self.action_std = torch.ones(self.T, self.total_q, device=dev, dtype=dt) * float(self.init_std)
        self.last_action_mean = torch.zeros_like(self.action_mean)

        # buffers
        self.current_plan = torch.zeros_like(self.action_mean)
        self.batch_max_abs_tilt_angle = (
            torch.tensor(self.max_abs_tilt_angle, device=dev, dtype=dt).repeat(self.batch_size, 1)
        )

    def _init_collision_model(self) -> None:
        self.tensor_args = TensorDeviceType()
        self.curobo_config = RobotWorldConfig.load_from_config(
            self.curobo_robot_file,
            self.curobo_world_file,
            collision_activation_distance=self.min_collision_distance,
            self_collision_activation_distance=self.min_self_collision_distance,
        )
        self.curobo_fn = RobotWorld(self.curobo_config)
        self.curobo_fn2 = RobotWorld(self.curobo_config)

    # --------------------------- runtime ---------------------------
    def update_joint_states(self) -> None:
        self.robot1_q, self.robot2_q = self.ros_module.read_joint_state()
        dev, dt = self.device, self.dtype
        self.b_r1_q = torch.tensor(self.robot1_q, device=dev, dtype=dt).repeat(self.batch_size, 1)
        self.b_r2_q = torch.tensor(self.robot2_q, device=dev, dtype=dt).repeat(self.batch_size, 1)

    def get_collision_cost(self, weight: float) -> torch.Tensor:
        # why: 将任何激活的碰撞/自碰撞转换为固定惩罚，避免距离尺度影响梯度
        q = torch.cat((self.b_r1_q, self.b_r2_q), dim=1)
        q_mid = torch.cat(((self.last_b_r1_q + self.b_r1_q) / 2.0, (self.last_b_r2_q + self.b_r2_q) / 2.0), dim=1)
        d_world1, d_self1 = self.curobo_fn.get_world_self_collision_distance_from_joints(q)
        d_world2, d_self2 = self.curobo_fn.get_world_self_collision_distance_from_joints(q_mid)
        d_new = d_world1 + d_world2 + d_self1 + d_self2
        d_new[d_new != 0] = weight
        return d_new.view(d_new.size(0), 1)

    # --------------------------- iCEM core ---------------------------
    def cem_worker(self) -> Tuple[torch.Tensor, torch.Tensor]:  # API 保持不变
        """Run iCEM planning; return (plan_seq [T,total_q], min_energy).

        与标准 CEM 的差异：迭代内基于精英重采样、跨步 shift 记忆、最后执行 best-action。
        """
        return self._icem_worker()

    def _icem_worker(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dev, dt = self.device, self.dtype
        torch.manual_seed(int(self.seed))

        elite_k = max(1, int(self.cem_elite_frac * self.batch_size))
        total_q = self.total_q

        # 以上一控制步的均值序列为“平滑参考”
        batch_last_seq = self.last_action_mean.repeat(self.batch_size, 1, 1)

        # 分布参数
        plan_mean = self.action_mean.clone()
        plan_std = self.action_std.clone()

        # 供下一控制步使用的 shift elites
        out_shift_pool: Optional[torch.Tensor] = None

        # 迭代内精英池（iCEM: keep elites）
        iter_elite_pool = torch.empty(0, self.T, total_q, device=dev, dtype=dt)

        min_energy: Optional[torch.Tensor] = None
        best_seq: Optional[torch.Tensor] = None

        for it in range(self.cem_iters):
            seqs = self._icem_sample_population(
                plan_mean,
                plan_std,
                iter_elite_pool,
                self.prev_shifted_elites,
                include_mean=(self.icem_include_mean_last_iter and it == self.cem_iters - 1),
            )

            # clamp by dq limits
            seqs = torch.max(seqs, torch.cat((self.g_r1_dq_min, self.g_r2_dq_min)).view(1, 1, -1))
            seqs = torch.min(seqs, torch.cat((self.g_r1_dq_max, self.g_r2_dq_max)).view(1, 1, -1))

            # rollout reset
            self.last_b_r1_q = self.b_r1_q.clone()
            self.last_b_r2_q = self.b_r2_q.clone()
            self.first_b_r1_q = self.b_r1_q.clone()
            self.first_b_r2_q = self.b_r2_q.clone()

            stage_cost = torch.zeros(self.batch_size, 1, device=dev, dtype=dt)

            # rollout
            for i in range(self.T):
                if i == 0:
                    _, b_abs_pose, b_rel_jac, b_abs_position, b_angle = rel_abs_pose_rel_jac(
                        self.g_r1_dh,
                        self.g_r2_dh,
                        self.g_r1_base.repeat(self.batch_size, 1),
                        self.g_r2_base.repeat(self.batch_size, 1),
                        self.g_r1_eff.repeat(self.batch_size, 1),
                        self.g_r2_eff.repeat(self.batch_size, 1),
                        self.b_r1_q,
                        self.b_r2_q,
                        self.b_line_d,
                        self.b_quat_line_ref,
                        self.robot1_q_num,
                        self.robot2_q_num,
                        self.robot1_dh_type,
                        self.robot2_dh_type,
                    )
                    b_null = get_rel_jacobian_null(b_rel_jac, self.robot1_q_num, self.robot2_q_num, self.batch_size)

                curr = seqs[:, i, :]
                b_r1_dq = curr[:, : self.robot1_q_num]
                b_r2_dq = curr[:, self.robot1_q_num :]

                # null-space projection for relative task
                b_r1_proj, b_r2_proj = get_proj_qd(
                    b_r1_dq, b_r2_dq, self.robot1_q_num, self.robot2_q_num, b_null
                )

                # dq limits
                b_r1_proj = torch.clamp(b_r1_proj, self.g_r1_dq_min, self.g_r1_dq_max)
                b_r2_proj = torch.clamp(b_r2_proj, self.g_r2_dq_min, self.g_r2_dq_max)

                # integrate with position limits
                self.last_b_r1_q = self.b_r1_q
                self.last_b_r2_q = self.b_r2_q
                self.b_r1_q, self.b_r2_q, b_r1_proj, b_r2_proj = update_joint_position_with_limits(
                    self.b_r1_q,
                    self.b_r2_q,
                    b_r1_proj,
                    b_r2_proj,
                    self.g_r1_q_min,
                    self.g_r1_q_max,
                    self.g_r2_q_min,
                    self.g_r2_q_max,
                    float(self.dt),
                )

                # kinematics for cost
                _, b_abs_pose, b_rel_jac, b_abs_position, b_angle = rel_abs_pose_rel_jac(
                    self.g_r1_dh,
                    self.g_r2_dh,
                    self.g_r1_base.repeat(self.batch_size, 1),
                    self.g_r2_base.repeat(self.batch_size, 1),
                    self.g_r1_eff.repeat(self.batch_size, 1),
                    self.g_r2_eff.repeat(self.batch_size, 1),
                    self.b_r1_q,
                    self.b_r2_q,
                    self.b_line_d,
                    self.b_quat_line_ref,
                    self.robot1_q_num,
                    self.robot2_q_num,
                    self.robot1_dh_type,
                    self.robot2_dh_type,
                )
                b_null = get_rel_jacobian_null(b_rel_jac, self.robot1_q_num, self.robot2_q_num, self.batch_size)

                # costs
                abs_cost = get_abs_cost(
                    self.g_abs_pose.repeat(self.batch_size, 1),
                    b_abs_pose,
                    self.g_abs_position.repeat(self.batch_size, 1),
                    b_abs_position,
                    self.abs_weight,
                    self.abs_position_weight,
                )
                vel_cost = get_vel_cost(b_r1_proj, b_r2_proj, self.q_vel_weight)
                smooth_cost = get_acc_cost(
                    b_r1_proj,
                    b_r2_proj,
                    batch_last_seq,  # continuity to previous plan
                    self.robot1_q_num,
                    self.robot2_q_num,
                    i,
                    self.q_acc_weight,
                )
                tilt_cost = get_tilt_constraint_cost(
                    b_angle, self.batch_max_abs_tilt_angle, self.tilt_constraint_weight
                )
                collision_cost = self.get_collision_cost(self.collision_constraint_weight)

                stage_cost += abs_cost + vel_cost + smooth_cost + tilt_cost + collision_cost

            # terminal & stagnation
            joint_change = (
                torch.square(self.first_b_r1_q - self.b_r1_q).sum(dim=1, keepdim=True)
                + torch.square(self.first_b_r2_q - self.b_r2_q).sum(dim=1, keepdim=True)
            )
            joint_change = torch.clamp(joint_change, min=0.001)

            term_abs_cost = get_abs_cost(
                self.g_abs_pose.repeat(self.batch_size, 1),
                b_abs_pose,
                self.g_abs_position.repeat(self.batch_size, 1),
                b_abs_position,
                self.terminal_abs_weight,
                self.terminal_abs_position_weight,
            )
            stage_cost += term_abs_cost + term_abs_cost / (self.stagnation_weight * joint_change)

            # select elites
            flat = stage_cost.squeeze(-1)
            best_vals, best_idx = torch.topk(flat, k=elite_k, largest=False)
            elites = seqs[best_idx]  # [K, T, Q]

            # track best sequence for execution (iCEM: execute best-action)
            cur_min, argmin = torch.min(best_vals, dim=0)
            best_seq = elites[argmin]
            min_energy = cur_min if min_energy is None else torch.minimum(min_energy, cur_min)

            # update distribution by elites (smoothing)
            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0, unbiased=False).clamp_min(1e-6)
            plan_mean = (1.0 - self.cem_alpha) * plan_mean + self.cem_alpha * new_mean
            plan_std = (1.0 - self.cem_alpha) * plan_std + self.cem_alpha * new_std

            # keep部分精英进入下一迭代采样池
            keep_n = max(1, int(self.icem_keep_frac * elite_k))
            iter_elite_pool = elites[:keep_n].detach()

            # 准备跨控制步的 shift elites（仅保留一次，以最后一迭代为准）
            out_shift_pool = elites[: max(1, int(self.icem_shift_frac * elite_k))].detach()

        # 保存本步分布与计划（用于下次 warm-start）
        self.current_plan = plan_mean.clamp(
            torch.cat((self.g_r1_dq_min, self.g_r2_dq_min)),
            torch.cat((self.g_r1_dq_max, self.g_r2_dq_max)),
        )
        self.current_mppi_result = self.current_plan.clone()  # 与下游接口保持一致的命名

        # 相对任务引导（与旧模块相同）
        self._apply_relative_guidance()

        # 记忆：shift elites 存到下一控制步
        if out_shift_pool is not None and out_shift_pool.numel() > 0:
            # 时间移位：去掉首个动作，尾部补噪声动作
            noise_last = torch.randn(out_shift_pool.size(0), 1, total_q, device=dev, dtype=dt) * (
                self.icem_elite_noise_scale * plan_std[-1:, :]
            ) + plan_mean[-1:, :]
            self.prev_shifted_elites = torch.cat([out_shift_pool[:, 1:, :], noise_last], dim=1).detach()
        else:
            self.prev_shifted_elites = torch.empty(0, self.T, total_q, device=dev, dtype=dt)

        # 滚动：均值右移；std 衰减
        self.last_action_mean = self.current_mppi_result.clone()
        self.action_mean[:-1, :] = self.current_mppi_result[1:, :]
        self.action_mean[-1, :].zero_()
        self.action_std *= self.decay

        # 存储 best 序列的首个动作以供执行
        self._last_best_u0 = (
            best_seq[0].detach().cpu().numpy().tolist() if best_seq is not None else None
        )

        return self.current_mppi_result, min_energy if min_energy is not None else torch.tensor(0.0, device=dev)

    def _icem_sample_population(
        self,
        plan_mean: torch.Tensor,
        plan_std: torch.Tensor,
        elite_pool: torch.Tensor,
        shifted_pool: torch.Tensor,
        include_mean: bool,
    ) -> torch.Tensor:
        """Compose the iCEM population for the *current* iteration.

        why: 复用精英（keep/shift）可显著降低预算，提高样本效率；在最后一次迭代把 mean 也加入候选，避免从未评估的均值被执行。
        """
        B = self.batch_size
        dev, dt = self.device, self.dtype

        # 1) 基础样本：全局 N(mean, std)
        base = torch.randn(B, self.T, self.total_q, device=dev, dtype=dt) * plan_std + plan_mean

        # 2) 从迭代内精英附近重采样
        n_resample = int(self.icem_beta * B)
        if elite_pool is not None and elite_pool.numel() > 0 and n_resample > 0:
            idx = torch.randint(0, elite_pool.size(0), (n_resample,), device=dev)
            seeds = elite_pool[idx]
            noise = torch.randn_like(seeds) * (self.icem_elite_noise_scale * plan_std)
            base[:n_resample] = (seeds + noise)

        # 3) 注入上一步时间移位的精英（若存在），用少量噪声扰动
        if shifted_pool is not None and shifted_pool.numel() > 0:
            m = min(shifted_pool.size(0), B // 4)  # 适度注入
            if m > 0:
                noise = torch.randn_like(shifted_pool[:m]) * (0.5 * self.icem_elite_noise_scale * plan_std)
                base[n_resample : n_resample + m] = shifted_pool[:m] + noise

        # 4) 在最后一次迭代显式注入均值序列（只占 1 个样本）
        if include_mean:
            base[-1] = plan_mean

        return base

    # --------------------------- control API ---------------------------
    def warm_up(self) -> None:
        for _ in range(5):
            self.update_joint_states()
            self._icem_worker()

    def warm_up2(self) -> None:
        for _ in range(3):
            self.update_joint_states()
            self._icem_worker()

    def play_once(self) -> None:
        self.update_joint_states()
        plan, _ = self._icem_worker()
        # iCEM: 执行 best-action，而非 mean-action
        if getattr(self, "_last_best_u0", None) is not None:
            u0 = self._last_best_u0
        else:
            u0 = plan[0].detach().cpu().numpy().tolist()
        self.ros_module.write_high_u(u0)

    # ------------------ guidance (CPU DQ) ------------------
    def _apply_relative_guidance(self) -> None:
        """Blend absolute task (planned) with relative-task regulation via nullspace.

        why: 在相对任务雅可比零空间中注入绝对任务规划量，确保双臂相对位姿约束优先。
        """
        dual_q = np.concatenate((self.robot1_q, self.robot2_q))
        for i in range(self.T):
            rel_fb = vec8(self.cpu_dual.relative_pose(dual_q))
            rel_err = self.cpu_desire_rel_pose - rel_fb
            Jrel = self.cpu_dual.relative_pose_jacobian(dual_q)
            Jrel_pinv = Jrel.T @ np.linalg.pinv(Jrel @ Jrel.T + 1e-7 * np.eye(8))
            v_rel = self.high_rel_gain * (Jrel_pinv @ rel_err)

            v_plan = self.current_mppi_result[i, :].detach().cpu().numpy()
            v_plan_ns = (np.eye(self.total_q) - Jrel_pinv @ Jrel) @ v_plan

            v_joint = v_rel + v_plan_ns
            self.current_mppi_result[i, :] = torch.tensor(v_joint, device=self.device, dtype=self.dtype)
            dual_q = dual_q + self.dt * v_joint


# ==========================
# TORCH UTILS（与原 CEM 模块一致）
# ==========================

@torch.jit.script
def update_joint_position_with_limits(
    b_r1_q: torch.Tensor,
    b_r2_q: torch.Tensor,
    b_r1_dq: torch.Tensor,
    b_r2_dq: torch.Tensor,
    r1_q_min: torch.Tensor,
    r1_q_max: torch.Tensor,
    r2_q_min: torch.Tensor,
    r2_q_max: torch.Tensor,
    dt: float,
):
    updated_r1_q = b_r1_q + b_r1_dq * dt
    updated_r2_q = b_r2_q + b_r2_dq * dt

    clamped_r1_q = torch.clamp(updated_r1_q, r1_q_min, r1_q_max)
    clamped_r2_q = torch.clamp(updated_r2_q, r2_q_min, r2_q_max)

    allowed_r1_dq = (clamped_r1_q - b_r1_q) / dt
    allowed_r2_dq = (clamped_r2_q - b_r2_q) / dt
    return clamped_r1_q, clamped_r2_q, allowed_r1_dq, allowed_r2_dq


@torch.jit.script
def get_proj_qd(
    b_r1_qd: torch.Tensor,
    b_r2_qd: torch.Tensor,
    r1_q_num: int,
    r2_q_num: int,
    b_null: torch.Tensor,
):
    both = torch.cat((b_r1_qd.unsqueeze(2), b_r2_qd.unsqueeze(2)), dim=1)
    b_first = b_null[:, :r1_q_num, :]
    b_last = b_null[:, r1_q_num:, :]
    r1_proj = torch.matmul(b_first, both).squeeze(2)
    r2_proj = torch.matmul(b_last, both).squeeze(2)
    return r1_proj, r2_proj


@torch.jit.script
def get_current_vel(r1_dq_seq: torch.Tensor, r2_dq_seq: torch.Tensor, i: int):
    r1_dq = r1_dq_seq[:, i, :]
    r2_dq = r2_dq_seq[:, i, :]
    return r1_dq, r2_dq


@torch.jit.script
def get_abs_cost(
    desire_abs_pose: torch.Tensor,
    abs_pose: torch.Tensor,
    desire_abs_position: torch.Tensor,
    abs_position: torch.Tensor,
    rot_weight: float,
    position_weight: float,
):
    quat_difference = rot_weight * torch.abs(desire_abs_pose - abs_pose)
    position_difference = position_weight * torch.abs(desire_abs_position - abs_position)
    result = quat_difference.sum(dim=1, keepdim=True) + position_difference.sum(dim=1, keepdim=True)
    return result


@torch.jit.script
def get_vel_cost(b_r1_dq: torch.Tensor, b_r2_dq: torch.Tensor, weight: float):
    diff = torch.abs(b_r1_dq) + torch.abs(b_r2_dq)
    return weight * diff.sum(dim=1, keepdim=True)


@torch.jit.script
def get_tilt_constraint_cost(angle: torch.Tensor, max_abs_tilt_angle: torch.Tensor, weight: float):
    vmin = (angle < -max_abs_tilt_angle).float()
    vmax = (angle > max_abs_tilt_angle).float()
    total = vmin + vmax
    return weight * total.sum(dim=1, keepdim=True)


@torch.jit.script
def get_acc_cost(
    b_r1_dq: torch.Tensor,
    b_r2_dq: torch.Tensor,
    ref_seq: torch.Tensor,  # [B,T,Q]
    r1_q_num: int,
    r2_q_num: int,
    i: int,
    weight: float,
):
    ref_r1 = ref_seq[:, i, :r1_q_num]
    ref_r2 = ref_seq[:, i, r1_q_num:]
    diff = torch.abs(b_r1_dq - ref_r1) + torch.abs(b_r2_dq - ref_r2)
    return weight * diff.sum(dim=1, keepdim=True)


@torch.jit.script
def get_rel_jacobian_null(jac_batch: torch.Tensor, r1_q_num: int, r2_q_num: int, batch_size: int):
    I = torch.eye(r1_q_num + r2_q_num, dtype=torch.float64, device="cuda:0").repeat(batch_size, 1, 1)
    eps = 1e-16 * torch.eye(8, dtype=torch.float64, device="cuda:0").repeat(batch_size, 1, 1)
    J = jac_batch.to(torch.float64)
    Jt = J.transpose(-2, -1)
    JJt = torch.matmul(J, Jt)
    Jpinv = Jt @ torch.inverse(JJt + eps)
    return (I - torch.matmul(Jpinv, J)).to(torch.float32)

