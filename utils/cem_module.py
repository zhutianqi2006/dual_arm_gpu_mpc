"""
file: utils/cem_an_module.py

Cross-Entropy Method (CEM) controller for dual-arm DQ robotics, refactored from the MPPI + KMeans module.
- Drops MPPI and KMeans; keeps collision, joint-limit, tilt constraints, absolute pose/position tracking.
- Same external interface as the MPPI module so it can be a drop-in replacement:
    - __init__(config, desire_abs_pose, desire_abs_position, desire_rel_pose, desire_line_d, desire_quat_line_ref)
    - warm_up(), warm_up2(), play_once()

Notes:
- Uses curobo for collision distances; dq_torch for fast batched kinematics.
- Plans in joint velocity space over horizon T with receding-horizon execution.
- Distribution: factorized Gaussian over (T, dof).
- Elite refit with smoothing (cem_alpha) and optional decay.
"""

# ==========================
# PSEUDOCODE / PLAN
# ==========================
# 1) Initialize:
#    - Copy robot/config params (T, dt, limits, weights...)
#    - Build CPU DQ models and CUDA tensors (targets, DH, bases, etc.)
#    - Build curobo world for collision queries
#    - Start ROS HighROSModule thread
#    - Init CEM distribution: action_mean, action_std (shape [T, total_q])
#
# 2) Update joint states:
#    - Read from ROS, replicate to batch for vectorized rollouts
#
# 3) CEM iteration (cem_worker):
#    for it in range(cem_iters):
#       3.1) Sample batch of action sequences from N(mean, std)
#       3.2) Simulate rollouts over horizon:
#            for i in 0..T-1:
#               - If i==0 compute rel_jacobian null projector
#               - Project joint velocities into nullspace
#               - Clamp velocities, update joint positions within limits
#               - Evaluate stage costs: abs pose+position, vel, smoothness to last plan,
#                 tilt angle constraint, collision cost
#       3.3) Add terminal costs and stagnation penalty
#       3.4) Select top-k elites and refit mean/std with smoothing
#    - Save current mean as planned sequence; apply relative-task guidance (same as MPPI module)
#    - Receding-horizon shift for next tick
#
# 4) warm_up(): run a few cem_worker passes to stabilize distribution
# 5) play_once(): recompute plan, execute first control via ROS


# ==========================
# CODE
# ==========================
import os
import time
import math
import threading
from typing import Tuple, List, Dict, Union, Optional

import numpy as np
import torch

# curobo for collision detection
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# DQ Robotics (CPU)
from dqrobotics import DQ, vec8, vec4
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


class CEMModule:
    """Cross-Entropy-Method based controller compatible with the MPPI module's public API."""

    def __init__(
        self,
        config: ConfigModule,
        desire_abs_pose: torch.Tensor,
        desire_abs_position: torch.Tensor,
        desire_rel_pose: torch.Tensor,
        desire_line_d: torch.Tensor,
        desire_quat_line_ref: torch.Tensor,
    ) -> None:
        # Basic types/devices
        self.dtype = torch.float32
        self.device = "cuda:0"

        # Horizon and sampling
        self.T = config.mppi_T
        self.dt = config.mppi_dt
        self.seed = config.mppi_seed
        self.batch_size = config.batch_size

        # Limits & constraints
        self.min_collision_distance = config.min_collision_distance
        self.min_self_collision_distance = config.min_self_collision_distance
        self.max_acc_abs_value = config.max_acc_abs_value
        self.max_abs_tilt_angle = config.max_abs_tilt_angle

        # Weights
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

        # Robots
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

        # Joint limits
        self.robot1_q_min = config.robot1_q_min
        self.robot1_q_max = config.robot1_q_max
        self.robot2_q_min = config.robot2_q_min
        self.robot2_q_max = config.robot2_q_max

        # Velocity limits
        self.robot1_dq_min = config.robot1_dq_min
        self.robot1_dq_max = config.robot1_dq_max
        self.robot2_dq_min = config.robot2_dq_min
        self.robot2_dq_max = config.robot2_dq_max

        # Acceleration limits
        self.robot1_ddq_min = config.robot1_ddq_min
        self.robot1_ddq_max = config.robot1_ddq_max
        self.robot2_ddq_min = config.robot2_ddq_min
        self.robot2_ddq_max = config.robot2_ddq_max

        # Targets & aux params
        self.desire_abs_pose = desire_abs_pose
        self.desire_abs_position = desire_abs_position
        self.desire_rel_pose = desire_rel_pose
        self.desire_line_d = desire_line_d
        self.desire_quat_line_ref = desire_quat_line_ref
        self.high_rel_gain = config.high_rel_gain
        self.high_abs_gain = config.high_abs_gain

        # Controller switching (match MPPIAdpAnModule behavior)
        self.c_abs_max = getattr(config, "c_abs_max", 0.4)
        self.c_eta = getattr(config, "c_eta", 0.0)
        self.c = 0.0

        # curobo config files
        self.curobo_world_file = config.curobo_world_file
        self.curobo_robot_file = config.curobo_robot_file

        # CEM hyper-parameters (with sensible defaults)
        self.cem_elite_frac = getattr(config, "cem_elite_frac", 0.01)
        self.cem_iters = getattr(config, "cem_iters", 100)
        self.cem_alpha = getattr(config, "cem_alpha", 0.8)  # higher -> track elites more
        self.init_std = getattr(config, "cem_init_std", 3.0)
        self.decay = getattr(config, "cem_decay", 1.0)      # std decay per control step

        # One-pole low-pass smoothing on planned sequence (match request: replace moving-average)
        # y[t] = a*y[t-1] + (1-a)*x[t]
        self.cem_lpf_alpha = float(getattr(config, "cem_lpf_alpha", 0.0))

        # Internal buffers
        self._init_cpu_dq_model()
        self._init_tensors()
        self._init_collision_model()

        # ROS
        self.ros_module = HighROSModule(config)
        self.ros_thread = threading.Thread(target=self.ros_module.run)
        self.ros_thread.start()
        self.start_time = time.time()

    def moving_average_filter(self, xx: torch.Tensor, window_size: int) -> torch.Tensor:
        """Apply moving average filter for smoothing input sequence, using numpy internally."""
        if window_size <= 1:
            return xx
        xx_np = xx.detach().cpu().numpy()
        b = np.ones(int(window_size)) / float(window_size)
        num_steps, num_controls = xx_np.shape
        xx_mean_np = np.zeros_like(xx_np)
        for d in range(num_controls):
            xx_mean_np[:, d] = np.convolve(xx_np[:, d], b, mode="same")
        return torch.from_numpy(xx_mean_np).to(xx.device)

    def low_pass_filter(self, xx: torch.Tensor, alpha: float) -> torch.Tensor:
        """One-pole IIR low-pass filter along time dimension (T x Q)."""
        a = float(alpha)
        if not (a > 0.0):
            return xx
        if a >= 1.0:
            a = 0.999

        yy = xx.clone()
        for t in range(1, yy.shape[0]):
            yy[t] = a * yy[t - 1] + (1.0 - a) * xx[t]
        return yy

    # ------------------ init helpers ------------------
    def _init_cpu_dq_model(self) -> None:
        # Targets
        self.cpu_desire_abs_pose = DQ(self.desire_abs_pose)
        self.cpu_desire_abs_pose = self.cpu_desire_abs_pose.normalize()
        self.cpu_desire_abs_pose = vec8(self.cpu_desire_abs_pose)
        self.cpu_desire_rel_pose = DQ(self.desire_rel_pose)
        self.cpu_desire_rel_pose = self.cpu_desire_rel_pose.normalize()
        self.cpu_desire_rel_pose = vec8(self.cpu_desire_rel_pose)
        self.cpu_desire_line_d = DQ(self.desire_line_d)
        self.cpu_desire_quat_line_ref = DQ(self.desire_quat_line_ref)
        
        # robot1
        robot1_config_dh_mat = np.array(self.robot1_dh_mat)
        self.cpu_robot1_dh_mat =  robot1_config_dh_mat.T
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
        # robot2
        robot2_config_dh_mat = np.array(self.robot2_dh_mat)
        self.cpu_robot2_dh_mat =  robot2_config_dh_mat.T
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
        # robot2 and robot1
        self.cpu_dq_dual_arm_model = DQ_CooperativeDualTaskSpace(self.cpu_robot1, self.cpu_robot2)

    def _init_tensors(self) -> None:
        dev, dt = self.device, self.dtype

        # Targets (GPU)
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

        # Limits (GPU)
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

        # Batch expansions
        self.b_abs_pose = self.g_abs_pose.repeat(self.batch_size, 1)
        self.b_abs_pos = self.g_abs_position.repeat(self.batch_size, 1)
        self.b_rel_pose = self.g_rel_pose.repeat(self.batch_size, 1)
        self.b_r1_base = self.g_r1_base.repeat(self.batch_size, 1)
        self.b_r2_base = self.g_r2_base.repeat(self.batch_size, 1)
        self.b_r1_eff = self.g_r1_eff.repeat(self.batch_size, 1)
        self.b_r2_eff = self.g_r2_eff.repeat(self.batch_size, 1)

        # Distribution over action sequences (joint velocities)
        self.total_q = self.robot1_q_num + self.robot2_q_num
        self.action_mean = torch.zeros(self.T, self.total_q, device=dev, dtype=dt)
        self.action_std = torch.ones(self.T, self.total_q, device=dev, dtype=dt) * float(self.init_std)
        self.last_action_mean = torch.zeros_like(self.action_mean)

        # Buffers used during rollout
        self.current_plan = torch.zeros_like(self.action_mean)
        self.batch_eps8 = 1e-8 * torch.eye(8, device=dev, dtype=dt).repeat(self.batch_size, 1, 1)
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

    # ------------------ runtime ------------------
    def update_joint_states(self) -> None:
        self.robot1_q, self.robot2_q = self.ros_module.read_joint_state()
        dev, dt = self.device, self.dtype
        self.b_r1_q = torch.tensor(self.robot1_q, device=dev, dtype=dt).repeat(self.batch_size, 1)
        self.b_r2_q = torch.tensor(self.robot2_q, device=dev, dtype=dt).repeat(self.batch_size, 1)

    def get_collision_cost(self, weight: float) -> torch.Tensor:
        q = torch.cat((self.b_r1_q, self.b_r2_q), dim=1)
        q_mid = torch.cat(((self.last_b_r1_q + self.b_r1_q) / 2.0, (self.last_b_r2_q + self.b_r2_q) / 2.0), dim=1)
        d_world1, d_self1 = self.curobo_fn.get_world_self_collision_distance_from_joints(q)
        d_world2, d_self2 = self.curobo_fn.get_world_self_collision_distance_from_joints(q_mid)
        d_new = d_world1 + d_world2 + d_self1 + d_self2
        d_new[d_new != 0] = weight  # why: convert any activation into fixed penalty
        return d_new.view(d_new.size(0), 1)

    def cem_worker(
        self, capture_variance: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]]:
        """Run CEM planning and optionally capture per-iteration std statistics."""
        dev, dt = self.device, self.dtype
        torch.manual_seed(int(self.seed))

        # Important: each CEM iteration must rollout from the same *current* state.
        start_b_r1_q = self.b_r1_q.clone()
        start_b_r2_q = self.b_r2_q.clone()

        elite_k = max(1, int(self.cem_elite_frac * self.batch_size))
        total_q = self.total_q

        # Reference sequence: last velocity plan (used as rollout nominal + smoothness reference)
        batch_last_seq = self.last_action_mean.repeat(self.batch_size, 1, 1)

        # In this CEM variant we optimize in acceleration-space (actually Δdq = ddq * dt), like MPPI.
        # We sample eps sequences and add them to the last velocity plan to obtain dq for rollouts.
        eps_min = (torch.cat((self.g_r1_ddq_min, self.g_r2_ddq_min)) * float(self.dt)).view(1, 1, -1)
        eps_max = (torch.cat((self.g_r1_ddq_max, self.g_r2_ddq_max)) * float(self.dt)).view(1, 1, -1)
        if float(self.max_acc_abs_value) > 0.0:
            abs_bound = torch.full((1, 1, total_q), float(self.dt) * float(self.max_acc_abs_value), device=dev, dtype=dt)
            eps_min = torch.maximum(eps_min, -abs_bound)
            eps_max = torch.minimum(eps_max, abs_bound)

        # CEM iterations
        plan_mean = self.action_mean.clone()
        plan_std = self.action_std.clone()
        min_energy = None

        variance_history: Optional[List[Dict[str, object]]] = [] if capture_variance else None

        for iter_idx in range(self.cem_iters):
            # Sample eps (Δdq) sequences: [B, T, Q]
            seqs = torch.randn(self.batch_size, self.T, total_q, device=dev, dtype=dt) * plan_std + plan_mean

            # Clamp eps by acceleration bounds (ddq*dt) and optional absolute bound
            seqs = torch.max(seqs, eps_min)
            seqs = torch.min(seqs, eps_max)

            # Reset rollout state (per-iteration)
            self.b_r1_q = start_b_r1_q.clone()
            self.b_r2_q = start_b_r2_q.clone()
            self.last_b_r1_q = self.b_r1_q.clone()
            self.last_b_r2_q = self.b_r2_q.clone()
            self.first_b_r1_q = self.b_r1_q.clone()
            self.first_b_r2_q = self.b_r2_q.clone()

            stage_cost = torch.zeros(self.batch_size, 1, device=dev, dtype=dt)

            # Rollout
            for i in range(self.T):
                # Compute kinematics and nullspace projector at the CURRENT state
                if i == 0:
                    _, b_abs_pose, b_rel_jac, b_abs_position, b_angle = rel_abs_pose_rel_jac(
                        self.g_r1_dh,
                        self.g_r2_dh,
                        self.b_r1_base,
                        self.b_r2_base,
                        self.b_r1_eff,
                        self.b_r2_eff,
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

                # Current joint-velocity command from the sample
                curr_eps = seqs[:, i, :]
                curr_dq = batch_last_seq[:, i, :] + curr_eps
                b_r1_dq = curr_dq[:, : self.robot1_q_num]
                b_r2_dq = curr_dq[:, self.robot1_q_num :]

                # Project into relative-task nullspace
                b_r1_proj, b_r2_proj = get_proj_qd(
                    b_r1_dq, b_r2_dq, self.robot1_q_num, self.robot2_q_num, b_null
                )

                # Respect dq limits
                b_r1_proj = torch.clamp(b_r1_proj, self.g_r1_dq_min, self.g_r1_dq_max)
                b_r2_proj = torch.clamp(b_r2_proj, self.g_r2_dq_min, self.g_r2_dq_max)

                # Integrate with position limits
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

                # Kinematics for cost
                _, b_abs_pose, b_rel_jac, b_abs_position, b_angle = rel_abs_pose_rel_jac(
                    self.g_r1_dh,
                    self.g_r2_dh,
                    self.b_r1_base,
                    self.b_r2_base,
                    self.b_r1_eff,
                    self.b_r2_eff,
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

                # Costs
                abs_cost = get_abs_cost(self.b_abs_pose, b_abs_pose, self.b_abs_pos, b_abs_position, self.abs_weight, self.abs_position_weight)
                vel_cost = get_vel_cost(b_r1_proj, b_r2_proj, self.q_vel_weight)
                smooth_cost = get_acc_cost(
                    b_r1_proj,
                    b_r2_proj,
                    batch_last_seq,  # encourages continuity to previous plan
                    self.robot1_q_num,
                    self.robot2_q_num,
                    i,
                    self.q_acc_weight,
                )
                tilt_cost = get_tilt_constraint_cost(b_angle, self.batch_max_abs_tilt_angle, self.tilt_constraint_weight)
                collision_cost = self.get_collision_cost(self.collision_constraint_weight)

                stage_cost += abs_cost + vel_cost + smooth_cost + tilt_cost + collision_cost

            # Terminal & stagnation
            joint_change = (
                torch.square(self.first_b_r1_q - self.b_r1_q).sum(dim=1, keepdim=True)
                + torch.square(self.first_b_r2_q - self.b_r2_q).sum(dim=1, keepdim=True)
            )
            joint_change = torch.clamp(joint_change, min=0.001)

            term_abs_cost = get_abs_cost(self.b_abs_pose, b_abs_pose, self.b_abs_pos, b_abs_position, self.terminal_abs_weight, self.terminal_abs_position_weight)
            # Only add the terminal/stagnation ratio term on the first CEM iteration.
            # (Matches MPPIKmeansAdpAnModule behavior for the first planning refinement; later CEM iters don't need it.)
            stage_cost += term_abs_cost
            if iter_idx == 0:
                stage_cost += term_abs_cost / (self.stagnation_weight * joint_change)

            # Elite selection
            flat = stage_cost.squeeze(-1)
            best_vals, best_idx = torch.topk(flat, k=elite_k, largest=False)
            elites = seqs[best_idx]  # [K, T, Q]

            # Update distribution with smoothing
            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0, unbiased=False).clamp_min(1e-6)
            plan_mean = new_mean
            plan_std = (1.0 - self.cem_alpha) * plan_std + self.cem_alpha * new_std

            if capture_variance:
                step_mean_std = plan_std.mean(dim=1).detach().cpu().tolist()
                variance_history.append(
                    {
                        "iteration": int(iter_idx),
                        "mean_std": float(plan_std.mean().detach().item()),
                        "min_std": float(plan_std.min().detach().item()),
                        "max_std": float(plan_std.max().detach().item()),
                        "step_mean_std": step_mean_std,
                    }
                )

            # Track min energy
            cur_min = best_vals.min()
            min_energy = cur_min if min_energy is None else torch.minimum(min_energy, cur_min)

        # Save plan and apply relative-task guidance
        # Convert optimized eps-mean (Δdq) into a velocity plan by adding to last velocity plan
        self.current_plan = (self.last_action_mean + plan_mean).clamp(
            torch.cat((self.g_r1_dq_min, self.g_r2_dq_min)),
            torch.cat((self.g_r1_dq_max, self.g_r2_dq_max)),
        )

        # Smooth planned sequence (match MPPI: moving average on the planned sequence)
        self.current_mppi_result = self.current_plan.clone()  # reuse downstream method name
        self.current_mppi_result = self.moving_average_filter(self.current_mppi_result, int(self.T))
        self.current_mppi_result = self.current_mppi_result.clamp(
            torch.cat((self.g_r1_dq_min, self.g_r2_dq_min)),
            torch.cat((self.g_r1_dq_max, self.g_r2_dq_max)),
        )
        self._apply_relative_guidance()

        # Receding horizon: remember & shift
        self.last_action_mean = self.current_mppi_result.clone()
        # Keep distribution warm-start in eps-space
        self.action_mean[:-1, :] = plan_mean[1:, :]
        self.action_mean[-1, :].zero_()
        self.action_std *= self.decay

        min_energy = min_energy if min_energy is not None else torch.tensor(0.0, device=dev)
        if capture_variance:
            return self.current_mppi_result, min_energy, variance_history if variance_history is not None else []
        return self.current_mppi_result, min_energy

    def warm_up(self) -> None:
        # Match MPPIAdpAnModule warm-up: update switch state, then reset planner state.
        for _ in range(10):
            self.update_joint_states()
            _, cem_energy = self.cem_worker()
            _, p_energy = self.traditional_control_result()
            self.update_c(cem_energy, p_energy)

        # action_mean is eps(Δdq) mean; last_action_mean is the last velocity plan
        self.action_mean.zero_()
        self.last_action_mean.zero_()
        self.current_plan.zero_()
        self.current_mppi_result = torch.zeros_like(self.action_mean)
        self.action_std.fill_(float(self.init_std))
        self.c = 0.0
        self.start_time = time.time()

    def warm_up2(self) -> None:
        for _ in range(10):
            self.update_joint_states()
            _, cem_energy = self.cem_worker()
            _, p_energy = self.traditional_control_result()
            self.update_c(cem_energy, p_energy)

        self.c = 0.0
        self.start_time = time.time()

    def play_once(self) -> None:
        self.update_joint_states()
        plan, cem_energy = self.cem_worker()
        p_u0, p_energy = self.traditional_control_result()

        try:
            cem_energy_v = float(cem_energy.detach().item()) if isinstance(cem_energy, torch.Tensor) else float(cem_energy)
        except Exception:
            cem_energy_v = float("nan")
        try:
            p_energy_v = float(p_energy.detach().item()) if isinstance(p_energy, torch.Tensor) else float(p_energy)
        except Exception:
            p_energy_v = float("nan")

        u0_cem = plan[0].detach().cpu().numpy()
        flag = self.update_c(cem_energy, p_energy)

        print("cem_energy:", cem_energy_v)
        print("p_energy:", p_energy_v)
        print("c:", float(self.c), "switch_to_traditional:", bool(flag))
        if flag:
            u0 = p_u0
            # Reset planner state when falling back, like MPPIAdpAnModule
            # action_mean is eps(Δdq) mean; last_action_mean is the last velocity plan
            self.action_mean.zero_()
            self.last_action_mean.zero_()
            self.current_plan.zero_()
            self.current_mppi_result = torch.zeros_like(self.action_mean)
            self.action_std.fill_(float(self.init_std))
        else:
            u0 = u0_cem

        self.ros_module.write_high_u(u0.tolist())

    def update_c(self, cem_energy: Union[torch.Tensor, float], p_energy: Union[torch.Tensor, float]) -> bool:
        """Return True when switching to traditional controller is triggered."""
        if isinstance(cem_energy, torch.Tensor):
            cem_energy_v = float(cem_energy.detach().item())
        else:
            cem_energy_v = float(cem_energy)
        if isinstance(p_energy, torch.Tensor):
            p_energy_v = float(p_energy.detach().item())
        else:
            p_energy_v = float(p_energy)

        cem_energy_v = max(cem_energy_v, 1e-12)
        c_add = float(self.dt) * (p_energy_v / cem_energy_v - float(self.c_eta))
        self.c += c_add
        self.c = max(-0.05, min(float(self.c_abs_max), float(self.c)))
        return self.c < 0.0

    def traditional_control_result(self):
        dual_arm_joint_pos = np.concatenate((self.robot1_q, self.robot2_q))
        energy = 0
        for i in range(self.T):
            dual_arm_abs_feedback = vec8(self.cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos))
            dual_arm_rel_feedback = vec8(self.cpu_dq_dual_arm_model.relative_pose(dual_arm_joint_pos))
            dual_arm_rel_error = self.cpu_desire_rel_pose - dual_arm_rel_feedback
            dual_arm_abs_error = self.cpu_desire_abs_pose - dual_arm_abs_feedback
            dual_arm_rel_jacobian = self.cpu_dq_dual_arm_model.relative_pose_jacobian(dual_arm_joint_pos)
            dual_arm_rel_jacobian_roboust_inv = dual_arm_rel_jacobian.T @ np.linalg.pinv(np.matmul(dual_arm_rel_jacobian, dual_arm_rel_jacobian.T) + 0.0000001 * np.eye(8))
            # abs control
            dual_arm_abs_jacobian = self.cpu_dq_dual_arm_model.absolute_pose_jacobian(dual_arm_joint_pos)
            dual_arm_abs_feedback = vec8(self.cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos))
            dual_arm_abs_refer = vec8(DQ(self.desire_abs_pose).normalize())
            dual_arm_abs_error = dual_arm_abs_refer - dual_arm_abs_feedback
            dual_arm_abs_jacobian_roboust_inv = dual_arm_abs_jacobian.T @ np.linalg.pinv(np.matmul(dual_arm_abs_jacobian, dual_arm_abs_jacobian.T) + 0.0000001 * np.eye(8))
            dual_arm_abs_joint_vel = self.high_abs_gain * np.matmul(dual_arm_abs_jacobian_roboust_inv, (dual_arm_abs_error))
            # null space control
            dual_arm_joint_vel = np.matmul(np.eye(self.robot1_q_num+self.robot2_q_num)-dual_arm_rel_jacobian_roboust_inv@(dual_arm_rel_jacobian), dual_arm_abs_joint_vel)
            dual_arm_joint_vel = np.clip(dual_arm_joint_vel, -0.3, 0.3)           
            if i == 0:
                dual_arm_return = dual_arm_joint_vel 
            dual_arm_joint_pos +=  self.dt * dual_arm_joint_vel
            dual_arm_abs_feedback = self.cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos)
            abs_cost = self.abs_weight* np.linalg.norm(dual_arm_abs_refer - vec8(dual_arm_abs_feedback))
            abs_pose_p = dual_arm_abs_feedback.P()
            abs_pose_d = dual_arm_abs_feedback.D()
            abs_position = (2*abs_pose_d*abs_pose_p.conj())
            current_l_quat = abs_pose_p*self.cpu_desire_line_d*abs_pose_p.conj()
            current_l_quat = current_l_quat.normalize()
            self.cpu_desire_quat_line_ref = self.cpu_desire_quat_line_ref.normalize()
            angle = 57.2958*math.acos(vec4(current_l_quat).dot(vec4(self.cpu_desire_quat_line_ref)))
            if abs(angle) > self.max_abs_tilt_angle:
                tilt_cost = 1*self.tilt_constraint_weight
            else:
                tilt_cost = 0
            desire_abs_position = [0,self.desire_abs_position[0],self.desire_abs_position[1],self.desire_abs_position[2]]
            abs_position_cost = self.abs_position_weight * np.linalg.norm(desire_abs_position - vec4(abs_position))
            dual_arm_joint_pos_cuda = torch.from_numpy(dual_arm_joint_pos).view(1, (self.robot1_q_num+self.robot2_q_num)).cuda().float()
            d_world, d_self = self.curobo_fn2.get_world_self_collision_distance_from_joints(dual_arm_joint_pos_cuda)
            d_new = d_world + d_self
            d_new[d_new!=0] = self.collision_constraint_weight
            energy += abs_cost + abs_position_cost + d_new + tilt_cost
        terminal_abs_cost = self.terminal_abs_weight * np.linalg.norm(dual_arm_abs_refer - vec8(dual_arm_abs_feedback))
        terminal_abs_position_cost = self.terminal_abs_position_weight * np.linalg.norm(desire_abs_position - vec4(abs_position))
        energy += terminal_abs_cost + terminal_abs_position_cost +tilt_cost
        return  dual_arm_return, energy

    # ------------------ guidance (CPU DQ) ------------------
    def _apply_relative_guidance(self) -> None:
        """Blend absolute task (planned) with relative-task regulation via nullspace."""
        dual_q = np.concatenate((self.robot1_q, self.robot2_q))
        for i in range(self.T):
            # Relative-task feedback
            rel_fb = vec8(self.cpu_dq_dual_arm_model.relative_pose(dual_q))
            rel_err = self.cpu_desire_rel_pose - rel_fb
            Jrel = self.cpu_dq_dual_arm_model.relative_pose_jacobian(dual_q)
            Jrel_pinv = Jrel.T @ np.linalg.pinv(Jrel @ Jrel.T + 1e-7 * np.eye(8))
            v_rel = self.high_rel_gain * (Jrel_pinv @ rel_err)

            # Absolute-task planned joint vel (from CEM)
            v_plan = self.current_mppi_result[i, :].detach().cpu().numpy()
            v_plan_ns = (np.eye(self.total_q) - Jrel_pinv @ Jrel) @ v_plan

            v_joint = v_rel + v_plan_ns
            self.current_mppi_result[i, :] = torch.tensor(v_joint, device=self.device, dtype=self.dtype)
            dual_q = dual_q + self.dt * v_joint


# ==========================
# TORCH UTILS (ported from MPPI module; kept minimal)
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
    # Here both are the same source tensor; we split in projector
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
    ref_seq: torch.Tensor,  # [B,T,Q] previous plan (broadcasted)
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
