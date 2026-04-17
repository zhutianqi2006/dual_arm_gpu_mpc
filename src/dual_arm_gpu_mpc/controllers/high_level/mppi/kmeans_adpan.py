#!/usr/bin/env python
import os
import time
import math

import numpy as np
import torch
from dqrobotics import DQ, vec4, vec8
from dq_torch import mppi_project_step, rel_abs_pose_rel_jac
from dual_arm_gpu_mpc.common.paths import project_root
from dual_arm_gpu_mpc.config.loader import ConfigModule
from dual_arm_gpu_mpc.controllers.high_level.mppi.base import DualArmMPPICore
from dual_arm_gpu_mpc.controllers.high_level.mppi.common_ops import (
    compute_weights,
    compute_weights_k,
    epsilon_generator,
    epsilon_generator_colored,
    epsilon_generator_log,
    get_abs_cost,
    get_acc_cost,
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
from dual_arm_gpu_mpc.robotics.dq.manipulability import (
    dq_jacobian_to_twist_jacobian,
    manipulability_from_twist_jacobian_rows,
    manipulability_from_twist_jacobian_rows_pdet,
    manipulability_from_twist_jacobian_rows_gram_eig_scaled,
)
from kmeans_pytorch import kmeans

class MPPIKmeansAdpAnModule(DualArmMPPICore):
    def __init__(self, config: ConfigModule , desire_abs_pose: torch.Tensor, desire_abs_position: torch.Tensor,
                 desire_rel_pose: torch.Tensor, desire_line_d: torch.Tensor,  desire_quat_line_ref: torch.Tensor):
        self.dtype = torch.float32
        self.device = "cuda:0"
        self.manip_weight = getattr(config, "manip_weight", 0.5)

        # manipulability options (all optional, keep backward compatible)
        self.manip_enable = getattr(config, "manip_enable", self.manip_weight > 0.0)
        self.manip_metric = getattr(config, "manip_metric", "pdet")
        self.manip_twist_rows = getattr(config, "manip_twist_rows", "all6")
        self.manip_pdet_k = getattr(config, "manip_pdet_k", None)
        self.manip_gram_eig_last3_div = float(getattr(config, "manip_gram_eig_last3_div", 1.0))
        self.manip_gram_eig_last_k = int(getattr(config, "manip_gram_eig_last_k", 3))
        self.manip_eps = float(getattr(config, "manip_eps", 1e-12))

        # manipulability logging (optional)
        self.manip_log_enable = bool(getattr(config, "manip_log_enable", False))
        self.manip_log_flush_every = int(getattr(config, "manip_log_flush_every", 50))
        self.manip_log_path = getattr(config, "manip_log_path", None)
        self._manip_log_steps: list[int] = []
        self._manip_log_t: list[float] = []
        self._manip_log_manip: list[float] = []
        self._manip_log_calls = 0
        if self.manip_log_enable:
            self._init_manip_logger()
        self.num_clusters = config.num_clusters
        self.common_num = config.common_num
        self._init_common_state(
            config,
            desire_abs_pose,
            desire_abs_position,
            desire_rel_pose,
            desire_line_d,
            desire_quat_line_ref,
        )

    def _init_manip_logger(self):
        logs_dir = project_root() / "logs"
        os.makedirs(logs_dir, exist_ok=True)

        if self.manip_log_path is None or str(self.manip_log_path).strip() == "":
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.manip_log_path = str(logs_dir / f"manip_trace_{ts}.npz")

    def _append_current_manip_log(self, t: float, manip_value: float):
        if not self.manip_log_enable:
            return

        self._manip_log_calls += 1
        self._manip_log_steps.append(int(self._manip_log_calls))
        self._manip_log_t.append(float(t))
        self._manip_log_manip.append(float(manip_value))

        if self.manip_log_flush_every > 0 and (self._manip_log_calls % self.manip_log_flush_every) == 0:
            self._flush_manip_log()

    def _flush_manip_log(self):
        if not self.manip_log_enable:
            return
        if self.manip_log_path is None:
            return
        if len(self._manip_log_steps) == 0:
            return

        np.savez(
            self.manip_log_path,
            step=np.asarray(self._manip_log_steps, dtype=np.int64),
            t=np.asarray(self._manip_log_t, dtype=np.float64),
            manipulability=np.asarray(self._manip_log_manip, dtype=np.float64),
            metric=str(self.manip_metric),
            rows=str(self.manip_twist_rows),
            pdet_k=(-1 if self.manip_pdet_k is None else int(self.manip_pdet_k)),
            gram_eig_last_k=int(self.manip_gram_eig_last_k),
            gram_eig_last3_div=float(self.manip_gram_eig_last3_div),
            weight=float(self.manip_weight),
        )

    def flush_manip_log(self):
        """Manually flush manipulability log to disk (npz)."""
        self._flush_manip_log()

    def _compute_manip_from_twist_jacobian(self, j_twist: torch.Tensor) -> torch.Tensor:
        """Return manipulability (..., 1) from a 6xN twist Jacobian per current config."""
        metric = str(self.manip_metric).lower()
        rows = str(self.manip_twist_rows).lower()
        eps = self.manip_eps

        if metric in ("pdet", "svd", "pseudo_det", "pseudodet"):
            k = None if self.manip_pdet_k is None else int(self.manip_pdet_k)
            return manipulability_from_twist_jacobian_rows_pdet(j_twist, rows=rows, k=k, eps=eps)
        if metric in ("gram_eig_scaled", "eig_scaled", "gram_eig"):
            return manipulability_from_twist_jacobian_rows_gram_eig_scaled(
                j_twist,
                rows=rows,
                last_k=int(self.manip_gram_eig_last_k),
                div=float(self.manip_gram_eig_last3_div),
                eps=eps,
            )
        if metric in ("yoshikawa", "det"):
            return manipulability_from_twist_jacobian_rows(j_twist, rows=rows, eps=eps)

        raise ValueError(
            f"Unknown manip_metric={self.manip_metric!r} (expected 'pdet', 'gram_eig_scaled', or 'yoshikawa')"
        )

    def _log_current_manipulability(self):
        """Log manipulability at the current real joint state (batch=1)."""
        if not self.manip_log_enable:
            return

        # time base: prefer controller start_time if warm_up was called
        now = time.time()
        t = float(now - self.start_time) if hasattr(self, "start_time") else float(now)

        q1 = torch.tensor(self.robot1_q, device=self.device, dtype=self.dtype).view(1, -1)
        q2 = torch.tensor(self.robot2_q, device=self.device, dtype=self.dtype).view(1, -1)

        rel_pos, _, rel_jac, _, _ = rel_abs_pose_rel_jac(
            self.gpu_robot1_dh_mat,
            self.gpu_robot2_dh_mat,
            self.batch_robot1_base[:1],
            self.batch_robot2_base[:1],
            self.batch_robot1_effector[:1],
            self.batch_robot2_effector[:1],
            q1,
            q2,
            self.batch_line_d[:1],
            self.batch_quat_line_ref[:1],
            self.robot1_q_num,
            self.robot2_q_num,
            self.robot1_dh_type,
            self.robot2_dh_type,
        )

        j_twist = dq_jacobian_to_twist_jacobian(rel_pos, rel_jac)  # (1, 6, N)
        manip = self._compute_manip_from_twist_jacobian(j_twist)  # (1, 1)
        manip_value = float(manip.detach().to(torch.float64).view(-1)[0].item())
        self._append_current_manip_log(t=t, manip_value=manip_value)

    def _before_play_once(self):
        if self.manip_log_enable:
            self._log_current_manipulability()

    def _after_play_once(self):
        if self.manip_log_enable and self.manip_log_flush_every <= 0:
            self._flush_manip_log()

    def _select_warm_up2_result(self, mppi_u0, mppi_u):
        return mppi_u[0].cpu().numpy()

    def mppi_worker(self):
        # init ur3 dual quaternion modelwa
        batch_last_mppi_result = self.last_mppi_result.repeat(self.batch_size, 1, 1)
        # robot1_acc_explore_seq, robot2_acc_explore_seq = epsilon_generator_log(int(self.batch_size), self.robot1_q_num, self.robot2_q_num, self.mppi_T,
        #                                                                        self.mean, self.std, self.log_std, self.mppi_dt*self.max_acc_abs_value, self.mppi_seed)
        robot1_acc_explore_seq, robot2_acc_explore_seq = epsilon_generator_colored(int(self.batch_size), self.robot1_q_num, self.robot2_q_num, self.mppi_T,
                                                                               self.mean, self.std, self.gamma, self.mppi_dt*self.max_acc_abs_value, self.mppi_seed)
        batch_last_robot1_mppi_result = batch_last_mppi_result[:,:,:self.robot1_q_num]
        batch_last_robot2_mppi_result = batch_last_mppi_result[:,:, self.robot1_q_num:]
        batch_robot1_dq_seq = robot1_acc_explore_seq + batch_last_mppi_result[:,:,:self.robot1_q_num]
        batch_robot2_dq_seq = robot2_acc_explore_seq + batch_last_mppi_result[:,:, self.robot1_q_num:]
        self.last_batch_fake_robot1_q = self.batch_fake_robot1_q.clone()
        self.last_batch_fake_robot2_q = self.batch_fake_robot2_q.clone()
        self.first_batch_fake_robot1_q = self.batch_fake_robot1_q.clone()
        self.first_batch_fake_robot2_q = self.batch_fake_robot2_q.clone()
        self.stage_cost = torch.zeros(self.batch_size, 1, dtype = self.dtype, device = self.device)
        for i in range(self.mppi_T):
            # init to get first nullspace matrix
            if i == 0:
                rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(self.gpu_robot1_dh_mat, self.gpu_robot2_dh_mat,
                                                     self.batch_robot1_base,  self.batch_robot2_base, 
                                                     self.batch_robot1_effector, self.batch_robot2_effector, 
                                                     self.batch_fake_robot1_q, self.batch_fake_robot2_q,
                                                     self.batch_line_d, self.batch_quat_line_ref, self.robot1_q_num, self.robot2_q_num, self.robot1_dh_type, self.robot2_dh_type)
                bacth_rel_jacobian_null =  get_rel_jacobian_null(bacth_rel_jacobian, self.robot1_q_num, self.robot2_q_num, self.batch_size)

            batch_robot1_ith_dq, batch_robot2_ith_dq = get_current_vel(batch_robot1_dq_seq, batch_robot2_dq_seq, i)

            last_batch_robot1_mppi_proj, last_batch_robot2_mppi_proj = get_proj_qd(batch_last_robot1_mppi_result[:,i,:], batch_last_robot2_mppi_result[:,i,:], 
                                                                             self.robot1_q_num, self.robot2_q_num, bacth_rel_jacobian_null)
            self.last_batch_fake_robot1_q = self.batch_fake_robot1_q
            self.last_batch_fake_robot2_q = self.batch_fake_robot2_q
            self.batch_fake_robot1_q, self.batch_fake_robot2_q, batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq = mppi_project_step(
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
            robot1_acc_explore_seq[:,i,:self.robot1_q_num] = batch_robot1_ith_proj_dq- last_batch_robot1_mppi_proj
            robot2_acc_explore_seq[:,i,:self.robot2_q_num] = batch_robot2_ith_proj_dq- last_batch_robot2_mppi_proj
            robot1_acc_explore_seq[:,i,:self.robot1_q_num] = torch.clamp(robot1_acc_explore_seq[:,i,:self.robot1_q_num], self.mppi_dt*self.gpu_robot1_ddq_min, self.mppi_dt*self.gpu_robot1_ddq_max)
            robot2_acc_explore_seq[:,i,:self.robot2_q_num] = torch.clamp(robot2_acc_explore_seq[:,i,:self.robot2_q_num], self.mppi_dt*self.gpu_robot2_ddq_min, self.mppi_dt*self.gpu_robot2_ddq_max)
            
            rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(self.gpu_robot1_dh_mat, self.gpu_robot2_dh_mat,
                                                     self.batch_robot1_base,  self.batch_robot2_base, 
                                                     self.batch_robot1_effector, self.batch_robot2_effector, 
                                                     self.batch_fake_robot1_q, self.batch_fake_robot2_q,
                                                     self.batch_line_d, self.batch_quat_line_ref, self.robot1_q_num, self.robot2_q_num, self.robot1_dh_type, self.robot2_dh_type)
            bacth_rel_jacobian_null =  get_rel_jacobian_null(bacth_rel_jacobian, self.robot1_q_num, self.robot2_q_num, self.batch_size)
            abs_cost = get_abs_cost(self.batch_desire_abs_pose, bacth_abs_pos, self.batch_desire_abs_position, batch_abs_position, self.abs_weight, self.abs_position_weight)
            vel_cost = get_vel_cost(batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq, self.q_vel_weight)
            tilt_constraint_cost = get_tilt_constraint_cost(batch_angle, self.batch_max_abs_tilt_angle, self.tilt_constraint_weight)
            if i == 0:
                acc_cost = get_acc_cost(batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq, batch_last_mppi_result, self.robot1_q_num, self.robot2_q_num, i, self.q_acc_weight)
            collision_cost = self.get_collision_cost(self.collision_constraint_weight)
            self.stage_cost += (abs_cost + vel_cost+ collision_cost + tilt_constraint_cost)
        
        joint_change = torch.square(self.first_batch_fake_robot1_q-self.batch_fake_robot1_q).sum(dim=1, keepdim=True)+torch.square(self.first_batch_fake_robot2_q-self.batch_fake_robot2_q).sum(dim=1, keepdim=True)
        min_joint_change = 0.001
        joint_change = torch.clamp(joint_change, min=min_joint_change)
        stagnation_cost = self.stagnation_weight*joint_change
        abs_terminal_cost = get_abs_cost(self.batch_desire_abs_pose, bacth_abs_pos, self.batch_desire_abs_position, batch_abs_position, self.terminal_abs_weight, self.terminal_abs_position_weight)
        
        self.stage_cost += acc_cost + abs_terminal_cost + abs_terminal_cost/stagnation_cost
       
        
        stage_median_value = torch.median(self.stage_cost)
        # 计算最小值和最大值
        normalized_stage_cost = self.stage_cost - stage_median_value
        stage_min_value = torch.min(normalized_stage_cost)
        stage_max_value = torch.max(normalized_stage_cost)
        scaled_stage_cost =  (self.robot1_q_num+self.robot2_q_num)*(normalized_stage_cost - stage_min_value) / (2*stage_max_value - 2*stage_min_value)
        exp_cost= stagnation_cost
        exp_median_value = torch.median(exp_cost)
        normalized_exp_cost = exp_cost - exp_median_value
        exp_min_value = torch.min(normalized_exp_cost)
        exp_max_value = torch.max(normalized_exp_cost)
        scaled_exp_cost =  (normalized_exp_cost - exp_min_value) / (exp_max_value - exp_min_value)
        abs_median_terminal_cost = torch.median(abs_terminal_cost)
        normalized_abs_terminal_cost = abs_terminal_cost - abs_median_terminal_cost
        abs_min_terminal_cost = torch.min(normalized_abs_terminal_cost)
        abs_max_terminal_cost = torch.max(normalized_abs_terminal_cost)
        scaled_abs_terminal_cost =  (normalized_abs_terminal_cost - abs_min_terminal_cost) / (abs_max_terminal_cost - abs_min_terminal_cost)
        normalized_abs_pos = bacth_abs_pos
        abs_pos_min = torch.min(normalized_abs_pos)
        abs_pos_max = torch.max(normalized_abs_pos)
        scaled_abs_pos =  (normalized_abs_pos - abs_pos_min) / (abs_pos_max - abs_pos_min)

        robot1_q_change = self.first_batch_fake_robot1_q-self.batch_fake_robot1_q
        robot2_q_change = self.first_batch_fake_robot2_q-self.batch_fake_robot2_q
        robot1_q_change_min = torch.min(robot1_q_change)
        robot1_q_change_max = torch.max(robot1_q_change)
        robot2_q_change_min = torch.min(robot2_q_change)
        robot2_q_change_max = torch.max(robot2_q_change)
        scaled_robot1_q_change =  (robot1_q_change - robot1_q_change_min) / (robot1_q_change_max - robot1_q_change_min)
        scaled_robot2_q_change =  (robot2_q_change - robot2_q_change_min) / (robot2_q_change_max - robot2_q_change_min)

        num_clusters =  self.num_clusters 
        combined_tensor = torch.cat((scaled_stage_cost, scaled_robot1_q_change, scaled_robot2_q_change), dim=1)
        # combined_tensor = torch.cat((scaled_stage_cost, 2*scaled_exp_cost), dim=1)
        # combined_tensor = scaled_exp_cost
        min_energy = self.stage_cost.min()
        min_index = self.stage_cost.argmin()
        cluster_ids_x, cluster_centers = kmeans(
            X=combined_tensor, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'), tol=1e-4)
        stage_cost = self.stage_cost.squeeze()  # 转换为1D张量
        _, indices = torch.topk(stage_cost, k=self.common_num, largest=False)
        indices = indices.cpu()  
        # 获取对应的类别
        selected_clusters = cluster_ids_x[indices]
        # 统计出现次数
        counts = torch.bincount(selected_clusters, minlength=num_clusters)
        # scores = torch.arange(10, 0, -1, dtype=torch.float32).to(selected_clusters.device)
        # # 加权统计簇得分
        # cluster_scores = torch.bincount(
        #     selected_clusters,
        #     weights=scores,
        #     minlength=num_clusters
        # )
        # # 找到得分最高的簇
        # highest_scoring_cluster = torch.argmax(cluster_scores).item()
        most_common_cluster = torch.argmax(counts).item()
        epsilon = get_all_dq_seq(robot1_acc_explore_seq, robot2_acc_explore_seq)
        # combined_tensor = (5*joint_change)
       
        min_type = most_common_cluster
        selected_indices = (cluster_ids_x == min_type).nonzero().squeeze()
        selected_stage_cost = self.stage_cost[selected_indices]
        selected_epsilon = epsilon[selected_indices]
        selected_batch_size = selected_indices.size(0)
        w_epsilon = compute_weights_k(selected_epsilon, selected_stage_cost, selected_batch_size, self.param_lambda)
        w_epsilon = self.moving_average_filter(w_epsilon, int(self.mppi_T))
        self.current_mppi_result = w_epsilon + self.last_mppi_result
        self.current_mppi_result = torch.clamp(self.current_mppi_result, torch.cat((self.gpu_robot1_dq_min, self.gpu_robot2_dq_min)), torch.cat((self.gpu_robot1_dq_max, self.gpu_robot2_dq_max)))
        self.mppi_result_modefied()
        self.last_mppi_result = self.current_mppi_result
        return self.current_mppi_result, min_energy
    
    def mppi_worker2(self):
        # init ur3 dual quaternion modelwa
        last_u=(self.current_mppi_result[0]).cpu().numpy()
        # self.robot1_q_temp = self.robot1_q + self.mppi_dt*last_u[:self.robot1_q_num]
        # self.robot2_q_temp = self.robot2_q + self.mppi_dt*last_u[self.robot1_q_num:]
        self.robot1_q_temp = self.robot1_q
        self.robot2_q_temp = self.robot2_q
        self.batch_fake_robot1_q = torch.tensor(self.robot1_q_temp, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_fake_robot2_q = torch.tensor(self.robot2_q_temp, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        batch_last_mppi_result = self.last_mppi_result.repeat(self.batch_size, 1, 1)
        robot1_acc_explore_seq, robot2_acc_explore_seq = epsilon_generator_log(int(self.batch_size), self.robot1_q_num, self.robot2_q_num, self.mppi_T,
                                                                               self.mean, self.an_std, self.log_std, self.mppi_dt*self.max_acc_abs_value, self.mppi_seed)
        batch_last_robot1_mppi_result = batch_last_mppi_result[:,:,:self.robot1_q_num]
        batch_last_robot2_mppi_result = batch_last_mppi_result[:,:, self.robot1_q_num:]
        batch_robot1_dq_seq = robot1_acc_explore_seq + batch_last_mppi_result[:,:,:self.robot1_q_num]
        batch_robot2_dq_seq = robot2_acc_explore_seq + batch_last_mppi_result[:,:, self.robot1_q_num:]
        self.last_batch_fake_robot1_q = self.batch_fake_robot1_q.clone()
        self.last_batch_fake_robot2_q = self.batch_fake_robot2_q.clone()
        self.first_batch_fake_robot1_q = self.batch_fake_robot1_q.clone()
        self.first_batch_fake_robot2_q = self.batch_fake_robot2_q.clone()
        self.stage_cost = torch.zeros(self.batch_size, 1, dtype = self.dtype, device = self.device)
        for i in range(self.mppi_T):
            # init to get first nullspace matrix
            if i == 0:
                rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(self.gpu_robot1_dh_mat, self.gpu_robot2_dh_mat,
                                                     self.batch_robot1_base,  self.batch_robot2_base, 
                                                     self.batch_robot1_effector, self.batch_robot2_effector, 
                                                     self.batch_fake_robot1_q, self.batch_fake_robot2_q,
                                                     self.batch_line_d, self.batch_quat_line_ref, self.robot1_q_num, self.robot2_q_num, self.robot1_dh_type, self.robot2_dh_type)
                # print("angle", batch_angle[0])
                bacth_rel_jacobian_null =  get_rel_jacobian_null(bacth_rel_jacobian, self.robot1_q_num, self.robot2_q_num, self.batch_size)

            batch_robot1_ith_dq, batch_robot2_ith_dq = get_current_vel(batch_robot1_dq_seq, batch_robot2_dq_seq, i)

            last_batch_robot1_mppi_proj, last_batch_robot2_mppi_proj = get_proj_qd(batch_last_robot1_mppi_result[:,i,:], batch_last_robot2_mppi_result[:,i,:], 
                                                                             self.robot1_q_num, self.robot2_q_num, bacth_rel_jacobian_null)
            self.last_batch_fake_robot1_q = self.batch_fake_robot1_q
            self.last_batch_fake_robot2_q = self.batch_fake_robot2_q
            self.batch_fake_robot1_q, self.batch_fake_robot2_q, batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq = mppi_project_step(
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
            robot1_acc_explore_seq[:,i,:self.robot1_q_num] = batch_robot1_ith_proj_dq- last_batch_robot1_mppi_proj
            robot2_acc_explore_seq[:,i,:self.robot2_q_num] = batch_robot2_ith_proj_dq- last_batch_robot2_mppi_proj
            robot1_acc_explore_seq[:,i,:self.robot1_q_num] = torch.clamp(robot1_acc_explore_seq[:,i,:self.robot1_q_num], self.mppi_dt*self.gpu_robot1_ddq_min, self.mppi_dt*self.gpu_robot1_ddq_max)
            robot2_acc_explore_seq[:,i,:self.robot2_q_num] = torch.clamp(robot2_acc_explore_seq[:,i,:self.robot2_q_num], self.mppi_dt*self.gpu_robot2_ddq_min, self.mppi_dt*self.gpu_robot2_ddq_max)
            
            rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(self.gpu_robot1_dh_mat, self.gpu_robot2_dh_mat,
                                                     self.batch_robot1_base,  self.batch_robot2_base, 
                                                     self.batch_robot1_effector, self.batch_robot2_effector, 
                                                     self.batch_fake_robot1_q, self.batch_fake_robot2_q,
                                                     self.batch_line_d, self.batch_quat_line_ref, self.robot1_q_num, self.robot2_q_num, self.robot1_dh_type, self.robot2_dh_type)
            bacth_rel_jacobian_null =  get_rel_jacobian_null(bacth_rel_jacobian, self.robot1_q_num, self.robot2_q_num, self.batch_size)
            abs_cost = get_abs_cost(self.batch_desire_abs_pose, bacth_abs_pos, self.batch_desire_abs_position, batch_abs_position, self.abs_weight, self.abs_position_weight)
            vel_cost = get_vel_cost(batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq, self.q_vel_weight)
            tilt_constraint_cost = get_tilt_constraint_cost(batch_angle, self.batch_max_abs_tilt_angle, self.tilt_constraint_weight)
            acc_cost = get_acc_cost(batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq, batch_last_mppi_result, self.robot1_q_num, self.robot2_q_num, i, self.q_acc_weight)
            collision_cost = self.get_collision_cost(self.collision_constraint_weight)
            self.stage_cost += (abs_cost + vel_cost+ collision_cost + acc_cost + tilt_constraint_cost)
        joint_change = torch.square(self.first_batch_fake_robot1_q-self.batch_fake_robot1_q).sum(dim=1, keepdim=True)+torch.square(self.first_batch_fake_robot2_q-self.batch_fake_robot2_q).sum(dim=1, keepdim=True)
        abs_terminal_cost = get_abs_cost(self.batch_desire_abs_pose, bacth_abs_pos, self.batch_desire_abs_position, batch_abs_position, self.terminal_abs_weight, self.terminal_abs_position_weight)
        if self.manip_enable:
            manip_cost = self._terminal_manip_cost(rel_pos, bacth_rel_jacobian)
            self.stage_cost += manip_cost
        self.stage_cost += abs_terminal_cost + 0.01*joint_change
        min_energy = self.stage_cost.min()
        epsilon = get_all_dq_seq(robot1_acc_explore_seq, robot2_acc_explore_seq)
        w_epsilon = compute_weights(epsilon, self.stage_cost, self.batch_size, self.param_lambda)
        w_epsilon = self.moving_average_filter(w_epsilon, int(self.mppi_T))
        self.current_mppi_result = w_epsilon + self.last_mppi_result
        # constraints
        self.current_mppi_result = torch.clamp(self.current_mppi_result, torch.cat((self.gpu_robot1_dq_min, self.gpu_robot2_dq_min)), torch.cat((self.gpu_robot1_dq_max, self.gpu_robot2_dq_max)))
        self.mppi_result_modefied()
        self.last_mppi_result[:-1,:] = self.current_mppi_result[1:,:]
        self.last_mppi_result[-1,:] = self.current_mppi_result[-1,:]
        return self.current_mppi_result, min_energy

    def _terminal_manip_cost(self, rel_pose_vec8: torch.Tensor, rel_jacobian: torch.Tensor) -> torch.Tensor:
        """Penalize low manipulability at the final rollout state.

        Uses a 6xN twist Jacobian derived from the DQ pose Jacobian.
        """
        # rel_pose_vec8: (B, 8), rel_jacobian: (B, 8, N)
        j_twist = dq_jacobian_to_twist_jacobian(rel_pose_vec8, rel_jacobian)  # (B, 6, N)

        eps = self.manip_eps
        manip = self._compute_manip_from_twist_jacobian(j_twist)
        return float(self.manip_weight) / torch.clamp(manip, min=eps)
    
    def traditional_control_result(self):
        dual_arm_joint_pos = np.concatenate((self.robot1_q, self.robot2_q))
        energy = 0
        for i in range(self.mppi_T):
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
            dual_arm_joint_pos +=  self.mppi_dt * dual_arm_joint_vel
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

    def get_cost(self, sequence):
        dual_arm_joint_pos = np.concatenate((self.robot1_q, self.robot2_q))
        energy = 0
        for i in range(self.mppi_T):
            dual_arm_abs_feedback = vec8(self.cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos))
            dual_arm_rel_feedback = vec8(self.cpu_dq_dual_arm_model.relative_pose(dual_arm_joint_pos))
            dual_arm_abs_error = self.cpu_desire_abs_pose - dual_arm_abs_feedback
            dual_arm_rel_jacobian = self.cpu_dq_dual_arm_model.relative_pose_jacobian(dual_arm_joint_pos)
            dual_arm_rel_jacobian_roboust_inv = dual_arm_rel_jacobian.T @ np.linalg.pinv(np.matmul(dual_arm_rel_jacobian, dual_arm_rel_jacobian.T) + 0.0000001 * np.eye(8))
            # abs control
            dual_arm_abs_feedback = vec8(self.cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos))
            dual_arm_abs_refer = vec8(DQ(self.desire_abs_pose).normalize())
            dual_arm_abs_joint_vel = sequence[i].cpu().numpy()
            # null space control
            dual_arm_joint_vel = np.matmul(np.eye(self.robot1_q_num+self.robot2_q_num)-dual_arm_rel_jacobian_roboust_inv@(dual_arm_rel_jacobian), dual_arm_abs_joint_vel)      
            dual_arm_joint_pos +=  self.mppi_dt * dual_arm_joint_vel
            dual_arm_abs_feedback = self.cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos)
            abs_cost = self.abs_weight* np.linalg.norm(dual_arm_abs_refer - vec8(dual_arm_abs_feedback))
            abs_pose_p = dual_arm_abs_feedback.P()
            abs_pose_d = dual_arm_abs_feedback.D()
            abs_position = (2*abs_pose_d*abs_pose_p.conj())
            current_l_quat = abs_pose_p*self.cpu_desire_line_d*abs_pose_p.conj()
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
        return  energy
    
# module-level helpers are provided below to keep the policy self-contained.
