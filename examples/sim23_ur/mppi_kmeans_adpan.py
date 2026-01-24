#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import rclpy
from dq_torch import rel_abs_pose_rel_jac
from utils.config_module import ConfigModule
import utils.mppi_kmeans_adpan_module as mppi_mod
from utils.high_ros_module import HighROSModule



def build_parser():
    parser = argparse.ArgumentParser(description="Run dual-arm MPPI control or offline evaluations.")
    parser.add_argument("--mode", choices=["run", "eval"], default="eval", help="run: ROS loop; eval: offline variance study")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "ur3_and_ur3e.yaml"), help="Path to YAML config")
    parser.add_argument("--logdir", default=os.path.join(os.path.dirname(__file__), "../../logs/kl_eval"), help="Directory for evaluation logs")
    parser.add_argument("--variances", type=float, nargs="*", default=[3.0, 0.6], help="Std values to sweep in eval mode")
    parser.add_argument("--rollouts", type=int, default=3, help="Number of MPPI batches per variance")
    parser.add_argument("--hist-bins", type=int, default=40, help="Histogram bins for discrete KL")
    parser.add_argument("--ros-domain", default="16", help="ROS_DOMAIN_ID when mode=run")
    return parser


def run_control(args):
    import rclpy

    os.environ["ROS_DOMAIN_ID"] = args.ros_domain
    rclpy.init(args=None)

    desire_abs_pose = [-0.009809, -0.700866, -0.008828, 0.713171, 0.03289, -0.000662, -0.283115, -0.003703]
    desire_abs_position = [0.45, 0.0, 0.35]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.054285, -0.000927, -0.262089, -0.003409]
    desire_line_d = [0, 0, 0, 1]
    desire_quat_line_ref = [0, -0.9995, -0.026341, 0.017418]

    config = ConfigModule(args.config)
    mppi_module = mppi_mod.MPPIKmeansAdpAnModule(
        config,
        desire_abs_pose,
        desire_abs_position,
        desire_rel_pose,
        desire_line_d,
        desire_quat_line_ref,
    )
    mppi_module.warm_up()
    mppi_module.warm_up2()
    while True:
        mppi_module.play_once()


def simulate_rollout(module: mppi_mod.MPPIKmeansAdpAnModule, std_value: float):
    device = module.device
    dtype = module.dtype
    batch_size = module.batch_size
    nq = module.robot1_q_num + module.robot2_q_num

    module.std = std_value
    q1, q2 = module.ros_module.read_joint_state()
    module.robot1_q = q1
    module.robot2_q = q2
    module.batch_fake_robot1_q = torch.tensor(q1, device=device, dtype=dtype).repeat(batch_size, 1)
    module.batch_fake_robot2_q = torch.tensor(q2, device=device, dtype=dtype).repeat(batch_size, 1)

    batch_last_mppi = module.last_mppi_result.clone().repeat(batch_size, 1, 1)
    robot1_eps, robot2_eps = mppi_mod.epsilon_generator_colored(
        int(batch_size),
        module.robot1_q_num,
        module.robot2_q_num,
        module.mppi_T,
        module.mean,
        std_value,
        module.gamma,
        module.mppi_dt * module.max_acc_abs_value,
        module.mppi_seed,
    )

    batch_robot1_dq_seq = robot1_eps + batch_last_mppi[:, :, : module.robot1_q_num]
    batch_robot2_dq_seq = robot2_eps + batch_last_mppi[:, :, module.robot1_q_num :]

    module.last_batch_fake_robot1_q = module.batch_fake_robot1_q.clone()
    module.last_batch_fake_robot2_q = module.batch_fake_robot2_q.clone()
    module.first_batch_fake_robot1_q = module.batch_fake_robot1_q.clone()
    module.first_batch_fake_robot2_q = module.batch_fake_robot2_q.clone()

    stage_cost = torch.zeros(batch_size, 1, device=device, dtype=dtype)
    first_step_base = None
    acc_cost = None

    for i in range(module.mppi_T):
        if i == 0:
            rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(
                module.gpu_robot1_dh_mat,
                module.gpu_robot2_dh_mat,
                module.batch_robot1_base,
                module.batch_robot2_base,
                module.batch_robot1_effector,
                module.batch_robot2_effector,
                module.batch_fake_robot1_q,
                module.batch_fake_robot2_q,
                module.batch_line_d,
                module.batch_quat_line_ref,
                module.robot1_q_num,
                module.robot2_q_num,
                module.robot1_dh_type,
                module.robot2_dh_type,
            )
            bacth_rel_jacobian_null = mppi_mod.get_rel_jacobian_null(
                bacth_rel_jacobian, module.robot1_q_num, module.robot2_q_num, batch_size
            )

        batch_robot1_dq, batch_robot2_dq = mppi_mod.get_current_vel(batch_robot1_dq_seq, batch_robot2_dq_seq, i)

        batch_robot1_proj_dq, batch_robot2_proj_dq = mppi_mod.get_proj_qd(
            batch_robot1_dq,
            batch_robot2_dq,
            module.robot1_q_num,
            module.robot2_q_num,
            bacth_rel_jacobian_null,
        )
        last_robot1_proj, last_robot2_proj = mppi_mod.get_proj_qd(
            batch_last_mppi[:, i, : module.robot1_q_num],
            batch_last_mppi[:, i, module.robot1_q_num :],
            module.robot1_q_num,
            module.robot2_q_num,
            bacth_rel_jacobian_null,
        )

        batch_robot1_proj_dq = torch.clamp(batch_robot1_proj_dq, module.gpu_robot1_dq_min, module.gpu_robot1_dq_max)
        batch_robot2_proj_dq = torch.clamp(batch_robot2_proj_dq, module.gpu_robot2_dq_min, module.gpu_robot2_dq_max)

        module.last_batch_fake_robot1_q = module.batch_fake_robot1_q.clone()
        module.last_batch_fake_robot2_q = module.batch_fake_robot2_q.clone()

        module.batch_fake_robot1_q, module.batch_fake_robot2_q, batch_robot1_proj_dq, batch_robot2_proj_dq = (
            mppi_mod.update_joint_position_with_limits(
                module.batch_fake_robot1_q,
                module.batch_fake_robot2_q,
                batch_robot1_proj_dq,
                batch_robot2_proj_dq,
                module.gpu_robot1_q_min,
                module.gpu_robot1_q_max,
                module.gpu_robot2_q_min,
                module.gpu_robot2_q_max,
                module.mppi_dt,
            )
        )

        robot1_eps[:, i, : module.robot1_q_num] = batch_robot1_proj_dq - last_robot1_proj
        robot2_eps[:, i, : module.robot2_q_num] = batch_robot2_proj_dq - last_robot2_proj
        robot1_eps[:, i, : module.robot1_q_num] = torch.clamp(
            robot1_eps[:, i, : module.robot1_q_num],
            module.mppi_dt * module.gpu_robot1_ddq_min,
            module.mppi_dt * module.gpu_robot1_ddq_max,
        )
        robot2_eps[:, i, : module.robot2_q_num] = torch.clamp(
            robot2_eps[:, i, : module.robot2_q_num],
            module.mppi_dt * module.gpu_robot2_ddq_min,
            module.mppi_dt * module.gpu_robot2_ddq_max,
        )

        rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(
            module.gpu_robot1_dh_mat,
            module.gpu_robot2_dh_mat,
            module.batch_robot1_base,
            module.batch_robot2_base,
            module.batch_robot1_effector,
            module.batch_robot2_effector,
            module.batch_fake_robot1_q,
            module.batch_fake_robot2_q,
            module.batch_line_d,
            module.batch_quat_line_ref,
            module.robot1_q_num,
            module.robot2_q_num,
            module.robot1_dh_type,
            module.robot2_dh_type,
        )
        bacth_rel_jacobian_null = mppi_mod.get_rel_jacobian_null(
            bacth_rel_jacobian, module.robot1_q_num, module.robot2_q_num, batch_size
        )

        abs_cost = mppi_mod.get_abs_cost(
            module.batch_desire_abs_pose,
            bacth_abs_pos,
            module.batch_desire_abs_position,
            batch_abs_position,
            module.abs_weight,
            module.abs_position_weight,
        )
        vel_cost = mppi_mod.get_vel_cost(batch_robot1_proj_dq, batch_robot2_proj_dq, module.q_vel_weight)
        tilt_cost = mppi_mod.get_tilt_constraint_cost(batch_angle, module.batch_max_abs_tilt_angle, module.tilt_constraint_weight)

        if i == 0:
            acc_cost = mppi_mod.get_acc_cost(
                batch_robot1_proj_dq,
                batch_robot2_proj_dq,
                batch_last_mppi,
                module.robot1_q_num,
                module.robot2_q_num,
                i,
                module.q_acc_weight,
            )

        collision_cost = module.get_collision_cost(module.collision_constraint_weight)
        increment = abs_cost + vel_cost + collision_cost + tilt_cost
        stage_cost += increment

        if first_step_base is None:
            first_step_base = increment.clone()

    joint_change = (torch.square(module.first_batch_fake_robot1_q - module.batch_fake_robot1_q).sum(dim=1, keepdim=True) +
                    torch.square(module.first_batch_fake_robot2_q - module.batch_fake_robot2_q).sum(dim=1, keepdim=True))
    min_joint_change = 0.001
    joint_change = torch.clamp(joint_change, min=min_joint_change)
    stagnation_cost = module.stagnation_weight * joint_change
    abs_terminal_cost = mppi_mod.get_abs_cost(
        module.batch_desire_abs_pose,
        bacth_abs_pos,
        module.batch_desire_abs_position,
        batch_abs_position,
        module.terminal_abs_weight,
        module.terminal_abs_position_weight,
    )

    ratio_term = abs_terminal_cost / stagnation_cost
    stage_cost_pre_penalty = stage_cost.clone()
    penalty_term = acc_cost + abs_terminal_cost + ratio_term
    stage_cost += penalty_term

    epsilon = mppi_mod.get_all_dq_seq(robot1_eps, robot2_eps)
    traj_flat = epsilon.reshape(batch_size, module.mppi_T * nq)

    return {
        "traj": traj_flat.detach().cpu(),
        "stage_cost": stage_cost.detach().cpu().squeeze(1),
        "stage_cost_base": stage_cost_pre_penalty.detach().cpu().squeeze(1),
        "ratio_term": ratio_term.detach().cpu().squeeze(1),
        "step0_base": first_step_base.detach().cpu().squeeze(1),
        "step0_penalty": (first_step_base + penalty_term).detach().cpu().squeeze(1),
    }


def collect_statistics(module, std_value, rollouts):
    traj_list = []
    cost_list = []
    base_step = []
    pen_step = []
    base_cost = []
    ratio_terms = []

    for _ in range(rollouts):
        sample = simulate_rollout(module, std_value)
        traj_list.append(sample["traj"])
        cost_list.append(sample["stage_cost"])
        base_step.append(sample["step0_base"])
        pen_step.append(sample["step0_penalty"])
        base_cost.append(sample["stage_cost_base"])
        ratio_terms.append(sample["ratio_term"])

    traj_tensor = torch.cat(traj_list, dim=0)
    cost_tensor = torch.cat(cost_list, dim=0)
    base_tensor = torch.cat(base_step, dim=0)
    pen_tensor = torch.cat(pen_step, dim=0)
    base_cost_tensor = torch.cat(base_cost, dim=0)
    ratio_tensor = torch.cat(ratio_terms, dim=0)

    return {
        "traj": traj_tensor,
        "cost": cost_tensor,
        "step0_base": base_tensor,
        "step0_penalty": pen_tensor,
        "stage_cost_base": base_cost_tensor,
        "ratio_term": ratio_tensor,
    }


def gaussian_kl(samples_a: torch.Tensor, samples_b: torch.Tensor, eps: float = 1e-6):
    a = samples_a.double()
    b = samples_b.double()
    mu_a = a.mean(dim=0)
    mu_b = b.mean(dim=0)
    cov_a = torch.cov(a.T) + eps * torch.eye(a.shape[1], dtype=torch.double)
    cov_b = torch.cov(b.T) + eps * torch.eye(b.shape[1], dtype=torch.double)
    cov_b_inv = torch.linalg.inv(cov_b)
    d = a.shape[1]
    term_trace = torch.trace(cov_b_inv @ cov_a)
    diff = (mu_b - mu_a).unsqueeze(0)
    term_quad = diff @ cov_b_inv @ diff.T
    logdet = torch.logdet(cov_b) - torch.logdet(cov_a)
    return 0.5 * (logdet - d + term_trace + term_quad.squeeze())


def histogram_kl(samples_a: torch.Tensor, samples_b: torch.Tensor, bins: int):
    a = samples_a.cpu().numpy()
    b = samples_b.cpu().numpy()
    hist_a, edges = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=edges, density=True)
    hist_a = np.clip(hist_a, 1e-9, None)
    hist_b = np.clip(hist_b, 1e-9, None)
    return float(np.sum(hist_a * np.log(hist_a / hist_b)) * (edges[1] - edges[0]))


def run_eval(args):
    config = ConfigModule(args.config)
    mppi_mod.HighROSModule = HighROSModule

    desire_abs_pose = [-0.009809, -0.700866, -0.008828, 0.713171, 0.03289, -0.000662, -0.283115, -0.003703]
    desire_abs_position = [0.45, 0.0, 0.35]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.054285, -0.000927, -0.262089, -0.003409]
    desire_line_d = [0, 0, 0, 1]
    desire_quat_line_ref = [0, -0.9995, -0.026341, 0.017418]

    module = mppi_mod.MPPIKmeansAdpAnModule(
        config,
        desire_abs_pose,
        desire_abs_position,
        desire_rel_pose,
        desire_line_d,
        desire_quat_line_ref,
    )

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    stats_per_std = {}
    prev_std = None
    prev_data = None

    for std_value in args.variances:
        data = collect_statistics(module, std_value, args.rollouts)
        stats = {
            "traj_mean_norm": float(data["traj"].norm(dim=1).mean().item()),
            "traj_samples": int(data["traj"].shape[0]),
            "cost_mean": float(data["cost"].mean().item()),
            "cost_std": float(data["cost"].std(unbiased=False).item()),
            "step0_base_mean": float(data["step0_base"].mean().item()),
            "step0_penalty_mean": float(data["step0_penalty"].mean().item()),
            "stage_cost_base_var": float(data["stage_cost_base"].var(unbiased=False).item()),
            "stage_cost_penalty_var": float(data["cost"].var(unbiased=False).item()),
            "ratio_term_var": float(data["ratio_term"].var(unbiased=False).item()),
            "kl_cost_base_to_penalty": histogram_kl(data["stage_cost_base"], data["cost"], bins=args.hist_bins),
            "kl_cost_penalty_to_base": histogram_kl(data["cost"], data["stage_cost_base"], bins=args.hist_bins),
        }
        stats_per_std[str(std_value)] = stats

        if prev_data is not None:
            traj_kl = gaussian_kl(prev_data["traj"], data["traj"])
            cost_kl = histogram_kl(prev_data["cost"], data["cost"], bins=args.hist_bins)
            stats_per_std[f"kl_traj_{prev_std}_to_{std_value}"] = float(traj_kl.item())
            stats_per_std[f"kl_cost_{prev_std}_to_{std_value}"] = cost_kl

        prev_std = std_value
        prev_data = data

    out_file = logdir / f"kl_eval_{timestamp}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(stats_per_std, f, indent=2)

    print(f"Saved evaluation summary to {out_file}")


def main():
    args =None
    rclpy.init(args=args)
    args = build_parser().parse_args()
    os.environ['ROS_DOMAIN_ID'] = '16'
    
    run_eval(args)


if __name__ == "__main__":
    main()