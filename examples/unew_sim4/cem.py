#!/usr/bin/env python
import argparse
import json
import os
import time
import math
from datetime import datetime
from pathlib import Path

import rclpy
import torch

from utils.cem_module import CEMModule
from utils.config_module import ConfigModule


DESIRE_ABS_POSE = [0.00085, 0.923642, -0.383209, -0.005971, 0.187191, 0.157905, 0.379813, 0.076992]
DESIRE_ABS_POSITION = [-0.4, 0.0, 0.825]
DESIRE_REL_POSE = [9.63267947e-05,  7.07244290e-01, -7.06969239e-01, -3.67320509e-06, 3.03159877e-01,  1.23636280e-01,  1.23726146e-01, -8.79988859e-02]
DESIRE_LINE_D = [0, 0, 0, 1]
DESIRE_QUAT_LINE_REF = [0, -0.011682, 0.003006, -0.999927]


class CEMModuleDynamic(CEMModule):
    """CEM controller with dynamic obstacle updates (match unew_sim4 MPPI dynamic setup)."""

    def _init_collision_model(self) -> None:
        super()._init_collision_model()

        # obstacle 1
        self.init_obstacle_x = self.curobo_config.world_model.world_model.cuboid[0].pose[0]
        self.init_obstacle_x_dim = self.curobo_config.world_model.world_model.cuboid[0].dims[0]
        self.current_obstacle_x = self.init_obstacle_x
        self.last_obstacle_x = self.init_obstacle_x
        self.fake_obstacle_x = self.init_obstacle_x

        # obstacle 2
        self.init_obstacle_y = self.curobo_config.world_model.world_model.cuboid[1].pose[1]
        self.init_obstacle_y_dim = self.curobo_config.world_model.world_model.cuboid[1].dims[1]
        self.current_obstacle_y = self.init_obstacle_y
        self.last_obstacle_y = self.init_obstacle_y
        self.fake_obstacle_y = self.init_obstacle_y

    def update_curobo_world_model(self, time_elapsed: float) -> None:
        # obstacle 1
        self.last_obstacle_x = self.current_obstacle_x
        self.curobo_config.world_model.world_model.cuboid[0].dims[0] = self.init_obstacle_x_dim
        self.curobo_config.world_model.world_model.cuboid[0].pose[0] = self.init_obstacle_x + math.sin(0.15 * time_elapsed)

        # obstacle 2
        self.last_obstacle_y = self.current_obstacle_y
        self.curobo_config.world_model.world_model.cuboid[1].dims[1] = self.init_obstacle_y_dim
        self.curobo_config.world_model.world_model.cuboid[1].pose[1] = self.init_obstacle_y + math.cos(0.3 * time_elapsed)

        self.ros_module.write_obstacle(
            self.curobo_config.world_model.world_model.cuboid[0].pose[0:3]
            + self.curobo_config.world_model.world_model.cuboid[1].pose[0:3],
            2,
        )
        self.curobo_fn.update_world(self.curobo_config.world_model.world_model)
        self.curobo_fn2.update_world(self.curobo_config.world_model.world_model)

        self.current_obstacle_x = self.curobo_config.world_model.world_model.cuboid[0].pose[0]
        self.fake_obstacle_x = self.current_obstacle_x
        self.current_obstacle_y = self.curobo_config.world_model.world_model.cuboid[1].pose[1]
        self.fake_obstacle_y = self.current_obstacle_y

    def update_obstacle_velocity_estimate(self) -> None:
        # Match MPPI unew_sim4 velocity estimation.
        self.current_obstacle_x_velocity = (self.current_obstacle_x - self.last_obstacle_x) / 0.1
        self.current_obstacle_y_velocity = (self.current_obstacle_y - self.last_obstacle_y) / 0.1

    def update_fake_curobo_world_model(self, vel1: float, vel2: float) -> None:
        # Match MPPI unew_sim4 fake obstacle update (forward shift based on estimated velocity).
        _fake_obstacle_x_dim = 1.4 * self.T * abs(vel1) * self.dt
        self.fake_obstacle_x += 0.5 * vel1 * self.dt * self.T
        self.curobo_config.world_model.world_model.cuboid[0].pose[0] = self.fake_obstacle_x

        _fake_obstacle_y_dim = 1.4 * self.T * abs(vel2) * self.dt
        self.fake_obstacle_y += 0.5 * vel2 * self.dt * self.T
        self.curobo_config.world_model.world_model.cuboid[1].pose[1] = self.fake_obstacle_y

        self.curobo_fn.update_world(self.curobo_config.world_model.world_model)
        self.curobo_fn2.update_world(self.curobo_config.world_model.world_model)

    def play_once(self) -> None:
        self.update_curobo_world_model(time.time() - self.start_time)
        self.update_joint_states()
        self.update_obstacle_velocity_estimate()
        self.update_fake_curobo_world_model(self.current_obstacle_x_velocity, self.current_obstacle_y_velocity)

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
            self.action_mean.zero_()
            self.last_action_mean.zero_()
            self.current_plan.zero_()
            self.current_mppi_result = torch.zeros_like(self.action_mean)
            self.action_std.fill_(float(self.init_std))
        else:
            u0 = u0_cem

        self.ros_module.write_high_u(u0.tolist())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or evaluate the CEM dual-arm controller")
    default_config = os.path.join(os.path.dirname(__file__), "two_franka.yaml")
    default_logdir = os.path.join(os.path.dirname(__file__), "../../logs/cem_variance")
    parser.add_argument("--mode", choices=["run", "variance"], default="variance", help="Run ROS loop or log variance evolution")
    parser.add_argument("--config", default=default_config, help="Path to controller config file")
    parser.add_argument("--ros-domain", type=int, default=16, help="ROS_DOMAIN_ID to export before init")
    parser.add_argument("--logdir", default=default_logdir, help="Directory for variance logs (variance mode only)")
    parser.add_argument("--runs", type=int, default=1, help="Number of variance traces to record")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip the initial warm-up passes in variance mode")
    return parser


def create_module(config_path: str) -> CEMModule:
    config = ConfigModule(config_path)
    return CEMModuleDynamic(
        config,
        DESIRE_ABS_POSE,
        DESIRE_ABS_POSITION,
        DESIRE_REL_POSE,
        DESIRE_LINE_D,
        DESIRE_QUAT_LINE_REF,
    )


def run_closed_loop(module: CEMModule) -> None:
    module.warm_up()
    while True:
        module.play_once()


def run_variance_mode(module: CEMModule, args: argparse.Namespace) -> None:
    if not args.skip_warmup:
        module.warm_up()

    logdir = Path(args.logdir).expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    records = []
    for run_idx in range(args.runs):
        if hasattr(module, "update_curobo_world_model"):
            module.update_curobo_world_model(time.time() - module.start_time)
        module.update_joint_states()
        if hasattr(module, "update_obstacle_velocity_estimate"):
            module.update_obstacle_velocity_estimate()
        if hasattr(module, "update_fake_curobo_world_model"):
            module.update_fake_curobo_world_model(
                getattr(module, "current_obstacle_x_velocity", 0.0),
                getattr(module, "current_obstacle_y_velocity", 0.0),
            )
        _, energy, history = module.cem_worker(capture_variance=True)
        records.append(
            {
                "run": run_idx,
                "min_energy": float(energy.detach().cpu().item()),
                "iteration_history": history,
            }
        )

    payload = {
        "timestamp": timestamp,
        "init_std": float(module.init_std),
        "cem_iters": int(module.cem_iters),
        "batch_size": int(module.batch_size),
        "horizon": int(module.T),
        "records": records,
    }

    out_file = logdir / f"cem_variance_{timestamp}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved variance trace to {out_file}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain)
    rclpy.init(args=None)
    module = create_module(args.config)
    # run_variance_mode(module, args)
    run_closed_loop(module)

if __name__ == "__main__":
    main()