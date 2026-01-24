#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import rclpy

from utils.cem_module import CEMModule
from utils.config_module import ConfigModule


DESIRE_ABS_POSE = [0.00085, 0.923642, -0.383209, -0.005971, - 0.077555, 0.113491, 0.278521, - 0.330388]
DESIRE_ABS_POSITION = [0.40, 0.55, 0.60]
DESIRE_REL_POSE = [9.63267947e-05,  7.07244290e-01, -7.06969239e-01, -3.67320509e-06, 3.03159877e-01,  1.23636280e-01,  1.23726146e-01, -8.79988859e-02]
DESIRE_LINE_D = [0, 0, 0, 1]
DESIRE_QUAT_LINE_REF = [0, -0.011682, 0.003006, -0.999927]


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
    return CEMModule(
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
        module.update_joint_states()
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