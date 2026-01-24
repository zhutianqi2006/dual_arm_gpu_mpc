#!/usr/bin/env python
"""Benchmark manipulability computation using dq_torch Jacobians."""

import argparse
import math
import os
import time
from typing import Tuple

import torch

from dq_torch import rel_abs_pose_rel_jac
from utils.config_module import ConfigModule


DEFAULT_LINE_D = [0.0, 0.0, 0.0, 1.0]
DEFAULT_QUAT_LINE_REF = [0.0, -0.9995, -0.026341, 0.017418]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Dual-arm manipulability benchmark")
	default_config = os.path.join(os.path.dirname(__file__), "ur3_and_ur3e.yaml")
	parser.add_argument("--config", default=default_config, help="Path to the dual-arm config YAML")
	parser.add_argument("--samples", type=int, default=5000, help="Total joint samples to evaluate")
	parser.add_argument("--batch", type=int, default=1000, help="Batch size per Jacobian evaluation")
	parser.add_argument("--device", default=None, help="Torch device string, defaults to cuda if available")
	parser.add_argument("--seed", type=int, default=0, help="PRNG seed for joint sampling")
	parser.add_argument("--line-d", type=float, nargs=4, default=DEFAULT_LINE_D, metavar=("w", "x", "y", "z"))
	parser.add_argument("--quat-line", type=float, nargs=4, default=DEFAULT_QUAT_LINE_REF)
	return parser

@torch.no_grad()
def manipulability(rel_jac: torch.Tensor) -> torch.Tensor:
	gram = rel_jac @ rel_jac.transpose(1, 2)
	gram = 0.5 * (gram + gram.transpose(1, 2))  # enforce symmetry
	det = torch.clamp(torch.linalg.det(gram), min=0.0)
	return torch.sqrt(det)


def sample_uniform(low: torch.Tensor, high: torch.Tensor, batch: int, generator: torch.Generator) -> torch.Tensor:
	if batch <= 0:
		raise ValueError("Batch size must be positive")
	u = torch.rand((batch, low.numel()), device=low.device, dtype=low.dtype, generator=generator)
	return low + (high - low) * u


def prepare_robot_tensors(config: ConfigModule, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
	r1_dh = torch.tensor(config.robot1_dh_mat, device=device, dtype=torch.float32).reshape(-1).contiguous()
	r2_dh = torch.tensor(config.robot2_dh_mat, device=device, dtype=torch.float32).reshape(-1).contiguous()
	r1_base = torch.tensor(config.robot1_base, device=device, dtype=dtype)
	r2_base = torch.tensor(config.robot2_base, device=device, dtype=dtype)
	r1_eff = torch.tensor(config.robot1_effector, device=device, dtype=dtype)
	r2_eff = torch.tensor(config.robot2_effector, device=device, dtype=dtype)
	return r1_dh, r2_dh, r1_base, r2_base, r1_eff, r2_eff


def main() -> None:
	args = build_parser().parse_args()

	device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
	device = torch.device(device_str)
	dtype = torch.float32

	config = ConfigModule(args.config)

	r1_dh, r2_dh, r1_base, r2_base, r1_eff, r2_eff = prepare_robot_tensors(config, device, dtype)
	desire_line_d = torch.tensor(args.line_d, device=device, dtype=dtype)
	desire_quat_line = torch.tensor(args.quat_line, device=device, dtype=dtype)

	r1_q_min = torch.tensor(config.robot1_q_min, device=device, dtype=dtype)
	r1_q_max = torch.tensor(config.robot1_q_max, device=device, dtype=dtype)
	r2_q_min = torch.tensor(config.robot2_q_min, device=device, dtype=dtype)
	r2_q_max = torch.tensor(config.robot2_q_max, device=device, dtype=dtype)

	generator = torch.Generator(device=device)
	generator.manual_seed(args.seed)

	total_samples = max(1, args.samples)
	batch_size = max(1, args.batch)
	iterations = math.ceil(total_samples / batch_size)
	processed = 0
	manip_records = []

	if device.type == "cuda":
		torch.cuda.synchronize(device)
	start_time = time.perf_counter()

	for _ in range(iterations):
		current_batch = min(batch_size, total_samples - processed)
		if current_batch <= 0:
			break

		b_r1_q = sample_uniform(r1_q_min, r1_q_max, current_batch, generator)
		b_r2_q = sample_uniform(r2_q_min, r2_q_max, current_batch, generator)

		b_r1_base = r1_base.unsqueeze(0).repeat(current_batch, 1)
		b_r2_base = r2_base.unsqueeze(0).repeat(current_batch, 1)
		b_r1_eff = r1_eff.unsqueeze(0).repeat(current_batch, 1)
		b_r2_eff = r2_eff.unsqueeze(0).repeat(current_batch, 1)
		b_line_d = desire_line_d.unsqueeze(0).repeat(current_batch, 1)
		b_quat_line = desire_quat_line.unsqueeze(0).repeat(current_batch, 1)

		_, _, rel_jac, _, _ = rel_abs_pose_rel_jac(
			r1_dh,
			r2_dh,
			b_r1_base,
			b_r2_base,
			b_r1_eff,
			b_r2_eff,
			b_r1_q,
			b_r2_q,
			b_line_d,
			b_quat_line,
			config.robot1_q_num,
			config.robot2_q_num,
			config.robot1_dh_type,
			config.robot2_dh_type,
		)

		manip_vals = manipulability(rel_jac)
		manip_records.append(manip_vals.mean().item())
		processed += current_batch

	if device.type == "cuda":
		torch.cuda.synchronize(device)
	elapsed = time.perf_counter() - start_time

	avg_manip = sum(manip_records) / len(manip_records) if manip_records else 0.0
	per_sample = elapsed / processed if processed else 0.0

	print(f"Device: {device_str}")
	print(f"Samples processed: {processed}")
	print(f"Total time: {elapsed*1000:.3f} ms")
	print(f"Per-sample time: {per_sample*1e6:.3f} us")
	print(f"Average manipulability: {avg_manip:.6f}")


if __name__ == "__main__":
	main()
