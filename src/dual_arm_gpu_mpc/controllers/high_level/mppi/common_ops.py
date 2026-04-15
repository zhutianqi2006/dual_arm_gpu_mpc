import numpy as np
import torch


def moving_average_filter_tensor(xx: torch.Tensor, window_size: int) -> torch.Tensor:
    xx_np = xx.cpu().numpy()
    b = np.ones(window_size) / window_size
    num_steps, num_controls = xx_np.shape
    xx_mean_np = np.zeros_like(xx_np)

    for d in range(num_controls):
        xx_mean_np[:, d] = np.convolve(xx_np[:, d], b, mode="same")

    return torch.from_numpy(xx_mean_np).to(xx.device)


@torch.jit.script
def update_joint_position_with_limits(
    batch_robot1_q: torch.Tensor,
    batch_robot2_q: torch.Tensor,
    batch_robot1_dq: torch.Tensor,
    batch_robot2_dq: torch.Tensor,
    robot1_joint_min: torch.Tensor,
    robot1_joint_max: torch.Tensor,
    robot2_joint_min: torch.Tensor,
    robot2_joint_max: torch.Tensor,
    dt: float,
):
    updated_robot1_q = batch_robot1_q + batch_robot1_dq * dt
    updated_robot2_q = batch_robot2_q + batch_robot2_dq * dt

    clamped_robot1_q = torch.clamp(updated_robot1_q, robot1_joint_min, robot1_joint_max)
    clamped_robot2_q = torch.clamp(updated_robot2_q, robot2_joint_min, robot2_joint_max)

    allowed_robot1_dq = (clamped_robot1_q - batch_robot1_q) / dt
    allowed_robot2_dq = (clamped_robot2_q - batch_robot2_q) / dt

    return clamped_robot1_q, clamped_robot2_q, allowed_robot1_dq, allowed_robot2_dq


@torch.jit.script
def update_fake_joint_pos(
    batch_fake_robot1_q: torch.Tensor,
    batch_fake_robot2_q: torch.Tensor,
    batch_robot1_qd: torch.Tensor,
    batch_robot2_qd: torch.Tensor,
    dt: float,
):
    last_batch_fake_robot1_q = batch_fake_robot1_q.clone()
    last_batch_fake_robot2_q = batch_fake_robot2_q.clone()
    batch_fake_robot1_q = dt * batch_robot1_qd + batch_fake_robot1_q
    batch_fake_robot2_q = dt * batch_robot2_qd + batch_fake_robot2_q
    return last_batch_fake_robot1_q, last_batch_fake_robot2_q, batch_fake_robot1_q, batch_fake_robot2_q


@torch.jit.script
def compute_weights(epsilon: torch.Tensor, S: torch.Tensor, batch_size: int, p_lamada: float) -> torch.Tensor:
    w = torch.zeros(batch_size, device=S.device, dtype=S.dtype)
    rho = S.min()
    eta = torch.sum(torch.exp((-1.0 / p_lamada) * (S - rho)))
    while eta < 22 or eta > 30:
        if eta > 30:
            p_lamada *= 0.8
        else:
            p_lamada *= 1.2
        eta = torch.sum(torch.exp((-1.0 / p_lamada) * (S - rho)))
    w = (1.0 / eta) * torch.exp((-1.0 / p_lamada) * (S - rho))
    w_epsilon = torch.sum(w.view(batch_size, 1, 1) * epsilon, dim=0)
    return w_epsilon


@torch.jit.script
def compute_weights_k(epsilon: torch.Tensor, S: torch.Tensor, batch_size: int, p_lamada: float) -> torch.Tensor:
    w = torch.zeros(batch_size, device=S.device, dtype=S.dtype)
    rho = S.min()
    eta = torch.sum(torch.exp((-1.0 / p_lamada) * (S - rho)))
    while eta < 15 or eta > 20:
        if eta > 20:
            p_lamada *= 0.8
        else:
            p_lamada *= 1.2
        eta = torch.sum(torch.exp((-1.0 / p_lamada) * (S - rho)))
    w = (1.0 / eta) * torch.exp((-1.0 / p_lamada) * (S - rho))
    w_epsilon = torch.sum(w.view(batch_size, 1, 1) * epsilon, dim=0)
    return w_epsilon


@torch.jit.script
def get_proj_qd(
    batch_robot1_qd: torch.Tensor,
    batch_robot2_qd: torch.Tensor,
    robot1_q_num: int,
    robot2_q_num: int,
    batch_null_space_mat: torch.Tensor,
):
    batch_two_robot_qd = torch.cat((batch_robot1_qd.unsqueeze(2), batch_robot2_qd.unsqueeze(2)), dim=1)
    batch_null_space_mat_first = batch_null_space_mat[:, :robot1_q_num, :]
    batch_null_space_mat_last = batch_null_space_mat[:, robot1_q_num:, :]
    batch_robot1_proj_qd = torch.matmul(batch_null_space_mat_first, batch_two_robot_qd).squeeze(2)
    batch_robot2_proj_qd = torch.matmul(batch_null_space_mat_last, batch_two_robot_qd).squeeze(2)
    return batch_robot1_proj_qd, batch_robot2_proj_qd


@torch.jit.script
def get_current_vel(robot1_dq_seq: torch.Tensor, robot2_dq_seq: torch.Tensor, i: int):
    robot1_dq = robot1_dq_seq[:, i, :]
    robot2_dq = robot2_dq_seq[:, i, :]
    return robot1_dq, robot2_dq


@torch.jit.script
def get_all_q(batch_robot1_q: torch.Tensor, batch_robot2_q: torch.Tensor):
    return torch.cat((batch_robot1_q, batch_robot2_q), dim=1)


@torch.jit.script
def get_all_dq_seq(batch_robot1_dq_rollout_seq: torch.Tensor, batch_robot2_dq_rollout_seq: torch.Tensor):
    return torch.cat((batch_robot1_dq_rollout_seq, batch_robot2_dq_rollout_seq), dim=2)


@torch.jit.script
def get_abs_cost(
    desire_abs_pos: torch.Tensor,
    abs_pos: torch.Tensor,
    desire_abs_position: torch.Tensor,
    abs_position: torch.Tensor,
    rot_weight: float,
    position_weight: float,
):
    quat_difference = rot_weight * torch.abs(desire_abs_pos - abs_pos)
    position_difference = position_weight * torch.abs(desire_abs_position - abs_position)
    result = quat_difference.sum(dim=1, keepdim=True) + position_difference.sum(dim=1, keepdim=True)
    return result


@torch.jit.script
def get_vel_cost(batch_robot1_dq: torch.Tensor, batch_robot2_dq: torch.Tensor, weight: float):
    difference = torch.abs(batch_robot1_dq) + torch.abs(batch_robot2_dq)
    result = weight * difference.sum(dim=1, keepdim=True)
    return result


@torch.jit.script
def get_vel_constraint_cost(
    batch_robot1_dq: torch.Tensor,
    batch_robot2_dq: torch.Tensor,
    batch_robot1_dq_min: torch.Tensor,
    batch_robot1_dq_max: torch.Tensor,
    batch_robot2_dq_min: torch.Tensor,
    batch_robot2_dq_max: torch.Tensor,
    weight: float,
):
    robot1_violation_min = (batch_robot1_dq < batch_robot1_dq_min).float()
    robot1_violation_max = (batch_robot1_dq > batch_robot1_dq_max).float()
    robot2_violation_min = (batch_robot2_dq < batch_robot2_dq_min).float()
    robot2_violation_max = (batch_robot2_dq > batch_robot2_dq_max).float()
    total_violations = robot1_violation_min + robot1_violation_max + robot2_violation_min + robot2_violation_max
    total_cost = weight * total_violations.sum(dim=1, keepdim=True)
    return total_cost


@torch.jit.script
def get_pos_constraint_cost(
    batch_robot1_q: torch.Tensor,
    batch_robot2_q: torch.Tensor,
    batch_robot1_q_min: torch.Tensor,
    batch_robot1_q_max: torch.Tensor,
    batch_robot2_q_min: torch.Tensor,
    batch_robot2_q_max: torch.Tensor,
    weight: float,
):
    robot1_violation_min = (batch_robot1_q < batch_robot1_q_min).float()
    robot1_violation_max = (batch_robot1_q > batch_robot1_q_max).float()
    robot2_violation_min = (batch_robot2_q < batch_robot2_q_min).float()
    robot2_violation_max = (batch_robot2_q > batch_robot2_q_max).float()
    total_violations = robot1_violation_min + robot1_violation_max + robot2_violation_min + robot2_violation_max
    total_cost = weight * total_violations.sum(dim=1, keepdim=True)
    return total_cost


@torch.jit.script
def get_tilt_constraint_cost(angle: torch.Tensor, batch_max_abs_tilt_angle: torch.Tensor, weight: float):
    robot1_violation_min = (angle < -batch_max_abs_tilt_angle).float()
    robot1_violation_max = (angle > batch_max_abs_tilt_angle).float()
    total_violations = robot1_violation_min + robot1_violation_max
    total_cost = weight * total_violations.sum(dim=1, keepdim=True)
    return total_cost


@torch.jit.script
def get_acc_cost(
    batch_robot1_dq: torch.Tensor,
    batch_robot2_dq: torch.Tensor,
    batch_last_mppi_result: torch.Tensor,
    robot1_q_num: int,
    robot2_q_num: int,
    i: int,
    weight: float,
):
    last_robot1_dq = batch_last_mppi_result[:, i, :robot1_q_num]
    last_robot2_dq = batch_last_mppi_result[:, i, robot1_q_num:]
    difference = torch.abs(batch_robot1_dq - last_robot1_dq) + torch.abs(batch_robot2_dq - last_robot2_dq)
    result = weight * difference.sum(dim=1, keepdim=True)
    return result


@torch.jit.script
def get_rel_jacobian_null(jac_batch: torch.Tensor, robot1_q_num: int, robot2_q_num: int, batch_size: int):
    batch_i = torch.eye(robot1_q_num + robot2_q_num, dtype=torch.float64, device="cuda:0").repeat(batch_size, 1, 1)
    batch_eps = 1e-16 * torch.eye(8, dtype=torch.float64, device="cuda:0").repeat(batch_size, 1, 1)
    jac_batch = jac_batch.to(torch.float64)
    jac_batch_t = jac_batch.transpose(-2, -1)
    J_J_t = torch.matmul(jac_batch, jac_batch_t)
    J_pinv = jac_batch_t @ torch.inverse(J_J_t + batch_eps)
    J_pinv_J = torch.matmul(J_pinv, jac_batch)
    rel_jacobian_null = batch_i - J_pinv_J
    return rel_jacobian_null.to(torch.float32)


@torch.jit.script
def epsilon_generator(
    batch_size: int,
    robot1_q_num: int,
    robot2_q_num: int,
    mppi_T: int,
    mean: float,
    std: float,
    max_abs_value: float,
    mppi_seed: int,
):
    torch.manual_seed(mppi_seed)
    total_q_num = robot1_q_num + robot2_q_num
    data = torch.randn(batch_size, mppi_T, total_q_num, dtype=torch.float32, device="cuda:0")
    data = data * std + mean
    data = torch.clamp(data, min=-max_abs_value, max=max_abs_value)
    part1, part2 = data.split([robot1_q_num, robot2_q_num], dim=2)
    return part1, part2


@torch.jit.script
def epsilon_generator_log(
    batch_size: int,
    robot1_q_num: int,
    robot2_q_num: int,
    mppi_T: int,
    mean: float,
    std: float,
    log_std: float,
    max_abs_value: float,
    mppi_seed: int,
):
    total_q_num = robot1_q_num + robot2_q_num
    torch.manual_seed(mppi_seed)
    data = torch.randn(batch_size, mppi_T, total_q_num, dtype=torch.float32, device="cuda:0")
    torch.manual_seed(mppi_seed)
    log_normal_data = torch.exp(
        torch.randn(batch_size, mppi_T, total_q_num, dtype=torch.float32, device="cuda:0") * log_std
    )
    combined_data = data * log_normal_data
    combined_data = combined_data * std + mean
    combined_data = torch.clamp(combined_data, min=-max_abs_value, max=max_abs_value)
    part1, part2 = combined_data.split([robot1_q_num, robot2_q_num], dim=2)
    return part1, part2


@torch.jit.script
def epsilon_generator_colored(
    batch_size: int,
    robot1_q_num: int,
    robot2_q_num: int,
    mppi_T: int,
    mean: float,
    std: float,
    gamma: float,
    max_abs_value: float,
    mppi_seed: int,
):
    total_q_num = robot1_q_num + robot2_q_num
    device = "cuda:0"
    T = mppi_T
    N = T // 2 + 1

    f_min = 1.0 / N
    n_range = torch.arange(1, N - 1, device=device, dtype=torch.float32)
    sum_term = 4 * torch.sum(torch.pow(n_range, -gamma))
    zeta = (T**-2) * (N**gamma) * (1 + sum_term)

    torch.manual_seed(mppi_seed)
    real_part = torch.randn(batch_size, total_q_num, N, device=device)
    torch.manual_seed(mppi_seed)
    imag_part = (
        torch.randn(batch_size, total_q_num, N, device=device)
        if T % 2 != 0
        else torch.zeros(batch_size, total_q_num, N, device=device)
    )

    freq_indices = torch.arange(N, device=device).float()
    scaled_freq = torch.clamp_min(freq_indices / N, f_min)
    variance = (scaled_freq ** (-gamma)) * (std**2) / zeta
    real_part *= torch.sqrt(variance)
    imag_part *= torch.sqrt(variance) if T % 2 != 0 else torch.zeros_like(imag_part)

    full_spectrum = torch.zeros(batch_size, total_q_num, T, dtype=torch.complex64, device=device)
    full_spectrum[..., :N] = torch.complex(real_part, imag_part)
    if T % 2 == 0:
        full_spectrum[..., N:] = torch.conj(torch.flip(full_spectrum[..., 1 : N - 1], dims=[-1]))
    else:
        full_spectrum[..., N:] = torch.conj(torch.flip(full_spectrum[..., 1:N], dims=[-1]))

    time_domain = torch.fft.ifft(full_spectrum, n=T, norm="forward").real * T
    noise = time_domain.permute(0, 2, 1)
    noise = noise * std + mean
    noise = torch.clamp(noise, -max_abs_value, max_abs_value)

    part1, part2 = noise.split([robot1_q_num, robot2_q_num], dim=2)
    return part1, part2
