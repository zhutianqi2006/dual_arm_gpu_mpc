import pytest
import torch

from dual_arm_gpu_mpc.controllers.high_level.mppi.common_ops import (
    compute_weights,
    compute_weights_k,
    get_abs_cost,
    get_rel_jacobian_null,
    update_joint_position_with_limits,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_update_joint_position_with_limits_clamps_positions_and_velocity():
    batch_robot1_q = torch.tensor([[0.0, 0.5]], device="cuda:0")
    batch_robot2_q = torch.tensor([[0.0, -0.5]], device="cuda:0")
    batch_robot1_dq = torch.tensor([[2.0, -4.0]], device="cuda:0")
    batch_robot2_dq = torch.tensor([[3.0, -3.0]], device="cuda:0")
    robot1_joint_min = torch.tensor([-0.2, -0.2], device="cuda:0")
    robot1_joint_max = torch.tensor([0.1, 0.6], device="cuda:0")
    robot2_joint_min = torch.tensor([-0.1, -0.6], device="cuda:0")
    robot2_joint_max = torch.tensor([0.2, -0.2], device="cuda:0")

    q1, q2, dq1, dq2 = update_joint_position_with_limits(
        batch_robot1_q,
        batch_robot2_q,
        batch_robot1_dq,
        batch_robot2_dq,
        robot1_joint_min,
        robot1_joint_max,
        robot2_joint_min,
        robot2_joint_max,
        0.1,
    )

    torch.testing.assert_close(q1, torch.tensor([[0.1, 0.1]], device="cuda:0"))
    torch.testing.assert_close(q2, torch.tensor([[0.2, -0.6]], device="cuda:0"))
    torch.testing.assert_close(dq1, torch.tensor([[1.0, -4.0]], device="cuda:0"))
    torch.testing.assert_close(dq2, torch.tensor([[2.0, -1.0]], device="cuda:0"))


def test_compute_weights_returns_finite_weighted_sequence():
    batch_size = 32
    epsilon = torch.linspace(
        0.0,
        1.0,
        steps=batch_size * 2 * 2,
        device="cuda:0",
    ).view(batch_size, 2, 2)
    stage_cost = torch.linspace(1.0, 3.0, steps=batch_size, device="cuda:0").view(batch_size, 1)

    weighted = compute_weights(epsilon, stage_cost, batch_size, 0.5)

    assert weighted.shape == (2, 2)
    assert torch.isfinite(weighted).all()
    assert (weighted >= 0.0).all()
    assert (weighted <= 1.0).all()


def test_compute_weights_k_returns_finite_weighted_sequence():
    batch_size = 20
    epsilon = torch.linspace(
        0.1,
        0.9,
        steps=batch_size * 2 * 2,
        device="cuda:0",
    ).view(batch_size, 2, 2)
    stage_cost = torch.linspace(0.3, 1.8, steps=batch_size, device="cuda:0").view(batch_size, 1)

    weighted = compute_weights_k(epsilon, stage_cost, batch_size, 0.4)

    assert weighted.shape == (2, 2)
    assert torch.isfinite(weighted).all()


def test_get_abs_cost_sums_pose_and_position_error_per_batch():
    desire_abs_pos = torch.zeros((2, 8), device="cuda:0")
    abs_pos = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        device="cuda:0",
    )
    desire_abs_position = torch.zeros((2, 4), device="cuda:0")
    abs_position = torch.tensor(
        [
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 2.0, 0.0],
        ],
        device="cuda:0",
    )

    result = get_abs_cost(
        desire_abs_pos,
        abs_pos,
        desire_abs_position,
        abs_position,
        2.0,
        0.5,
    )

    expected = torch.tensor([[5.0 + 1.0], [4.0 + 1.0]], device="cuda:0")
    torch.testing.assert_close(result, expected)


def test_get_rel_jacobian_null_returns_float32_batch_square_matrix():
    jacobian = torch.zeros((2, 8, 4), device="cuda:0", dtype=torch.float32)
    jacobian[0, 0, 0] = 1.0
    jacobian[0, 1, 1] = 1.0
    jacobian[1, 2, 2] = 1.0
    jacobian[1, 3, 3] = 1.0

    null = get_rel_jacobian_null(jacobian, 2, 2, 2)

    assert null.shape == (2, 4, 4)
    assert null.dtype == torch.float32
    assert torch.isfinite(null).all()
    torch.testing.assert_close(null[0], null[0].transpose(0, 1), atol=1e-5, rtol=1e-5)
