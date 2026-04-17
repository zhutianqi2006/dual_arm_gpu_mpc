from __future__ import annotations

import pytest

from dual_arm_gpu_mpc.common.moving_obstacle import PingPongObstacleTrajectory


def test_ping_pong_obstacle_moves_between_endpoints_and_bounces():
    trajectory = PingPongObstacleTrajectory(
        start=[0.0, 0.0, 0.0],
        end=[1.0, 0.0, 0.0],
        speed=0.5,
    )

    first = trajectory.step(1.0)
    second = trajectory.step(1.0)
    third = trajectory.step(1.0)

    assert pytest.approx(first) == [0.5, 0.0, 0.0]
    assert pytest.approx(second) == [1.0, 0.0, 0.0]
    assert pytest.approx(third) == [0.5, 0.0, 0.0]


def test_ping_pong_obstacle_rejects_degenerate_paths():
    with pytest.raises(ValueError, match="start and end cannot be the same"):
        PingPongObstacleTrajectory(start=[0.0, 0.0, 0.0], end=[0.0, 0.0, 0.0], speed=0.1)
