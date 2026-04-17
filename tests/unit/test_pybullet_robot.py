from __future__ import annotations

import pytest

from dual_arm_gpu_mpc.common.pybullet_robot import BulletJointResetRobot


class _FakePyBullet:
    JOINT_FIXED = 4

    def __init__(self):
        self.reset_calls = []

    def getNumJoints(self, body_id, physicsClientId):
        assert body_id == 11
        assert physicsClientId == 7
        return 4

    def getJointInfo(self, body_id, joint_index, physicsClientId):
        joint_types = [0, self.JOINT_FIXED, 0, 1]
        return (None, None, joint_types[joint_index])

    def resetJointState(self, body_id, joint_index, joint_position, physicsClientId):
        self.reset_calls.append((body_id, joint_index, joint_position, physicsClientId))


def test_bullet_joint_reset_robot_skips_fixed_joints():
    fake_pyb = _FakePyBullet()
    robot = BulletJointResetRobot(body_id=11, client_id=7, pybullet_module=fake_pyb)

    robot.reset_joint_configuration([0.1, 0.2, 0.3])

    assert fake_pyb.reset_calls == [
        (11, 0, 0.1, 7),
        (11, 2, 0.2, 7),
        (11, 3, 0.3, 7),
    ]


def test_bullet_joint_reset_robot_validates_joint_count():
    fake_pyb = _FakePyBullet()
    robot = BulletJointResetRobot(body_id=11, client_id=7, pybullet_module=fake_pyb)

    with pytest.raises(ValueError, match="Joint position count does not match movable joints"):
        robot.reset_joint_configuration([0.1, 0.2])
