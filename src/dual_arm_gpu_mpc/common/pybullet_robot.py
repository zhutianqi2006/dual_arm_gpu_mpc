from __future__ import annotations

from collections.abc import Sequence


def load_pybullet_modules():
    import pybullet as pyb
    import pybullet_data

    return pyb, pybullet_data


class BulletJointResetRobot:
    def __init__(self, body_id: int, client_id: int, pybullet_module):
        self.body_id = body_id
        self.client_id = client_id
        self._pyb = pybullet_module
        self._movable_joint_indices = tuple(
            joint_index
            for joint_index in range(self._pyb.getNumJoints(self.body_id, physicsClientId=self.client_id))
            if self._pyb.getJointInfo(self.body_id, joint_index, physicsClientId=self.client_id)[2]
            != self._pyb.JOINT_FIXED
        )

    def reset_joint_configuration(self, joint_positions: Sequence[float]) -> None:
        if len(joint_positions) != len(self._movable_joint_indices):
            raise ValueError(
                "Joint position count does not match movable joints: "
                f"{len(joint_positions)} != {len(self._movable_joint_indices)}"
            )

        for joint_index, joint_position in zip(self._movable_joint_indices, joint_positions):
            self._pyb.resetJointState(
                self.body_id,
                joint_index,
                float(joint_position),
                physicsClientId=self.client_id,
            )
