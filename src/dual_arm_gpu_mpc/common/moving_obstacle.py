from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PingPongObstacleTrajectory:
    start: list[float]
    end: list[float]
    speed: float
    _distance_along_segment: float = field(init=False, default=0.0)
    _forward: bool = field(init=False, default=True)
    _segment_vector: np.ndarray = field(init=False, repr=False)
    _segment_unit: np.ndarray = field(init=False, repr=False)
    _segment_length: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        start = np.asarray(self.start, dtype=np.float64)
        end = np.asarray(self.end, dtype=np.float64)

        self._segment_vector = end - start
        self._segment_length = float(np.linalg.norm(self._segment_vector))
        if self._segment_length <= 1.0e-9:
            raise ValueError("PingPongObstacleTrajectory start and end cannot be the same")
        if self.speed <= 0.0:
            raise ValueError("PingPongObstacleTrajectory speed must be positive")

        self.start = start.tolist()
        self.end = end.tolist()
        self._segment_unit = self._segment_vector / self._segment_length

    @property
    def position(self) -> list[float]:
        start = np.asarray(self.start, dtype=np.float64)
        current = start + self._segment_unit * self._distance_along_segment
        return current.tolist()

    def step(self, dt: float) -> list[float]:
        remaining = max(0.0, float(self.speed) * max(0.0, float(dt)))
        if remaining == 0.0:
            return self.position

        while remaining > 0.0:
            if self._forward:
                distance_to_boundary = self._segment_length - self._distance_along_segment
                direction = 1.0
            else:
                distance_to_boundary = self._distance_along_segment
                direction = -1.0

            travel = min(remaining, distance_to_boundary)
            self._distance_along_segment += direction * travel
            remaining -= travel

            if remaining > 0.0:
                self._forward = not self._forward

        return self.position
