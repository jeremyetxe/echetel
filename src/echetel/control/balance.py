"""PD balance controller.

Converts body tilt (roll, pitch) into a Cartesian correction vector that
is added to every foot's target position, effectively shifting the body
to restore balance.
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from ..sensors.imu import Orientation


class BalanceController:
    def __init__(self, cfg: dict) -> None:
        self.kp_roll = cfg["kp_roll"]
        self.kd_roll = cfg["kd_roll"]
        self.kp_pitch = cfg["kp_pitch"]
        self.kd_pitch = cfg["kd_pitch"]
        self.max_correction = math.radians(cfg["max_correction_deg"])
        self._prev_roll = 0.0
        self._prev_pitch = 0.0

    def compute(self, orientation: Orientation) -> NDArray:
        """Return a (3,) body-frame correction vector [dx, dy, dz] in metres."""
        roll_r = math.radians(orientation.roll_deg)
        pitch_r = math.radians(orientation.pitch_deg)
        roll_rate = math.radians(orientation.roll_rate_dps)
        pitch_rate = math.radians(orientation.pitch_rate_dps)

        roll_corr = -(self.kp_roll * roll_r + self.kd_roll * roll_rate)
        pitch_corr = -(self.kp_pitch * pitch_r + self.kd_pitch * pitch_rate)

        roll_corr = max(-self.max_correction, min(self.max_correction, roll_corr))
        pitch_corr = max(-self.max_correction, min(self.max_correction, pitch_corr))

        # Translate angular corrections to foot-position offsets.
        # Approximation valid for small angles: Δz ≈ 0, lateral/fore offset.
        body_height = 0.20  # approximate standing height, metres
        dx = math.tan(pitch_corr) * body_height
        dy = math.tan(roll_corr) * body_height

        self._prev_roll = roll_r
        self._prev_pitch = pitch_r

        return np.array([dx, dy, 0.0])
