"""Forward kinematics for a single 3-DOF leg.

Joint order: hip_yaw (θ₀), hip_pitch (θ₁), knee_pitch (θ₂).
All angles in radians. Outputs foot position in the leg-root frame.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def rotation_x(a: float) -> NDArray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rotation_y(a: float) -> NDArray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotation_z(a: float) -> NDArray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


class LegFK:
    """Forward kinematics for one leg."""

    def __init__(self, coxa: float, femur: float, tibia: float) -> None:
        self.coxa = coxa
        self.femur = femur
        self.tibia = tibia

    def foot_position(self, angles: NDArray) -> NDArray:
        """Return foot (x, y, z) in the leg-root frame.

        angles: [hip_yaw_rad, hip_pitch_rad, knee_pitch_rad]
        """
        θ0, θ1, θ2 = angles

        # Hip yaw rotates the whole leg around the vertical axis.
        R_yaw = rotation_z(θ0)

        # After yaw, extend coxa along local x.
        p_knee_root = R_yaw @ np.array([self.coxa, 0.0, 0.0])

        # Hip pitch tilts femur down from the coxa tip.
        R_pitch = R_yaw @ rotation_y(θ1)
        p_ankle_root = p_knee_root + R_pitch @ np.array([self.femur, 0.0, 0.0])

        # Knee pitch continues from the ankle.
        R_knee = R_pitch @ rotation_y(θ2)
        p_foot_root = p_ankle_root + R_knee @ np.array([self.tibia, 0.0, 0.0])

        return p_foot_root

    def jacobian(self, angles: NDArray) -> NDArray:
        """3×3 geometric Jacobian (numerical, central differences)."""
        eps = 1e-5
        J = np.zeros((3, 3))
        for i in range(3):
            da = np.zeros(3)
            da[i] = eps
            J[:, i] = (self.foot_position(angles + da) - self.foot_position(angles - da)) / (2 * eps)
        return J
