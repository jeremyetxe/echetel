"""Inverse kinematics for a single 3-DOF leg.

Analytical closed-form solution followed by a Newton-Raphson refinement step.
All lengths in metres, angles in radians.
"""

from __future__ import annotations
import math
import numpy as np
from numpy.typing import NDArray

from .forward import LegFK


class IKError(RuntimeError):
    """Raised when no valid IK solution exists for the requested foot position."""


class LegIK:
    def __init__(self, coxa: float, femur: float, tibia: float, joint_limits: dict) -> None:
        self.coxa = coxa
        self.femur = femur
        self.tibia = tibia
        self.fk = LegFK(coxa, femur, tibia)
        lim = joint_limits
        self._limits = np.array([
            [math.radians(lim["hip_yaw"][0]),   math.radians(lim["hip_yaw"][1])],
            [math.radians(lim["hip_pitch"][0]),  math.radians(lim["hip_pitch"][1])],
            [math.radians(lim["knee_pitch"][0]), math.radians(lim["knee_pitch"][1])],
        ])

    def solve(self, foot_target: NDArray) -> NDArray:
        """Return joint angles [hip_yaw, hip_pitch, knee_pitch] in radians."""
        angles = self._analytical(foot_target)
        angles = self._refine(angles, foot_target)
        self._check_limits(angles)
        return angles

    # ------------------------------------------------------------------

    def _analytical(self, p: NDArray) -> NDArray:
        x, y, z = p

        # Hip yaw: angle of the foot projected onto the horizontal plane.
        θ0 = math.atan2(y, x)

        # Project into the sagittal plane after removing coxa.
        horizontal = math.hypot(x, y) - self.coxa
        d = math.hypot(horizontal, z)  # femur-tibia reach

        L, l = self.femur, self.tibia
        if d > L + l:
            # Clamp to maximum reach rather than raising immediately; the
            # Newton-Raphson refinement will not improve beyond this, but we
            # let _check_limits decide whether to reject the pose.
            d = L + l - 1e-6

        # Law of cosines for knee angle.
        cos_knee = (L**2 + l**2 - d**2) / (2 * L * l)
        cos_knee = max(-1.0, min(1.0, cos_knee))
        # Convention: knee_pitch is measured as the supplement of the interior angle.
        θ2 = math.pi - math.acos(cos_knee)

        # Hip pitch: angle to reach the foot.
        γ = math.atan2(-z, horizontal)
        cos_α = (L**2 + d**2 - l**2) / (2 * L * d)
        cos_α = max(-1.0, min(1.0, cos_α))
        α = math.acos(cos_α)
        θ1 = γ + α

        return np.array([θ0, θ1, θ2])

    def _refine(self, angles: NDArray, target: NDArray, iterations: int = 5) -> NDArray:
        for _ in range(iterations):
            error = target - self.fk.foot_position(angles)
            if np.linalg.norm(error) < 1e-6:
                break
            J = self.fk.jacobian(angles)
            try:
                delta = np.linalg.lstsq(J, error, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            angles = angles + delta
        return angles

    def _check_limits(self, angles: NDArray) -> None:
        for i, (lo, hi) in enumerate(self._limits):
            if not lo <= angles[i] <= hi:
                raise IKError(
                    f"Joint {i} angle {math.degrees(angles[i]):.1f}° outside "
                    f"limits [{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°]"
                )
