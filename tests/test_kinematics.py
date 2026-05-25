"""Unit tests for forward and inverse kinematics."""

import math
import numpy as np
import pytest

from echetel.kinematics.forward import LegFK
from echetel.kinematics.inverse import LegIK, IKError

COXA = 0.055
FEMUR = 0.110
TIBIA = 0.130
LIMITS = {
    "hip_yaw": [-45.0, 45.0],
    "hip_pitch": [-30.0, 70.0],
    "knee_pitch": [10.0, 150.0],
}


@pytest.fixture
def fk():
    return LegFK(COXA, FEMUR, TIBIA)


@pytest.fixture
def ik():
    return LegIK(COXA, FEMUR, TIBIA, LIMITS)


class TestForwardKinematics:
    def test_zero_angles_places_foot_along_x(self, fk):
        """All joints at zero should put the foot along the +x axis."""
        foot = fk.foot_position(np.zeros(3))
        assert foot[1] == pytest.approx(0.0, abs=1e-6)
        assert foot[2] == pytest.approx(0.0, abs=1e-6)
        assert foot[0] == pytest.approx(COXA + FEMUR + TIBIA, abs=1e-6)

    def test_hip_yaw_rotates_horizontally(self, fk):
        angles = np.array([math.radians(90), 0.0, 0.0])
        foot = fk.foot_position(angles)
        # Foot should now be along +y, not +x
        assert foot[0] == pytest.approx(0.0, abs=1e-4)
        assert foot[1] > 0

    def test_hip_pitch_lowers_foot(self, fk):
        angles = np.array([0.0, math.radians(45), 0.0])
        foot = fk.foot_position(angles)
        assert foot[2] < 0  # foot should be below the hip

    def test_jacobian_shape(self, fk):
        J = fk.jacobian(np.zeros(3))
        assert J.shape == (3, 3)

    def test_jacobian_numerical_consistency(self, fk):
        angles = np.array([0.1, 0.3, 0.8])
        J = fk.jacobian(angles)
        # Verify by finite-difference on first column
        eps = 1e-7
        delta = np.array([eps, 0.0, 0.0])
        fd = (fk.foot_position(angles + delta) - fk.foot_position(angles - delta)) / (2 * eps)
        np.testing.assert_allclose(J[:, 0], fd, atol=1e-5)


class TestInverseKinematics:
    def test_roundtrip(self, fk, ik):
        """IK followed by FK should recover the original foot position."""
        test_angles = np.array([math.radians(10), math.radians(30), math.radians(80)])
        target = fk.foot_position(test_angles)
        recovered = ik.solve(target)
        actual = fk.foot_position(recovered)
        np.testing.assert_allclose(actual, target, atol=1e-4)

    def test_reachable_positions(self, fk, ik):
        for yaw_deg in [-20, 0, 20]:
            for pitch_deg in [10, 30, 50]:
                angles_in = np.array([math.radians(yaw_deg), math.radians(pitch_deg), math.radians(60)])
                target = fk.foot_position(angles_in)
                recovered = ik.solve(target)
                actual = fk.foot_position(recovered)
                np.testing.assert_allclose(actual, target, atol=1e-3,
                                           err_msg=f"Failed for yaw={yaw_deg}° pitch={pitch_deg}°")

    def test_unreachable_raises(self, ik):
        far_target = np.array([10.0, 0.0, 0.0])  # well beyond arm reach
        with pytest.raises(IKError):
            ik.solve(far_target)
