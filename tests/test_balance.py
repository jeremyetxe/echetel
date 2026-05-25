"""Tests for the balance controller."""

import math
import numpy as np
import pytest

from echetel.control.balance import BalanceController
from echetel.sensors.imu import Orientation


@pytest.fixture
def balance():
    cfg = {
        "kp_roll": 12.0,
        "kd_roll": 1.8,
        "kp_pitch": 14.0,
        "kd_pitch": 2.0,
        "max_correction_deg": 8.0,
    }
    return BalanceController(cfg)


class TestBalanceController:
    def test_zero_tilt_zero_correction(self, balance):
        o = Orientation(0, 0, 0, 0, 0, 0)
        corr = balance.compute(o)
        np.testing.assert_allclose(corr, [0.0, 0.0, 0.0], atol=1e-9)

    def test_positive_pitch_moves_forward(self, balance):
        o = Orientation(pitch_deg=5.0, roll_deg=0, yaw_deg=0,
                        pitch_rate_dps=0, roll_rate_dps=0, yaw_rate_dps=0)
        corr = balance.compute(o)
        assert corr[0] < 0  # nose-up → push CoM backward (negative dx)

    def test_positive_roll_moves_laterally(self, balance):
        o = Orientation(pitch_deg=0, roll_deg=5.0, yaw_deg=0,
                        pitch_rate_dps=0, roll_rate_dps=0, yaw_rate_dps=0)
        corr = balance.compute(o)
        assert corr[1] < 0  # roll right → shift left (negative dy)

    def test_correction_is_clamped(self, balance):
        o = Orientation(pitch_deg=90.0, roll_deg=90.0, yaw_deg=0,
                        pitch_rate_dps=0, roll_rate_dps=0, yaw_rate_dps=0)
        corr = balance.compute(o)
        # Correction magnitude should not exceed tan(max_correction_deg) * body_height
        max_disp = math.tan(math.radians(8.0)) * 0.20
        assert abs(corr[0]) <= max_disp + 1e-6
        assert abs(corr[1]) <= max_disp + 1e-6
