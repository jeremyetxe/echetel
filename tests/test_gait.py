"""Tests for the flat-surface tripod gait."""

import math
import numpy as np
import pytest

from echetel.gait.tripod_gait import TripodGait


@pytest.fixture
def gait():
    cfg = {
        "step_height": 0.035,
        "step_length": 0.080,
        "duty_factor": 0.667,
        "cycle_period": 0.9,
    }
    nominal = [
        np.array([0.20, 0.0, -0.15]),
        np.array([-0.10, 0.17, -0.15]),
        np.array([-0.10, -0.17, -0.15]),
    ]
    return TripodGait(cfg, nominal)


class TestTripodGait:
    def test_returns_three_foot_positions(self, gait):
        targets = gait.step(0.01, np.array([1.0, 0.0, 0.0]), np.zeros(3))
        assert len(targets) == 3
        for t in targets:
            assert t.shape == (3,)

    def test_swing_leg_height_increases(self, gait):
        """During the swing phase the active foot should rise above nominal z."""
        max_z = -999.0
        for _ in range(100):
            targets = gait.step(0.003, np.array([1.0, 0.0, 0.0]), np.zeros(3))
            for t in targets:
                max_z = max(max_z, t[2])
        assert max_z > -0.15  # must have risen above the nominal stance z

    def test_phase_wraps(self, gait):
        """Phase should stay in [0, 1) after many steps."""
        for _ in range(1000):
            gait.step(0.01, np.array([1.0, 0.0, 0.0]), np.zeros(3))
        assert 0.0 <= gait.state.phase < 1.0

    def test_balance_correction_applied(self, gait):
        correction = np.array([0.02, 0.0, 0.0])
        targets_no_corr = gait.step(0.01, np.array([1.0, 0.0, 0.0]), np.zeros(3))
        targets_corr = gait.step(0.01, np.array([1.0, 0.0, 0.0]), correction)
        # Each foot should differ by the correction amount.
        for no_c, c in zip(targets_no_corr, targets_corr):
            assert abs(c[0] - no_c[0] - correction[0]) < 0.01
