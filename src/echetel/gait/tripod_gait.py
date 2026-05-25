"""Tripod gait for flat-surface walking.

With three legs at 120° spacing the robot is always statically stable: at
any instant two legs are in stance and one is in swing, cycling through
legs 0 → 1 → 2 → 0.  The duty factor (fraction of cycle in stance) is
configurable; the default 2/3 gives exactly one leg airborne at all times.
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field


@dataclass
class TripodGaitState:
    phase: float = 0.0              # 0–1, advances each call to step()
    active_swing_leg: int = 0       # which leg is currently swinging


class TripodGait:
    def __init__(self, cfg: dict, nominal_foot_positions: list[NDArray]) -> None:
        self.step_height = cfg["step_height"]
        self.step_length = cfg["step_length"]
        self.duty_factor = cfg["duty_factor"]
        self.cycle_period = cfg["cycle_period"]
        self.nominal = [p.copy() for p in nominal_foot_positions]
        self.state = TripodGaitState()

    def step(self, dt: float, direction: NDArray, balance_correction: NDArray) -> list[NDArray]:
        """Advance gait by *dt* seconds and return target foot positions."""
        self.state.phase = (self.state.phase + dt / self.cycle_period) % 1.0
        phase = self.state.phase

        targets = []
        for leg_idx in range(3):
            leg_phase = (phase + leg_idx / 3.0) % 1.0
            if leg_phase < (1.0 - self.duty_factor):
                # Swing phase
                foot = self._swing_trajectory(leg_idx, leg_phase / (1.0 - self.duty_factor))
            else:
                # Stance phase — push body forward over planted foot.
                stance_t = (leg_phase - (1.0 - self.duty_factor)) / self.duty_factor
                foot = self.nominal[leg_idx] - direction * self.step_length * (stance_t - 0.5)
            foot = foot + balance_correction
            targets.append(foot)
        return targets

    def _swing_trajectory(self, leg: int, t: float) -> NDArray:
        """Cubic spline foot trajectory for the swing phase (t ∈ [0, 1])."""
        start = self.nominal[leg] - np.array([self.step_length / 2, 0, 0])
        end = self.nominal[leg] + np.array([self.step_length / 2, 0, 0])
        # Cubic blend for horizontal motion.
        s = 3 * t**2 - 2 * t**3
        pos = start + s * (end - start)
        # Sine arc for vertical lift.
        pos[2] += self.step_height * math.sin(math.pi * t)
        return pos
