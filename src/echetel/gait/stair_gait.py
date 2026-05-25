"""Stair-climbing gait for ascent and descent.

The strategy: one leg steps up (or down) to the next tread while the other
two remain planted, ensuring static stability at all times.  Body height is
lowered to keep the CoM below the step edge during the transition.
"""

from __future__ import annotations

import math
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray


class StairPhase(Enum):
    APPROACH = auto()       # walking to the first step edge
    STEP_UP = auto()        # lifting one leg onto the next tread
    BODY_SHIFT = auto()     # shifting body weight over planted feet
    STEP_COMPLETE = auto()  # brief double-support dwell
    RETREAT = auto()        # symmetric descent phase
    DONE = auto()


class StairGait:
    """Manages a single stair-climbing manoeuvre (ascent or descent)."""

    def __init__(self, cfg: dict, nominal_foot_positions: list[NDArray]) -> None:
        flat_cfg = cfg["gait"]["flat"]
        stair_cfg = cfg["gait"]["stair"]
        self.base_step_height = flat_cfg["step_height"]
        self.direction = "ascent"  # or "descent"

        self.ascent_cfg = stair_cfg["ascent"]
        self.descent_cfg = stair_cfg["descent"]
        self.nominal = [p.copy() for p in nominal_foot_positions]

        self._phase = StairPhase.APPROACH
        self._active_leg = 0
        self._manoeuvre_t = 0.0
        self._manoeuvre_duration = 1.2  # seconds per step-up
        self._steps_remaining = 0
        self._riser = 0.0
        self._tread = 0.0

    def configure(self, direction: str, step_count: int, riser_m: float, tread_m: float) -> None:
        self.direction = direction
        self._steps_remaining = step_count
        self._riser = riser_m
        self._tread = tread_m
        self._phase = StairPhase.APPROACH
        self._active_leg = 0
        self._manoeuvre_t = 0.0

    def step(self, dt: float, balance_correction: NDArray) -> list[NDArray]:
        if self._phase == StairPhase.DONE:
            return [p.copy() + balance_correction for p in self.nominal]

        self._manoeuvre_t += dt
        t = min(self._manoeuvre_t / self._manoeuvre_duration, 1.0)
        targets = [p.copy() for p in self.nominal]

        if self.direction == "ascent":
            targets = self._ascent_targets(t, targets)
        else:
            targets = self._descent_targets(t, targets)

        if t >= 1.0:
            self._advance_phase()

        return [tgt + balance_correction for tgt in targets]

    def done(self) -> bool:
        return self._phase == StairPhase.DONE

    # ------------------------------------------------------------------

    def _ascent_targets(self, t: float, targets: list[NDArray]) -> list[NDArray]:
        cfg = self.ascent_cfg
        step_h = self.base_step_height * cfg["step_height_multiplier"]
        reach = cfg["forward_reach"]

        leg = self._active_leg
        if self._phase == StairPhase.STEP_UP:
            # Swing the active leg up onto the next tread.
            start = self.nominal[leg].copy()
            end = self.nominal[leg] + np.array([reach, 0.0, self._riser])
            s = 3 * t**2 - 2 * t**3
            targets[leg] = start + s * (end - start)
            targets[leg][2] += step_h * math.sin(math.pi * t)
        elif self._phase == StairPhase.BODY_SHIFT:
            # Push body forward so CoM is above the planted higher foot.
            for i in range(3):
                targets[i] = self.nominal[i] + np.array([(1 - t) * reach * 0.5, 0.0, 0.0])
        return targets

    def _descent_targets(self, t: float, targets: list[NDArray]) -> list[NDArray]:
        cfg = self.descent_cfg
        step_h = self.base_step_height * cfg["step_height_multiplier"]
        reach = cfg["forward_reach"]

        leg = self._active_leg
        if self._phase == StairPhase.STEP_UP:
            start = self.nominal[leg].copy()
            end = self.nominal[leg] + np.array([reach, 0.0, -self._riser])
            s = 3 * t**2 - 2 * t**3
            targets[leg] = start + s * (end - start)
            targets[leg][2] += step_h * math.sin(math.pi * t)
        return targets

    def _advance_phase(self) -> None:
        self._manoeuvre_t = 0.0
        if self._phase == StairPhase.APPROACH:
            self._phase = StairPhase.STEP_UP
        elif self._phase == StairPhase.STEP_UP:
            self._phase = StairPhase.BODY_SHIFT
        elif self._phase == StairPhase.BODY_SHIFT:
            self._phase = StairPhase.STEP_COMPLETE
        elif self._phase == StairPhase.STEP_COMPLETE:
            self._steps_remaining -= 1
            self._active_leg = (self._active_leg + 1) % 3
            if self._steps_remaining <= 0:
                self._phase = StairPhase.DONE
            else:
                self._phase = StairPhase.STEP_UP
