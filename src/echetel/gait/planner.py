"""Gait planner: selects between flat-surface and stair gaits."""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from .tripod_gait import TripodGait
from .stair_gait import StairGait, StairPhase
from ..kinematics.inverse import LegIK
from ..navigation.stair_detection import StairInfo


class GaitPlanner:
    FLAT = "flat"
    ASCENT = "ascent"
    DESCENT = "descent"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        seg = cfg["legs"]["segments"]
        lim = cfg["legs"]["joint_limits"]

        body_radius = cfg["robot"]["body_radius"]
        offsets_deg = cfg["legs"]["offsets_deg"]

        # Nominal foot positions in body frame (legs extended horizontally).
        reach = seg["coxa_length"] + seg["femur_length"] * 0.7
        self.nominal_feet = [
            np.array([
                body_radius * math.cos(math.radians(d)) + reach,
                body_radius * math.sin(math.radians(d)),
                -seg["tibia_length"],
            ])
            for d in offsets_deg
        ]

        self.ik_solvers = [
            LegIK(seg["coxa_length"], seg["femur_length"], seg["tibia_length"], lim)
            for _ in range(3)
        ]

        self.flat_gait = TripodGait(cfg["gait"]["flat"], self.nominal_feet)
        self.stair_gait = StairGait(cfg, self.nominal_feet)
        self._mode = self.FLAT
        self._dt = 1.0 / 100  # matches RobotController.CONTROL_HZ
        self._direction = np.array([1.0, 0.0, 0.0])

    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def update_terrain(self, stair_info: Optional[StairInfo]) -> None:
        if stair_info is None:
            return
        if stair_info.detected and self._mode == self.FLAT:
            direction = "ascent" if stair_info.ascending else "descent"
            self._mode = direction
            self.stair_gait.configure(
                direction=direction,
                step_count=stair_info.step_count,
                riser_m=stair_info.riser_m,
                tread_m=stair_info.tread_m,
            )

    def step(self, balance_correction: NDArray) -> list[list[float]]:
        """Return joint angles for all three legs."""
        if self._mode == self.FLAT:
            foot_targets = self.flat_gait.step(self._dt, self._direction, balance_correction)
        else:
            foot_targets = self.stair_gait.step(self._dt, balance_correction)
            if self.stair_gait.done():
                self._mode = self.FLAT

        joint_targets = []
        for i, (ik, foot) in enumerate(zip(self.ik_solvers, foot_targets)):
            try:
                angles = ik.solve(foot)
            except Exception:
                angles = np.zeros(3)
            joint_targets.append(angles.tolist())
        return joint_targets

    def segment_done(self) -> bool:
        if self._mode == self.FLAT:
            return True
        return self.stair_gait.done()
