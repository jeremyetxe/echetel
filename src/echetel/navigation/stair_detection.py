"""Stair detection from a RealSense depth frame.

Algorithm:
1. Crop to a configurable region-of-interest in front of the robot.
2. Build a row-average depth profile.
3. Detect discontinuities (step edges) whose magnitude is within the
   allowed riser range.
4. Measure tread depth between consecutive edges.
5. Return a StairInfo if a consistent stair pattern is found.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..sensors.depth_camera import DepthFrame

log = logging.getLogger(__name__)


@dataclass
class StairInfo:
    detected: bool
    ascending: bool       # True = steps going up away from robot
    step_count: int
    riser_m: float        # average riser height
    tread_m: float        # average tread depth
    first_edge_m: float   # distance to nearest step edge


class StairDetector:
    def __init__(self, cfg: dict) -> None:
        self.edge_threshold = cfg["edge_threshold_m"]
        self.max_rise = cfg["max_rise_m"]
        self.min_tread = cfg["min_tread_m"]
        roi = cfg["scan_region"]
        self._roi = roi  # {top, bottom, left, right} as fractions

    def process(self, frame: DepthFrame) -> Optional[StairInfo]:
        depth = frame.depth
        h, w = depth.shape
        r = self._roi
        top = int(r["top"] * h)
        bot = int(r["bottom"] * h)
        left = int(r["left"] * w)
        right = int(r["right"] * w)

        roi_depth = depth[top:bot, left:right]
        profile = self._row_profile(roi_depth)
        if profile is None:
            return None

        edges = self._detect_edges(profile, frame.intrinsics)
        if not edges:
            return None

        return self._classify(edges)

    # ------------------------------------------------------------------

    def _row_profile(self, roi: NDArray) -> Optional[NDArray]:
        """Mean depth per row, ignoring zeros (invalid pixels)."""
        valid = np.where(roi > 0, roi, np.nan)
        profile = np.nanmean(valid, axis=1)
        if np.all(np.isnan(profile)):
            return None
        return profile

    def _detect_edges(self, profile: NDArray, intrinsics: dict) -> list[dict]:
        """Find rows where depth jumps by more than the edge threshold."""
        diff = np.diff(profile)
        edges = []
        for i, d in enumerate(diff):
            if abs(d) >= self.edge_threshold:
                edges.append({
                    "row": i,
                    "depth_m": float(np.nanmean(profile[i:i+2])),
                    "jump_m": float(d),
                })
        return edges

    def _classify(self, edges: list[dict]) -> Optional[StairInfo]:
        if len(edges) < 1:
            return None

        risers = [abs(e["jump_m"]) for e in edges]
        avg_riser = sum(risers) / len(risers)

        if avg_riser > self.max_rise:
            log.debug("Riser %.3f m exceeds max; ignoring", avg_riser)
            return None

        # Ascending = depth increases as rows go down (steps going away).
        ascending = edges[0]["jump_m"] > 0

        # Estimate tread from consecutive edge distances (approximation).
        if len(edges) >= 2:
            tread = abs(edges[1]["depth_m"] - edges[0]["depth_m"])
        else:
            tread = self.min_tread

        if tread < self.min_tread:
            return None

        return StairInfo(
            detected=True,
            ascending=ascending,
            step_count=len(edges),
            riser_m=avg_riser,
            tread_m=tread,
            first_edge_m=edges[0]["depth_m"],
        )
