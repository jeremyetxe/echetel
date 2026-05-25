"""High-level path planner.

Maps floor-to-floor navigation requests into a sequence of motion segments
(flat walking, stair ascent, stair descent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class Segment(NamedTuple):
    mode: str           # "flat", "ascent", "descent"
    description: str


class PathPlanner:
    def plan(self, current_floor: int, target_floor: int) -> list[Segment]:
        """Return an ordered list of motion segments to reach *target_floor*."""
        if current_floor == target_floor:
            return []

        segments: list[Segment] = []
        delta = 1 if target_floor > current_floor else -1

        floor = current_floor
        while floor != target_floor:
            # Walk to the staircase on the current floor.
            segments.append(Segment("flat", f"Walk to staircase on floor {floor}"))
            if delta > 0:
                segments.append(Segment("ascent", f"Climb from floor {floor} to {floor + 1}"))
            else:
                segments.append(Segment("descent", f"Descend from floor {floor} to {floor - 1}"))
            floor += delta
            # Walk from the staircase landing to the destination.
            if floor == target_floor:
                segments.append(Segment("flat", f"Walk to destination on floor {floor}"))

        return segments
