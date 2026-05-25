"""Foot trajectory primitives used by both gaits."""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray


def cubic_blend(t: float) -> float:
    """Smooth step (3t²-2t³) for t ∈ [0, 1]."""
    t = max(0.0, min(1.0, t))
    return 3 * t * t - 2 * t * t * t


def swing_arc(start: NDArray, end: NDArray, lift: float, t: float) -> NDArray:
    """Interpolate a foot along a raised arc from *start* to *end*.

    t ∈ [0, 1]: 0 = start, 1 = end.
    The peak is at t = 0.5 with height *lift* above the start/end Z.
    """
    s = cubic_blend(t)
    pos = start + s * (end - start)
    pos = pos.copy()
    pos[2] += lift * math.sin(math.pi * t)
    return pos


def body_trajectory(
    start: NDArray, end: NDArray, t: float, height_profile: NDArray | None = None
) -> NDArray:
    """Interpolate the body position along a straight-line path.

    Optional *height_profile* is a 1-D array sampled at t=0…1 that adjusts
    the Z coordinate (useful for crouching on stair transitions).
    """
    s = cubic_blend(t)
    pos = start + s * (end - start)
    if height_profile is not None:
        idx = int(t * (len(height_profile) - 1))
        pos = pos.copy()
        pos[2] += height_profile[idx]
    return pos
