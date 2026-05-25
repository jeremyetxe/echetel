"""Tests for stair detection from depth frames."""

import numpy as np
import pytest

from echetel.navigation.stair_detection import StairDetector, StairInfo
from echetel.sensors.depth_camera import DepthFrame


INTRINSICS = {"fx": 386.0, "fy": 386.0, "cx": 320.0, "cy": 240.0}
CFG = {
    "edge_threshold_m": 0.05,
    "max_rise_m": 0.220,
    "min_tread_m": 0.220,
    "scan_region": {"top": 0.3, "bottom": 0.75, "left": 0.15, "right": 0.85},
}


def make_frame(depth: np.ndarray) -> DepthFrame:
    color = np.zeros((*depth.shape, 3), dtype=np.uint8)
    return DepthFrame(depth, color, INTRINSICS)


@pytest.fixture
def detector():
    return StairDetector(CFG)


class TestStairDetector:
    def test_flat_floor_not_detected(self, detector):
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        result = detector.process(make_frame(depth))
        assert result is None or not result.detected

    def test_ascending_stairs_detected(self, detector):
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        # Simulate two ascending step edges in the ROI rows 144–360.
        h, w = depth.shape
        for row_frac, jump in [(0.45, 0.15), (0.60, 0.15)]:
            row = int(row_frac * h)
            depth[row:, :] += jump
        result = detector.process(make_frame(depth))
        assert result is not None
        assert result.detected
        assert result.ascending

    def test_riser_too_large_not_detected(self, detector):
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        h = depth.shape[0]
        row = int(0.5 * h)
        depth[row:, :] += 0.35  # 350 mm > max 220 mm
        result = detector.process(make_frame(depth))
        assert result is None or not result.detected
