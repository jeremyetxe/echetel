"""Intel RealSense D435 depth camera interface."""

from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


class DepthFrame:
    """Container for a single aligned depth + colour frame pair."""

    def __init__(self, depth: NDArray, color: NDArray, intrinsics: dict) -> None:
        self.depth = depth          # (H, W) float32, metres
        self.color = color          # (H, W, 3) uint8
        self.intrinsics = intrinsics


class DepthCamera:
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._pipeline = None
        self._align = None
        self._intrinsics: dict = {}
        self._stub_mode = False
        self._init_pipeline()

    def read(self) -> DepthFrame:
        if self._stub_mode:
            return self._stub_frame()
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=200)
            aligned = self._align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_frame.get_units()
            color = np.asanyarray(color_frame.get_data())
            return DepthFrame(depth, color, self._intrinsics)
        except Exception as exc:
            log.warning("Camera read error: %s", exc)
            return self._stub_frame()

    def stop(self) -> None:
        if self._pipeline and not self._stub_mode:
            self._pipeline.stop()

    # ------------------------------------------------------------------

    def _init_pipeline(self) -> None:
        cfg = self._cfg
        try:
            import pyrealsense2 as rs
            pipeline = rs.pipeline()
            config = rs.config()
            if cfg.get("serial"):
                config.enable_device(cfg["serial"])
            config.enable_stream(rs.stream.depth, cfg["width"], cfg["height"], rs.format.z16, cfg["fps"])
            config.enable_stream(rs.stream.color, cfg["width"], cfg["height"], rs.format.bgr8, cfg["fps"])
            profile = pipeline.start(config)
            sensor = profile.get_device().first_depth_sensor()
            sensor.set_option(rs.option.min_distance, int(cfg["min_depth_m"] * 1000))
            align_to = rs.stream.color
            self._align = rs.align(align_to)
            self._pipeline = pipeline
            depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            intr = depth_stream.get_intrinsics()
            self._intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}
            log.info("RealSense D435 started (%dx%d @ %d fps)", cfg["width"], cfg["height"], cfg["fps"])
        except Exception as exc:
            log.warning("RealSense unavailable (%s) — using stub depth frames", exc)
            self._stub_mode = True
            self._intrinsics = {
                "fx": 386.0, "fy": 386.0,
                "cx": self._cfg["width"] / 2,
                "cy": self._cfg["height"] / 2,
            }

    def _stub_frame(self) -> DepthFrame:
        h, w = self._cfg["height"], self._cfg["width"]
        depth = np.full((h, w), 1.0, dtype=np.float32)
        color = np.zeros((h, w, 3), dtype=np.uint8)
        return DepthFrame(depth, color, self._intrinsics)
