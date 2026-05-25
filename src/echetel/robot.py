"""Top-level robot controller."""

from __future__ import annotations

import signal
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import yaml

from .control.servo import ServoInterface
from .control.balance import BalanceController
from .gait.planner import GaitPlanner
from .navigation.stair_detection import StairDetector
from .navigation.path_planner import PathPlanner
from .sensors.imu import IMU
from .sensors.depth_camera import DepthCamera
from .sensors.load_cell import LoadCell
from .payload.laundry_basket import LaundryBasket

log = logging.getLogger(__name__)


class RobotController:
    """Orchestrates all subsystems and runs the main control loop."""

    CONTROL_HZ = 100  # inner loop frequency
    SENSOR_HZ = 30    # depth camera update frequency

    def __init__(self, config_path: str | Path) -> None:
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self._running = False
        self._tick = 0

        log.info("Initialising hardware interfaces…")
        self.servos = ServoInterface(self.cfg["servos"])
        self.imu = IMU(self.cfg["imu"])
        self.camera = DepthCamera(self.cfg["depth_camera"])
        self.load_cell = LoadCell(self.cfg["load_cell"])
        self.basket = LaundryBasket(self.cfg["payload"], self.servos)

        log.info("Initialising control stack…")
        self.balance = BalanceController(self.cfg["balance"])
        self.gait = GaitPlanner(self.cfg)
        self.stair_detector = StairDetector(self.cfg["stair_detection"])
        self.path_planner = PathPlanner()

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def go_to_floor(self, target_floor: int, current_floor: int) -> None:
        """Navigate from *current_floor* to *target_floor* via the staircase."""
        route = self.path_planner.plan(current_floor, target_floor)
        log.info("Route: %s", route)
        for segment in route:
            self._execute_segment(segment)

    def run(self) -> None:
        """Block and run the control loop until interrupted."""
        self._running = True
        log.info("Control loop started at %d Hz", self.CONTROL_HZ)
        period = 1.0 / self.CONTROL_HZ
        next_sensor_tick = 0
        sensor_period = self.CONTROL_HZ // self.SENSOR_HZ

        while self._running:
            t0 = time.monotonic()

            orientation = self.imu.read()
            self._safety_check(orientation)

            if self._tick % sensor_period == next_sensor_tick:
                depth_frame = self.camera.read()
                stair_info = self.stair_detector.process(depth_frame)
                self.gait.update_terrain(stair_info)

            balance_correction = self.balance.compute(orientation)
            joint_targets = self.gait.step(balance_correction)
            self.servos.write_all(joint_targets)
            self.basket.level(orientation)

            self._tick += 1
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
            else:
                log.warning("Control loop overrun by %.1f ms", (elapsed - period) * 1e3)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_segment(self, segment: dict) -> None:
        mode = segment["mode"]
        log.info("Executing segment: %s", mode)
        self.gait.set_mode(mode)
        while not self._segment_complete(segment):
            time.sleep(0.01)

    def _segment_complete(self, segment: dict) -> bool:
        return self.gait.segment_done()

    def _safety_check(self, orientation) -> None:
        max_tilt = self.cfg["safety"]["max_body_tilt_deg"]
        if abs(orientation.roll_deg) > max_tilt or abs(orientation.pitch_deg) > max_tilt:
            log.critical(
                "Body tilt exceeded safety limit (roll=%.1f°, pitch=%.1f°) — ESTOP",
                orientation.roll_deg, orientation.pitch_deg,
            )
            self._shutdown(None, None)

    def _shutdown(self, signum, frame) -> None:
        log.info("Shutting down…")
        self._running = False
        self.servos.disable_all()
        sys.exit(0)


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the Echetel robot controller")
    parser.add_argument(
        "--config",
        default="config/robot_config.yaml",
        help="Path to robot_config.yaml",
    )
    args = parser.parse_args()
    robot = RobotController(args.config)
    robot.run()


if __name__ == "__main__":
    main()
