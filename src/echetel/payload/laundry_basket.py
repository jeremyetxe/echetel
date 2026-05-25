"""Laundry basket mount controller.

The basket sits on a two-axis gimbal driven by two servos:
- servo 10: lock/release latch
- servo 11: tilt axis (compensates body pitch during stair climbing)

The basket tilts opposite to the body pitch so the laundry remains level
and doesn't shift its weight distribution mid-climb.
"""

from __future__ import annotations

import math
import logging

from ..sensors.imu import Orientation

log = logging.getLogger(__name__)


class LaundryBasket:
    def __init__(self, cfg: dict, servos) -> None:
        self._lock_id = cfg["basket_lock_servo_id"]
        self._tilt_id = cfg["tilt_servo_id"]
        self._max_tilt = math.radians(cfg["max_tilt_deg"])
        self._servos = servos
        self._locked = True
        self.lock()

    def lock(self) -> None:
        self._write_servo(self._lock_id, 0.0)   # 0 rad = locked position
        self._locked = True
        log.debug("Basket locked")

    def release(self) -> None:
        self._write_servo(self._lock_id, math.radians(45.0))
        self._locked = False
        log.debug("Basket released")

    def level(self, orientation: Orientation) -> None:
        """Tilt the basket to cancel the body's pitch angle."""
        body_pitch = math.radians(orientation.pitch_deg)
        correction = -body_pitch
        correction = max(-self._max_tilt, min(self._max_tilt, correction))
        self._write_servo(self._tilt_id, correction)

    # ------------------------------------------------------------------

    def _write_servo(self, servo_id: int, angle_rad: float) -> None:
        # Reach into the servo interface's lower-level method.
        try:
            tick = self._servos._radians_to_ticks(angle_rad)
            self._servos._send_packet(
                servo_id,
                0x03,  # INST_WRITE
                __import__("struct").pack("<Hi", 116, tick),  # ADDR_GOAL_POSITION
            )
        except Exception as exc:
            log.debug("Basket servo write failed: %s", exc)
