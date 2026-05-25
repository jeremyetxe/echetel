"""Servo interface — Dynamixel-compatible protocol over serial.

Supports the Dynamixel Protocol 2.0 packet format used by most mid-range
brushless servos.  The CAN back-end uses the same logical interface.
"""

from __future__ import annotations

import logging
import math
import struct
import time
from typing import Sequence

import serial

log = logging.getLogger(__name__)

# Protocol 2.0 constants
_HEADER = b"\xff\xff\xfd\x00"
_INST_WRITE = 0x03
_INST_SYNC_WRITE = 0x83
_ADDR_GOAL_POSITION = 116  # 4 bytes, unit = 0.088°
_ADDR_TORQUE_ENABLE = 64   # 1 byte
_POSITION_UNIT = 0.088     # degrees per tick
_TICKS_PER_REV = 4096


class ServoInterface:
    def __init__(self, cfg: dict) -> None:
        self._ids: list[list[int]] = cfg["ids"]   # [leg][joint]
        self._port = self._open_port(cfg)
        self._torque_limit = cfg.get("torque_limit", 0.85)
        self._enable_all()

    def write_all(self, joint_targets: list[list[float]]) -> None:
        """Send goal positions for all legs in a single sync-write packet.

        joint_targets: list of 3 legs, each a list of 3 joint angles (radians).
        """
        data = []
        for leg_idx, joints in enumerate(joint_targets):
            for joint_idx, angle_rad in enumerate(joints):
                servo_id = self._ids[leg_idx][joint_idx]
                tick = self._radians_to_ticks(angle_rad)
                data.append((servo_id, tick))
        self._sync_write_positions(data)

    def disable_all(self) -> None:
        for row in self._ids:
            for sid in row:
                self._write_byte(sid, _ADDR_TORQUE_ENABLE, 0)
        log.info("All servos disabled")

    # ------------------------------------------------------------------

    def _enable_all(self) -> None:
        for row in self._ids:
            for sid in row:
                self._write_byte(sid, _ADDR_TORQUE_ENABLE, 1)

    def _sync_write_positions(self, id_ticks: list[tuple[int, int]]) -> None:
        # Build sync write payload.
        data_length = 4  # goal position is 4 bytes
        payload = struct.pack("<HH", _ADDR_GOAL_POSITION, data_length)
        for servo_id, tick in id_ticks:
            payload += struct.pack("<Bi", servo_id, tick)
        self._send_packet(0xFE, _INST_SYNC_WRITE, payload)

    def _write_byte(self, servo_id: int, address: int, value: int) -> None:
        payload = struct.pack("<HB", address, value)
        self._send_packet(servo_id, _INST_WRITE, payload)

    def _send_packet(self, servo_id: int, instruction: int, params: bytes) -> None:
        length = len(params) + 3  # instruction + params + 2-byte CRC
        header = _HEADER + struct.pack("<BHB", servo_id, length, instruction)
        body = header + params
        crc = self._crc16(body)
        packet = body + struct.pack("<H", crc)
        try:
            self._port.write(packet)
        except Exception as exc:
            log.debug("Servo write error: %s", exc)

    @staticmethod
    def _crc16(data: bytes) -> int:
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    @staticmethod
    def _radians_to_ticks(angle: float) -> int:
        degrees = math.degrees(angle)
        ticks = int(round(degrees / _POSITION_UNIT + _TICKS_PER_REV / 2))
        return max(0, min(_TICKS_PER_REV - 1, ticks))

    @staticmethod
    def _open_port(cfg: dict):
        iface = cfg.get("interface", "serial")
        if iface == "serial":
            try:
                port = serial.Serial(
                    cfg["port"],
                    baudrate=cfg["baudrate"],
                    timeout=0.01,
                )
                log.info("Servo serial port %s opened at %d baud", cfg["port"], cfg["baudrate"])
                return port
            except Exception as exc:
                log.warning("Serial port unavailable (%s) — using stub", exc)
                return _StubPort()
        raise ValueError(f"Unknown servo interface: {iface!r}")


class _StubPort:
    def write(self, data): pass
    def read(self, n): return b"\x00" * n


def calibrate_main() -> None:
    import argparse, yaml
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Servo calibration utility")
    parser.add_argument("--config", default="config/robot_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    iface = ServoInterface(cfg["servos"])
    log.info("Moving all servos to zero position for 3 seconds…")
    zero = [[0.0, 0.0, 0.0]] * 3
    iface.write_all(zero)
    time.sleep(3.0)
    iface.disable_all()
    log.info("Calibration complete")
