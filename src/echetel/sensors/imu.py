"""ICM-42688-P IMU driver (SPI).

Reads raw accelerometer and gyroscope data and integrates them into roll/pitch
orientation using a complementary filter.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class Orientation:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_rate_dps: float = 0.0
    pitch_rate_dps: float = 0.0
    yaw_rate_dps: float = 0.0


# ICM-42688-P register addresses
_REG_PWR_MGMT0 = 0x4E
_REG_GYRO_CONFIG0 = 0x4F
_REG_ACCEL_CONFIG0 = 0x50
_REG_TEMP_DATA1 = 0x1D
_REG_ACCEL_DATA_X1 = 0x1F
_REG_GYRO_DATA_X1 = 0x25

_ACCEL_SCALE = 9.80665 / 2048.0   # ±16 g range, 2048 LSB/g
_GYRO_SCALE = 1.0 / 16.4           # ±2000 dps range, 16.4 LSB/dps

_COMPLEMENTARY_ALPHA = 0.98        # weight on gyro vs. accelerometer


class IMU:
    def __init__(self, cfg: dict) -> None:
        self._orientation = Orientation()
        self._last_t = time.monotonic()
        self._trim = cfg.get("orientation_offset", [0.0, 0.0, 0.0])
        self._spi = self._open_spi(cfg["device"])

    def read(self) -> Orientation:
        raw_accel = self._read_accel()
        raw_gyro = self._read_gyro()
        self._integrate(raw_accel, raw_gyro)
        return self._orientation

    # ------------------------------------------------------------------

    def _integrate(self, accel: list[float], gyro: list[float]) -> None:
        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now

        ax, ay, az = accel
        gx, gy, gz = gyro

        # Accelerometer-derived tilt (stable long-term).
        accel_roll = math.degrees(math.atan2(ay, math.hypot(ax, az)))
        accel_pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))

        # Gyroscope integration (stable short-term).
        gyro_roll = self._orientation.roll_deg + gx * dt
        gyro_pitch = self._orientation.pitch_deg + gy * dt

        # Complementary filter.
        self._orientation.roll_deg = (
            _COMPLEMENTARY_ALPHA * gyro_roll + (1 - _COMPLEMENTARY_ALPHA) * accel_roll
            + self._trim[0]
        )
        self._orientation.pitch_deg = (
            _COMPLEMENTARY_ALPHA * gyro_pitch + (1 - _COMPLEMENTARY_ALPHA) * accel_pitch
            + self._trim[1]
        )
        self._orientation.yaw_deg += gz * dt + self._trim[2]
        self._orientation.roll_rate_dps = gx
        self._orientation.pitch_rate_dps = gy
        self._orientation.yaw_rate_dps = gz

    def _read_accel(self) -> list[float]:
        raw = self._spi_read(_REG_ACCEL_DATA_X1, 6)
        return [
            self._to_signed16(raw[0], raw[1]) * _ACCEL_SCALE,
            self._to_signed16(raw[2], raw[3]) * _ACCEL_SCALE,
            self._to_signed16(raw[4], raw[5]) * _ACCEL_SCALE,
        ]

    def _read_gyro(self) -> list[float]:
        raw = self._spi_read(_REG_GYRO_DATA_X1, 6)
        return [
            self._to_signed16(raw[0], raw[1]) * _GYRO_SCALE,
            self._to_signed16(raw[2], raw[3]) * _GYRO_SCALE,
            self._to_signed16(raw[4], raw[5]) * _GYRO_SCALE,
        ]

    @staticmethod
    def _to_signed16(high: int, low: int) -> int:
        val = (high << 8) | low
        return val - 65536 if val >= 32768 else val

    def _spi_read(self, reg: int, length: int) -> list[int]:
        # SPI read: assert CS, send reg | 0x80 (read bit), then clock in bytes.
        try:
            result = self._spi.xfer2([reg | 0x80] + [0x00] * length)
            return result[1:]
        except Exception:
            return [0] * length

    @staticmethod
    def _open_spi(device: str):
        try:
            import spidev
            spi = spidev.SpiDev()
            bus, dev = (int(x) for x in device.replace("/dev/spidev", "").split("."))
            spi.open(bus, dev)
            spi.max_speed_hz = 10_000_000
            spi.mode = 0
            return spi
        except Exception:
            # Return a stub when running without hardware.
            return _StubSpi()


class _StubSpi:
    def xfer2(self, data):
        return [0] * len(data)
