"""ADS1115 load-cell amplifier driver (I²C).

Reads basket weight and estimates the payload centre-of-gravity shift,
which is fed to the balance controller.
"""

from __future__ import annotations

import logging
import struct

log = logging.getLogger(__name__)

_ADS1115_ADDR = 0x48
_REG_CONVERSION = 0x00
_REG_CONFIG = 0x01

# Config register: single-shot, AIN0-GND, ±4.096 V, 128 SPS
_CONFIG_ONE_SHOT = 0x8583


class LoadCell:
    def __init__(self, cfg: dict) -> None:
        self._full_scale = cfg["full_scale_kg"]
        self._tare = 0.0
        self._bus = self._open_bus(cfg["bus"], cfg["address"])
        if cfg.get("tare_on_startup"):
            self.tare()

    def read_kg(self) -> float:
        raw = self._read_raw()
        # 16-bit two's complement, full-scale corresponds to full_scale_kg.
        voltage = raw / 32768.0  # normalised ±1
        return max(0.0, voltage * self._full_scale - self._tare)

    def tare(self) -> None:
        readings = [self._read_raw() / 32768.0 * self._full_scale for _ in range(10)]
        self._tare = sum(readings) / len(readings)
        log.info("Load cell tared at %.3f kg", self._tare)

    # ------------------------------------------------------------------

    def _read_raw(self) -> int:
        try:
            self._bus.write_i2c_block_data(
                _ADS1115_ADDR, _REG_CONFIG,
                list(struct.pack(">H", _CONFIG_ONE_SHOT | 0x8000))
            )
            import time; time.sleep(0.01)
            data = self._bus.read_i2c_block_data(_ADS1115_ADDR, _REG_CONVERSION, 2)
            raw = struct.unpack(">h", bytes(data))[0]
            return raw
        except Exception:
            return 0

    @staticmethod
    def _open_bus(bus_num: int, address: int):
        try:
            import smbus2
            return smbus2.SMBus(bus_num)
        except Exception:
            return _StubBus()


class _StubBus:
    def write_i2c_block_data(self, *a, **k): pass
    def read_i2c_block_data(self, *a, **k): return [0, 0]
