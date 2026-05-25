#!/usr/bin/env python3
"""Servo calibration utility — moves all joints to zero and disables torque."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from echetel.control.servo import calibrate_main

if __name__ == "__main__":
    calibrate_main()
