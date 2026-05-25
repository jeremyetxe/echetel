# Echetel

Echetel is the control software for a three-legged stair-climbing robot designed to transport laundry between floors. The robot uses a stable tripod gait adapted for stair ascent and descent, with real-time balance control and stair-edge detection.

## Overview

```
     [basket]
    /   |   \
  leg0 leg1 leg2
```

The robot body mounts a laundry basket and three articulated legs arranged 120° apart. Each leg has three degrees of freedom (hip yaw, hip pitch, knee pitch) driven by brushless servo motors. An onboard IMU, depth camera, and toe force sensors close the balance loop and guide stair detection.

## Architecture

```
src/echetel/
├── robot.py            # Top-level RobotController
├── gait/
│   ├── tripod_gait.py  # Flat-surface tripod walking
│   ├── stair_gait.py   # Stair ascent/descent gait
│   └── planner.py      # Mode-switching gait planner
├── kinematics/
│   ├── forward.py      # Forward kinematics (joint → foot position)
│   └── inverse.py      # Inverse kinematics (foot position → joints)
├── sensors/
│   ├── imu.py          # IMU orientation + angular rate
│   ├── depth_camera.py # Stair geometry from depth image
│   └── load_cell.py    # Basket weight & CG estimation
├── control/
│   ├── balance.py      # PD balance compensator
│   ├── trajectory.py   # Foot trajectory generation
│   └── servo.py        # Low-level servo interface (USB/CAN)
├── navigation/
│   ├── stair_detection.py  # Detect step edges and measure riser/tread
│   └── path_planner.py     # High-level route planner (floor → floor)
└── payload/
    └── laundry_basket.py   # Basket lock/tilt actuator
```

## Requirements

- Python 3.10+
- numpy, scipy, pyserial (or python-can for CAN bus)
- pyrealsense2 (Intel RealSense depth camera)
- smbus2 (IMU over I²C)

Install:

```bash
pip install -e .
```

## Quick start

```bash
# Calibrate servos before first run
python scripts/calibrate.py

# Run the robot (will wait for a navigation goal)
python scripts/run_robot.py --config config/robot_config.yaml
```

## Configuration

All hardware parameters (leg geometry, servo IDs, PID gains, stair geometry limits) live in `config/robot_config.yaml`. See that file for annotated fields.

## Running tests

```bash
pytest tests/
```

## Hardware

| Component | Spec |
|---|---|
| Leg count | 3 |
| DOF per leg | 3 (hip yaw, hip pitch, knee pitch) |
| Servo | 30 kg·cm brushless, 12 V |
| IMU | ICM-42688-P (SPI) |
| Depth sensor | Intel RealSense D435 |
| Onboard computer | Raspberry Pi 5 |
| Battery | 22.2 V 6S LiPo, 5 Ah |
| Payload | Up to 5 kg laundry basket |
| Stair spec | Rise ≤ 220 mm, run ≥ 220 mm |

## License

MIT
