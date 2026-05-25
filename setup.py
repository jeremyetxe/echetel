from setuptools import setup, find_packages

setup(
    name="echetel",
    version="0.1.0",
    description="Control software for a three-legged stair-climbing laundry robot",
    author="Jeremy Green",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "scipy>=1.12",
        "pyserial>=3.5",
        "python-can>=4.3",
        "smbus2>=0.4",
        "PyYAML>=6.0",
    ],
    extras_require={
        "camera": ["pyrealsense2>=2.55"],
        "dev": ["pytest>=8.0"],
    },
    entry_points={
        "console_scripts": [
            "echetel-run=echetel.robot:main",
            "echetel-calibrate=echetel.control.servo:calibrate_main",
        ]
    },
)
