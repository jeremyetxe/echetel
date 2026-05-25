#!/usr/bin/env python3
"""Start the Echetel robot controller."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from echetel.robot import main

if __name__ == "__main__":
    main()
