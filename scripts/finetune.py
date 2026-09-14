#!/usr/bin/env python
"""Fine-tune the detector to fix real-photograph false positives."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.finetune import main

if __name__ == "__main__":
    main()
