import sys
from pathlib import Path

# Allow running directly (python scripts/<file>.py) from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#!/usr/bin/env python
"""Fit temperature scaling + threshold on validation data."""
from src.training.calibrate import main

if __name__ == "__main__":
    main()
