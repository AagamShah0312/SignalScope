import sys
from pathlib import Path

# Allow running directly (python scripts/<file>.py) from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#!/usr/bin/env python
"""Analyse a single image with SignalScope.

Usage:
    python scripts/predict.py path/to/image.jpg
    python scripts/predict.py path/to/image.jpg --robustness
"""
from app.inference import _main

if __name__ == "__main__":
    raise SystemExit(_main())
