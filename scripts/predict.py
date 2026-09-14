#!/usr/bin/env python
"""Analyse a single image with SignalScope.

Usage:
    python scripts/predict.py path/to/image.jpg
    python scripts/predict.py path/to/image.jpg --robustness
"""
from app.inference import _main

if __name__ == "__main__":
    raise SystemExit(_main())
