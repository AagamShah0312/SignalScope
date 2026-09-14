import sys
from pathlib import Path

# Allow running directly (python scripts/<file>.py) from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#!/usr/bin/env python
"""Evaluate on the public CIFAKE test split (delegates to src.evaluation.evaluate)."""
from src.evaluation.evaluate import main

if __name__ == "__main__":
    main()
