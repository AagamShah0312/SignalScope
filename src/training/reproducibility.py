"""Deterministic seeding helpers for reproducible training/evaluation.

Reproducibility caveats (documented, not hidden):

* PyTorch training on CUDA is **not bit-for-bit deterministic** because of
  non-deterministic GPU kernels.  ``torch.use_deterministic_algorithms`` would
  make many cuDNN operations unavailable and slow training dramatically, so we
  only enable the parts that are practical.
* DataLoader workers add their own randomness unless seeded.
* Weight-initialization and augmentation draw from RNGs seeded here, so the
  *split, initialization and augmentation order* are reproducible; floating
  point reductions may still vary run-to-run on GPU.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs (CPU + CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
