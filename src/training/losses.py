"""Loss functions for SignalScope training."""

from __future__ import annotations

import torch.nn as nn


def build_criterion(name: str = "cross_entropy", **kwargs):
    """Build a classification criterion by name."""
    name = name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=kwargs.get("label_smoothing", 0.0))
    raise ValueError(f"Unknown loss: {name}")
