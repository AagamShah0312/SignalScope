"""Ensemble inference over multiple EfficientNet-B0 checkpoints.

The two available models specialise in different failure modes:

* the CIFAKE-trained baseline detects synthetic imagery but over-flags real
  photographs (its REAL class was 32x32 CIFAR-10),
* the fine-tuned "real-photo adaptation" model almost never flags real photos
  but is slightly weaker on high-quality synthetic images.

Averaging their logits (soft-voting) recovers the best of both.  The member
weights are configurable in ``config.yaml`` (``model.ensemble``) and were
selected on the small public adaptation evaluation set — see
``reports/real_photo_adaptation.md``.  They should be re-estimated once
full-scale CIFAKE + Defactify validation data is available.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.models.model import create_model, load_checkpoint


class EnsembleClassifier:
    """Weighted logit-space average of several 2-class EfficientNet models."""

    def __init__(
        self,
        models: Sequence[nn.Module],
        weights: Optional[Sequence[float]] = None,
        device: Optional[torch.device] = None,
    ):
        if len(models) == 0:
            raise ValueError("EnsembleClassifier requires at least one model.")
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models = [m.to(device) for m in models]
        for m in self.models:
            m.eval()

        if weights is None:
            weights = [1.0] * len(models)
        total = sum(float(w) for w in weights)
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        self.weights = [float(w) / total for w in weights]
        self.device = device

    def to(self, device):
        self.models = [m.to(device) for m in self.models]
        self.device = device
        return self

    @property
    def primary(self) -> nn.Module:
        """Model used for Grad-CAM explanations (highest-weight member)."""
        return self.models[max(range(len(self.weights)), key=lambda i: self.weights[i])]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted average of member logits."""
        outputs = []
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                outputs.append(weight * model(x))
        return sum(outputs)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities of the combined logits."""
        return torch.softmax(self.logits(x), dim=1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


def build_ensemble_from_checkpoints(
    checkpoint_paths: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
) -> EnsembleClassifier:
    """Load an ensemble from checkpoint files (each an EfficientNet-B0)."""
    models: List[nn.Module] = []
    for path in checkpoint_paths:
        model = create_model(num_classes=num_classes, pretrained=False)
        load_checkpoint(model, str(path), device=device)
        models.append(model)
    return EnsembleClassifier(models, weights=weights, device=device)
