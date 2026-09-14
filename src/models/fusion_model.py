"""Fusion model: EfficientNet-B0 spatial embedding + frequency features.

The spatial-only EfficientNet-B0 remains the default.  This fusion model is an
*experimental candidate* used to test whether frequency-domain features improve
unseen-generator generalisation (see src/evaluation/evaluate_unseen.py).  It is
only selected if experiments actually justify it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.models.frequency_features import FREQUENCY_FEATURE_DIM


class FusionClassifier(nn.Module):
    """EfficientNet-B0 features + FFT frequency features + classification head."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Reuse the convolutional trunk and pooling; drop the ImageNet head.
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self._spatial_dim = backbone.classifier[1].in_features  # 1280

        self.dropout = nn.Dropout(p=dropout)
        self.fusion = nn.Linear(self._spatial_dim + FREQUENCY_FEATURE_DIM, num_classes)

    def _spatial_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor, freq_features: torch.Tensor) -> torch.Tensor:
        spatial = self._spatial_embedding(x)
        fused = torch.cat([spatial, freq_features], dim=1)
        fused = self.dropout(fused)
        return self.fusion(fused)


def create_fusion_model(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> FusionClassifier:
    """Instantiate the spatial+frequency fusion model."""
    return FusionClassifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
