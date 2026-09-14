"""EfficientNet-B0 model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create an EfficientNet-B0 with an ``num_classes`` classification head."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_device() -> torch.device:
    """Return the best available device (CUDA if present, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load weights into ``model``, tolerating both plain state_dict and the
    checkpoint-dict format produced by the training script."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}\n"
            "Train a model (scripts/train.py) or download the weights — "
            "see README.md."
        )

    device = device or get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model.to(device)


if __name__ == "__main__":
    device = get_device()
    model = create_model().to(device)
    print("Device:", device)
    print("Model: EfficientNet-B0")
    print("Parameters:", sum(p.numel() for p in model.parameters()))
