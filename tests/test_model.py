"""Tests for model construction and checkpoint loading."""

from __future__ import annotations

import torch

from src.models.model import create_model, load_checkpoint


def test_model_outputs_two_classes():
    model = create_model(num_classes=2, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 2)


def test_model_parameter_count_is_efficientnet_b0():
    model = create_model(num_classes=2, pretrained=False)
    n = sum(p.numel() for p in model.parameters())
    # EfficientNet-B0 is ~4.0M parameters with a 2-class head.
    assert 3_500_000 < n < 4_500_000


def test_load_committed_checkpoint():
    """The shipped checkpoint must load into a fresh model."""
    model = create_model(num_classes=2, pretrained=False)
    loaded = load_checkpoint(model, "src/models/best_efficientnet_b0.pth", device=torch.device("cpu"))
    assert isinstance(loaded, torch.nn.Module)


def test_missing_checkpoint_raises(tmp_path):
    model = create_model(num_classes=2, pretrained=False)
    try:
        load_checkpoint(model, str(tmp_path / "nope.pth"))
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
