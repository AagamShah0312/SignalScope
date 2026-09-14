"""Tests for model construction and checkpoint loading."""

from __future__ import annotations

import torch

from src.models.model import create_model, load_checkpoint, resolve_checkpoint


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


def test_resolve_checkpoint_returns_existing():
    """Resolution must return a checkpoint that actually exists on disk."""
    resolved = resolve_checkpoint()
    assert resolved.exists()
    assert resolved.suffix == ".pth"


def test_resolve_checkpoint_prefers_explicit():
    """An explicitly passed checkpoint (even missing) is preferred first."""
    resolved = resolve_checkpoint("src/models/best_efficientnet_b0.pth")
    assert resolved.name == "best_efficientnet_b0.pth"
    assert resolved.exists()


def test_ensemble_averages_logits():
    """Weighted logit-space average of two models behaves as expected."""
    from src.models.ensemble import EnsembleClassifier

    m1 = create_model(num_classes=2, pretrained=False)
    m2 = create_model(num_classes=2, pretrained=False)
    ensemble = EnsembleClassifier([m1, m2], weights=[0.3, 0.7])
    x = torch.randn(1, 3, 224, 224)
    logits = ensemble.logits(x)
    assert logits.shape == (1, 2)
    proba = ensemble.predict_proba(x)
    assert proba.shape == (1, 2)
    assert torch.allclose(proba.sum(dim=1), torch.ones(1), atol=1e-6)


def test_ensemble_requires_at_least_one_model():
    from src.models.ensemble import EnsembleClassifier

    try:
        EnsembleClassifier([])
        raised = False
    except ValueError:
        raised = True
    assert raised

