"""Tests for frequency-domain feature extraction and the fusion model."""

from __future__ import annotations

import torch

from src.models.frequency_features import (
    FREQUENCY_FEATURE_DIM,
    extract_frequency_features,
    fft_magnitude,
    _to_grayscale,
)
from src.models.fusion_model import create_fusion_model


def test_frequency_feature_shape():
    x = torch.randn(3, 3, 224, 224)
    f = extract_frequency_features(x)
    assert f.shape == (3, FREQUENCY_FEATURE_DIM)
    assert torch.isfinite(f).all()


def test_fft_magnitude_shape():
    gray = _to_grayscale(torch.randn(2, 3, 128, 128))
    mag = fft_magnitude(gray)
    assert mag.shape == (2, 128, 128)


def test_fusion_model_forward():
    model = create_fusion_model(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    f = extract_frequency_features(x)
    out = model(x, f)
    assert out.shape == (2, 2)
