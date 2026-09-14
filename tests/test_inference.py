"""Tests for the inference pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from app.inference import ModelNotFoundError, SignalScopePredictor
from app.preprocessing import ImageValidationError, validate_file
from src.training.calibration import TemperatureScaler


@pytest.fixture(scope="module")
def predictor():
    return SignalScopePredictor()


def test_predict_schema(predictor, sample_png_bytes):
    result = predictor.predict(sample_png_bytes, filename="test.png", include_explanation=False)
    for key in ("verdict", "binary_label", "probability_ai", "probability_real",
                "confidence", "threshold", "status"):
        assert key in result
    assert 0.0 <= result["probability_ai"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["binary_label"] in ("REAL", "AI_GENERATED")
    assert result["verdict"] in ("likely_ai_generated", "likely_real", "inconclusive")


def test_confidence_between_0_and_1(predictor, sample_png_bytes):
    result = predictor.predict(sample_png_bytes, filename="t.png", include_explanation=False)
    assert 0.0 <= result["confidence"] <= 1.0


def test_threshold_logic(predictor, sample_png_bytes):
    result = predictor.predict(sample_png_bytes, filename="t.png", include_explanation=False)
    expected = "AI_GENERATED" if result["probability_ai"] >= result["threshold"] else "REAL"
    assert result["binary_label"] == expected


def test_gradcam_generates_output(predictor, sample_png_bytes, tmp_path):
    result = predictor.predict(
        sample_png_bytes, filename="t.png", include_explanation=True,
        output_dir=str(tmp_path), base_name="g",
    )
    assert result["visualization"] is not None
    assert result["visualization"]["overlay"]
    assert result["explanation"] is not None
    assert isinstance(result["explanation"]["evidence"], list)


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        SignalScopePredictor(checkpoint_path=str(tmp_path / "missing.pth"))


def test_validate_invalid_file_rejected():
    with pytest.raises(ImageValidationError):
        validate_file(b"not an image", filename="x.png")


def test_calibration_temperature_identity():
    scaler = TemperatureScaler(temperature=1.0)
    logits = np.array([[1.0, -1.0], [-1.0, 1.0]])
    probs = scaler.calibrate(logits)
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
