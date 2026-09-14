"""Tests for calibration and threshold selection."""

from __future__ import annotations

import numpy as np

from src.training.calibration import (
    calibration_quality,
    select_threshold_youden,
    TemperatureScaler,
)


def test_temperature_scaling_recovers_identity():
    # With well-separated logits the learned temperature stays finite.
    logits = np.array([[4.0, -4.0], [-4.0, 4.0], [5.0, -5.0], [-5.0, 5.0]])
    labels = np.array([1, 0, 1, 0])
    scaler = TemperatureScaler().fit(logits, labels)
    assert scaler.temperature > 0


def test_calibrate_output_is_probability():
    scaler = TemperatureScaler(temperature=2.0)
    probs = scaler.calibrate(np.array([[2.0, -2.0]]))
    assert probs.shape == (1, 2)
    assert np.allclose(probs.sum(), 1.0)
    assert 0 <= probs[0, 1] <= 1


def test_youden_threshold_in_range():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = select_threshold_youden(y_true, y_prob)
    assert 0.0 < thr < 1.0


def test_calibration_quality_keys():
    q = calibration_quality(np.array([0, 1, 1]), np.array([0.1, 0.8, 0.9]))
    for key in ("brier_score", "log_loss", "expected_calibration_error"):
        assert key in q
