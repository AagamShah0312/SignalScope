"""Tests for the central metrics module."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_metrics, confusion_parts


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_metrics(y_true, y_prob)
    assert m["roc_auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["fpr"] == 0.0


def test_division_by_zero_safe():
    # All negatives (no positives): FPR denominator FP+TN is non-zero, TPR=0.
    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3])
    m = compute_metrics(y_true, y_prob)
    assert m["confusion_matrix"] == [[3, 0], [0, 0]]


def test_confusion_parts():
    cm = np.array([[8, 2], [1, 9]])
    parts = confusion_parts(cm)
    assert parts == {"TN": 8, "FP": 2, "FN": 1, "TP": 9}


def test_single_class_auc_is_nan():
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    m = compute_metrics(y_true, y_prob)
    assert np.isnan(m["roc_auc"])
