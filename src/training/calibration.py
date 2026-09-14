"""Confidence calibration (temperature scaling) and threshold selection.

The model's raw softmax output must NOT be treated as a trustworthy
probability.  We calibrate with temperature scaling fitted **only on
validation data** (never on the official held-out test set).

Two artifacts are produced and persisted:

* ``model/calibration.json``  — temperature + decision thresholds
* ``reports/calibration_results.json`` — calibration-quality metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.config import load_config


class TemperatureScaler:
    """Single-parameter temperature scaling over logits.

    ``p_calibrated = softmax(logits / T)``.  T=1.0 is the identity.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, epochs: int = 200):
        """Learn T on validation logits by minimising NLL (no labels leak)."""
        labels = np.asarray(labels).reshape(-1)
        if len(np.unique(labels)) < 2:
            self.temperature = 1.0
            return self

        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        temperature = torch.nn.Parameter(torch.ones(1) * self.temperature)
        optimizer = torch.optim.LBFGS([temperature], lr=lr)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_t / temperature, labels_t)
            loss.backward()
            return loss

        for _ in range(epochs):
            optimizer.step(closure)

        self.temperature = float(temperature.detach().cpu().item())
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        with torch.no_grad():
            return torch.softmax(logits_t / self.temperature, dim=1).numpy()


class IsotonicCalibrator:
    """Non-parametric isotonic regression on the AI-class probability."""

    def __init__(self):
        self.model: Optional[IsotonicRegression] = None

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model.fit(probs, labels)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if self.model is None:
            return probs
        calibrated = self.model.predict(probs)
        return np.stack([1.0 - calibrated, calibrated], axis=1)


def select_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Choose the threshold maximising Youden's J = TPR - FPR."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best = int(np.argmax(j))
    # roc_curve thresholds may contain >1 or <=0 entries; clamp to (0,1).
    thr = float(thresholds[best])
    return min(max(thr, 1e-6), 1.0 - 1e-6)


def select_threshold_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.05) -> float:
    """Choose the threshold giving (at most) a target false-positive rate."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.5
    idx = valid[-1]  # highest threshold still meeting the FPR budget
    thr = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return min(max(thr, 1e-6), 1.0 - 1e-6)


def fit_calibration(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    method: str = "temperature_scaling",
) -> Tuple[object, float]:
    """Fit a calibrator on validation data and return (calibrator, threshold)."""
    val_labels = np.asarray(val_labels).reshape(-1)

    if method == "isotonic":
        # Isotonic works on the raw AI probability.
        probs = _softmax(val_logits)[:, 1]
        calibrator = IsotonicCalibrator().fit(probs, val_labels)
        y_prob = calibrator.calibrate(probs)[:, 1]
    else:
        calibrator = TemperatureScaler().fit(val_logits, val_labels)
        y_prob = calibrator.calibrate(val_logits)[:, 1]

    threshold = select_threshold_youden(val_labels, y_prob)
    return calibrator, threshold


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def calibration_quality(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Brier score, log loss and ECE (binned) for a set of probabilities."""
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    ece, n_bins = 0.0, 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)

    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))),
        "expected_calibration_error": float(ece),
    }


def save_calibration(calibrator, threshold: float, method: str, quality: Dict[str, float]) -> None:
    """Persist calibration parameters to ``model/calibration.json``."""
    cfg = load_config()
    model_dir = cfg.resolve(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": method,
        "threshold": float(threshold),
        "parameters": {
            "temperature": float(getattr(calibrator, "temperature", 1.0)),
        },
        "quality": quality,
    }
    with open(model_dir / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_calibration() -> Dict:
    """Load calibration parameters, returning identity defaults if absent."""
    cfg = load_config()
    path = cfg.resolve(cfg.paths.model_dir) / "calibration.json"
    if not path.exists():
        return {
            "method": "none",
            "temperature": float(cfg.calibration.temperature),
            "threshold": float(cfg.evaluation.threshold),
            "calibrated": False,
        }
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["calibrated"] = True
    return data
