"""Centralised evaluation metrics for binary real-vs-AI classification.

Label convention (kept consistent across the whole project):
    0 = REAL
    1 = AI_GENERATED
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_numpy(y):
    return np.asarray(y).reshape(-1)


def _auc(y_true, y_prob) -> float:
    y_true = _as_numpy(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _fpr_from_cm(cm) -> float:
    """False positive rate = FP / (FP + TN), safe against zero denominators."""
    tn, fp = cm[0, 0], cm[0, 1]
    denom = fp + tn
    return float(fp / denom) if denom > 0 else float("nan")


def _tpr_from_cm(cm) -> float:
    """True positive rate = TP / (TP + FN)."""
    fn, tp = cm[1, 0], cm[1, 1]
    denom = tp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def confusion_parts(cm) -> Dict[str, int]:
    """Return TN/FP/FN/TP counts from a 2x2 confusion matrix."""
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def compute_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
    include_confusion: bool = True,
) -> Dict[str, object]:
    """Compute the standard binary-classification metrics.

    Args:
        y_true: ground-truth labels (0 = REAL, 1 = AI_GENERATED).
        y_prob: probability of the positive class (AI_GENERATED).
        threshold: decision threshold applied to ``y_prob``.

    Returns a dict with: auc, macro_f1, accuracy, precision, recall, fpr, tpr,
    threshold and (optionally) confusion_matrix + tn/fp/fn/tp.
    """
    y_true = _as_numpy(y_true)
    y_prob = _as_numpy(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Empty inputs.")

    y_pred = (y_prob >= threshold).astype(int)

    auc = _auc(y_true, y_prob)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    parts = confusion_parts(cm)

    result: Dict[str, object] = {
        "roc_auc": auc,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": _fpr_from_cm(cm),
        "tpr": _tpr_from_cm(cm),
        "threshold": float(threshold),
    }
    if include_confusion:
        result["confusion_matrix"] = cm.tolist()
        result.update(parts)
    return result


def format_metrics(metrics: Dict[str, object]) -> str:
    """Pretty-print a metrics dict for the console."""
    lines = [
        f"ROC-AUC:   {metrics['roc_auc']:.4f}",
        f"Macro-F1:  {metrics['macro_f1']:.4f}",
        f"Accuracy:  {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall:    {metrics['recall']:.4f}",
        f"FPR:       {metrics['fpr']:.4f}",
        f"TPR:       {metrics['tpr']:.4f}",
        f"Threshold: {metrics['threshold']:.4f}",
    ]
    if "confusion_matrix" in metrics:
        lines.append("Confusion Matrix [[TN, FP], [FN, TP]]:")
        lines.append(f"  {metrics['confusion_matrix']}")
    return "\n".join(lines)
