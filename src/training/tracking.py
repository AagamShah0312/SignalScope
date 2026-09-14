"""Lightweight experiment logging (CSV append) — prevents random experimentation.

Each run appends a row to ``reports/experiments.csv`` so that every training
configuration and its result are recorded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import load_config

COLUMNS = [
    "experiment_name",
    "timestamp",
    "dataset",
    "train_size",
    "validation_size",
    "test_size",
    "model",
    "pretrained",
    "epochs",
    "learning_rate",
    "batch_size",
    "optimizer",
    "weight_decay",
    "augmentation",
    "frequency_features",
    "calibration",
    "seed",
    "val_auc",
    "val_loss",
    "val_macro_f1",
    "val_accuracy",
    "notes",
]


def log_experiment(record: Dict[str, object], path: str = "reports/experiments.csv") -> None:
    """Append one experiment record to the CSV log."""
    cfg = load_config()
    output = Path(path)
    if not output.is_absolute():
        output = cfg.resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    row = {col: record.get(col, "") for col in COLUMNS}
    row.setdefault("timestamp", pd.Timestamp.now().isoformat(timespec="seconds"))

    if output.exists() and os.path.getsize(output) > 0:
        existing = pd.read_csv(output)
    else:
        existing = pd.DataFrame(columns=COLUMNS)

    # Reconcile columns in case the schema evolves.
    for col in COLUMNS:
        if col not in existing.columns:
            existing[col] = ""

    new_row = pd.DataFrame([row], columns=COLUMNS)
    existing = pd.concat([existing, new_row], ignore_index=True)
    existing.to_csv(output, index=False)
