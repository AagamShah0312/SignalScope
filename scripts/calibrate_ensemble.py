#!/usr/bin/env python
"""Fit temperature scaling on locally-available held-out data (the ensemble).

The standard CLI (`src/training/calibrate.py --dataset mixed`) requires the
CIFAKE + Defactify datasets, which are not downloadable in this environment.
This script does the same thing against the small real-photo adaptation
validation set (data/processed/finetune_val.csv, 12 real + 4 AI), using the
**same ensemble that serves inference** (config model.ensemble), so the fitted
temperature is applied to the exact logits the API produces.

The result is deliberately labelled as provisional: it is fitted on a 16-image
public adaptation set, NOT the official CIFAKE validation split. Re-fit with
`python -m src.training.calibrate --dataset mixed` once the datasets are local.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import load_config
from src.data.transforms import get_eval_transforms
from src.models.ensemble import EnsembleClassifier
from src.models.model import create_model, load_checkpoint
from src.training.calibration import (
    TemperatureScaler,
    calibration_quality,
    save_calibration,
    select_threshold_youden,
)

REPO = Path(__file__).resolve().parent.parent
VAL_CSV = REPO / "data" / "processed" / "finetune_val.csv"


def build_ensemble(cfg, device):
    members = cfg.model.get("ensemble", {}).get("members", [])
    if len(members) < 2:
        raise SystemExit("Config model.ensemble needs >= 2 members.")
    models, weights = [], []
    for member in members:
        path = cfg.resolve(member["checkpoint"])
        if not path.exists():
            raise SystemExit(f"Ensemble member missing: {path}")
        model = create_model(num_classes=int(cfg.data.num_classes), pretrained=False)
        load_checkpoint(model, str(path), device=device)
        model.eval()
        models.append(model)
        weights.append(float(member.get("weight", 1.0)))
    return EnsembleClassifier(models, weights=weights, device=device)


def collect_ensemble_logits(ensemble, val_df, image_size, device):
    transform = get_eval_transforms(image_size=image_size)
    logits, labels = [], []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            image = Image.open(row["path"]).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits.append(ensemble.logits(tensor).cpu().numpy())
            labels.append(int(row["label"]))
    return np.concatenate(logits, axis=0), np.array(labels)


def main() -> None:
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(cfg.data.image_size)

    if not VAL_CSV.exists():
        raise SystemExit(f"{VAL_CSV} not found. Run scripts/build_finetune_dataset.py first.")

    val_df = pd.read_csv(VAL_CSV)
    ensemble = build_ensemble(cfg, device)
    logits, labels = collect_ensemble_logits(ensemble, val_df, image_size, device)

    # Raw (uncalibrated) probabilities and quality.
    raw_probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()[:, 1]
    before = calibration_quality(labels, raw_probs)
    print(f"Validation set: {len(labels)} images ({int((labels == 0).sum())} real, "
          f"{int((labels == 1).sum())} AI)")
    print(f"BEFORE (T=1.0): {json.dumps(before, indent=2)}")

    # Fit temperature on the held-out logits.
    scaler = TemperatureScaler().fit(logits, labels)
    cal_probs = scaler.calibrate(logits)[:, 1]
    after = calibration_quality(labels, cal_probs)
    threshold = select_threshold_youden(labels, cal_probs)

    print(f"\nFitted temperature T = {scaler.temperature:.4f}")
    print(f"AFTER  (T={scaler.temperature:.4f}): {json.dumps(after, indent=2)}")
    print(f"Youden-J threshold on val: {threshold:.4f}")

    quality = dict(after)
    quality["brier_score_before"] = before["brier_score"]
    quality["ece_before"] = before["expected_calibration_error"]
    save_calibration(scaler, threshold, "temperature_scaling", quality)
    print("\nWrote model/calibration.json")


if __name__ == "__main__":
    main()
