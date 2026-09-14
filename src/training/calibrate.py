"""Fit temperature scaling + threshold on validation data (CLI)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from src.data.loaders import create_dataloaders, create_mixed_dataloaders
from src.models.model import create_model, load_checkpoint
from src.training.calibration import (
    calibration_quality,
    fit_calibration,
    save_calibration,
)


def collect_val_logits(model, val_loader, device):
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, y in tqdm(val_loader, desc="Collecting validation logits"):
            images = images.to(device, non_blocking=True)
            out = model(images)
            logits.append(out.cpu().numpy())
            labels.extend(y.numpy())
    return np.concatenate(logits), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Calibrate on validation data.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", choices=["cifake", "mixed"], default="cifake")
    parser.add_argument("--method", choices=["temperature_scaling", "isotonic"],
                        default="temperature_scaling")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mixed":
        _, val_loader = create_mixed_dataloaders()
    else:
        _, val_loader, _ = create_dataloaders()

    checkpoint = args.checkpoint or str(cfg.resolve(cfg.paths.checkpoint))
    model = create_model(num_classes=int(cfg.data.num_classes), pretrained=False)
    load_checkpoint(model, checkpoint, device=device)

    val_logits, val_labels = collect_val_logits(model, val_loader, device)
    calibrator, threshold = fit_calibration(val_logits, val_labels, method=args.method)

    y_prob = calibrator.calibrate(val_logits)[:, 1]
    quality = calibration_quality(val_labels, y_prob)
    quality["threshold"] = float(threshold)
    if hasattr(calibrator, "temperature"):
        quality["temperature"] = calibrator.temperature

    save_calibration(calibrator, threshold, args.method, quality)

    reports_dir = cfg.resolve(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "calibration_results.json", "w", encoding="utf-8") as fh:
        json.dump(quality, fh, indent=2)

    print("\nCalibration complete.")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
