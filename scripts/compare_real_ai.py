#!/usr/bin/env python
"""Compare checkpoints on real photos + AI images (public adaptation eval set).

Reports the real-photo false-positive rate and the AI-image detection rate for
the CIFAKE baseline, the fine-tuned model, and the config-driven ensemble, and
writes ``reports/real_photo_adaptation.csv``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.inference import SignalScopePredictor

REPO = Path(__file__).resolve().parent.parent
REAL_DIRS = [REPO / "data" / "real_photos"]
AI_DIRS = [REPO / "data" / "ai_synthetic"]
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
BASELINE = str(REPO / "src" / "models" / "best_efficientnet_b0.pth")
FINETUNED = str(REPO / "src" / "models" / "fine_tuned_model.pth")


def images_in(dirs):
    out = []
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in EXT:
                out.append(p)
    return out


def probabilities(predictor, image_paths):
    return np.array([predictor.predict(str(p), include_explanation=False)["probability_ai"]
                     for p in image_paths])


def main() -> None:
    real = images_in(REAL_DIRS)
    ai = images_in(AI_DIRS)

    baselines = {
        "baseline": SignalScopePredictor(checkpoint_path=BASELINE),
        "fine_tuned": SignalScopePredictor(checkpoint_path=FINETUNED),
        "ensemble": SignalScopePredictor(),  # config-driven ensemble
    }

    rows = []
    print(f"{'model':12s} {'real_FP%':>9s} {'AI_TP%':>8s} {'balanced%':>10s} {'meanP(AI)_real':>14s}")
    print("-" * 60)
    for name, pred in baselines.items():
        rp = probabilities(pred, real)
        ap = probabilities(pred, ai)
        fp = (rp >= 0.5).mean() * 100
        tp = (ap >= 0.5).mean() * 100
        print(f"{name:12s} {fp:8.1f}% {tp:7.1f}% {(100 - fp + tp) / 2:9.1f}% {rp.mean():14.4f}")
        rows.append({
            "model": name,
            "n_real": len(real),
            "n_ai": len(ai),
            "real_false_positive_pct": round(float(fp), 2),
            "ai_detection_pct": round(float(tp), 2),
            "balanced_accuracy_pct": round(float((100 - fp + tp) / 2), 2),
            "mean_p_ai_on_real": round(float(rp.mean()), 4),
            "mean_p_ai_on_ai": round(float(ap.mean()), 4),
        })

    out = REPO / "reports" / "real_photo_adaptation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
