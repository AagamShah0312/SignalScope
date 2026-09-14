#!/usr/bin/env python
"""Run the degradation (robustness) benchmark on a handful of demo images.

This produces honest per-image robustness numbers without needing the large
public datasets: it runs the shipped ensemble on the three committed
AI-generated demo images plus four license-clean real photographs bundled with
scikit-image (in memory, no network), under every standard degradation
(JPEG, resize, blur, noise, brightness/contrast, crop, screenshot).

Output: reports/robustness_demo_results.csv

This is a *per-image demonstration* — the dataset-level benchmark
(scripts/evaluate.py --robustness / src.evaluation.evaluate_robustness)
still requires the public data.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from app.inference import SignalScopePredictor
from src.robustness.degradations import default_degradations

REPO = Path(__file__).resolve().parent.parent


def _load_sources():
    sources: dict = {}
    # Committed AI-generated demo images.
    for name in ["ai_ceramic_mug", "ai_landscape", "ai_abstract_art"]:
        p = REPO / "demo" / "ai_generated" / f"{name}.jpg"
        sources[f"{name} (AI)"] = (1, Image.open(p).convert("RGB"))

    # Real photographs bundled with scikit-image (no network needed).
    from skimage import data
    for name in ["astronaut", "coffee", "chelsea", "horse"]:
        arr = np.asarray(getattr(data, name)())
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        sources[f"{name} (REAL)"] = (0, Image.fromarray(arr.astype(np.uint8)).convert("RGB"))
    return sources


def _encode(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    predictor = SignalScopePredictor()
    conditions = default_degradations()
    sources = _load_sources()

    rows = []
    for label, (gt, original) in sources.items():
        for cond_name, fn in conditions.items():
            try:
                degraded = fn(original)
                result = predictor.predict(_encode(degraded), filename="demo.png",
                                           include_explanation=False, include_metadata=False)
                rows.append({
                    "image": label,
                    "ground_truth": "AI" if gt == 1 else "REAL",
                    "condition": cond_name,
                    "probability_ai": round(result["probability_ai"], 4),
                    "binary_label": result["binary_label"],
                    "verdict": result["verdict"],
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "image": label,
                    "ground_truth": "AI" if gt == 1 else "REAL",
                    "condition": cond_name,
                    "probability_ai": "",
                    "binary_label": "ERROR",
                    "verdict": str(exc),
                })

    out = REPO / "reports" / "robustness_demo_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary: how often the binary label flips from the clean image.
    print(f"{'image':22s} {'clean P(AI)':>11s} {'flips under degradation':>24s}")
    print("-" * 60)
    for label, (gt, original) in sources.items():
        clean = predictor.predict(_encode(original), filename="demo.png",
                                  include_explanation=False, include_metadata=False)
        base_label = clean["binary_label"]
        base_p = clean["probability_ai"]
        flips = 0
        for r in rows:
            if r["image"] == label and r["binary_label"] not in ("ERROR", base_label):
                flips += 1
        print(f"{label:22s} {base_p:11.4f} {flips:>20d}/{len(conditions)}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
