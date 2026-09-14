#!/usr/bin/env python
"""Reproduce and quantify the known failure mode: real photographs that are
out-of-distribution for the CIFAKE-trained baseline get misclassified as
AI-generated.

The baseline model (src/models/best_efficientnet_b0.pth) was trained only on
CIFAKE, whose REAL class is 32x32 CIFAR-10 images. High-resolution real
photographs fall outside that distribution and are frequently assigned a high
P(AI-generated).

This script uses license-clean, non-identifiable real photographs bundled with
scientific Python (scikit-image / scikit-learn / scipy), so it can run with no
external data.  For your own images, point it at a folder with `--images`.

Usage:
    python scripts/reproduce_real_photo_false_positives.py
    python scripts/reproduce_real_photo_false_positives.py --images path/to/real_photos
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

# Allow running directly (python scripts/<file>.py) from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from app.inference import SignalScopePredictor


def _bundled_real_photos() -> dict:
    """Return {name: RGB ndarray} of real photographs bundled with scipy-ecosystem."""
    photos = {}

    from skimage import data
    photos["coffee"] = np.asarray(data.coffee())            # real photo, coffee cup
    photos["horse"] = np.asarray(data.horse())              # real photo, horse
    photos["rocket"] = np.asarray(data.rocket())            # real photo, rocket launch
    photos["coins"] = np.stack([np.asarray(data.coins())] * 3, axis=-1)
    photos["moon"] = np.stack([np.asarray(data.moon())] * 3, axis=-1)
    photos["chelsea"] = np.asarray(data.chelsea())          # real photo, cat

    from sklearn.datasets import load_sample_image
    photos["china"] = load_sample_image("china.jpg")        # real photo, garden/landscape
    photos["flower"] = load_sample_image("flower.jpg")      # real photo, flower

    try:
        from scipy.datasets import face
        photos["raccoon"] = np.asarray(face())              # real photo, raccoon
    except Exception:
        pass  # raccoon face may require a network fetch; skip if unavailable

    return photos


def _encode(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=None,
                        help="Optional directory of your own real photos to test")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    predictor = SignalScopePredictor(checkpoint_path=args.checkpoint)

    rows = []
    sources: dict = {}
    if args.images:
        for p in sorted(Path(args.images).glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                sources[p.name] = np.asarray(Image.open(p).convert("RGB"))
    sources.update(_bundled_real_photos())

    print(f"{'image':22s} {'size':12s} {'P(AI)':>8s}  verdict")
    print("-" * 60)
    for name, arr in sources.items():
        result = predictor.predict(_encode(arr), filename=f"{name}.png",
                                   include_explanation=False)
        h, w = arr.shape[:2]
        print(f"{name:22s} {f'{w}x{h}':12s} {result['probability_ai']:8.4f}  {result['verdict_label']}")
        rows.append({
            "image": name,
            "width": w,
            "height": h,
            "ground_truth": "REAL",
            "probability_ai": round(result["probability_ai"], 4),
            "verdict": result["verdict"],
            "binary_label": result["binary_label"],
            "misclassified": int(result["binary_label"] == "AI_GENERATED"),
        })

    out = Path("reports/real_photo_false_positives.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    wrong = [r for r in rows if r["misclassified"]]
    print("-" * 60)
    print(f"Misclassified as AI: {len(wrong)}/{len(rows)} real photographs")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
