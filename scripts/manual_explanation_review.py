#!/usr/bin/env python
"""Produce Grad-CAM regions + explanation text for 10 sample images for
MANUAL review (the user grades these themselves — this script only extracts
measured quantities and saves the overlays).

For each image it prints: factual description, verdict, P(AI), the strongest
Grad-CAM activation region (bbox + position label + relative area + mean
activation), attention concentration, and the generated explanation text.
Overlays/heatmaps are saved under reports/explanation_samples/manual_review/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from app.inference import SignalScopePredictor
from app.preprocessing import preprocess_image, to_numpy_rgb, validate_file
from src.explainability.evidence import (
    activation_concentration,
    region_position_label,
    top_activation_regions,
)
from src.explainability.gradcam import GradCAM, save_visualization

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "explanation_samples" / "manual_review"

SAMPLES = [
    # (label, path, factual description)
    ("real/coffee", "data/real_photos/bundled/coffee.png",
     "Photograph of a coffee mug (scikit-image sample)"),
    ("real/chelsea", "data/real_photos/bundled/chelsea.png",
     "Photograph of a cat (scikit-image sample)"),
    ("real/horse", "data/real_photos/bundled/horse.png",
     "Photograph of a horse (scikit-image sample)"),
    ("real/butterfly", "data/real_photos/opencv/butterfly.jpg",
     "Photograph of a butterfly (OpenCV sample data)"),
    ("real/fruits", "data/real_photos/opencv/fruits.jpg",
     "Photograph of fruits (OpenCV sample data)"),
    ("ai/ceramic_mug", "demo/ai_generated/ai_ceramic_mug.jpg",
     "AI-generated ceramic mug (project demo)"),
    ("ai/landscape", "demo/ai_generated/ai_landscape.jpg",
     "AI-generated landscape (project demo)"),
    ("ai/abstract_art", "demo/ai_generated/ai_abstract_art.jpg",
     "AI-generated abstract art (project demo)"),
    ("ai/forest", "data/ai_synthetic/ai_01_forest.jpg",
     "AI-generated forest scene (project training data)"),
    ("ai/cat", "data/ai_synthetic/ai_07_cat.jpg",
     "AI-generated cat (project training data)"),
]


def main() -> None:
    predictor = SignalScopePredictor()
    # Use the same model the predictor uses for Grad-CAM (ensemble primary).
    model = predictor.model
    device = predictor.device
    image_size = predictor.image_size

    report = []
    for label, rel_path, desc in SAMPLES:
        path = REPO / rel_path
        result = predictor.predict(str(path), include_explanation=True,
                                   include_metadata=False,
                                   output_dir=str(OUT / label),
                                   base_name="gradcam")

        # Recompute the raw CAM to extract the actual region boxes.
        image = validate_file(path.read_bytes(), filename=path.name)
        tensor = preprocess_image(image, image_size).to(device)
        gradcam = GradCAM(model)
        try:
            cam, _, _ = gradcam.generate(tensor)
        finally:
            gradcam.remove_hooks()
        cam_np = cam.numpy()
        orig_np = to_numpy_rgb(image)
        regions = top_activation_regions(cam_np, orig_np.shape[:2])
        for r in regions:
            r["position"] = region_position_label(r["bbox"], orig_np.shape[:2])
        conc = activation_concentration(cam_np)

        entry = {
            "label": label,
            "description": desc,
            "path": rel_path,
            "verdict": result["verdict_label"],
            "p_ai": round(result["probability_ai"], 4),
            "strongest_region": regions[0] if regions else None,
            "n_regions": len(regions),
            "concentration": conc,
            "explanation_summary": result["explanation"]["summary"],
            "explanation_evidence": result["explanation"]["evidence"],
        }
        report.append(entry)

        print(f"\n{'=' * 70}\n{label}  ({desc})\n{'=' * 70}")
        print(f"verdict={result['verdict_label']}  P(AI)={result['probability_ai']:.4f}  "
              f"status={result['status']}")
        if regions:
            r0 = regions[0]
            print(f"strongest region: bbox={r0['bbox']}  position={r0['position']}  "
                  f"rel_area={r0['relative_area']:.3f}  mean_act={r0['mean_activation']}")
        else:
            print("strongest region: (none above threshold — diffuse attention)")
        print(f"concentration: top5%={conc['top5pct_concentration']}  "
              f"mean={conc['mean_activation']:.4f}  max={conc['max_activation']:.4f}")
        print("explanation:")
        for e in result["explanation"]["evidence"]:
            print(f"  - {e}")
        print(f"  summary: {result['explanation']['summary']}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "review.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved overlays under {OUT}/<label>/ and review.json")


if __name__ == "__main__":
    main()
