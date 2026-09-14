"""Dataset-level robustness benchmark.

For each degradation condition, run the detector over a labelled image set and
aggregate accuracy / ROC-AUC / Macro-F1 / confidence change versus the original.

Produces ``reports/robustness_results.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.inference import SignalScopePredictor
from src.config import load_config
from src.robustness.degradations import apply_degradations, default_degradations


def main():
    parser = argparse.ArgumentParser(description="Robustness benchmark over a labelled manifest.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None,
                        help="Manifest CSV (default: Defactify eval manifest if present)")
    parser.add_argument("--limit", type=int, default=None, help="Max images to evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = cfg.resolve(cfg.data.defactify_eval_manifest)

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}. Provide --manifest.")

    df = pd.read_csv(manifest_path)
    if args.limit:
        df = df.head(args.limit)

    predictor = SignalScopePredictor(config_path=args.config)
    conditions = default_degradations()

    rows = []
    for idx, row in df.iterrows():
        path = Path(row["path"])
        if not path.exists():
            continue
        original = predictor.predict(path, include_explanation=False, include_metadata=False)
        base_p = original["probability_ai"]

        # original row
        rows.append({"image": path.name, "condition": "original", "label": int(row["label"]),
                     "probability_ai": base_p,
                     "correct": int((base_p >= original["threshold"]) == int(row["label"]))})

        from PIL import Image
        img = Image.open(path).convert("RGB")
        for name, degraded in apply_degradations(img, conditions).items():
            if isinstance(degraded, Exception):
                continue
            import io
            buf = io.BytesIO()
            degraded.save(buf, format="PNG")
            pred = predictor.predict(buf.getvalue(), filename=path.name,
                                     include_explanation=False, include_metadata=False)
            p = pred["probability_ai"]
            rows.append({"image": path.name, "condition": name, "label": int(row["label"]),
                         "probability_ai": p,
                         "correct": int((p >= pred["threshold"]) == int(row["label"]))})

    results = pd.DataFrame(rows)
    if results.empty:
        raise SystemExit("No images evaluated.")

    from src.evaluation.metrics import compute_metrics

    summary = []
    for condition, group in results.groupby("condition"):
        labels = group["label"].values
        probs = group["probability_ai"].values
        m = compute_metrics(labels, probs)
        summary.append({
            "condition": condition,
            "accuracy": round(m["accuracy"], 4),
            "auc": round(m["roc_auc"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "fpr": round(m["fpr"], 4),
            "mean_probability_ai": round(float(np.mean(probs)), 4),
            "confidence_delta_vs_original": round(
                float(group["probability_ai"].mean() - results.loc[results["condition"] == "original",
                                                                  "probability_ai"].mean()), 4),
        })

    out = cfg.resolve(cfg.paths.reports_dir) / "robustness_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
