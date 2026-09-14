"""Evaluate a trained SignalScope model on the frozen CIFAKE test split.

Produces:
    reports/evaluation_results.json  (metrics + confusion matrix)
    reports/roc_curve.png            (ROC curve)

The CIFAKE test split is a *public* benchmark, not the official SIH held-out
set.  Results here must never be presented as the official score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from src.data.loaders import create_dataloaders
from src.evaluation.metrics import compute_metrics, format_metrics
from src.models.model import create_model, load_checkpoint


def run_inference(model, loader, device):
    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def save_roc_curve(y_true, y_prob, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"SignalScope (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — public CIFAKE test split")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on the public CIFAKE test split.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = args.threshold if args.threshold is not None else float(cfg.evaluation.threshold)

    _, _, test_loader = create_dataloaders()

    checkpoint = args.checkpoint or str(cfg.resolve(cfg.paths.checkpoint))
    model = create_model(num_classes=int(cfg.data.num_classes), pretrained=False)
    load_checkpoint(model, checkpoint, device=device)

    y_true, y_prob = run_inference(model, test_loader, device)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)

    print("\n" + "=" * 58)
    print("CIFAKE TEST RESULTS (public benchmark — not the SIH score)")
    print("=" * 58)
    print(format_metrics(metrics))

    reports_dir = cfg.resolve(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "evaluation_results.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    save_roc_curve(y_true, y_prob, reports_dir / "roc_curve.png")

    print(f"\nSaved: {reports_dir / 'evaluation_results.json'}")
    print(f"Saved: {reports_dir / 'roc_curve.png'}")


if __name__ == "__main__":
    main()
