"""Leave-one-generator-out cross-generator evaluation.

This is the most important public experiment for the SignalScope task, whose
core challenge is generalisation to **unseen generators**.

For each AI generator present in the public Defactify evaluation set, we:

1. train a fresh EfficientNet-B0 on CIFAKE + Defactify **excluding** that
   generator's training images,
2. evaluate on that generator's held-out images (plus real images so that
   ROC-AUC and FPR are well-defined),
3. record ROC-AUC, Macro-F1, accuracy, FPR and threshold.

Results are written to ``reports/unseen_generator_results.csv`` and clearly
labelled as a **public cross-generator experiment** — NOT the official SIH
held-out score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import load_config
from src.data.loaders import create_mixed_dataloaders
from src.data.manifests import AI_GENERATORS, load_defactify_manifest
from src.data.transforms import get_eval_transforms
from src.evaluation.metrics import compute_metrics
from src.models.model import create_model
from src.training.reproducibility import seed_everything
from src.training.train import _build_optimizer, train_model


def _evaluate_on_generator(model, df, device, image_size, threshold=0.5):
    """Run the model over the eval rows for one generator + real images."""
    transform = get_eval_transforms(image_size=image_size)
    labels, probs = [], []
    model.eval()
    with torch.no_grad():
        for _, row in df.iterrows():
            path = Path(row["path"])
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            outputs = model(tensor)
            p = torch.softmax(outputs, dim=1)[0, 1].item()
            labels.append(int(row["label"]))
            probs.append(p)
    if not labels:
        return None
    metrics = compute_metrics(np.array(labels), np.array(probs), threshold=threshold)
    metrics["n"] = len(labels)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Leave-one-generator-out evaluation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per fold (default from config)")
    parser.add_argument("--generators", nargs="*", default=None,
                        help="Subset of generators to run (default: all available)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    epochs = args.epochs if args.epochs is not None else int(cfg.training.epochs)
    image_size = int(cfg.data.image_size)
    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)
    threshold = float(cfg.evaluation.threshold)

    # Defactify eval manifest (contains per-generator labels + real images).
    eval_df = load_defactify_manifest("eval")
    available = sorted(set(eval_df["generator"]) - {"real"})
    available = [g for g in AI_GENERATORS if g in available]
    generators = args.generators or available
    if not generators:
        raise SystemExit("No Defactify AI generators found. Run the Defactify downloader first.")

    reports_dir = cfg.resolve(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for unseen in generators:
        print("\n" + "=" * 62)
        print(f"Experiment: unseen generator = {unseen}")
        print("=" * 62)

        seed_everything(int(cfg.training.seed))

        train_loader, val_loader = create_mixed_dataloaders(
            batch_size=batch_size, num_workers=num_workers, exclude_generator=unseen
        )

        model = create_model(num_classes=int(cfg.data.num_classes), pretrained=True).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = _build_optimizer(str(cfg.training.optimizer).lower(), model.parameters(),
                                     float(cfg.training.learning_rate), float(cfg.training.weight_decay))

        tmp_ckpt = reports_dir / f".tmp_ckpt_{unseen}.pth"
        train_model(model, train_loader, val_loader, criterion, optimizer, device,
                    epochs=epochs, use_freq=False, selection_metric="val_auc",
                    checkpoint_path=tmp_ckpt, quiet=False)

        # Evaluate on the unseen generator's images + real images (for AUC/FPR).
        eval_subset = eval_df[(eval_df["generator"] == unseen) | (eval_df["generator"] == "real")]
        metrics = _evaluate_on_generator(model, eval_subset, device, image_size, threshold)
        if tmp_ckpt.exists():
            tmp_ckpt.unlink()

        if metrics is None:
            print(f"  No evaluation images found for {unseen}; skipping.")
            continue

        print(f"  Unseen {unseen}: AUC {metrics['roc_auc']:.4f} | "
              f"F1 {metrics['macro_f1']:.4f} | Acc {metrics['accuracy']:.4f} | "
              f"FPR {metrics['fpr']:.4f}")
        rows.append({
            "experiment": f"train_without_{unseen}",
            "unseen_generator": unseen,
            "auc": round(metrics["roc_auc"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "fpr": round(metrics["fpr"], 4),
            "threshold": threshold,
            "n": metrics["n"],
        })

    if rows:
        results = pd.DataFrame(rows)
        out_csv = reports_dir / "unseen_generator_results.csv"
        results.to_csv(out_csv, index=False)
        with open(reports_dir / "unseen_generator_results.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print("\nSaved public cross-generator results to:")
        print(f"  {out_csv}")
        print("\nNOTE: these are PUBLIC cross-generator experiments, not the official SIH score.")

    else:
        print("No experiments completed.")


if __name__ == "__main__":
    main()
