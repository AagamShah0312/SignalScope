"""SignalScope training entrypoint (config-driven).

All important hyperparameters come from ``config.yaml`` and can be overridden
on the command line.  Usage:

    python -m src.training.train                       # defaults from config.yaml
    python -m src.training.train --dataset mixed        # CIFAKE + Defactify
    python -m src.training.train --epochs 10 --batch-size 32
    python -m src.training.train --use-frequency-features
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import load_config
from src.data.loaders import create_dataloaders, create_mixed_dataloaders
from src.evaluation.metrics import compute_metrics
from src.models.frequency_features import extract_frequency_features
from src.models.fusion_model import create_fusion_model
from src.models.model import create_model, get_device
from src.training.losses import build_criterion
from src.training.reproducibility import seed_everything
from src.training.tracking import log_experiment


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the SignalScope detector.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--dataset", choices=["cifake", "mixed"], default=None,
                        help="cifake (CIFAKE only) or mixed (CIFAKE + Defactify)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-frequency-features", action="store_true", default=None)
    parser.add_argument("--no-frequency-features", action="store_true", default=None)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default=None)
    parser.add_argument("--output-dir", default=None, help="Override model_dir")
    parser.add_argument("--tag", default=None, help="Human-readable experiment name")
    return parser.parse_args()


def _run_epoch(model, loader, criterion, device, optimizer=None, training=True, use_freq=False):
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels: list = []
    all_probs: list = []

    for batch in tqdm(loader, desc="Training" if training else "Validation", leave=False):
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            if use_freq:
                freq = extract_frequency_features(images)
                outputs = model(images, freq)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            if training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = float(epoch_loss)
    return metrics


def _build_optimizer(name, parameters, lr, weight_decay):
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    use_freq,
    selection_metric="val_auc",
    checkpoint_path=None,
    quiet=False,
):
    """Shared training loop (used by the CLI and leave-one-generator-out runs).

    Returns (best_score, best_epoch, history).
    """
    best_score = -1.0 if ("auc" in selection_metric or "f1" in selection_metric) else float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        if not quiet:
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)

        train_m = _run_epoch(model, train_loader, criterion, device, optimizer, True, use_freq)
        val_m = _run_epoch(model, val_loader, criterion, device, None, False, use_freq)

        if not quiet:
            print(
                f"Train | loss {train_m['loss']:.4f} | AUC {train_m['roc_auc']:.4f} | "
                f"F1 {train_m['macro_f1']:.4f} | Acc {train_m['accuracy']:.4f}"
            )
            print(
                f"Val   | loss {val_m['loss']:.4f} | AUC {val_m['roc_auc']:.4f} | "
                f"F1 {val_m['macro_f1']:.4f} | Acc {val_m['accuracy']:.4f}"
            )

        history.append({"epoch": epoch, "train": train_m, "validation": val_m})

        if selection_metric == "val_loss":
            score, is_better = val_m["loss"], val_m["loss"] < best_score
        elif selection_metric == "val_macro_f1":
            score, is_better = val_m["macro_f1"], val_m["macro_f1"] > best_score
        else:
            score, is_better = val_m["roc_auc"], val_m["roc_auc"] > best_score

        if is_better:
            best_score = score
            best_epoch = epoch
            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "selection_metric": selection_metric,
                        "best_score": best_score,
                    },
                    checkpoint_path,
                )

    return best_score, best_epoch, history


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Resolve configuration (CLI overrides -> config.yaml)
    # ------------------------------------------------------------------
    dataset_name = args.dataset or "cifake"
    epochs = args.epochs if args.epochs is not None else int(cfg.training.epochs)
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.training.batch_size)
    lr = args.learning_rate if args.learning_rate is not None else float(cfg.training.learning_rate)
    wd = args.weight_decay if args.weight_decay is not None else float(cfg.training.weight_decay)
    seed = args.seed if args.seed is not None else int(cfg.training.seed)
    optimizer_name = args.optimizer or str(cfg.training.optimizer).lower()
    num_workers = int(cfg.training.num_workers)

    if args.use_frequency_features:
        use_freq = True
    elif args.no_frequency_features:
        use_freq = False
    else:
        use_freq = bool(cfg.model.use_frequency_features)

    pretrained = bool(cfg.model.pretrained)
    image_size = int(cfg.data.image_size)
    dropout = float(cfg.model.dropout)
    selection_metric = str(cfg.training.selection_metric).lower()
    aug = cfg.training.augmentation
    augmentation_label = (
        "stronger" if bool(aug.get("stronger", False))
        else "baseline" if bool(aug.get("baseline", True))
        else "none"
    )

    model_dir = cfg.resolve(args.output_dir) if args.output_dir else cfg.resolve(cfg.paths.model_dir)
    exp_dir = cfg.resolve(cfg.paths.experiments_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed_everything(seed)

    device = get_device()
    print("=" * 62)
    print("SignalScope Training")
    print("=" * 62)
    print(f"Device:       {device}")
    if device.type == "cuda":
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"Dataset:      {dataset_name}")
    print(f"Epochs:       {epochs}")
    print(f"Batch size:   {batch_size}")
    print(f"Learning rate:{lr}")
    print(f"Weight decay: {wd}")
    print(f"Optimizer:    {optimizer_name}")
    print(f"Seed:         {seed}")
    print(f"Freq features:{use_freq}")
    print(f"Augmentation: {augmentation_label}")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if dataset_name == "mixed":
        train_loader, val_loader = create_mixed_dataloaders(batch_size=batch_size, num_workers=num_workers)
    else:
        train_loader, val_loader, _ = create_dataloaders(batch_size=batch_size, num_workers=num_workers)

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    # ------------------------------------------------------------------
    # Model / loss / optimizer
    # ------------------------------------------------------------------
    if use_freq:
        model = create_fusion_model(num_classes=int(cfg.data.num_classes), pretrained=pretrained, dropout=dropout)
    else:
        model = create_model(num_classes=int(cfg.data.num_classes), pretrained=pretrained)
    model = model.to(device)

    criterion = build_criterion("cross_entropy")
    optimizer = _build_optimizer(optimizer_name, model.parameters(), lr, wd)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_score, best_epoch, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        use_freq=use_freq,
        selection_metric=selection_metric,
        checkpoint_path=model_dir / "best_model.pth",
    )

    # ------------------------------------------------------------------
    # Persist metadata / history
    # ------------------------------------------------------------------
    metadata = {
        "architecture": "EfficientNet-B0" + (" + frequency features" if use_freq else ""),
        "pretrained": pretrained,
        "image_size": image_size,
        "classes": ["REAL", "AI_GENERATED"],
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": wd,
        "optimizer": optimizer_name,
        "augmentation": augmentation_label,
        "frequency_features": use_freq,
        "selection_metric": selection_metric,
        "best_validation_score": float(best_score),
        "dataset": dataset_name,
        "train_size": train_size,
        "validation_size": val_size,
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": (model_dir / "best_model.pth").as_posix(),
    }
    with open(model_dir / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    with open(exp_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, default=float)

    # ------------------------------------------------------------------
    # Experiment log
    # ------------------------------------------------------------------
    tag = args.tag or f"{dataset_name}-{'freq' if use_freq else 'spatial'}-{augmentation_label}"
    log_experiment({
        "experiment_name": tag,
        "dataset": dataset_name,
        "train_size": train_size,
        "validation_size": val_size,
        "model": metadata["architecture"],
        "pretrained": pretrained,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "weight_decay": wd,
        "augmentation": augmentation_label,
        "frequency_features": use_freq,
        "calibration": "none",
        "seed": seed,
        "val_auc": float(history[-1]["validation"]["roc_auc"]),
        "val_loss": float(history[-1]["validation"]["loss"]),
        "val_macro_f1": float(history[-1]["validation"]["macro_f1"]),
        "val_accuracy": float(history[-1]["validation"]["accuracy"]),
        "notes": f"best_{selection_metric}={best_score:.4f} at epoch {best_epoch}",
    })

    print("\n" + "=" * 62)
    print("Training complete.")
    print(f"Best {selection_metric}: {best_score:.4f} (epoch {best_epoch})")
    print(f"Checkpoint: {model_dir / 'best_model.pth'}")
    print(f"Metadata:   {model_dir / 'model_metadata.json'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
