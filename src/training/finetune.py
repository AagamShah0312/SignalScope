"""Fine-tune the SignalScope detector to fix real-photograph false positives.

Rationale (measured): the shipped CIFAKE-trained baseline assigns P(AI) > 0.98
to ordinary real photographs because CIFAKE's REAL class is 32x32 CIFAR-10
images.  This script adapts the model on a small balanced set of real
photographs (bundled scientific-Python samples + OpenCV sample data) and
synthetic images, in two stages:

    1. retrain only the classification head (backbone frozen), then
    2. unfreeze the last two EfficientNet blocks for a short fine-tune.

This is a *quick adaptation*, clearly documented, not a substitute for
full-scale CIFAKE+Defactify training (which needs the public datasets + GPU).

Usage:
    python -m src.training.finetune
    python -m src.training.finetune --epochs-head 15 --epochs-finetune 5
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

from src.config import REPO_ROOT, load_config
from src.data.dataset import SignalScopeDataset
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.evaluation.metrics import compute_metrics
from src.models.model import create_model, load_checkpoint
from src.training.reproducibility import seed_everything
from src.training.tracking import log_experiment


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--base-checkpoint", default=None,
                   help="Baseline checkpoint to adapt (default: config checkpoint)")
    p.add_argument("--train-manifest", default="data/processed/finetune_train.csv")
    p.add_argument("--val-manifest", default="data/processed/finetune_val.csv")
    p.add_argument("--epochs-head", type=int, default=15)
    p.add_argument("--epochs-finetune", type=int, default=5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default=None, help="Output checkpoint path")
    return p.parse_args()


def _run_epoch(model, loader, criterion, device, optimizer=None, training=True):
    if training:
        model.train()
    else:
        model.eval()
    running_loss, labels, probs = 0.0, [], []
    for images, y in tqdm(loader, desc="train" if training else "val", leave=False):
        images, y = images.to(device), y.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            out = model(images)
            loss = criterion(out, y)
            if training:
                loss.backward()
                optimizer.step()
        running_loss += loss.item() * images.size(0)
        labels.extend(y.detach().cpu().numpy())
        probs.extend(torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy())
    m = compute_metrics(np.array(labels), np.array(probs))
    m["loss"] = running_loss / len(loader.dataset)
    return m


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.training.seed)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(cfg.data.image_size)

    # ---- data ----
    train_ds = SignalScopeDataset(args.train_manifest, transform=get_train_transforms(
        image_size=image_size, baseline=True, stronger=False))
    val_ds = SignalScopeDataset(args.val_manifest, transform=get_eval_transforms(image_size))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    # ---- model ----
    from src.models.model import resolve_checkpoint

    base = args.base_checkpoint or str(resolve_checkpoint(cfg.paths.checkpoint))
    model = create_model(num_classes=int(cfg.data.num_classes), pretrained=False)
    load_checkpoint(model, base, device=device)

    # Replace the CIFAKE-biased head with a fresh one.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, int(cfg.data.num_classes)).to(device)

    # Freeze backbone; train head only.
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    def make_opt(lr):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=1e-4)

    # ---- stage 1: head only ----
    opt = make_opt(args.lr_head)
    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs_head + 1):
        _run_epoch(model, train_loader, criterion, device, opt, True)
        vm = _run_epoch(model, val_loader, criterion, device, None, False)
        print(f"[head] epoch {epoch:2d}/{args.epochs_head}  val_acc={vm['accuracy']:.4f} "
              f"val_auc={vm['roc_auc']:.4f}  loss={vm['loss']:.4f}")
        if vm["accuracy"] >= best_acc:
            best_acc = vm["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- stage 2: unfreeze last two blocks + head ----
    for name, param in model.features.named_parameters():
        param.requires_grad = name.startswith(("7.", "6.")) or name.startswith("8.")
    for param in model.classifier.parameters():
        param.requires_grad = True

    opt = make_opt(args.lr_finetune)
    for epoch in range(1, args.epochs_finetune + 1):
        _run_epoch(model, train_loader, criterion, device, opt, True)
        vm = _run_epoch(model, val_loader, criterion, device, None, False)
        print(f"[finetune] epoch {epoch:2d}/{args.epochs_finetune}  val_acc={vm['accuracy']:.4f} "
              f"val_auc={vm['roc_auc']:.4f}  loss={vm['loss']:.4f}")
        if vm["accuracy"] >= best_acc:
            best_acc = vm["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- save ----
    out = Path(args.output) if args.output else REPO_ROOT / "src" / "models" / "fine_tuned_model.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "base_checkpoint": base,
                "val_accuracy": best_acc}, out)

    metadata = {
        "architecture": "EfficientNet-B0 (fine-tuned for real-photo adaptation)",
        "pretrained": True,
        "base_checkpoint": base,
        "image_size": image_size,
        "classes": ["REAL", "AI_GENERATED"],
        "seed": seed,
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
        "val_accuracy": best_acc,
        "epochs_head": args.epochs_head,
        "epochs_finetune": args.epochs_finetune,
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(out),
    }
    with open(out.parent / "fine_tuned_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    log_experiment({
        "experiment_name": "finetune-real-photo-adaptation",
        "dataset": "real_photos+ai_synthetic(small)",
        "train_size": len(train_ds),
        "validation_size": len(val_ds),
        "model": metadata["architecture"],
        "pretrained": True,
        "epochs": args.epochs_head + args.epochs_finetune,
        "batch_size": args.batch_size,
        "optimizer": "adamw",
        "augmentation": "baseline",
        "frequency_features": False,
        "calibration": "none",
        "seed": seed,
        "val_accuracy": best_acc,
        "notes": "head retrain + last-2-block finetune to fix real-photo false positives",
    })

    print(f"\nSaved fine-tuned model: {out}  (val_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
