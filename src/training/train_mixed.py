from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from src.data.loaders_mixed import create_mixed_dataloaders
from src.models.model import create_model


# ==============================================================
# Configuration
# ==============================================================

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path(
    "model/best_efficientnet_b0_mixed.pth"
)

HISTORY_PATH = Path(
    "experiments/runs/mixed_training_history.json"
)


# ==============================================================
# Metrics
# ==============================================================

def calculate_metrics(labels, probabilities):

    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(
        labels,
        probabilities
    )

    f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "auc": float(auc),
        "macro_f1": float(f1),
        "accuracy": float(accuracy),
    }


# ==============================================================
# One epoch
# ==============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    training=True,
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    all_labels = []
    all_probabilities = []

    progress = tqdm(
        loader,
        desc="Training" if training else "Validation"
    )

    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            if training:
                loss.backward()
                optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.detach().cpu().numpy()
        )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    all_labels = np.array(
        all_labels
    )

    all_probabilities = np.array(
        all_probabilities
    )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities
    )

    metrics["loss"] = float(
        average_loss
    )

    return metrics


# ==============================================================
# Main
# ==============================================================

def main():

    print("\n" + "=" * 60)
    print("SignalScope - Mixed Dataset Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    print("=" * 60 + "\n")

    # ----------------------------------------------------------
    # Data
    # ----------------------------------------------------------

    train_loader, val_loader = (
        create_mixed_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    model = create_model(
        num_classes=2,
        pretrained=True,
    )

    model = model.to(DEVICE)

    # ----------------------------------------------------------
    # Loss + optimizer
    # ----------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    best_val_auc = -1.0

    history = []

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(1, EPOCHS + 1):

        print(
            f"\n{'=' * 20} "
            f"Epoch {epoch}/{EPOCHS} "
            f"{'=' * 20}"
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            training=True,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            training=False,
        )

        print("\nTraining:")
        print(
            f"  Loss: {train_metrics['loss']:.4f}"
        )
        print(
            f"  AUC:  {train_metrics['auc']:.4f}"
        )
        print(
            f"  F1:   {train_metrics['macro_f1']:.4f}"
        )
        print(
            f"  Acc:  {train_metrics['accuracy']:.4f}"
        )

        print("\nValidation:")
        print(
            f"  Loss: {val_metrics['loss']:.4f}"
        )
        print(
            f"  AUC:  {val_metrics['auc']:.4f}"
        )
        print(
            f"  F1:   {val_metrics['macro_f1']:.4f}"
        )
        print(
            f"  Acc:  {val_metrics['accuracy']:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }

        history.append(
            epoch_record
        )

        # ------------------------------------------------------
        # Save best checkpoint
        # ------------------------------------------------------

        if val_metrics["auc"] > best_val_auc:

            best_val_auc = (
                val_metrics["auc"]
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                },
                CHECKPOINT_PATH,
            )

            print(
                f"\n✓ Saved best model:"
                f" {CHECKPOINT_PATH}"
            )

    # ----------------------------------------------------------
    # Save history
    # ----------------------------------------------------------

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation AUC: "
        f"{best_val_auc:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"History: "
        f"{HISTORY_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()