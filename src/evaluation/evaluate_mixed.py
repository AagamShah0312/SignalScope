from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from src.models.model import create_model
from src.data.transforms import get_eval_transforms


# ==============================================================
# Configuration
# ==============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path(
    "model/best_efficientnet_b0_mixed.pth"
)

EVAL_MANIFEST = Path(
    "data/raw/defactify_eval/manifest.csv"
)

OUTPUT_PATH = Path(
    "reports/mixed_defactify_results.csv"
)

BATCH_SIZE = 16


# ==============================================================
# Dataset
# ==============================================================

class DefactifyEvalDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = Image.open(
            row["path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])

        return image, label


# ==============================================================
# Load model
# ==============================================================

def load_model():

    print("\nLoading model...")

    model = create_model(
        num_classes=2,
        pretrained=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    # Our training script saves a checkpoint dictionary.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    print(
        f"Loaded: {CHECKPOINT_PATH}"
    )

    return model


# ==============================================================
# Evaluate
# ==============================================================

def main():

    print("\n" + "=" * 60)
    print("SignalScope - Mixed Model Defactify Evaluation")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print("=" * 60)

    # ----------------------------------------------------------
    # Load evaluation manifest
    # ----------------------------------------------------------

    df = pd.read_csv(
        EVAL_MANIFEST
    )

    print(
        f"\nEvaluation images: {len(df):,}"
    )

    print("\nGenerator distribution:")

    if "generator" in df.columns:
        print(
            df["generator"].value_counts().to_string()
        )

    # ----------------------------------------------------------
    # Dataset / loader
    # ----------------------------------------------------------

    dataset = DefactifyEvalDataset(
        df,
        transform=get_eval_transforms(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    model = load_model()

    # ----------------------------------------------------------
    # Predictions
    # ----------------------------------------------------------

    all_labels = []
    all_probabilities = []

    print("\nRunning evaluation...")

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            all_labels.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    labels = np.array(
        all_labels
    )

    probabilities = np.array(
        all_probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ----------------------------------------------------------
    # Overall metrics
    # ----------------------------------------------------------

    auc = roc_auc_score(
        labels,
        probabilities
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(
        f"ROC-AUC:   {auc:.4f}"
    )

    print(
        f"Macro-F1:  {macro_f1:.4f}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"FPR:       {fpr:.4f}"
    )

    print(
        f"FNR:       {fnr:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    # ----------------------------------------------------------
    # Per-generator detection
    # ----------------------------------------------------------

    if "generator" in df.columns:

        results_df = df.copy()

        results_df["probability_ai"] = (
            probabilities
        )

        results_df["prediction"] = (
            predictions
        )

        print("\n" + "=" * 60)
        print("PER-GENERATOR RESULTS")
        print("=" * 60)

        generator_rows = []

        for generator, group in results_df.groupby(
            "generator"
        ):

            group_predictions = (
                group["prediction"].values
            )

            group_probabilities = (
                group["probability_ai"].values
            )

            group_labels = (
                group["label"].values
            )

            # Detection rate for AI generators
            if group_labels.mean() == 1:

                detection_rate = (
                    group_predictions == 1
                ).mean()

                mean_ai_probability = (
                    group_probabilities.mean()
                )

                print(
                    f"{generator:25s} "
                    f"Detection rate: "
                    f"{detection_rate:.4f}  "
                    f"Mean AI probability: "
                    f"{mean_ai_probability:.4f}"
                )

                generator_rows.append(
                    {
                        "generator": generator,
                        "type": "AI",
                        "detection_rate": detection_rate,
                        "mean_ai_probability":
                            mean_ai_probability,
                    }
                )

            # Real-image false-positive rate
            else:

                false_positive_rate = (
                    group_predictions == 1
                ).mean()

                mean_ai_probability = (
                    group_probabilities.mean()
                )

                print(
                    f"{generator:25s} "
                    f"False positive rate: "
                    f"{false_positive_rate:.4f}  "
                    f"Mean AI probability: "
                    f"{mean_ai_probability:.4f}"
                )

                generator_rows.append(
                    {
                        "generator": generator,
                        "type": "Real",
                        "detection_rate":
                            false_positive_rate,
                        "mean_ai_probability":
                            mean_ai_probability,
                    }
                )

        generator_results = pd.DataFrame(
            generator_rows
        )

    else:

        generator_results = pd.DataFrame()

    # ----------------------------------------------------------
    # Save detailed predictions
    # ----------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Detailed predictions saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("=" * 60)


if __name__ == "__main__":
    main()