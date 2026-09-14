from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset

from src.data.dataset import SignalScopeDataset
from src.data.transforms import get_train_transforms, get_eval_transforms


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

CIFAKE_MANIFEST = Path("data/processed/cifake_manifest.csv")
DEFACTIFY_MANIFEST = Path("data/raw/defactify_train/manifest.csv")


# ------------------------------------------------------------------
# Mixed dataset loader
# ------------------------------------------------------------------

def create_mixed_dataloaders(
    batch_size=16,
    num_workers=2,
):
    """
    Create training and validation DataLoaders using:

    1. CIFAKE train/validation data
    2. Defactify training sample

    The Defactify 600-image evaluation set is NOT used here.
    """

    # ==============================================================
    # 1. Load CIFAKE manifest
    # ==============================================================

    cifake_df = pd.read_csv(CIFAKE_MANIFEST)

    cifake_train_df = cifake_df[
        cifake_df["split"] == "train"
    ].copy()

    cifake_test_df = cifake_df[
        cifake_df["split"] == "test"
    ].copy()

    # Reproduce the same 90/10 train/validation split
    cifake_train_df = cifake_train_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    val_size = int(len(cifake_train_df) * 0.10)

    cifake_val_df = cifake_train_df.iloc[:val_size].copy()
    cifake_train_df = cifake_train_df.iloc[val_size:].copy()

    # ==============================================================
    # 2. Load Defactify training sample
    # ==============================================================

    defactify_df = pd.read_csv(DEFACTIFY_MANIFEST)

    # Safety check:
    # Only training images should exist in this manifest.
    if "split" in defactify_df.columns:
        defactify_train_df = defactify_df[
            defactify_df["split"] == "train"
        ].copy()
    else:
        defactify_train_df = defactify_df.copy()

    # Defactify manifest stores the filename, while the existing
    # SignalScopeDataset expects a full "path" column.
    defactify_train_df["path"] = (
        "data/raw/defactify_train/images/"
        + defactify_train_df["filename"].astype(str)
    )

    # ==============================================================
    # 3. Create temporary manifests
    # ==============================================================

    mixed_train_df = pd.concat(
        [
            cifake_train_df,
            defactify_train_df,
        ],
        ignore_index=True,
    )

    mixed_train_df = mixed_train_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Save manifests for reproducibility
    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    mixed_train_manifest = Path(
        "data/processed/mixed_train_manifest.csv"
    )

    val_manifest = Path(
        "data/processed/mixed_val_manifest.csv"
    )

    mixed_train_df.to_csv(
        mixed_train_manifest,
        index=False
    )

    cifake_val_df.to_csv(
        val_manifest,
        index=False
    )

    # ==============================================================
    # 4. Create datasets
    # ==============================================================

    train_dataset = SignalScopeDataset(
        mixed_train_manifest,
        transform=get_train_transforms()
    )

    val_dataset = SignalScopeDataset(
        val_manifest,
        transform=get_eval_transforms()
    )

    # ==============================================================
    # 5. Create DataLoaders
    # ==============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ==============================================================
    # 6. Print useful information
    # ==============================================================

    print("\n" + "=" * 60)
    print("MIXED DATASET")
    print("=" * 60)

    print(f"CIFAKE train:       {len(cifake_train_df):,}")
    print(f"Defactify train:    {len(defactify_train_df):,}")
    print(f"Mixed train total:  {len(mixed_train_df):,}")
    print(f"CIFAKE validation:  {len(cifake_val_df):,}")

    print("=" * 60)

    print("\nMixed training labels:")

    label_counts = mixed_train_df["label"].value_counts().sort_index()

    for label, count in label_counts.items():
        label_name = (
            "Real"
            if int(label) == 0
            else "AI Generated"
        )

        print(
            f"  {label_name}: {count:,}"
        )

    print("=" * 60 + "\n")

    return train_loader, val_loader