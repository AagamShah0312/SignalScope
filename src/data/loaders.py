"""DataLoader construction for the CIFAKE pipeline.

Loaders read the **frozen** split manifests produced by
:mod:`src.data.manifests` — the split is computed once and reused, so
training, validation and test sets never drift between runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import SignalScopeDataset
from src.data.manifests import build_split_manifests
from src.data.transforms import get_eval_transforms, get_train_transforms


def _resolve_batch(batch_size: Optional[int]) -> int:
    if batch_size is not None:
        return batch_size
    return int(load_config().training.batch_size)


def create_dataloaders(
    manifest_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    image_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders using the frozen CIFAKE split.

    Args:
        manifest_path: (deprecated) if given, splits are derived from this
            single manifest instead of the canonical frozen split. Kept for
            backward compatibility with older callers.
    """
    cfg = load_config()
    batch_size = _resolve_batch(batch_size)
    num_workers = num_workers if num_workers is not None else int(cfg.training.num_workers)
    image_size = image_size or int(cfg.data.image_size)

    if manifest_path is not None:
        # Legacy single-manifest path (splits on the fly).
        import pandas as pd

        df = pd.read_csv(manifest_path)
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()
        train_df = train_df.sample(frac=1, random_state=int(cfg.data.split_seed)).reset_index(drop=True)
        val_size = max(1, int(len(train_df) * float(cfg.data.validation_fraction)))
        val_df = train_df.iloc[:val_size].copy()
        train_df = train_df.iloc[val_size:].copy()

        manifest_dir = cfg.resolve(cfg.data.manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        train_manifest = manifest_dir / "train_manifest.csv"
        val_manifest = manifest_dir / "val_manifest.csv"
        test_manifest = manifest_dir / "test_manifest.csv"
        train_df.to_csv(train_manifest, index=False)
        val_df.to_csv(val_manifest, index=False)
        test_df.to_csv(test_manifest, index=False)
    else:
        splits = build_split_manifests()
        train_manifest, val_manifest, test_manifest = (
            splits["train"], splits["val"], splits["test"],
        )

    aug = cfg.training.augmentation
    train_transform = get_train_transforms(
        image_size=image_size,
        baseline=bool(aug.get("baseline", True)),
        stronger=bool(aug.get("stronger", False)),
        jpeg_quality_low=int(aug.get("jpeg_quality_low", 70)),
        blur_sigma=float(aug.get("blur_sigma", 0.5)),
        noise_std=float(aug.get("noise_std", 0.01)),
    )
    eval_transform = get_eval_transforms(image_size=image_size)

    train_dataset = SignalScopeDataset(train_manifest, transform=train_transform)
    val_dataset = SignalScopeDataset(val_manifest, transform=eval_transform)
    test_dataset = SignalScopeDataset(test_manifest, transform=eval_transform)

    common = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common)

    print(f"Training images:   {len(train_dataset):,}")
    print(f"Validation images: {len(val_dataset):,}")
    print(f"Test images:       {len(test_dataset):,}")

    return train_loader, val_loader, test_loader


def create_mixed_dataloaders(
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    exclude_generator: Optional[str] = None,
):
    """Create DataLoaders over a mixed CIFAKE + Defactify training set.

    Args:
        exclude_generator: withhold a Defactify generator's training images
            (for leave-one-generator-out experiments).
    """
    cfg = load_config()
    batch_size = _resolve_batch(batch_size)
    num_workers = num_workers if num_workers is not None else int(cfg.training.num_workers)
    image_size = int(cfg.data.image_size)

    from src.data.manifests import build_mixed_train_manifest

    manifests = build_mixed_train_manifest(exclude_generator=exclude_generator)

    aug = cfg.training.augmentation
    train_transform = get_train_transforms(
        image_size=image_size,
        baseline=bool(aug.get("baseline", True)),
        stronger=bool(aug.get("stronger", False)),
        jpeg_quality_low=int(aug.get("jpeg_quality_low", 70)),
        blur_sigma=float(aug.get("blur_sigma", 0.5)),
        noise_std=float(aug.get("noise_std", 0.01)),
    )
    eval_transform = get_eval_transforms(image_size=image_size)

    train_dataset = SignalScopeDataset(manifests["train"], transform=train_transform)
    val_dataset = SignalScopeDataset(manifests["val"], transform=eval_transform)

    common = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common)

    print(f"Mixed training images:   {len(train_dataset):,}")
    print(f"Mixed validation images: {len(val_dataset):,}")
    return train_loader, val_loader
