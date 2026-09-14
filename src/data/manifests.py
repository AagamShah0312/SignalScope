"""Manifest generation and frozen train/validation/test splitting.

The official SIH held-out test set is NEVER touched by any of this code — it
only works on public datasets (CIFAKE, Defactify).

Key guarantee: the train/validation split is computed **once**, persisted to
CSV, and reused.  This prevents silent re-splitting (which can leak validation
images into training across runs) and makes the split reproducible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import REPO_ROOT, load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Canonical generator-name normalisation used by both Defactify downloaders,
# which currently disagree on names (e.g. "sdxl" vs "stable_diffusion_xl").
GENERATOR_ALIASES = {
    "stable_diffusion_2_1": "stable_diffusion_2_1",
    "stable_diffusion_2.1": "stable_diffusion_2_1",
    "sd2.1": "stable_diffusion_2_1",
    "sdxl": "stable_diffusion_xl",
    "stable_diffusion_xl": "stable_diffusion_xl",
    "stable_diffusion_3": "stable_diffusion_3",
    "stable_diffusion_3.0": "stable_diffusion_3",
    "dalle3": "dalle_3",
    "dalle_3": "dalle_3",
    "dalle-3": "dalle_3",
    "midjourney_6": "midjourney_6",
    "midjourney6": "midjourney_6",
    "real": "real",
}

AI_GENERATORS = [
    "stable_diffusion_2_1",
    "stable_diffusion_xl",
    "stable_diffusion_3",
    "dalle_3",
    "midjourney_6",
]


def normalize_generator(name: object) -> str:
    """Normalize a generator label to a canonical name."""
    key = str(name).strip().lower()
    return GENERATOR_ALIASES.get(key, key)


def _cfg_paths():
    cfg = load_config()
    manifest_dir = cfg.resolve(cfg.data.manifest_dir)
    cifake_root = cfg.resolve(cfg.data.cifake_root)
    return cfg, manifest_dir, cifake_root


def collect_cifake_images(root: Path) -> pd.DataFrame:
    """Walk the CIFAKE folder structure (train/test x REAL/FAKE) into a DF."""
    rows: List[Dict[str, object]] = []
    for split in ("train", "test"):
        for label_name, label in (("REAL", 0), ("FAKE", 1)):
            folder = root / split / label_name
            if not folder.exists():
                continue
            for image_path in sorted(folder.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({
                        "path": image_path.as_posix(),
                        "label": label,
                        "generator": "stable_diffusion_v1_4" if label == 1 else "real_cifar10",
                        "source": "CIFAKE",
                        "split": split,
                    })
    return pd.DataFrame(rows)


def build_cifake_manifest(root: Optional[Path] = None) -> pd.DataFrame:
    """Create (or load) the full CIFAKE manifest."""
    cfg, manifest_dir, cifake_root = _cfg_paths()
    root = Path(root) if root else cifake_root
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output = manifest_dir / "cifake_manifest.csv"
    if output.exists():
        return pd.read_csv(output)
    if not root.exists():
        raise FileNotFoundError(
            f"CIFAKE dataset not found at {root}. "
            "Download it and re-run, or point data.cifake_root at it."
        )
    df = collect_cifake_images(root)
    df.to_csv(output, index=False)
    return df


def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _split_fingerprint(df: pd.DataFrame) -> str:
    """Deterministic fingerprint of the (path, label) pairs used to decide
    whether a persisted split matches the current data."""
    joined = "|".join(f"{p}:{l}" for p, l in zip(df["path"].astype(str), df["label"].astype(str)))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def build_split_manifests(
    val_fraction: Optional[float] = None,
    seed: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """Build and persist frozen ``train/val/test`` manifests from CIFAKE.

    * ``test`` = CIFAKE's official test split (never used for training/tuning).
    * ``val``  = a deterministic fraction carved out of CIFAKE's train split.
    * ``train``= the remainder of CIFAKE's train split.

    The split is recomputed only when the underlying data changes
    (fingerprint mismatch) or when ``force=True``.
    """
    cfg, manifest_dir, _ = _cfg_paths()
    val_fraction = val_fraction if val_fraction is not None else float(cfg.data.validation_fraction)
    seed = seed if seed is not None else int(cfg.data.split_seed)

    cifake = build_cifake_manifest()
    train_pool = cifake[cifake["split"] == "train"].copy()
    test_df = cifake[cifake["split"] == "test"].copy()

    if train_pool.empty:
        raise ValueError("CIFAKE train split is empty — check data/raw/cifake layout.")

    fingerprint = _split_fingerprint(train_pool)
    train_path = manifest_dir / "train_manifest.csv"
    val_path = manifest_dir / "val_manifest.csv"
    test_path = manifest_dir / "test_manifest.csv"
    fingerprint_path = manifest_dir / ".split_fingerprint"

    persisted = fingerprint_path.exists() and fingerprint_path.read_text().strip() == fingerprint
    if persisted and train_path.exists() and val_path.exists() and test_path.exists() and not force:
        return {"train": train_path, "val": val_path, "test": test_path}

    # Stratified shuffle by label, deterministic via a per-row stable hash so
    # that re-running on the same files always produces the same assignment.
    train_pool = train_pool.copy()
    train_pool["_h"] = [_stable_hash(str(p)) for p in train_pool["path"]]
    train_pool = train_pool.sort_values(["_h"]).reset_index(drop=True)

    val_size = max(1, int(round(len(train_pool) * val_fraction)))
    val_df = train_pool.iloc[:val_size].drop(columns=["_h"]).copy()
    train_df = train_pool.iloc[val_size:].drop(columns=["_h"]).copy()

    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    fingerprint_path.write_text(fingerprint)

    return {"train": train_path, "val": val_path, "test": test_path}


def load_defactify_manifest(kind: str = "train") -> pd.DataFrame:
    """Load a Defactify manifest (train or eval) and normalise its columns.

    Both downloader variants are supported: one stores ``filename``, the other
    stores an absolute/relative ``path``.  The returned frame always has
    ``path``, ``label``, ``generator`` and ``source`` columns.
    """
    cfg = load_config()
    if kind == "train":
        manifest_path = cfg.resolve(cfg.data.defactify_train_manifest)
        image_dir = cfg.resolve(cfg.data.defactify_train_image_dir)
    elif kind == "eval":
        manifest_path = cfg.resolve(cfg.data.defactify_eval_manifest)
        image_dir = manifest_path.parent / "images"
    else:
        raise ValueError(f"Unknown Defactify kind: {kind}")

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Defactify {kind} manifest not found: {manifest_path}. "
            "Run the downloader scripts in src/data/ first."
        )

    df = pd.read_csv(manifest_path)
    if "path" not in df.columns and "filename" in df.columns:
        df["path"] = df["filename"].astype(str).map(
            lambda name: (image_dir / name).as_posix()
        )
    elif "path" in df.columns:
        # Make relative paths resolve from the repo root.
        df["path"] = df["path"].astype(str).map(
            lambda p: p if Path(p).is_absolute() else (REPO_ROOT / p).as_posix()
        )
    else:
        raise ValueError(f"Defactify {kind} manifest has no path/filename column.")

    if "generator" in df.columns:
        df["generator"] = df["generator"].map(normalize_generator)
    if "source" not in df.columns:
        df["source"] = "Defactify"
    return df


def build_mixed_train_manifest(
    val_fraction: Optional[float] = None,
    seed: Optional[int] = None,
    exclude_generator: Optional[str] = None,
) -> Dict[str, Path]:
    """Build a mixed CIFAKE+Defactify training manifest.

    Args:
        exclude_generator: if given, that Defactify generator's training images
            are withheld (used by leave-one-generator-out experiments).  Real
            images are never excluded.
    """
    cfg, manifest_dir, _ = _cfg_paths()
    val_fraction = val_fraction if val_fraction is not None else float(cfg.data.validation_fraction)
    seed = seed if seed is not None else int(cfg.data.split_seed)

    splits = build_split_manifests(val_fraction, seed)
    cifake_train = pd.read_csv(splits["train"])
    cifake_val = pd.read_csv(splits["val"])

    defactify = load_defactify_manifest("train")
    defactify = defactify[defactify["label"].isin([0, 1])].copy()
    if exclude_generator is not None:
        canon = normalize_generator(exclude_generator)
        kept = defactify[(defactify["generator"] == "real") | (defactify["generator"] != canon)]
        defactify = kept.copy()

    needed = {"path", "label"}
    missing = needed - set(defactify.columns)
    if missing:
        raise ValueError(f"Defactify manifest missing columns: {missing}")

    mixed = pd.concat([cifake_train, defactify[["path", "label", "generator", "source"]]],
                      ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    mixed_path = manifest_dir / "mixed_train_manifest.csv"
    val_path = manifest_dir / "mixed_val_manifest.csv"
    mixed.to_csv(mixed_path, index=False)
    cifake_val.to_csv(val_path, index=False)

    return {"train": mixed_path, "val": val_path}
