"""PyTorch Dataset for SignalScope manifest CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SignalScopeDataset(Dataset):
    """Dataset over a manifest CSV.

    Expected CSV columns:
        path      -> image file path (absolute, or relative to repo root)
        label     -> 0 for REAL, 1 for AI_GENERATED
        generator -> (optional) generator/source name
        split     -> (optional) train / val / test
    """

    def __init__(
        self,
        csv_file: str,
        transform=None,
        return_path: bool = False,
    ):
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.return_path = return_path

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {self.csv_file}")

        self.data = pd.read_csv(self.csv_file)

        required_columns = {"path", "label"}
        missing = required_columns - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Tolerate manifests whose paths were written relative to the repo root
        # on another machine.
        from src.config import REPO_ROOT

        self.data["path"] = self.data["path"].astype(str).map(
            lambda p: p if Path(p).is_absolute() else (REPO_ROOT / p).as_posix()
        )

    def __len__(self) -> int:
        return len(self.data)

    def _row(self, index: int):
        row = self.data.iloc[index]
        image_path = Path(row["path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        label = int(row["label"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(image_path)

    def __getitem__(self, index: int):
        image, label, path = self._row(index)
        if self.return_path:
            return image, label, path
        return image, label
