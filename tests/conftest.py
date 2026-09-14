"""Shared pytest fixtures for SignalScope tests."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def sample_rgb_image() -> Image.Image:
    """A deterministic 224x224 RGB test image (no external data needed)."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[..., 0] = 60
    arr[..., 1] = 110
    arr[..., 2] = 200
    # A few light blobs so the image is not uniform.
    for cx, cy in [(40, 40), (180, 150), (120, 200)]:
        arr[cy - 8:cy + 8, cx - 8:cx + 8] = (255, 255, 255)
    return Image.fromarray(arr)


@pytest.fixture(scope="session")
def sample_png_bytes(sample_rgb_image) -> bytes:
    import io

    buf = io.BytesIO()
    sample_rgb_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def tmp_manifest_dir(tmp_path_factory):
    """A manifest directory containing a tiny labelled image set."""
    import pandas as pd

    base = tmp_path_factory.mktemp("dataset")
    images = base / "images"
    images.mkdir()

    rows = []
    for i in range(8):
        label = 0 if i < 4 else 1
        img = Image.fromarray(
            np.full((32, 32, 3), fill_value=(i * 20) % 255, dtype=np.uint8)
        )
        path = images / f"img_{i}.png"
        img.save(path)
        rows.append({"path": str(path), "label": label, "generator": "test", "source": "fixture"})

    df = pd.DataFrame(rows)
    manifest = base / "manifest.csv"
    df.to_csv(manifest, index=False)
    return manifest
