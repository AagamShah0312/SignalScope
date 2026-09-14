"""Tests for dataset loading and transforms."""

from __future__ import annotations

import pytest

from src.data.dataset import SignalScopeDataset
from src.data.transforms import get_eval_transforms, get_train_transforms


def test_dataset_length_and_labels(tmp_manifest_dir):
    dataset = SignalScopeDataset(str(tmp_manifest_dir), transform=get_eval_transforms())
    assert len(dataset) == 8
    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert label in (0, 1)


def test_dataset_missing_file_raises(tmp_path):
    import pandas as pd

    manifest = tmp_path / "bad.csv"
    pd.DataFrame([{"path": "/nonexistent/img.png", "label": 0}]).to_csv(manifest, index=False)
    dataset = SignalScopeDataset(str(manifest), transform=get_eval_transforms())
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_dataset_missing_columns_raises(tmp_path):
    import pandas as pd

    manifest = tmp_path / "badcols.csv"
    pd.DataFrame([{"label": 0}]).to_csv(manifest, index=False)
    with pytest.raises(ValueError):
        SignalScopeDataset(str(manifest))


def test_transforms_output_shape(sample_rgb_image):
    train = get_train_transforms(image_size=224, baseline=True, stronger=False)
    eval_t = get_eval_transforms(image_size=224)
    assert train(sample_rgb_image).shape == (3, 224, 224)
    assert eval_t(sample_rgb_image).shape == (3, 224, 224)


def test_stronger_augmentation_shape(sample_rgb_image):
    transform = get_train_transforms(image_size=224, baseline=True, stronger=True)
    out = transform(sample_rgb_image)
    assert out.shape == (3, 224, 224)
