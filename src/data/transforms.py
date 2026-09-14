"""Image transformations for training and evaluation.

Training transforms are config-driven via ``config.yaml``
(``training.augmentation``).  Evaluation transforms are fixed and identical
across training/evaluation/inference so that the model always sees the same
preprocessing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from torchvision import transforms

# ImageNet statistics — these are FIXED (never fitted on our data) and match
# the EfficientNet pretrained weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic preprocessing used for evaluation and inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class _JPEGCompression:
    """Random mild JPEG re-compression (realistic online degradation)."""

    def __init__(self, quality_low: int = 70, quality_high: int = 95, p: float = 0.5):
        self.quality_low = quality_low
        self.quality_high = quality_high
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            quality = int(np.random.randint(self.quality_low, self.quality_high + 1))
            import io
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")
        return img


class _AddNoise:
    """Add mild Gaussian noise to the image tensor (un-normalised space)."""

    def __init__(self, std: float = 0.01, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if np.random.rand() < self.p:
            tensor = tensor + torch_randn_like(tensor) * self.std
            tensor = tensor.clamp(0.0, 1.0)
        return tensor


def torch_randn_like(tensor):
    import torch
    return torch.randn_like(tensor)


def get_train_transforms(
    image_size: int = 224,
    baseline: bool = True,
    stronger: bool = False,
    jpeg_quality_low: int = 70,
    blur_sigma: float = 0.5,
    noise_std: float = 0.01,
) -> transforms.Compose:
    """Training augmentation.

    ``baseline``: resize + horizontal flip + small rotation + color jitter.
    ``stronger``: additionally applies mild JPEG compression, Gaussian blur
        and Gaussian noise — realistic, online-style degradations that improve
        generalisation without changing the task distribution.
    """
    ops = [transforms.Resize((image_size, image_size))]
    if baseline:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        ]
    if stronger:
        ops += [
            _JPEGCompression(quality_low=jpeg_quality_low),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, blur_sigma)),
        ]
    ops += [transforms.ToTensor()]
    if stronger:
        ops += [_AddNoise(std=noise_std)]
    ops += [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(ops)
