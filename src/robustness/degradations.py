"""Controlled image degradations for robustness benchmarking.

Each degradation is configurable and designed to mimic realistic online
transforms (re-compression, resizing, blur, noise, brightness/contrast,
cropping, screenshot-like re-encoding).  None of them destroy the image with
unrealistic transformations.
"""

from __future__ import annotations

import io
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def jpeg_compression(quality: int) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        _rgb(img).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    return fn


def resize_to(scale: float) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return _rgb(img).resize((new_w, new_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return fn


def gaussian_blur(radius: float) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        return _rgb(img).filter(ImageFilter.GaussianBlur(radius=radius))
    return fn


def add_noise(std: float) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        arr = np.asarray(_rgb(img), dtype=np.float32)
        noise = np.random.normal(0.0, std * 255.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return fn


def brightness(factor: float) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(_rgb(img)).enhance(factor)
    return fn


def contrast(factor: float) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(_rgb(img)).enhance(factor)
    return fn


def center_crop(fraction: float) -> Callable[[Image.Image], Image.Image]:
    """Crop to the center ``fraction`` and resize back (crop + upscale)."""
    def fn(img: Image.Image) -> Image.Image:
        img = _rgb(img)
        w, h = img.size
        cw, ch = int(w * fraction), int(h * fraction)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)
    return fn


def screenshot_simulation(scale: float = 0.5, quality: int = 80) -> Callable[[Image.Image], Image.Image]:
    """Downscale + JPEG re-encode, approximating a screenshot pipeline."""
    def fn(img: Image.Image) -> Image.Image:
        img = _rgb(img)
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        buffer = io.BytesIO()
        small.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").resize((w, h), Image.BILINEAR)
    return fn


# ---------------------------------------------------------------------------
# Registry of standard benchmark conditions
# ---------------------------------------------------------------------------

def default_degradations() -> Dict[str, Callable[[Image.Image], Image.Image]]:
    """The standard, documented robustness benchmark suite."""
    return {
        "jpeg_q90": jpeg_compression(90),
        "jpeg_q70": jpeg_compression(70),
        "jpeg_q50": jpeg_compression(50),
        "resize_0.5": resize_to(0.5),
        "resize_1.5": resize_to(1.5),
        "blur_r1": gaussian_blur(1.0),
        "noise_s10": add_noise(10 / 255.0),
        "brightness_0.8": brightness(0.8),
        "brightness_1.2": brightness(1.2),
        "contrast_0.8": contrast(0.8),
        "contrast_1.2": contrast(1.2),
        "crop_0.7": center_crop(0.7),
        "screenshot": screenshot_simulation(),
    }


def apply_degradations(
    img: Image.Image,
    conditions: Dict[str, Callable[[Image.Image], Image.Image]],
) -> Dict[str, Image.Image]:
    """Apply every condition, returning a dict name -> degraded image."""
    results = {}
    for name, fn in conditions.items():
        try:
            results[name] = fn(img)
        except Exception as exc:  # a broken transform must not kill the benchmark
            results[name] = exc
    return results


def list_conditions() -> List[str]:
    return list(default_degradations().keys())
