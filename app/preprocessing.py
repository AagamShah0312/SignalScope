"""Image validation and preprocessing for the inference pipeline."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.data.transforms import get_eval_transforms

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp",
    "image/jpg",  # some clients send this
}

# Sanity limits — reject absurd inputs early (decompression-bomb guard).
MAX_DIMENSION = 8000
MAX_PIXELS = 40_000_000  # ~40 MP
DEFAULT_MAX_BYTES = 15 * 1024 * 1024


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""


def validate_extension(filename: Optional[str]) -> Optional[str]:
    """Return the lowercase extension or None."""
    if not filename:
        return None
    return Path(filename).suffix.lower()


def validate_file(
    contents: bytes,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Image.Image:
    """Validate image bytes and return a decoded RGB PIL image.

    Raises :class:`ImageValidationError` on any failure so the API can return
    a clean 4xx response.
    """
    if not contents:
        raise ImageValidationError("Empty file.")

    if len(contents) > max_bytes:
        raise ImageValidationError(
            f"File too large ({len(contents) / (1024 * 1024):.1f} MB). "
            f"Maximum allowed is {max_bytes / (1024 * 1024):.0f} MB."
        )

    ext = validate_extension(filename)
    if ext is not None and ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '{ext}'. Use JPEG, PNG, WebP or BMP."
        )

    if mime_type and mime_type not in ALLOWED_MIME_TYPES:
        raise ImageValidationError(f"Unsupported MIME type '{mime_type}'.")

    # Open the header only (lazy).  ``Image.open`` does not decode pixels, so
    # we can validate dimensions from the header *before* any expensive or
    # memory-hungry decode — this is the decompression-bomb defence.
    try:
        image = Image.open(io.BytesIO(contents))
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombError) as exc:
        raise ImageValidationError(
            "Could not read the image (corrupt or unsupported)."
        ) from exc

    # ---- dimension checks from the header (before decoding pixels) ----
    if image.width < 16 or image.height < 16:
        raise ImageValidationError(
            f"Image is too small ({image.width}x{image.height}). Minimum is 16x16."
        )
    if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions too large ({image.width}x{image.height})."
        )
    if image.width * image.height > MAX_PIXELS:
        raise ImageValidationError(
            f"Image has too many pixels ({image.width * image.height})."
        )

    try:
        image.load()  # force full decode (catches truncated/corrupt files)
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombError) as exc:
        raise ImageValidationError(
            "Could not read the image (corrupt or unsupported)."
        ) from exc

    return image.convert("RGB")


def preprocess_image(
    image: Image.Image,
    image_size: int = 224,
) -> torch.Tensor:
    """Convert a PIL RGB image to a normalized (1,3,H,W) inference tensor."""
    transform = get_eval_transforms(image_size=image_size)
    return transform(image).unsqueeze(0)


def to_numpy_rgb(image: Image.Image) -> np.ndarray:
    """PIL RGB image -> uint8 numpy array (H,W,3)."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def validate_dimensions(image: Image.Image, min_dim: int = 16) -> Tuple[int, int]:
    """Return (width, height) after checking the minimum dimension."""
    if image.width < min_dim or image.height < min_dim:
        raise ImageValidationError("Image dimensions are below the supported minimum.")
    return image.width, image.height
