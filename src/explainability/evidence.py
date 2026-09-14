"""Grounded visual evidence extraction from Grad-CAM maps.

The evidence is strictly tied to the *observable* model attention: we locate
the strongest activation regions and describe them neutrally.  We do **not**
assert that a region contains a specific AI artifact unless a dedicated,
reliable detector exists — and none is asserted here.

Design rule (from the problem statement):

    "The model's strongest activation is concentrated around this region."

is a safe claim.  "The AI drew an impossible handle" is not.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def top_activation_regions(
    cam: np.ndarray,
    image_shape: Tuple[int, int],
    top_k: int = 3,
    threshold: float = 0.5,
    min_area_frac: float = 0.001,
) -> List[Dict[str, object]]:
    """Extract up to ``top_k`` connected high-activation regions.

    Args:
        cam: Grad-CAM map (H,W) in [0,1].
        image_shape: original image (height, width) for coordinate mapping.
        threshold: activation threshold for binarisation.
        min_area_frac: ignore blobs smaller than this fraction of the image.

    Returns a list of regions, each with ``bbox`` (x0,y0,x1,y1 in original
    pixels), ``relative_area``, and ``mean_activation``.
    """
    import cv2

    h, w = cam.shape
    img_h, img_w = image_shape

    binary = (cam >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_area = min_area_frac * (img_h * img_w)
    regions = []
    for i in range(1, num_labels):  # skip background label 0
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        if area / float(img_h * img_w) > 0.95:
            continue  # degenerate: activation covers everything

        mask = labels == i
        mean_activation = float(cam[mask].mean())

        # Map from CAM coordinates to original image coordinates.
        x0 = int(x / w * img_w)
        y0 = int(y / h * img_h)
        x1 = int((x + bw) / w * img_w)
        y1 = int((y + bh) / h * img_h)

        regions.append({
            "bbox": [x0, y0, x1, y1],
            "relative_area": float(area / (img_h * img_w)),
            "mean_activation": round(mean_activation, 4),
        })

    regions.sort(key=lambda r: (-r["mean_activation"], -r["relative_area"]))
    return regions[:top_k]


def activation_concentration(cam: np.ndarray) -> Dict[str, float]:
    """Summaries of how concentrated the model's attention is.

    Returns mean/max activation and the Gini-like concentration of the top
    5% of pixels.  Useful for detecting "diffuse" (uninformative) heatmaps.
    """
    flat = cam.reshape(-1).astype(np.float32)
    top_frac = 0.05
    k = max(1, int(len(flat) * top_frac))
    top = np.sort(flat)[-k:]
    concentration = float(top.mean() / (flat.mean() + 1e-8))
    return {
        "mean_activation": float(flat.mean()),
        "max_activation": float(flat.max()),
        "top5pct_concentration": round(concentration, 4),
    }


def region_position_label(bbox: List[int], image_shape: Tuple[int, int]) -> str:
    """A coarse natural-language position label for a bounding box."""
    img_w, img_h = image_shape[1], image_shape[0]
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h

    vertical = "upper" if cy < 0.34 else "lower" if cy > 0.66 else "central"
    horizontal = "left" if cx < 0.34 else "right" if cx > 0.66 else "center"

    if horizontal == "center" and vertical == "central":
        return "the center of the image"
    return f"the {vertical}-{horizontal} region"
