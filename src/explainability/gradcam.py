"""Grad-CAM for EfficientNet-B0.

The implementation:

1. targets the final convolutional layer (``model.features[-1][0]``),
2. registers forward + backward hooks,
3. captures activations and gradients,
4. computes the weighted activation map (global-average-pooled gradients),
5. applies ReLU and normalises to [0, 1],
6. resizes and overlays the heatmap on the original image.

The heatmap is a **model explanation**, not proof of an artifact.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def find_target_layer(model: nn.Module) -> nn.Module:
    """Return the final convolutional layer of an EfficientNet backbone."""
    return model.features[-1][0]


class GradCAM:
    """Standard Grad-CAM (Selvaraju et al., 2017) over a target conv layer."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or find_target_layer(model)
        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Compute the Grad-CAM map for a single image tensor (1,3,H,W).

        Returns (cam[H,W], predicted_class, probabilities[2]).
        """
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = int(torch.argmax(probabilities, dim=1).item())
        target = predicted_class if target_class is None else target_class

        score = output[:, target]
        score.backward()

        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam - cam.min()
        denom = cam.max() + 1e-8
        cam = cam / denom

        if was_training:
            self.model.train()
        else:
            self.model.eval()

        return cam[0].detach().cpu(), predicted_class, probabilities[0].detach().cpu()

    def remove_hooks(self) -> None:
        self.forward_hook.remove()
        self.backward_hook.remove()


def overlay_heatmap(
    image: np.ndarray,
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap (H,W in [0,1]) onto an RGB image (H,W,3)."""
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def save_visualization(
    original_rgb: np.ndarray,
    cam: np.ndarray,
    output_dir: str,
    base_name: str = "gradcam",
) -> dict:
    """Save original/heatmap/overlay images (JPEG for small artifacts).

    Returns a dict of absolute paths keyed by ``original``/``heatmap``/``overlay``.
    """
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    original_path = out / f"{base_name}_original.jpg"
    heatmap_path = out / f"{base_name}_heatmap.jpg"
    overlay_path = out / f"{base_name}_overlay.jpg"

    Image.fromarray(original_rgb).save(original_path, quality=92)

    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    cv2.imwrite(str(heatmap_path), heatmap, [cv2.IMWRITE_JPEG_QUALITY, 92])

    overlay = overlay_heatmap(original_rgb, cam)
    Image.fromarray(overlay).save(overlay_path, quality=92)

    return {
        "original": str(original_path),
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
    }
