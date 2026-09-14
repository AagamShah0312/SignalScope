"""Frequency-domain feature extraction (FFT-based).

The official SIH problem statement highlights frequency/artifact features as an
innovation opportunity.  This module extracts a compact, fixed-length vector of
spectral statistics from an RGB image tensor.  The features are designed to be
cheap to compute (a single 2-D FFT per image) and to be concatenated with the
EfficientNet spatial embedding in :mod:`src.models.fusion_model`.

Extracted statistics (all on the log-magnitude spectrum of a grayscale image):

* radial band energy ratios (low / mid / high frequency energy share)
* high-frequency energy ratio (above the half-radius)
* spectral flatness / entropy (how concentrated energy is)
* azimuthal anisotropy (directional energy imbalance)

Whether these features actually *help* is an empirical question — the fusion
model must be compared against the spatial-only baseline before any claim is
made.  Nothing here assumes frequency features are useful by themselves.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

FREQUENCY_FEATURE_DIM = 16
EPS = 1e-8


def _to_grayscale(images: torch.Tensor) -> torch.Tensor:
    """RGB tensor (B,3,H,W) -> grayscale (B,H,W) using Rec.601 luma."""
    if images.ndim != 4:
        raise ValueError("Expected a batch tensor of shape (B, 3, H, W).")
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def fft_magnitude(gray: torch.Tensor) -> torch.Tensor:
    """Log-magnitude spectrum (B,H,W) with the DC component shifted to center."""
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray.float()))
    magnitude = torch.log(spectrum.abs() + EPS)
    return magnitude


def radial_profile(magnitude: torch.Tensor) -> np.ndarray:
    """Radially-averaged profile (B, R) of the spectrum, where R = floor(diag/2)."""
    import torch.nn.functional as F  # noqa: F401 (kept import local for clarity)

    b, h, w = magnitude.shape
    radius = int(np.floor(min(h, w) / 2.0))
    cy, cx = h / 2.0, w / 2.0

    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=magnitude.device),
        torch.arange(w, dtype=torch.float32, device=magnitude.device),
        indexing="ij",
    )
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    profiles = []
    for i in range(b):
        flat_mag = magnitude[i].reshape(-1)
        flat_dist = dist.reshape(-1)
        prof = torch.zeros(radius, dtype=torch.float32, device=magnitude.device)
        counts = torch.zeros(radius, dtype=torch.float32, device=magnitude.device)
        # Accumulate by nearest integer radius bin.
        bins = flat_dist.floor().long().clamp(0, radius - 1)
        prof.index_add_(0, bins, flat_mag)
        counts.index_add_(0, bins, torch.ones_like(flat_mag))
        profiles.append(prof / (counts + EPS))
    return torch.stack(profiles)


def extract_frequency_features(images: torch.Tensor) -> torch.Tensor:
    """Extract a fixed-length (B, FREQUENCY_FEATURE_DIM) feature vector.

    Args:
        images: RGB image batch of shape (B, 3, H, W) in *normalized* space
            (any normalization is fine — the FFT only needs consistent input).
    """
    gray = _to_grayscale(images)
    magnitude = fft_magnitude(gray)          # (B, H, W)
    b, h, w = magnitude.shape
    radius = int(np.floor(min(h, w) / 2.0))

    # ---- radial band energies ----
    prof = radial_profile(magnitude)          # (B, R)
    total = prof.sum(dim=1, keepdim=True) + EPS
    normalized = prof / total                 # energy share per radius

    def band_energy(lo: float, hi: float) -> torch.Tensor:
        """Sum of normalized radial energy in the relative radius window."""
        start = int(lo * radius)
        end = min(int(hi * radius), radius)
        if end <= start:
            return torch.zeros(b, device=images.device)
        return normalized[:, start:end].sum(dim=1)

    low_energy = band_energy(0.0, 0.25)
    mid_energy = band_energy(0.25, 0.5)
    high_energy = band_energy(0.5, 1.0)
    high_ratio = high_energy / (low_energy + mid_energy + high_energy + EPS)

    # ---- spectral entropy / flatness ----
    flat = normalized * torch.log(normalized + EPS)  # (B, R)
    entropy = -flat.sum(dim=1)
    flatness = torch.exp(torch.mean(torch.log(normalized + EPS), dim=1)) / \
        (torch.mean(normalized, dim=1) + EPS)

    # ---- azimuthal anisotropy ----
    # Directional energy in 4 orientation bands via FFT quadrant asymmetry.
    center_y, center_x = h // 2, w // 2
    quadrants = [
        magnitude[:, :center_y, :center_x],
        magnitude[:, :center_y, center_x:],
        magnitude[:, center_y:, :center_x],
        magnitude[:, center_y:, center_x:],
    ]
    quad_means = torch.stack([q.mean(dim=(1, 2)) for q in quadrants], dim=1)  # (B,4)
    quad_total = quad_means.sum(dim=1, keepdim=True) + EPS
    quad_share = quad_means / quad_total
    anisotropy = quad_share.std(dim=1)

    # ---- radial statistics of the profile ----
    prof_mean = prof.mean(dim=1)
    prof_std = prof.std(dim=1)
    prof_max = prof.max(dim=1).values

    features = torch.stack([
        low_energy, mid_energy, high_energy, high_ratio,
        entropy, flatness, anisotropy,
        prof_mean, prof_std, prof_max,
        quad_share[:, 0], quad_share[:, 1], quad_share[:, 2], quad_share[:, 3],
        total.squeeze(1).log(),
        (high_energy / (mid_energy + EPS)),
    ], dim=1)  # (B, 16)

    if torch.isnan(features).any() or torch.isinf(features).any():
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def frequency_feature_dim() -> int:
    return FREQUENCY_FEATURE_DIM
