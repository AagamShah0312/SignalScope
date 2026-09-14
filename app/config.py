"""Application-level configuration facade.

Delegates to :mod:`src.config` and adds device resolution plus the
user-facing class-name mapping used by the API and CLI.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.config import REPO_ROOT, SignalScopeConfig, load_config

__all__ = ["REPO_ROOT", "SignalScopeConfig", "load_config", "resolve_device"]

# Canonical label mapping — the single place that defines how internal class
# indices map to user-facing labels.
CLASS_NAMES = {0: "REAL", 1: "AI_GENERATED"}


def resolve_device(preference: Optional[str] = None) -> torch.device:
    """Resolve the compute device from a preference string.

    ``preference`` may be ``auto`` (default), ``cpu`` or ``cuda``.  Falls back
    to CPU whenever CUDA is unavailable.
    """
    preference = (preference or "auto").lower()
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is not available.")
    if preference == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_settings(config_path: Optional[str] = None) -> SignalScopeConfig:
    """Load the project configuration (cached per process)."""
    return load_config(config_path)
