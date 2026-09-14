"""Central configuration loading for SignalScope.

Every module in the project should obtain configuration through
:func:`load_config` (or the ``SignalScopeConfig`` dataclass) instead of
hardcoding hyperparameters.  The single source of truth is ``config.yaml``.

``SignalScopeConfig`` is a thin, dict-like wrapper that keeps the YAML
structure accessible both with attribute access (``cfg.training.epochs``)
and key access (``cfg["training"]["epochs"]``), so existing code that expects
a plain dict keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class ConfigDict(dict):
    """A ``dict`` that also exposes keys as attributes."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict):
            value = ConfigDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        value = super().get(key, default)
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[key] = value
        return value


@dataclass
class SignalScopeConfig:
    """Validated, resolved configuration for the whole project."""

    raw: ConfigDict = field(default_factory=ConfigDict)
    config_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Convenience accessors (read-only views of the resolved config)
    # ------------------------------------------------------------------
    @property
    def project(self) -> ConfigDict:
        return self.raw.get("project")

    @property
    def data(self) -> ConfigDict:
        return self.raw.get("data")

    @property
    def training(self) -> ConfigDict:
        return self.raw.get("training")

    @property
    def model(self) -> ConfigDict:
        return self.raw.get("model")

    @property
    def evaluation(self) -> ConfigDict:
        return self.raw.get("evaluation")

    @property
    def calibration(self) -> ConfigDict:
        return self.raw.get("calibration")

    @property
    def inference(self) -> ConfigDict:
        return self.raw.get("inference")

    @property
    def paths(self) -> ConfigDict:
        return self.raw.get("paths")

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __contains__(self, key: str) -> bool:
        return key in self.raw

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # ------------------------------------------------------------------
    # Path helpers (all resolved relative to the repo root)
    # ------------------------------------------------------------------
    def resolve(self, value: Any) -> Path:
        """Resolve a possibly-relative path against the repo root."""
        if value is None:
            raise ValueError("Path value is None")
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path


def _deep_convert(obj: Any) -> Any:
    """Recursively convert nested dicts/lists to ConfigDict."""
    if isinstance(obj, dict):
        return ConfigDict({k: _deep_convert(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_deep_convert(v) for v in obj]
    return obj


def _apply_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in any missing keys with safe defaults (so config.yaml can stay
    short while the code can rely on every key existing)."""
    defaults: Dict[str, Any] = {
        "data": {"validation_fraction": 0.10, "split_seed": 42},
        "training": {"batch_size": 16, "epochs": 30, "learning_rate": 1e-4,
                     "weight_decay": 1e-4, "optimizer": "adamw",
                     "num_workers": 2, "seed": 42, "selection_metric": "val_auc",
                     "augmentation": {"baseline": True, "stronger": False}},
        "model": {"pretrained": True, "dropout": 0.2,
                  "use_frequency_features": False},
        "evaluation": {"threshold": 0.5},
        "calibration": {"method": "temperature_scaling", "temperature": 1.0},
        "inference": {"high_threshold": 0.80, "low_threshold": 0.20,
                      "decision_threshold": 0.5, "device": "auto",
                      "max_upload_mb": 15},
        "paths": {"model_dir": "model", "reports_dir": "reports",
                  "experiments_dir": "experiments/runs", "output_dir": "output",
                  "checkpoint": "model/best_efficientnet_b0.pth"},
    }

    def merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    return merge(defaults, raw)


def load_config(config_path: Optional[str] = None) -> SignalScopeConfig:
    """Load and validate configuration from YAML.

    Args:
        config_path: Optional path to a YAML file. Defaults to
            ``<repo-root>/config.yaml``.

    Returns:
        A :class:`SignalScopeConfig` with attribute + key access.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Expected a config.yaml at the repository root."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid configuration in {path}: expected a mapping.")

    resolved = _apply_defaults(raw)
    return SignalScopeConfig(
        raw=_deep_convert(resolved),
        config_path=path,
    )
