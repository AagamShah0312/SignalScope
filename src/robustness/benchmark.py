"""Robustness benchmarking: run the detector under controlled degradations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from src.robustness.degradations import apply_degradations, default_degradations

from app.preprocessing import to_numpy_rgb, validate_file


def run_robustness_check(
    predictor,
    image_source,
    original_bytes: bytes,
    filename: Optional[str] = None,
    conditions: Optional[Dict] = None,
) -> Dict[str, object]:
    """Run the predictor on the original + degraded versions of one image.

    Returns ``{"included": True, "conditions": {name: {probability_ai, ...}}}``.
    Exceptions from individual degradations are recorded as ``error``.
    """
    conditions = conditions or default_degradations()

    if isinstance(image_source, (bytes, bytearray)):
        original = validate_file(original_bytes, filename=filename)
    else:
        original = validate_file(original_bytes, filename=Path(image_source).name)

    results: Dict[str, object] = {}
    for name, degraded in apply_degradations(original, conditions).items():
        if isinstance(degraded, Exception):
            results[name] = {"error": str(degraded)}
            continue
        try:
            buf = _encode(degraded)
            pred = predictor.predict(buf, filename=filename, include_explanation=False,
                                     include_metadata=False)
            results[name] = {
                "probability_ai": round(pred["probability_ai"], 4),
                "binary_label": pred["binary_label"],
                "verdict": pred["verdict"],
            }
        except Exception as exc:  # a degraded image must not crash the run
            results[name] = {"error": str(exc)}

    return {"included": True, "conditions": results}


def _encode(image) -> bytes:
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
