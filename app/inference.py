"""SignalScope inference pipeline — independent of training.

The :class:`SignalScopePredictor` implements the full prediction flow:

    1. validate file / decode image
    2. convert to RGB + evaluation preprocessing
    3. run the detector (model.eval + torch.no_grad)
    4. apply calibration (temperature scaling)
    5. apply the decision threshold (binary REAL vs AI_GENERATED)
    6. apply the responsible verdict policy (likely AI / likely real / inconclusive)
    7. generate Grad-CAM + grounded explanation (optional)
    8. extract provenance metadata (EXIF + C2PA, optional)

Usage (CLI):

    python -m app.inference path/to/image.jpg
    python -m app.inference path/to/image.jpg --no-explanation --json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from app.config import CLASS_NAMES, load_config, resolve_device
from app.preprocessing import (
    ImageValidationError,
    preprocess_image,
    to_numpy_rgb,
    validate_file,
)
from src.evaluation.metrics import compute_metrics  # noqa: F401 (re-export convenience)
from src.explainability.evidence import (
    activation_concentration,
    region_position_label,
    top_activation_regions,
)
from src.explainability.explanation import build_explanation, build_verdict
from src.explainability.gradcam import GradCAM, save_visualization
from src.models.model import create_model, load_checkpoint
from src.provenance.c2pa import detect_c2pa
from src.provenance.exif import extract_exif
from src.training.calibration import load_calibration


class ModelNotFoundError(RuntimeError):
    """Raised when the model checkpoint cannot be located."""


class SignalScopePredictor:
    """Loads the detector once and runs calibrated, explainable inference."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.cfg = load_config(config_path)
        self.device = device or resolve_device(self.cfg.inference.get("device", "auto"))
        self.image_size = int(self.cfg.data.image_size)
        self.num_classes = int(self.cfg.data.num_classes)

        checkpoint = self._resolve_checkpoint(checkpoint_path)
        if not checkpoint.exists():
            raise ModelNotFoundError(
                f"Model checkpoint not found: {checkpoint}\n"
                "Train a model (scripts/train.py) or download the weights — "
                "see README.md."
            )

        self.model = create_model(num_classes=self.num_classes, pretrained=False)
        load_checkpoint(self.model, str(checkpoint), device=self.device)
        self.model.eval()

        self.calibration = load_calibration()
        self.temperature = float(self.calibration.get("temperature", 1.0))
        self.calibrated = bool(self.calibration.get("calibrated", False))

        self.decision_threshold = float(self.cfg.evaluation.threshold)
        self.high_threshold = float(self.cfg.inference.high_threshold)
        self.low_threshold = float(self.cfg.inference.low_threshold)

        self.architecture = "EfficientNet-B0"

    def _resolve_checkpoint(self, checkpoint_path: Optional[str]) -> Path:
        """Prefer an explicit path, then newly-trained weights, then the
        committed baseline checkpoint."""
        candidates = []
        if checkpoint_path:
            candidates.append(Path(checkpoint_path))
        else:
            model_dir = self.cfg.resolve(self.cfg.paths.model_dir)
            candidates += [
                model_dir / "best_model.pth",
                model_dir / "best_efficientnet_b0.pth",
                self.cfg.resolve("src/models/best_efficientnet_b0.pth"),
            ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else Path(checkpoint_path or "")

    # ------------------------------------------------------------------
    # Probability + verdict
    # ------------------------------------------------------------------
    def _calibrated_probabilities(self, logits: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(logits / max(self.temperature, 1e-6), dim=1).cpu().numpy()

    def _decide(self, p_ai: float) -> Dict[str, object]:
        verdict = build_verdict(p_ai, self.high_threshold, self.low_threshold)
        binary_label = "AI_GENERATED" if p_ai >= self.decision_threshold else "REAL"
        return {
            "verdict": verdict["verdict"],
            "verdict_label": verdict["label"],
            "status": verdict["status"],
            "binary_label": binary_label,
            "confidence": float(
                p_ai if verdict["verdict"] == "likely_ai_generated"
                else (1.0 - p_ai) if verdict["verdict"] == "likely_real"
                else max(p_ai, 1.0 - p_ai)
            ),
        }

    # ------------------------------------------------------------------
    # Grad-CAM (isolated: requires gradients, so it runs outside no_grad)
    # ------------------------------------------------------------------
    def _gradcam(self, tensor: torch.Tensor):
        gradcam = GradCAM(self.model)
        try:
            cam, predicted_class, _ = gradcam.generate(tensor)
            return cam, predicted_class
        finally:
            gradcam.remove_hooks()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def predict(
        self,
        image_source: Union[str, Path, bytes],
        filename: Optional[str] = None,
        include_explanation: bool = True,
        include_metadata: bool = True,
        include_robustness: bool = False,
        output_dir: Optional[str] = None,
        base_name: str = "gradcam",
    ) -> Dict[str, object]:
        timing: Dict[str, float] = {}

        # ---- 1. load & validate -------------------------------------------
        t0 = time.perf_counter()
        if isinstance(image_source, (bytes, bytearray)):
            original_bytes = bytes(image_source)
            image = validate_file(original_bytes, filename=filename)
        else:
            path = Path(image_source)
            original_bytes = path.read_bytes()
            image = validate_file(original_bytes, filename=path.name)

        timing["validation_ms"] = (time.perf_counter() - t0) * 1000

        # ---- 2. preprocess -------------------------------------------------
        tensor = preprocess_image(image, self.image_size).to(self.device)

        # ---- 3. inference --------------------------------------------------
        t1 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
        timing["inference_ms"] = (time.perf_counter() - t1) * 1000

        probabilities = self._calibrated_probabilities(logits)
        p_ai = float(probabilities[0, 1])
        p_real = float(probabilities[0, 0])

        decision = self._decide(p_ai)

        # ---- 4. Grad-CAM + explanation ------------------------------------
        visualization: Optional[Dict[str, str]] = None
        explanation: Optional[Dict[str, object]] = None
        regions = []

        if include_explanation:
            t2 = time.perf_counter()
            cam, _ = self._gradcam(tensor)
            cam_np = cam.numpy()
            original_np = to_numpy_rgb(image)

            regions = top_activation_regions(cam_np, original_np.shape[:2])
            concentration = activation_concentration(cam_np)
            for r in regions:
                r["position"] = region_position_label(r["bbox"], original_np.shape[:2])

            explanation = build_explanation(
                probability_ai=p_ai,
                regions=regions,
                concentration=concentration,
                image_shape=original_np.shape[:2],
                verdict=decision["verdict"],
                high_threshold=self.high_threshold,
                low_threshold=self.low_threshold,
            )

            if output_dir:
                visualization = save_visualization(original_np, cam_np, output_dir, base_name)
            timing["gradcam_ms"] = (time.perf_counter() - t2) * 1000

        # ---- 5. provenance -------------------------------------------------
        provenance: Optional[Dict[str, object]] = None
        if include_metadata:
            exif = extract_exif(image)
            c2pa = detect_c2pa(original_bytes)
            provenance = {"exif": exif, "c2pa": c2pa}

        # ---- 6. robustness (optional, secondary) ---------------------------
        robustness: Optional[Dict[str, object]] = None
        if include_robustness:
            robustness = self.robustness_check(image_source, original_bytes, filename)

        timing["total_ms"] = (time.perf_counter() - t0) * 1000

        return {
            "verdict": decision["verdict"],
            "verdict_label": decision["verdict_label"],
            "binary_label": decision["binary_label"],
            "probability_ai": p_ai,
            "probability_real": p_real,
            "confidence": decision["confidence"],
            "threshold": self.decision_threshold,
            "status": decision["status"],
            "model": {"name": "SignalScope", "architecture": self.architecture},
            "explanation": explanation,
            "visualization": visualization,
            "provenance": provenance,
            "robustness": robustness,
            "latency_ms": timing,
            "disclaimer": (
                "SignalScope provides a likelihood assessment based on visual evidence. "
                "It should not be treated as definitive proof."
            ),
        }

    # ------------------------------------------------------------------
    # Robustness check
    # ------------------------------------------------------------------
    def robustness_check(
        self,
        image_source: Union[str, Path, bytes],
        original_bytes: bytes,
        filename: Optional[str] = None,
    ) -> Dict[str, object]:
        from src.robustness.benchmark import run_robustness_check

        return run_robustness_check(
            predictor=self,
            image_source=image_source,
            original_bytes=original_bytes,
            filename=filename,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result: Dict[str, object]) -> None:
    print("\n" + "=" * 58)
    print("SignalScope Analysis")
    print("=" * 58)
    print(f"\nVerdict:\n  {result['verdict_label']}")
    print(f"\nConfidence:\n  {result['confidence'] * 100:.1f}%")
    print(f"\nP(AI-generated): {result['probability_ai']:.4f}")
    print(f"P(real):         {result['probability_real']:.4f}")
    print(f"Threshold:       {result['threshold']:.4f}")
    print(f"\nModel:\n  {result['model']['architecture']}")

    if result.get("explanation"):
        print("\nExplanation:")
        for line in result["explanation"]["evidence"]:
            print(f"  - {line}")
        print(f"\n  {result['explanation']['summary']}")

    if result.get("visualization"):
        print("\nVisualization:")
        for kind, path in result["visualization"].items():
            print(f"  {kind}: {path}")

    if result.get("provenance"):
        prov = result["provenance"]
        exif = prov["exif"]
        c2pa = prov["c2pa"]
        print("\nProvenance:")
        print(f"  EXIF: {'Present' if exif['present'] else 'Not detected'}")
        if exif["present"]:
            for k, v in exif["fields"].items():
                print(f"    {k}: {v}")
        print(f"  Content Credentials: {c2pa['status'].replace('_', ' ')}")

    if result.get("latency_ms"):
        lat = result["latency_ms"]
        print(f"\nLatency: inference {lat.get('inference_ms', 0):.1f} ms | "
              f"total {lat.get('total_ms', 0):.1f} ms")

    print(f"\n{result['disclaimer']}")
    print("=" * 58)


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SignalScope image analysis CLI")
    parser.add_argument("image", help="Path to the image to analyze")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--no-explanation", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--robustness", action="store_true", help="Run degradation checks")
    parser.add_argument("--output-dir", default=None, help="Directory for Grad-CAM images")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args(argv)

    try:
        predictor = SignalScopePredictor(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        )
    except ModelNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        result = predictor.predict(
            args.image,
            include_explanation=not args.no_explanation,
            include_metadata=not args.no_metadata,
            include_robustness=args.robustness,
            output_dir=args.output_dir or "output",
        )
    except ImageValidationError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
