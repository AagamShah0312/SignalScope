"""Pydantic response schemas for the SignalScope API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str = "SignalScope"
    architecture: str = "EfficientNet-B0"


class EvidenceRegion(BaseModel):
    bbox: List[int] = Field(description="[x0, y0, x1, y1] in original pixels")
    relative_area: float
    mean_activation: float
    position: str


class Explanation(BaseModel):
    summary: str
    evidence: List[str] = []
    uncertainty: str
    confidence: float


class Visualization(BaseModel):
    original: Optional[str] = None
    heatmap: Optional[str] = None
    overlay: Optional[str] = None


class ExifInfo(BaseModel):
    present: bool = False
    fields: Dict[str, object] = Field(default_factory=dict)


class C2paInfo(BaseModel):
    present: bool = False
    status: str = "not_detected"
    note: str = "No machine-readable provenance credentials were found."


class Provenance(BaseModel):
    exif: ExifInfo = ExifInfo(present=False)
    c2pa: C2paInfo = C2paInfo()


class RobustnessCheck(BaseModel):
    included: bool = False
    conditions: Dict[str, object] = {}


class PredictResponse(BaseModel):
    verdict: str
    label: str
    binary_label: Optional[str] = None
    probability_ai: float
    probability_real: float
    confidence: float
    threshold: float
    status: str
    model: ModelInfo
    explanation: Optional[Explanation] = None
    visualization: Optional[Visualization] = None
    provenance: Optional[Provenance] = None
    robustness: Optional[RobustnessCheck] = None
    latency_ms: Optional[Dict[str, float]] = None
    disclaimer: str = (
        "SignalScope provides a likelihood assessment based on visual evidence. "
        "It should not be treated as definitive proof."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    architecture: str


class ErrorResponse(BaseModel):
    detail: str
