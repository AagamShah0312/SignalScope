"""SignalScope FastAPI backend.

Endpoints:
    GET  /            service info
    GET  /health      health check
    POST /predict     analyse an image (multipart/form-data)

The model is loaded once at startup (never per-request) and inference uses
``model.eval()`` + ``torch.no_grad()``.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference import ModelNotFoundError, SignalScopePredictor
from app.preprocessing import ImageValidationError
from app.schemas import ErrorResponse, HealthResponse, PredictResponse
from src.config import REPO_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("signalscope.api")

cfg = load_config()

# Serve generated visualization artifacts (Grad-CAM overlays/heatmaps) that
# the frontend requests after a prediction.
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="SignalScope API",
    description=(
        "AI-generated image detection backend for SignalScope — "
        "Telling Real From Synthetic in the Age of Generative Media."
    ),
    version="2.0.0",
)

# The preview/proxy host is dynamic; keep CORS permissive but document it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------
predictor = None
model_error = None
try:
    predictor = SignalScopePredictor()
    logger.info("Model loaded: %s on %s", predictor.architecture, predictor.device)
except ModelNotFoundError as exc:
    model_error = str(exc)
    logger.warning("Model not loaded: %s", model_error)
except Exception as exc:  # noqa: BLE001 — surface any load failure clearly
    model_error = f"Model failed to load: {exc}"
    logger.exception("Model failed to load")


@app.exception_handler(ImageValidationError)
async def image_validation_handler(request, exc: ImageValidationError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/", response_model=HealthResponse)
def root():
    return health()


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
        "device": str(predictor.device) if predictor else "unavailable",
        "architecture": predictor.architecture if predictor else "EfficientNet-B0",
    }


@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}})
async def predict(
    file: UploadFile = File(...),
    include_explanation: bool = Query(True),
    include_metadata: bool = Query(True),
    include_robustness: bool = Query(False),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail=model_error or "Model unavailable.")

    request_id = uuid.uuid4().hex[:12]
    max_mb = float(cfg.inference.get("max_upload_mb", 15))
    logger.info("request=%s received filename=%r size_header=%r", request_id, file.filename, file.size)

    # Stream-safe read with a hard size limit (upload bomb protection).
    contents = await file.read()
    if len(contents) > int(max_mb * 1024 * 1024):
        logger.info("request=%s rejected: too large (%d bytes)", request_id, len(contents))
        raise HTTPException(status_code=413, detail="File too large.")

    logger.info("request=%s validating image", request_id)
    try:
        # Use a generated, safe output directory for visualizations.
        output_dir = OUTPUT_DIR / "api" / request_id
        result = predictor.predict(
            contents,
            filename=file.filename,
            include_explanation=include_explanation,
            include_metadata=include_metadata,
            include_robustness=include_robustness,
            output_dir=str(output_dir) if include_explanation else None,
            base_name="gradcam",
        )
    except ImageValidationError as exc:
        logger.info("request=%s validation error: %s", request_id, exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("request=%s inference error", request_id)
        raise HTTPException(status_code=500, detail="Internal inference error.")

    logger.info(
        "request=%s done verdict=%s p_ai=%.3f total=%.1fms",
        request_id, result["verdict"], result["probability_ai"], result["latency_ms"]["total_ms"],
    )

    # Map internal paths to relative URLs so the frontend can fetch them.
    if result.get("visualization"):
        vis = {}
        for key, value in result["visualization"].items():
            if value:
                rel = Path(value).resolve().relative_to(OUTPUT_DIR.resolve())
                vis[key] = f"/files/{rel.as_posix()}"
        result["visualization"] = vis

    response = {
        "verdict": result["verdict"],
        "label": result["verdict_label"],
        "binary_label": result["binary_label"],
        "probability_ai": result["probability_ai"],
        "probability_real": result["probability_real"],
        "confidence": result["confidence"],
        "threshold": result["threshold"],
        "status": result["status"],
        "model": result["model"],
        "explanation": result.get("explanation"),
        "visualization": result.get("visualization"),
        "provenance": result.get("provenance"),
        "robustness": result.get("robustness"),
        "latency_ms": result.get("latency_ms"),
        "disclaimer": result["disclaimer"],
    }
    return PredictResponse(**response)


# Serve the generated visualization files (mounted after route definitions).
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")
