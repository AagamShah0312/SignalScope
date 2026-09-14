# SignalScope

**Telling Real From Synthetic in the Age of Generative Media**

SignalScope is a real-vs-AI-generated image classifier built for the SIH 2026
internal hackathon (Problem Statement 2). It accepts a single image, returns a
calibrated likelihood verdict, and explains its decision with Grad-CAM — while
communicating uncertainty honestly and never making absolute accusations.

---

## Problem

AI-generated imagery is increasingly realistic. Synthetic media can erode
trust in what we see online, yet most detectors overfit to a single generator
and fail silently on unseen ones. The core engineering challenge is
**generalisation to unseen generators** — not just high accuracy on a single
benchmark like CIFAKE.

## Solution

SignalScope is a detector + explanation system:

- an **EfficientNet-B0** backbone (ImageNet-pretrained) for spatial features,
- an optional **frequency-domain** feature branch (experimental),
- **temperature-scaling calibration** so the confidence score is honest,
- a responsible **verdict policy** (`Likely AI-generated` / `Likely real` /
  `Inconclusive`),
- **Grad-CAM** visualisations with grounded, non-hallucinated explanations,
- a **robustness** benchmark (JPEG / resize / blur / noise / crop / screenshot),
- optional **provenance** reporting (EXIF + C2PA presence),
- a FastAPI backend + React frontend + a dependency-free CLI.

## Core task

1. Accept a single image.
2. Classify it as **REAL** or **AI-GENERATED**.
3. Output a confidence score.
4. Evaluate on a held-out public test set (CIFAKE / Defactify).
5. Report ROC-AUC, Macro-F1, accuracy, FPR and confusion matrix.

The official SIH held-out test set is **never** used for training, tuning or
threshold selection. See [DATASETS.md](DATASETS.md).

---

## Bonus modules

| Module | Status |
|---|---|
| A. Faithful explanation (Grad-CAM + grounded evidence) | **IMPLEMENTED** |
| B. Generator attribution | NOT IMPLEMENTED |
| C. Robustness (degradation benchmark) | **IMPLEMENTED** |
| D. Provenance / metadata (EXIF + C2PA presence) | **IMPLEMENTED** (C2PA presence check, not cryptographic verification) |
| E. Multimodal (caption consistency) | NOT IMPLEMENTED |
| F. Deployable interface (API + frontend + CLI) | **IMPLEMENTED** |
| G. Active defence analysis | PARTIAL (degradation failure analysis only; no adversarial attacks) |

## Architecture

```
Image
  ↓
Preprocessing (RGB, resize 224×224, ImageNet normalisation)
  ↓
Spatial features (EfficientNet-B0)  ──┬── (optional) Frequency features (FFT)
  ↓                                   ↓
Feature fusion → classification head
  ↓
Calibration (temperature scaling)
  ↓
Verdict policy  →  Likely AI-generated / Likely real / Inconclusive
  ↓
Grad-CAM → grounded explanation → responsible UI
```

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/adit301206/SignalScope.git
cd SignalScope
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> CPU-only machine? Install a smaller PyTorch first:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
> then `pip install -r requirements.txt`.

**Model weights.** Two committed checkpoints (EfficientNet-B0, 16 MB each)
make prediction work immediately after install:

- `src/models/best_efficientnet_b0.pth` — CIFAKE baseline
- `src/models/fine_tuned_model.pth` — real-photo adaptation

The default predictor is a **soft-voting ensemble** of the two (config
`model.ensemble`, weights 0.3 / 0.7) that fixes the known failure where the
CIFAKE-only baseline labelled ordinary real photographs as AI-generated. See
[reports/real_photo_adaptation.md](reports/real_photo_adaptation.md). To train
a better model, see [Training](#training). If weights are missing, every
entrypoint fails with a clear, controlled error (and the ensemble falls back to
whichever members are present).

---

## Quick start — prediction

### CLI

```bash
python -m app.inference demo/ai_generated/ai_ceramic_mug.jpg
# JSON output:
python -m app.inference demo/ai_generated/ai_ceramic_mug.jpg --json
# also run robustness checks:
python -m app.inference demo/ai_generated/ai_ceramic_mug.jpg --robustness
```

### API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Then open the interactive Swagger docs at <http://localhost:8000/docs> and use
`POST /predict` (multipart `file`). Or with curl:

```bash
curl -F "file=@demo/ai_generated/ai_ceramic_mug.jpg" http://localhost:8000/predict
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 (proxies to the API on :8000)
```

---

## Dataset

| Dataset | Source | License | Use |
|---|---|---|---|
| CIFAKE | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | CC BY 2.0 (see dataset card) | Training + public test (CIFAR-10 real vs Stable Diffusion v1.4) |
| Defactify | [Hugging Face `Rajarshi-Roy-research/Defactify_Image_Dataset`](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) | As published on HF | Public cross-generator training/eval (SD 2.1 / SDXL / SD3 / DALL-E 3 / Midjourney 6) |

Downloaders live in `src/data/` (`download_defactify_train.py`,
`download_defactify_eval.py`, `select_defactify_train_local.py`). Full details
in [DATASETS.md](DATASETS.md).

**The organizer-held-out SIH test set is never downloaded or used here.**

## Training

All hyperparameters come from `config.yaml`. Defaults: EfficientNet-B0,
ImageNet-pretrained, AdamW, lr 1e-4, weight decay 1e-4, batch 16, 30 epochs.

```bash
python -m src.training.train                 # CIFAKE only
python -m src.training.train --dataset mixed # CIFAKE + Defactify
python -m src.training.train --use-frequency-features   # spatial + frequency
python -m src.training.train --epochs 10 --batch-size 32 --learning-rate 1e-4
```

The split is frozen on first run (`data/processed/{train,val,test}_manifest.csv`)
and reused. Outputs: `model/best_model.pth`, `model/model_metadata.json`,
`experiments/runs/training_history.json`, and a row in
`reports/experiments.csv`.

### Real-photo adaptation (small-data fine-tune)

```bash
python scripts/build_finetune_dataset.py        # builds the adaptation set
python -m src.training.finetune                 # trains src/models/fine_tuned_model.pth
python scripts/compare_real_ai.py               # baseline vs fine-tuned vs ensemble
python scripts/reproduce_real_photo_false_positives.py   # original bug reproduction
```

`build_finetune_dataset.py` fetches OpenCV `samples/data` (via git), extracts
bundled scikit-image/scikit-learn photos, and augments the committed
`data/ai_synthetic/*.jpg` images to balance the classes. Full details in
[reports/real_photo_adaptation.md](reports/real_photo_adaptation.md).

## Calibration

```bash
python -m src.training.calibrate --dataset mixed
```

Fits temperature scaling on **validation data only**, writes
`model/calibration.json` and `reports/calibration_results.json`.

## Evaluation

```bash
python -m src.evaluation.evaluate          # public CIFAKE test split
python -m src.evaluation.evaluate_unseen   # leave-one-generator-out (Defactify)
python -m src.evaluation.evaluate_robustness --limit 200
```

## Model

EfficientNet-B0 (~4.0 M parameters) with a 2-class head. Pretrained on
ImageNet, fine-tuned for real-vs-AI. Chosen for a strong accuracy/throughput
tradeoff — a judge can run a prediction on CPU in well under a minute.

The default detector is a soft-voting ensemble of two specialisations:

| Member | Role |
|---|---|
| CIFAKE baseline | strong at flagging synthetic imagery |
| Real-photo adaptation | strong at not flagging real photographs |

This was introduced after a reproduced failure (6/8 real photos labelled
AI-generated by the baseline); the ensemble lowers real-photo false positives
from 94.8% to 5.2% while detecting 92.3% of synthetic images on the public
adaptation set. See [reports/real_photo_adaptation.md](reports/real_photo_adaptation.md).

## Metrics

Measured results (shipped models, **public data only** — the official SIH
held-out score is computed separately by the organizers):

| Public benchmark | ROC-AUC | Macro-F1 | Accuracy | FPR (thr 0.5) |
|---|---:|---:|---:|---:|
| CIFAKE test (baseline, 1 epoch) | 0.9973 | 0.9723 | 0.9723 | 0.0432 |
| Defactify eval (mixed model, 5 epochs) | 0.8790 | 0.7303 | 0.8200 | 0.2700 |

Confusion matrices `[[TN, FP], [FN, TP]]`:

- CIFAKE baseline: `[[9568, 432], [122, 9878]]`
- Defactify mixed model: `[[73, 27], [81, 419]]`

**Overall vs unseen-generator AUC** (kept separate, on purpose):

- **Overall / public-benchmark AUC** — the two rows above, on CIFAKE's and
  Defactify's own public test splits.
- **Unseen-generator (leave-one-out) AUC** — the metric the task is actually
  judged on — is **still pending**: the runner
  (`src/evaluation/evaluate_unseen.py`) is implemented and writes
  `reports/unseen_generator_results.csv`, but the Defactify dataset cannot be
  downloaded in the evaluation environment (`datasets-server.huggingface.co`
  → TLS connection closed). It is **not fabricated**.

The 30-epoch retrain, frequency-feature comparison, and leave-one-generator-out
results are **pending** the public datasets. See
[reports/model_report.md](reports/model_report.md).

## Calibration status

Temperature scaling (and isotonic regression) are implemented and fitted on
**validation data only** (`src/training/calibration.py`, `scripts/calibrate.py`).

A real fit has been produced against the **local public adaptation validation
set** (16 images) via `scripts/calibrate_ensemble.py`:

| | Before (T=1.0) | After (T=0.7220) |
|---|---:|---:|
| Expected calibration error | 0.1969 | 0.1859 |
| Brier score | 0.1103 | 0.1133 |
| Log loss | 0.3277 | 0.3183 |

`model/calibration.json` now records `temperature: 0.7220`, `calibrated: true`
(previously the identity 1.0 / `calibrated: false`). This fit is **provisional**
— it uses the 16-image adaptation set because the official CIFAKE/Defactify
validation split is not downloadable here; re-fit with
`python -m src.training.calibrate --dataset mixed` when the datasets are local.

## Generalisation

The unseen-generator experiment (`src/evaluation/evaluate_unseen.py`) trains
without one generator and evaluates on it, reporting AUC / Macro-F1 / accuracy /
FPR. These are **public cross-generator experiments**, clearly labelled as such
— not the official SIH score.

## Explainability

Grad-CAM highlights the spatial regions that drive the prediction. Explanations
are assembled from measured quantities only (calibrated probability, activation
regions, attention concentration). No LLM and no hallucinated evidence. The
heatmap is a model explanation, **not proof** of any specific artifact.

## Robustness

`src/robustness/` applies realistic degradations (JPEG, resize, blur, noise,
brightness/contrast, crop, screenshot) and reports how the prediction changes.
Results are reported honestly, including failures. A measured per-image demo
run is in [`reports/robustness_demo_results.csv`](reports/robustness_demo_results.csv)
(regenerate with `python scripts/robustness_demo.py`); the dataset-level
benchmark is `python -m src.evaluation.evaluate_robustness --limit 200`
(pending public data).

## Provenance

EXIF (camera / software / timestamps — never GPS) and a best-effort C2PA
presence check are shown **separately** from the visual verdict. Missing
metadata is never treated as evidence of AI generation.

## Limitations

- Unseen generators can still fool the model.
- Compression and resizing can reduce detectable evidence.
- Metadata may be absent or stripped.
- Grad-CAM explains the model, not the world.
- Detectors carry dataset bias; probability is not absolute truth.
- Attribution (which exact generator) is not implemented.
- Synthetic images can be post-processed; real images can contain unusual artifacts.

## Demo

- Demo images: `demo/ai_generated/` (generic, no people).
- Grad-CAM sample set: `reports/explanation_samples/`.
- Demo video: **to be recorded** — follow [`demo/DEMO_SCRIPT.md`](demo/DEMO_SCRIPT.md)
  and paste the link here once published.

## Deployment

- Backend: `uvicorn app.api:app` (Render/Railway/AWS — CPU-friendly). A
  [`render.yaml`](render.yaml) blueprint deploys it to Render with a
  `/health` check.
- Frontend: `cd frontend && npm run build` (Vercel/Netlify). See
  [`frontend/vercel.json`](frontend/vercel.json) and set `VITE_API_BASE_URL`
  to the backend URL.
- Docker: `docker compose up --build` (see `Dockerfile`, `frontend/Dockerfile`,
  `docker-compose.yml`). **Note:** the Docker build was authored but not
  executed in the development environment.
- Keep-alive: a scheduled ping of the backend `/health` endpoint keeps Render's
  free tier warm — see [`deploy/keepalive.yml`](deploy/keepalive.yml) (GitHub
  Actions template) and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

Full step-by-step instructions: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

## Security

The upload path is hardened and covered by `tests/test_security.py` (31 tests):

- **Input validation** — empty/oversized files, disallowed extensions and MIME
  types, and content spoofing (a file claiming to be an image but containing
  HTML/scripts) are rejected with a clean 4xx.
- **Decompression-bomb defence** — image dimensions are checked from the file
  header *before* pixel decode, and Pillow's `DecompressionBombError` is caught.
- **Upload-bomb protection** — a hard byte limit (config `inference.max_upload_mb`)
  returns `413` for oversized uploads.
- **Path traversal** — the `/files` static mount blocks `../`; Grad-CAM output
  names are sanitized so they can never escape the output directory.
- **Metadata privacy** — EXIF extraction is a strict whitelist; GPS/location is
  never surfaced (only camera/software/date fields).
- **No information leakage** — error responses contain no stack traces or
  internal paths; filenames are never reflected into responses.
- **Responsible claims** — verdicts come from a fixed vocabulary
  (`Likely AI-generated` / `Likely real` / `Inconclusive`) with no absolute
  accusations.

## Team

*To be filled in by the team.*

## Responsible use

SignalScope provides a **likelihood assessment** based on visual evidence. It is
not a definitive determination, a deepfake detector, or a tool for identifying
real people. It must not be used for profiling, political claims, or as sole
evidence in any consequential decision.
