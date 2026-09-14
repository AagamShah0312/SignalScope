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

**Model weights.** A trained baseline checkpoint is committed at
`src/models/best_efficientnet_b0.pth` (EfficientNet-B0, 16 MB), so prediction
works immediately after install. To train a better model, see
[Training](#training). If weights are missing, every entrypoint fails with a
clear, controlled error.

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

## Metrics

Measured results (shipped baseline, public data only):

| Public benchmark | ROC-AUC | Macro-F1 | Accuracy | FPR |
|---|---:|---:|---:|---:|
| CIFAKE test (baseline, 1 epoch) | 0.9973 | 0.9723 | 0.9723 | 0.0432 |
| Defactify eval (mixed model, 5 epochs) | 0.8790 | 0.7303 | 0.8200 | 0.2700 |

New experiments (30-epoch training, frequency features, leave-one-generator-out)
are **pending** — the dataset is not present in this repository and results are
never fabricated. See [reports/model_report.md](reports/model_report.md).

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
Results are reported honestly, including failures.

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
- Demo video: to be added after the live recording.

## Deployment

- Backend: `uvicorn app.api:app` (Render/Railway/AWS — CPU-friendly).
- Frontend: `cd frontend && npm run build` (Vercel/Netlify).
- Docker: `docker compose up --build` (see `Dockerfile`, `frontend/Dockerfile`,
  `docker-compose.yml`). **Note:** the Docker build was authored but not
  executed in the development environment.

## Team

*To be filled in by the team.*

## Responsible use

SignalScope provides a **likelihood assessment** based on visual evidence. It is
not a definitive determination, a deepfake detector, or a tool for identifying
real people. It must not be used for profiling, political claims, or as sole
evidence in any consequential decision.
