# SignalScope — Originality Declaration

This project did not copy a public real-vs-fake notebook wholesale. It is an
original implementation assembled from standard, openly documented building
blocks, all credited below.

## Third-party dependencies (libraries & pretrained models)

- **PyTorch / torchvision** — deep-learning framework; EfficientNet-B0
  architecture and ImageNet weights (`EfficientNet_B0_Weights.DEFAULT`).
- **scikit-learn** — metrics (ROC-AUC, F1, accuracy, confusion matrix),
  isotonic regression for optional calibration.
- **NumPy / pandas / SciPy** — numeric and tabular data handling.
- **Pillow / opencv-python-headless** — image I/O and Grad-CAM overlay
  rendering (heatmap colormap, resize, `addWeighted`).
- **matplotlib** — ROC-curve plots.
- **FastAPI / Uvicorn / python-multipart** — prediction API.
- **piexif** — safe EXIF extraction.
- **PyYAML** — configuration loading.
- **pytest / httpx** — testing.

## Datasets & public references

- **CIFAKE** (Kaggle) — training and public testing.
- **Defactify Image Dataset** (Hugging Face) — public cross-generator
  training/evaluation.
- **OpenCV `samples/data`** (Apache-2.0) and **scikit-image / scikit-learn
  bundled sample images** (BSD-3-Clause) — genuine photographs used for the
  real-photo adaptation fine-tune (identifiable people excluded). See
  `reports/real_photo_adaptation.md`.
- **Project-generated synthetic images** — `data/ai_synthetic/*.jpg`
  (10 images) were generated with an image-generation model for the synthetic
  training class of the adaptation fine-tune. They depict generic scenes only
  (landscapes, animals, objects); no real or identifiable people.
- **Grad-CAM** — the explanation method follows Selvaraju et al., "Grad-CAM:
  Visual Explanations from Deep Networks via Gradient-based Localization"
  (ICCV 2017); the code is an original implementation (~70 lines) over
  torchvision's EfficientNet hooks.
- **Temperature scaling** — Guo et al., "On Calibration of Modern Neural
  Networks" (ICML 2017); implemented directly with PyTorch LBFGS.

## AI coding assistants

AI coding assistance (an agentic coding tool) was used during development to
audit the repository, refactor configuration/training/evaluation modules, write
tests, and draft documentation. An image-generation model produced the
synthetic training images in `data/ai_synthetic/` (disclosed above). All
generated code was reviewed and executed against the project's test suite.

## What SignalScope implemented independently

- Central configuration system (`config.yaml` + `src/config.py`).
- Frozen train/val/test split with leakage prevention.
- Unified config-driven training loop with experiment tracking.
- Central metrics module with safe FPR/TPR computation.
- Temperature-scaling calibration + threshold selection (validation only).
- Frequency-domain feature extraction (FFT) + fusion model (experimental).
- Leave-one-generator-out cross-generator evaluation harness.
- Responsible verdict policy (Likely AI / Likely real / Inconclusive).
- Grounded explanation pipeline (Grad-CAM → evidence regions → text).
- Robustness degradation suite + benchmark.
- EXIF + C2PA-presence provenance module.
- FastAPI backend, React frontend, CLI, tests, and all documentation.

## Attribution notes

Everything above that is third-party is marked as such in code comments and in
this file. No external code is presented as original SignalScope work.
