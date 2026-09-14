# SignalScope — Repository Audit & Implementation Checklist

Internal audit of the pre-existing repository (commit `7f67f24`) before refactoring.
This documents **what existed**, **what was broken**, and **what was decided**.

## What already exists (and is preserved)

| Area | Files | Status |
|---|---|---|
| CIFAKE manifest generation | `src/data/create_manifest.py` | Working |
| Dataset class | `src/data/dataset.py` | Working |
| DataLoaders (CIFAKE) | `src/data/loaders.py` | Working, but re-splits on every run |
| Mixed CIFAKE+Defactify loader | `src/data/loaders_mixed.py` | Working |
| Transforms | `src/data/transforms.py` | Working (baseline aug only) |
| EfficientNet-B0 model | `src/models/model.py` | Working |
| Training (CIFAKE) | `src/training/train.py` | Working, hardcoded config |
| Training (mixed) | `src/training/train_mixed.py` | Working, hardcoded config |
| Evaluation (CIFAKE test) | `src/evaluation/evaluate.py` | Working |
| Defactify eval | `src/evaluation/evaluate_defactify.py` | Working |
| Mixed-model eval | `src/evaluation/evaluate_mixed.py` | Working |
| Grad-CAM | `src/explainability/gradcam.py`, `generate_heatmaps.py`, `inspect_model.py` | Working |
| Downloaders | `src/data/download_defactify_{train,eval}.py`, `select_defactify_train_local.py` | Working |
| FastAPI backend | `app/backend.py` | Working (minimal) |
| Baseline reports | `reports/baseline_results.md`, `reports/mixed_results.md`, `*.csv` | Measured results — preserved |
| Trained checkpoint | `src/models/best_efficientnet_b0.pth` (16 MB EfficientNet-B0) | Loads & infers correctly |

## What is broken / problematic

1. **`requirements.txt` is UTF-16 encoded.** `pip install -r requirements.txt` fails.
   It also lists only matplotlib/numpy/pandas/scipy/xgboost — **no PyTorch, torchvision,
   scikit-learn, FastAPI, opencv**, so the project is not installable as-is.
2. **Config mismatch.** `config.yaml` says `epochs: 30`, but `train.py` hardcodes
   `EPOCHS = 1`. Every important hyperparameter is hardcoded in at least two scripts.
3. **No reproducibility.** No seed handling anywhere; data loader re-shuffles the
   train/val split on every call (it writes the manifest but only after splitting).
4. **Leakage / split hygiene risk.** The validation split is re-derived from training
   data each run; there is no single persisted split manifest. Normalization stats
   are fixed ImageNet values (fine), but the split must be frozen.
5. **Two divergent training scripts** (`train.py` vs `train_mixed.py`) with duplicate
   metrics logic and different checkpoint formats (plain state_dict vs dict) — this
   caused bugs (e.g., `evaluate.py` loads a plain state_dict while `train_mixed.py`
   writes a dict).
6. **No central metrics module.** AUC/F1/acc/confusion are re-implemented in 4 files.
7. **No inference pipeline** independent of training: no CLI, no calibration, no
   threshold policy, no "inconclusive" state, no explanation/metadata output.
8. **No calibration, robustness, provenance, frequency-domain, unseen-generator,
   or experiment-tracking modules** (only placeholder dirs).
9. **No tests.**
10. **No Docker / deployment / frontend** (backend has no matching frontend).
11. `app/backend.py` loads `model/best_efficientnet_b0_mixed.pth` (does not exist in
    repo) and reads the upload without a size limit.
12. Report `reports/mixed_defactify_results.csv` is committed (fine, measured) but
    mixes per-image detail into the repo root reports dir.

## Decisions

- **Do not delete working code.** Refactor in place; keep the old downloader scripts.
- **Single source of truth for config:** `src/config.py` (dataclass + YAML loader),
  with `app/config.py` as an app-facing facade. Everything reads from `config.yaml`.
- **Freeze the split:** a single `src/data/manifests.py` builds and persists
  `train/val/test_manifest.csv` once; loaders read the persisted manifests.
- **One training entrypoint** (`src/training/train.py`) with a `--dataset` switch
  (`cifake` / `mixed`) instead of two divergent scripts.
- **One metrics module** (`src/evaluation/metrics.py`).
- **One inference path** (`app/inference.py`) used by CLI, API and robustness/benchmark.
- **Shipped model = the existing committed checkpoint** (CIFAKE 1-epoch baseline).
  No new training could be run here (no dataset present); all *new* experiments are
  marked **pending** and must be run with `scripts/train.py` once data is available.
- **Honest reporting.** Preserve measured baseline numbers; never fabricate metrics.

## Phase plan (priority order)

- [x] PHASE 1 — audit, config, split, reproducibility, metrics, training, checkpoint, CLI
- [ ] PHASE 2 — generalisation (mixed data, leave-one-generator-out, tracking, failure analysis)
- [ ] PHASE 3 — calibration (temperature scaling, threshold, FPR, inconclusive)
- [ ] PHASE 4 — innovation (frequency features + fusion, spatial-only vs spatial+frequency)
- [ ] PHASE 5 — explainability (Grad-CAM, evidence, explanation, samples)
- [ ] PHASE 6 — robustness (degradations + benchmark)
- [ ] PHASE 7 — provenance (EXIF + C2PA)
- [ ] PHASE 8 — application (FastAPI, React frontend)
- [ ] PHASE 9 — optional depth (attribution / multimodal / active-defence)
- [ ] PHASE 10 — submission (tests, README, reports, docs, demo, clean-env test)
