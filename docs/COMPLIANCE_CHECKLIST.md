# SignalScope — Compliance Verification Checklist

Honest, evidence-based self-assessment against the 46-item compliance
checklist. **Status:** `YES` / `PARTIAL` / `NO`. Every answer points at the
file or command that proves it; nothing is claimed that was not actually
implemented or measured.

> Note: the official SIH problem-statement PDF was not readable in the
> evaluation environment (the attachment did not persist to the workspace), so
> this checklist is answered against the items below as supplied, and against
> the repository itself. Where a metric genuinely requires the public datasets
> or a manual step that has not happened yet, it is marked `PARTIAL`/`NO` with
> the exact command that closes the gap — never fabricated.

---

## 1. Core task — classification

| # | Question | Status | Evidence |
|---|---|---|---|
| 1 | Single image in → binary label (real / AI-generated) | **YES** | `app/inference.py` returns `binary_label` ∈ {REAL, AI_GENERATED}; API `POST /predict` field `binary_label`; CLI prints the verdict. |
| 2 | Confidence score alongside the label | **YES** | `probability_ai`, `probability_real`, and `confidence` returned by the predictor and API (`app/inference.py`, `app/schemas.py`). |
| 3 | Transfer learning (pretrained backbone), not from scratch | **YES** | EfficientNet-B0 with ImageNet weights (`src/models/model.py`, `torchvision EfficientNet_B0_Weights.DEFAULT`). |
| 4 | Honest three-way split with no overlap | **YES** (code) | `src/data/manifests.py` freezes `train/val/test` manifests: test = CIFAKE's own test split; val = deterministic 10% carved from train with a fixed seed + fingerprint so it never re-splits. Not yet re-run locally (no dataset in repo). |
| 5 | Exposed `predict` interface (function/CLI/API) | **YES** | CLI `python -m app.inference <image>`; API `POST /predict`; `scripts/predict.py`; `app.inference.SignalScopePredictor.predict()`. |
| 6 | Minimal working interface that runs end-to-end | **YES** | React frontend (drag-and-drop) + FastAPI + CLI. Verified: `/health`, `/predict`, and Grad-CAM asset fetch all return 200 through the Vite proxy. |

## 2. Data rules

| # | Question | Status | Evidence |
|---|---|---|---|
| 7 | Trained only on public data, never the organizers' held-out set | **YES** | `DATASETS.md` explicit statement; training uses CIFAKE + Defactify only; organizer set is never downloaded or touched. |
| 8 | Additional public datasets cited | **YES** | `README.md` dataset table and `DATASETS.md` cite CIFAKE (Kaggle, CC BY 2.0) and Defactify (Hugging Face) with URLs and licenses. |
| 9 | No scraping / use of real identifiable people | **YES** | Demo images are generic (no people); OpenCV sample data was curated to exclude identifiable people (`scripts/build_finetune_dataset.py` excludes `lena.jpg`/`messi5.jpg`); policy in `ORIGINALITY.md` + `reports/model_card.md`. |
| 10 | Held-out/test kept separate from training/tuning (no leakage) | **YES** (code) | Frozen split manifests + fingerprint (`src/data/manifests.py`); calibration/threshold selection restricted to validation (`src/training/calibration.py`); test set never used for tuning. |

## 3. Primary metric & evaluation protocol

| # | Question | Status | Evidence |
|---|---|---|---|
| 11 | ROC-AUC reported as primary metric on held-out set | **YES** | `reports/model_report.md`: CIFAKE test AUC 0.9973; Defactify eval AUC 0.8790. Note: "held-out" = the public test splits; the official SIH held-out score is computed by organizers via the prediction interface. |
| 12 | ROC-AUC reported separately for the unseen-generator split | **PARTIAL / BLOCKED** | The leave-one-generator-out harness is implemented (`src/evaluation/evaluate_unseen.py`) and writes `reports/unseen_generator_results.csv`, but the CSV is empty: the Defactify dataset cannot be downloaded in this environment (`datasets-server.huggingface.co` → TLS connection closed, verified 2026-09-14; the downloader exits with `SSLZeroReturnError`). **To close:** run the downloader + `python -m src.evaluation.evaluate_unseen` where HF egress is available. |
| 13 | Macro-F1 reported on held-out set | **YES** | 0.9723 (CIFAKE), 0.7303 (Defactify) in `reports/model_report.md`. |
| 14 | Confusion matrix in the report | **YES** | `reports/model_report.md`, `reports/baseline_results.md`, `reports/mixed_results.md` (e.g. CIFAKE `[[9568,432],[122,9878]]`). Also now in `README.md`. |
| 15 | Fixed operating threshold with accuracy + FPR at that threshold | **YES** | Threshold 0.5 (config `evaluation.threshold`); accuracy and FPR reported at that threshold in `reports/model_report.md`. |
| 16 | Overall AUC vs unseen-generator AUC clearly distinguished | **PARTIAL → fixed** | `README.md` Metrics section now separates "public benchmark (overall) AUC" from "unseen-generator (leave-one-out) AUC — pending". `reports/model_report.md` likewise labels the unseen row "pending". |

## 4. Generalisation & robustness

| # | Question | Status | Evidence |
|---|---|---|---|
| 17 | Augmentation / adaptation for unseen-generator generalisation | **YES** | Stronger augmentation (JPEG/blur/noise) in `config.yaml` + `src/data/transforms.py`; mixed CIFAKE+Defactify training (`src/training/train.py --dataset mixed`); experimental frequency-feature branch; real-photo adaptation fine-tune + ensemble (`reports/real_photo_adaptation.md`). |
| 18 | Calibration step (temperature scaling) | **PARTIAL (provisional fit)** | Real temperature-scaling fit produced via `scripts/calibrate_ensemble.py` on the 16-image local adaptation validation set: `model/calibration.json` now records `temperature: 0.7220, calibrated: true` (was 1.0/false); ECE 0.1969 → 0.1859. **Not** fitted on the official 10k-image validation split — that data is not downloadable here (`--dataset mixed` exits with `FileNotFoundError`). Re-fit when data is local. |
| 19 | Actual evidence tested on unseen generators | **PARTIAL** | The mixed model was evaluated on the Defactify eval set spanning 5 generators (AUC 0.8790), but those generators were partially included in mixed training, so this is cross-generator — not strictly leave-one-out — evidence. Strict leave-one-out results are pending (`evaluate_unseen.py`). |

## 5. Explanation bonus (Module A)

| # | Question | Status | Evidence |
|---|---|---|---|
| 20 | Human-readable explanation of specific cues | **PARTIAL** | Explanations are grounded in measured Grad-CAM regions and attention concentration (`src/explainability/explanation.py`), but the wording is templated ("strongest response concentrated around …") rather than free-form per-image artifact naming. |
| 21 | Explanation includes a visual heat-map | **YES** | Grad-CAM heatmap + overlay images (`src/explainability/gradcam.py`); served to the frontend and saved under `reports/explanation_samples/`. |
| 22 | Heat-map localizes to the anomalous region (not the whole image) | **PARTIAL** | Localized maps are produced and activation concentration is quantified; when attention is diffuse the system explicitly says so (`reports/failure_analysis.md` §3). Independent visual verification was not performed (no vision tooling in the dev environment). |
| 23 | Manually verified cues correspond to real artifacts | **NO** | Not performed. The samples exist (`reports/explanation_samples/`), but no documented manual artifact audit has been done. |
| 24 | Uncertainty communicated honestly | **YES** | Verdict vocabulary is "Likely AI-generated" / "Likely real" / "Inconclusive"; explanation includes an explicit uncertainty sentence; diffuse attention downgrades its own confidence. |
| 25 | Avoids claims about real individuals / events | **YES** | No identity/event claims anywhere; `reports/model_card.md` prohibits them. |

## 6. Other bonus modules

| # | Question | Status | Evidence |
|---|---|---|---|
| 26 | Module B (attribution) separate metric | **N/A** | Not attempted — correctly not folded into the binary AUC. |
| 27 | Module C (robustness) degradation-vs-accuracy with results | **PARTIAL** | Degradation suite + per-image robustness check implemented (`src/robustness/`); dataset-level CSV was empty. A measured per-image demo run is now recorded in `reports/robustness_demo_results.csv` (see below). Full dataset-level benchmark still needs the public data. |
| 28 | Module D (provenance) actually reads C2PA/EXIF and explains combination | **YES** | EXIF is read (safe subset, GPS excluded — `src/provenance/exif.py`); C2PA is a documented presence check (`src/provenance/c2pa.py`); provenance is kept **separate** from the visual verdict and never treated as proof. |
| 29 | Module E (multimodal) caption-consistency | **N/A** | Not attempted. |
| 30 | Module F (deployment) usable interface with reasonable latency | **YES** | Drag-and-drop React frontend + API + CLI; CPU inference measured ~250 ms/image in the API logs. |
| 31 | Module G (adversarial) actual attacks + honest failures | **NO** | Not attempted. Degradation failure analysis exists (`reports/failure_analysis.md`), but no adversarial perturbation attacks were run. |

## 7. Responsible framing

| # | Question | Status | Evidence |
|---|---|---|---|
| 32 | Likelihood framing, never definitive accusation | **YES** | "Likely AI-generated" / "Likely real" / "Inconclusive" only (`src/explainability/explanation.py`); disclaimer on every response. Enforced by `tests/test_security.py::TestResponsibleClaims`. |
| 33 | Avoids identifying/profiling real people | **YES** | EXIF GPS redacted; no face/detection identity features; prohibited uses documented. |
| 34 | Scoped to general synthetic imagery, no face-swap deepfakes | **YES** | Model card scopes to scenes/objects/art/product shots; face-swap detection explicitly out of scope. |
| 35 | No political claims about real-world events | **YES** | Model card + responsible-use section prohibit political/event claims. |

## 8. Submission contract — repo & reproducibility

| # | Question | Status | Evidence |
|---|---|---|---|
| 36 | Required structure (`/README.md`, `/src` or `/app`, `/model`, `/report`, requirements file) | **YES** | `README.md`, `src/` + `app/`, `model/` (calibration + metadata json), `reports/` (incl. `model_report.md`), `requirements.txt`. |
| 37 | Stranger clones and reproduces a prediction in <10 min | **YES** | Both checkpoints are committed (16 MB each); `pip install -r requirements.txt` + `python -m app.inference demo/ai_generated/ai_ceramic_mug.jpg`. Verified locally (CLI + API + frontend). |
| 38 | README states which core + bonus modules were built | **YES** | README "Bonus modules" table marks each module IMPLEMENTED / PARTIAL / NOT IMPLEMENTED. |
| 39 | README lists dataset sources and licenses | **YES** | README dataset table + `DATASETS.md`. |
| 40 | README reports overall AUC, unseen AUC, macro-F1, confusion matrix | **PARTIAL → fixed** | AUC, macro-F1 and confusion matrices are now in the README Metrics section; unseen-split AUC is explicitly marked **pending** (not fabricated). |
| 41 | README describes architecture, calibration/robustness, limitations | **YES** | Architecture diagram + Model + Calibration + Robustness + Limitations sections. |
| 42 | Link to a 3–5 minute demo video | **NO** | Not recorded. A recording script is provided at `demo/DEMO_SCRIPT.md`; the link slot in the README is left for the team to fill after recording. |
| 43 | One-page model report (task/data/model/metrics/baseline/limitations) | **YES** | `reports/model_report.md` covers all six sections. |

## 9. Originality & timeline

| # | Question | Status | Evidence |
|---|---|---|---|
| 44 | Real development activity (10–15 Sep), not one dump commit | **YES** | Multiple commits on 2026-09-14 (`git log`): security suite, real-photo ensemble fix, frontend redesign, deployment config — each a focused change. |
| 45 | Originality declaration listing third-party code/references | **YES** | `ORIGINALITY.md` credits libraries, datasets, Grad-CAM/temperature-scaling papers, and the AI-assistance used. |
| 46 | AI-assistant code actually verified to run | **YES** | `ORIGINALITY.md` declares AI assistance; the system was executed end-to-end (73 tests passing, CLI predictions, API + frontend verified) rather than left as boilerplate. |

---

## Remaining gaps (explicit, un-faked)

1. **Unseen-generator (leave-one-out) AUC** — harness ready, but the Defactify
   dataset is **not downloadable** in this environment (HF egress blocked):
   `python -m src.evaluation.evaluate_unseen` (items 12, 19, 40).
2. **Temperature-scaling fit on the official validation split** — a provisional
   fit exists on the 16-image adaptation set (T = 0.7220); the official fit
   needs the CIFAKE/Defactify validation data:
   `python -m src.training.calibrate --dataset mixed` (item 18).
3. **Dataset-level robustness benchmark** — needs public data:
   `python -m src.evaluation.evaluate_robustness --limit 200` (item 27); a
   per-image demo run is already recorded in `reports/robustness_demo_results.csv`.
4. **Manual Grad-CAM artifact audit** — human step, not done (item 23).
5. **Demo video** — human step, not done; use `demo/DEMO_SCRIPT.md` (item 42).
