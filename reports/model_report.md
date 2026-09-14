# SignalScope — Model Report

> One-page summary for SIH 2026. All metrics are from public data only; the
> official held-out SIH score is reported separately by the organizers.

## TASK

Binary classification: **REAL** vs **AI-GENERATED** for a single image, with a
calibrated confidence score.

## DATA & SPLIT

| Dataset | Source | License | Real / AI | Split |
|---|---|---|---|---|
| CIFAKE | Kaggle | CC BY 2.0 | CIFAR-10 real vs Stable Diffusion v1.4 | 90k train / 10k val / 20k test |
| Defactify | Hugging Face | as published | real + SD 2.1 / SDXL / SD3 / DALL-E 3 / Midjourney 6 | public cross-generator train/eval |

- Split is **frozen and persisted** (`data/processed/*_manifest.csv`).
- Validation is carved from training with a fixed seed (10%); the test set is
  CIFAKE's own test split.
- **Organizer held-out data: never used** for training/tuning/thresholds.

## MODEL / APPROACH

- Backbone: **EfficientNet-B0** (≈4.0 M params), ImageNet-pretrained.
- Input: 224×224 RGB, ImageNet normalisation.
- Head: 2-class linear layer.
- Optimizer: AdamW (lr 1e-4, weight decay 1e-4); batch 16; up to 30 epochs.
- Augmentation (baseline): resize + flip + rotation ±10° + color jitter.
- Augmentation (stronger, experimental): + JPEG compression, mild blur, mild noise.
- Frequency features: experimental FFT branch (radial band energies, entropy,
  anisotropy) fused with the spatial embedding — compared, not assumed useful.
- Calibration: temperature scaling on validation only; threshold via Youden's J.

## METRICS

Shipped baseline (measured, preserved from the pre-refactor experiments):

| Benchmark | ROC-AUC | Macro-F1 | Accuracy | FPR | Threshold |
|---|---:|---:|---:|---:|---:|
| CIFAKE test — baseline (1 epoch) | 0.9973 | 0.9723 | 0.9723 | 0.0432 | 0.5 |
| Defactify eval — mixed model (5 epochs) | 0.8790 | 0.7303 | 0.8200 | 0.2700 | 0.5 |

Confusion matrix (CIFAKE baseline): `[[9568, 432], [122, 9878]]`.
Confusion matrix (Defactify, mixed): `[[73, 27], [81, 419]]`.

Unseen-generator AUC (leave-one-generator-out): **pending / blocked** — the
runner (`src/evaluation/evaluate_unseen.py`) is implemented but the Defactify
dataset cannot be downloaded in the evaluation environment
(`datasets-server.huggingface.co` → TLS connection closed; verified
2026-09-14). No unseen-split number is fabricated here.

## CALIBRATION

Temperature scaling fitted on the **local public adaptation validation set**
(16 images: 12 real + 4 AI) via `scripts/calibrate_ensemble.py`
(2026-09-14, ensemble logits):

| Quantity | Before (T=1.0) | After (T=0.7220) |
|---|---:|---:|
| Expected calibration error (ECE, 10 bins) | 0.1969 | 0.1859 |
| Brier score | 0.1103 | 0.1133 |
| Log loss | 0.3277 | 0.3183 |
| Youden-J threshold (val) | — | 0.7498 |

`model/calibration.json` now records `temperature: 0.7220` and
`calibrated: true` (previously the identity 1.0 / `calibrated: false`).

> **Caveat:** this fit is on a 16-image public adaptation set, not the official
> CIFAKE/Defactify validation split (10k images), which is not downloadable
> here. It is a *real* fit but **provisional** — re-fit with
> `python -m src.training.calibrate --dataset mixed` once the datasets are local.

## BASELINE

| | CIFAKE test AUC | Defactify eval AUC |
|---|---:|---:|
| Existing baseline (1 epoch) | 0.9973 | 0.6174 |
| Mixed model (5 epochs) | — | 0.8790 |

The 1-epoch CIFAKE baseline overfits its training generator; adding diverse
public cross-generator data materially improved generalisation (0.6174 → 0.8790
on the Defactify eval). This is the direction the final model takes.

## ROBUSTNESS

Implemented degradation suite (JPEG q50/q70/q90, resize 0.5×/1.5×, blur, noise,
brightness/contrast, crop, screenshot). Dataset-level benchmark results are
**pending** local data; see `reports/robustness_results.csv`.

## LIMITATIONS

Unseen generators may still fool the model; compression reduces evidence;
metadata may be absent; Grad-CAM is an explanation not proof; dataset bias;
probability ≠ truth; no exact-generator attribution; post-processing and unusual
real-world artifacts remain hard cases.

## BONUS MODULES

- Explanation (Grad-CAM + grounded text): **implemented**.
- Robustness benchmark: **implemented**.
- Provenance (EXIF + C2PA presence): **implemented**.
- Attribution / multimodal / active-defence: **not implemented** (documented in
  the README module table).
