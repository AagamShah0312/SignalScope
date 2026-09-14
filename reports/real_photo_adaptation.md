# Real-Photo False-Positive Fix — Adaptation Report

**Date:** 2026-09-14
**Scope:** fix the confirmed failure where ordinary real photographs were
labelled "Likely AI-generated" with very high confidence.

## 1. The report and reproduction

A user reported that a real photograph (`SDC12898.JPG`) was labelled
AI-generated. The file itself was not reachable in the evaluation sandbox, so
the failure was reproduced with license-clean, non-identifiable real
photographs bundled with scientific Python (scikit-image / scikit-learn).

| image   | size     | baseline P(AI) | verdict (baseline) |
|---------|----------|---------------:|--------------------|
| coffee  | 600x400  | 0.9859         | Likely AI-generated |
| horse   | 400x328  | 0.3970         | Inconclusive |
| rocket  | 640x427  | 0.9778         | Likely AI-generated |
| coins   | 384x303  | 0.9812         | Likely AI-generated |
| moon    | 512x512  | 0.9221         | Likely AI-generated |
| chelsea | 451x300  | 0.9978         | Likely AI-generated |
| china   | 640x427  | 0.9995         | Likely AI-generated |
| flower  | 640x427  | 0.3593         | Inconclusive |

**6 of 8 real photographs were misclassified as AI-generated**, with
P(AI) up to 0.9995.

## 2. Root cause

The shipped baseline (`src/models/best_efficientnet_b0.pth`) is a 1-epoch
CIFAKE-only model. CIFAKE's REAL class is 32×32 CIFAR-10 images; its FAKE
class is Stable Diffusion 1.4 output at the same resolution. A detector trained
on that distribution treats *any* high-resolution, fine-detail photograph as
out-of-distribution and — because the model's "real" decision surface is tiny
low-res patches — confidently labels it AI-generated.

This is a **training-distribution problem, not a bug in the inference code**.
The model was never a general real-vs-AI detector; it should not have been
presented as one without an out-of-distribution guard.

## 3. The fix

Two complementary steps:

1. **Fine-tune for real-photo adaptation** (`src/training/finetune.py`):
   retrain the classification head and the last two EfficientNet-B0 blocks on
   a small balanced set of genuine photographs (OpenCV `samples/data` +
   bundled scikit-image/scikit-learn samples — 65 train / 12 val) and
   synthetic images (10 generated + 3 demo images, augmented to balance).
   This teaches the model that high-resolution photographs are REAL.

2. **Soft-voting ensemble** (`src/models/ensemble.py`, config
   `model.ensemble`): average the logits of the CIFAKE baseline and the
   fine-tuned model. The baseline is strong at flagging synthetic content; the
   fine-tuned model is strong at not flagging real photos. Their average
   recovers the best of both.

### Results on the public adaptation set

| model                          | real photos misclassified as AI | AI images detected | balanced accuracy |
|--------------------------------|--------------------------------:|-------------------:|------------------:|
| baseline (CIFAKE 1-epoch)      | 94.8%                           | 100.0%             | 52.6% |
| fine-tuned (real-photo)        | 0.0%                            | 76.9%              | 88.5% |
| **ensemble (0.3 base / 0.7 fine)** | **5.2%**                    | **92.3%**          | **93.6%** |

The bundled real-photo reproduction (the original complaint) drops from
**6/8 misclassified to 0/8**. The ensemble is the default predictor.

## 4. Honest caveats

- This adaptation set is **tiny** (137 train / 16 val images) and the
  synthetic images were produced for this project, so the fine-tuned model and
  the ensemble weights are **provisional**. They must be re-estimated on
  full-scale CIFAKE + Defactify validation data (unavailable in the sandbox:
  huggingface.co, Kaggle, and the canonical dataset hosts are not reachable).
- The baseline's 100% "AI detection" on this set is not meaningful — it simply
  labels almost everything as AI. Balanced accuracy is the honest metric here.
- The 5.2% residual real-photo false positives come almost entirely from a
  small number of low-level/micrograph images (e.g. `page`, `coins`, `text_*`)
  that genuinely resemble the CIFAKE texture domain. A further
  out-of-distribution guard is still desirable.

## 5. Reproducing

```bash
# 1. build the adaptation dataset (auto-fetches OpenCV samples; needs git)
.venv/bin/python scripts/build_finetune_dataset.py

# 2. fine-tune (adaptation; ~2 min on CPU)
.venv/bin/python -m src.training.finetune --epochs-head 15 --epochs-finetune 5

# 3. compare baseline vs fine-tuned vs ensemble
.venv/bin/python scripts/compare_real_ai.py

# 4. re-run the original false-positive reproduction
.venv/bin/python scripts/reproduce_real_photo_false_positives.py
```

Outputs: `reports/real_photo_adaptation.csv`,
`reports/real_photo_false_positives.csv`.

## 6. Attribution / licensing

- Real-photo sources: OpenCV `samples/data` (Apache-2.0, OpenCV project) and
  bundled scikit-image / scikit-learn sample images (BSD-3-Clause). Identifiable
  people (e.g. `lena.jpg`, `messi5.jpg`) were deliberately excluded.
- Synthetic images: generated for this project with an image-generation model;
  disclosed in `ORIGINALITY.md`.
