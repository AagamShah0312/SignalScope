# SignalScope — Dataset Documentation

This project uses **only public datasets**. The organizer-held-out SIH test set
is **not downloaded, not used for training, validation, threshold tuning, model
selection, or Grad-CAM development** — it is evaluated exclusively by the
organizers through the prediction interface.

## 1. CIFAKE

- **Name:** CIFAKE — Real and AI-Generated Synthetic Images.
- **URL / source:** <https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images>
- **License:** CC BY 2.0 (dataset card; the underlying CIFAR-10 is derived from
  images with their own provenance — do not redistribute the underlying real
  photos).
- **Contents:** 60,000 train + 20,000 test per class (REAL = CIFAR-10; FAKE =
  Stable Diffusion v1.4).
- **Classes:** REAL (0), AI_GENERATED (1).
- **Generators:** Stable Diffusion v1.4 (synthetic), CIFAR-10 (real).
- **Used for training:** yes (train split).
- **Used for validation:** yes (10% carved from train with a fixed seed).
- **Used for public testing:** yes (official CIFAKE test split).

## 2. Defactify Image Dataset

- **Name:** Defactify_Image_Dataset.
- **URL / source:** <https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset>
- **License:** as published on Hugging Face (see the dataset card).
- **Contents:** real + synthetic images across generators.
- **Classes:** REAL (0), AI_GENERATED (1) for the binary task; generator labels
  retained for cross-generator evaluation only.
- **Generators:** Stable Diffusion 2.1, Stable Diffusion XL, Stable Diffusion 3,
  DALL-E 3, Midjourney 6 (plus real).
- **Used for training:** optionally, as a small public cross-generator sample
  (`data/raw/defactify_train`).
- **Used for validation:** no (validation stays CIFAKE-based; the Defactify
  evaluation split is a public test).
- **Used for public testing:** yes (`data/raw/defactify_eval`), labelled
  "public cross-generator evaluation".

## Downloaders

- `src/data/download_defactify_train.py` — downloads a balanced training sample.
- `src/data/download_defactify_eval.py` — downloads a balanced evaluation sample.
- `src/data/select_defactify_train_local.py` — builds the same sample from a
  local copy of the full dataset (Parquet).

## Explicit statement

> **The organizer-held-out SIH test data was NOT used for training, validation,
> threshold tuning, or model selection.**
