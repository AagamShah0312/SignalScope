# SignalScope — Model Card

## Model details

- **Name:** SignalScope
- **Architecture:** EfficientNet-B0 (torchvision), 2-class head.
- **Pretrained weights:** ImageNet (torchvision `EfficientNet_B0_Weights.DEFAULT`).
- **Input:** 224×224 RGB image.
- **Output:** probability of AI-generated; calibrated verdict + confidence.
- **Checkpoint:** `src/models/best_efficientnet_b0.pth` (baseline).

## Intended use

Estimating whether a single generic image (scene, object, artwork, product
photograph) is likely real or AI-generated, with a calibrated confidence score
and a Grad-CAM explanation. Suitable for research, education and content-review
assistance.

## Prohibited use

- Identifying, profiling or making claims about **real identifiable people**.
- Face-swap / deepfake identification of specific individuals.
- Political claims or determining whether a real-world event occurred.
- Use as sole or definitive evidence in consequential decisions.
- Automated moderation with no human oversight.

## Training data

- CIFAKE (real: CIFAR-10; synthetic: Stable Diffusion v1.4).
- Optional public Defactify training sample (real + SD 2.1/SDXL/SD3/DALL-E 3/
  Midjourney 6) for cross-generator generalisation.

## Evaluation data

Public CIFAKE test split and public Defactify evaluation split. The organizer
held-out set is not used.

## Metrics

See [model_report.md](model_report.md). Baseline CIFAKE test: AUC 0.9973,
Macro-F1 0.9723, Accuracy 0.9723.

## Limitations & failure modes

- Degrades on unseen generators and heavily post-processed images.
- Confuses stylised/artwork real images with synthetic ones and vice versa.
- Grad-CAM attention may be diffuse and therefore less informative.
- Confidence may be miscalibrated for out-of-distribution inputs (mitigated by
  temperature scaling, but not eliminated).

## Ethical considerations

Detectors are not arbiters of truth. SignalScope uses likelihood language, keeps
provenance separate from the visual verdict, never treats missing metadata as
proof of AI, and does not identify people.

## Confidence interpretation

- `Likely AI-generated` / `Likely real`: probability beyond the configured
  thresholds (default 0.80 / 0.20).
- `Inconclusive`: probability between the thresholds — the model does not have
  enough evidence; this is an honest answer, not a failure.
