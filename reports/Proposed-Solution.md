# SignalScope — Proposed Solution

*SIH 2026 — proposed Solution*

## 2. Proposed Solution

SignalScope is an end-to-end image authenticity pipeline built around a calibrated CNN classifier
and an explainability layer, structured as follows:

```
Image → Preprocessing → EfficientNet-B0 → Probability → Calibration → Real/AI → Grad-CAM → Explanation
```

**Stage-by-stage:**

- **Input image** — a single image is accepted (photo, product shot, screenshot, etc.).
- **Preprocessing** — the image is resized, normalized, and (during training) augmented to
  improve robustness to compression, resizing, and screenshotting.
- **EfficientNet-B0 backbone** — a transfer-learning CNN extracts visual features. EfficientNet-B0
  is chosen for a strong accuracy-to-compute tradeoff, which keeps inference fast enough for a
  usable interface while still being expressive enough to pick up on subtle generative artifacts
  (texture, frequency, and lighting inconsistencies).
- **Probability score** — the backbone outputs a raw real-vs-AI-generated likelihood.
- **Calibration** — temperature scaling (or an equivalent calibration method) is applied so the
  reported confidence is honest — a "0.8 confidence" verdict should actually be right about 80% of
  the time, rather than an overconfident raw softmax score. This matters directly for the
  fixed-operating-point metric (accuracy and false-positive rate) the problem statement requires.
- **Real / AI-generated verdict** — the calibrated probability is thresholded into a final label
  with a confidence score, presented as a likelihood rather than a certainty.
- **Grad-CAM** — a gradient-based class activation map highlights the image regions that most
  influenced the model's decision, producing a visual heat-map rather than a black-box label.
- **Explanation** — the heat-map is translated into a short, human-readable explanation of the
  cues behind the verdict (e.g. "handle geometry is physically inconsistent; reflections don't
  match the light source"), so a non-expert can understand and act on the result.

**Why this design fits the challenge's grading priorities:**
- The **unseen-generator split AUC** is the primary metric, so the pipeline emphasizes
  augmentation and calibration to generalize rather than memorize artifacts specific to the
  training generators.
- The **explanation bonus (Module A)** is scored on faithfulness and localization, not fluency —
  which is why Grad-CAM sits directly upstream of the text explanation, grounding the language in
  an actual saliency region instead of a generic guess.
- The **responsible-framing requirement** (likelihood, not accusation) is built into the final
  presentation layer, not bolted on afterward.

This core pipeline can be extended with the optional bonus modules (generator attribution,
degradation robustness, provenance/metadata, image-text consistency, deployable UI, adversarial
robustness) without changing the core architecture.