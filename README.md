# SignalScope

**Telling Real From Synthetic in the Age of Generative Media**
SIH 2026 — Problem Statement 2 (C-433) — L. J. Institute of Engineering and Technology

SignalScope classifies an input image as **real** or **AI-generated**, reports performance on a
held-out test set that includes images from generators unseen during training, and produces a
faithful, human-readable explanation of the visual cues behind each verdict.

> Outputs are always framed as a likelihood ("likely AI-generated"), never an accusation. This
> project detects AI-generated imagery in general (scenes, objects, art, product shots) — it does
> **not** do face-swap deepfake detection of real individuals and does **not** adjudicate
> political claims or real-world events.

---

## 1. Modules built

| Module | Status |
|---|---|
| **Core** — real vs AI-generated classification | ✅ Required |
| A. Faithful explanation (Grad-CAM + grounded text) | ⬜ |
| B. Generator attribution | ⬜ |
| C. Robustness to degradation | ⬜ |
| D. Provenance & metadata (C2PA / EXIF) | ⬜ |
| E. Multimodal image–text consistency | ⬜ |
| F. Real-time / deployable interface | ⬜ |
| G. Active defence / adversarial analysis | ⬜ |

*Update the checkboxes above as modules are completed.*

## 2. Problem statement

Text-to-image models can now produce photorealistic images in seconds, which is fueling
misinformation, fraud, fake product listings, and manipulated "evidence." The hard part is not
just classifying real vs. fake — detectors that perform well on generators seen during training
often fail on images from a new, unseen generator, which is exactly the situation that matters in
the real world. A verdict alone is also not enough to build trust: without a clear explanation of
*why* an image was flagged, users have no way to judge or act on the result.

See [`report/01_Problem_Statement_and_Solution.md`](report/01_Problem_Statement_and_Solution.md)
for the full write-up.

## 3. Proposed solution / architecture

```
Image → Preprocessing → EfficientNet-B0 → Probability → Calibration → Real/AI → Grad-CAM → Explanation
```

- **Preprocessing** — resize, normalize, and augment for robustness to compression/resizing/screenshots.
- **EfficientNet-B0** — transfer-learning CNN backbone for feature extraction.
- **Probability** — raw real-vs-AI-generated likelihood from the backbone.
- **Calibration** — temperature scaling so confidence scores are honest, not just high.
- **Real / AI verdict** — calibrated probability thresholded into a label + confidence.
- **Grad-CAM** — saliency heat-map showing which regions drove the decision.
- **Explanation** — grounded, human-readable text describing the visual cues, generated from the heat-map.

Full rationale in [`report/01_Problem_Statement_and_Solution.md`](report/01_Problem_Statement_and_Solution.md).

## 4. Repository structure

```
SignalScope/
├── README.md                 ← you are here
├── requirements.txt
├── src/ or app/               ← source code
├── model/                     ← training/inference code + predict interface
│   └── weights/                ← via release link if large
├── report/
│   ├── 01_Problem_Statement_and_Solution.md
│   ├── 02_Project_Metadata.md
│   └── model_report.md        ← one-page model report (Section 7.3)
└── demo/                       ← link to demo video, screenshots
```

## 5. Setup & run instructions

```bash
# 1. Clone the repo
git clone https://github.com/adit301206/SignalScope.git
cd SignalScope

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a prediction on a new image
python model/predict.py --image path/to/image.jpg
```

*A judge should be able to go from clone to a prediction in under ~10 minutes. Update the
commands above once the actual entry point / CLI (or web app / notebook) is finalized.*

## 6. Datasets used

| Split | Source | Notes |
|---|---|---|
| Train / validation | Organizer-provided CIFAKE-style real-vs-synthetic set (real photos + images from disclosed generators such as Stable Diffusion) | MIT / open-licensed |
| Held-out test | Organizer-provided, evaluated only during judging | Includes real photos **and** synthetic images from generators *not* in the training set |
| Additional public data (if used) | e.g. GenImage | Cite source & license here |

Core task evaluation always runs on the organizers' held-out set via the `predict` interface —
never on data substituted by the team.

## 7. Reported metrics

*(Fill in after training — keep in sync with [`report/02_Project_Metadata.md`](report/02_Project_Metadata.md))*

| Metric | Value |
|---|---|
| Overall AUC (held-out) | — |
| Unseen-generator-split AUC (primary) | — |
| Macro-F1 | — |
| Accuracy @ chosen threshold | — |
| False-positive rate @ chosen threshold | — |
| Confusion matrix | see `report/model_report.md` |

## 8. Project metadata

Model name/version, dataset, training date, metrics, and commit history are tracked in
[`report/02_Project_Metadata.md`](report/02_Project_Metadata.md) — update this after every
training run.

## 9. Known limitations

*(Fill in honestly — which generators or degradations break the model, calibration edge cases, etc.)*

## 10. Demo video

🔗 [Link to 3–5 minute demo video] — shows the core running on a new image, plus any bonus modules.

## 11. Team & originality

- Team: SIH 2026, Problem Statement 2 (C-433), L. J. Institute of Engineering and Technology
- Third-party code/notebooks referenced: *(list here — required originality declaration)*
- AI coding assistants were used during development; the working system and its evaluation are what is scored.