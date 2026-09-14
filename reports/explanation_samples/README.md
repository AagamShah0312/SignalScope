# Explanation Samples

Real Grad-CAM outputs produced by the shipped SignalScope model
(`src/models/best_efficientnet_b0.pth`), generated with the CLI:

```bash
python -m app.inference demo/ai_generated/<name>.jpg \
    --output-dir reports/explanation_samples/<name>
```

Each folder contains `gradcam_original.jpg`, `gradcam_heatmap.jpg` and
`gradcam_overlay.jpg`.

## Current contents

| Sample | Source | Verdict (measured) | P(AI) |
|---|---|---|---|
| ai_ceramic_mug | generated demo image | Likely AI-generated | 0.997 |
| ai_landscape | generated demo image | Likely AI-generated | 0.827 |
| ai_abstract_art | generated demo image | Likely AI-generated | 0.939 |

## Note on real-image samples

Real-image explanation samples are **pending**: they require a real photo.
Populate them by running the same CLI on REAL images from the public CIFAKE
dataset (`data/raw/cifake/test/REAL/`) or the Defactify real set. The pipeline
is label-agnostic — it does not need a label to produce a heatmap and
explanation.
