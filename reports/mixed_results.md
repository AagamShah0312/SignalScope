# SignalScope Mixed-Data Experiment

## Experiment

The baseline SignalScope detector was trained using CIFAKE data only.

To improve cross-generator generalization, an additional 1,000 training images from the Defactify training split were added:

- 500 real images
- 100 Stable Diffusion 2.1
- 100 Stable Diffusion XL
- 100 Stable Diffusion 3
- 100 DALL-E 3
- 100 Midjourney 6

The Defactify evaluation set of 600 images was kept completely separate from training.

## Training

Model: EfficientNet-B0  
Image size: 224 × 224  
Batch size: 16  
Learning rate: 1e-4  
Epochs: 5  
Device: NVIDIA RTX 2050

Best CIFAKE validation ROC-AUC: 0.9990

## Cross-Generator Evaluation

The model was evaluated on 600 held-out Defactify images:

- 100 real
- 100 Stable Diffusion 2.1
- 100 Stable Diffusion XL
- 100 Stable Diffusion 3
- 100 DALL-E 3
- 100 Midjourney 6

### Overall Results

| Metric | CIFAKE Baseline | Mixed Model |
|---|---:|---:|
| ROC-AUC | 0.6174 | 0.8790 |
| Macro-F1 | 0.4545 | 0.7303 |
| Accuracy | 0.8333 | 0.8200 |
| FPR | 1.0000 | 0.2700 |
| FNR | 0.0000 | 0.1620 |

### Mixed Model Confusion Matrix

[[73, 27],
 [81, 419]]

## Per-Generator Detection

| Generator | Detection Rate |
|---|---:|
| DALL-E 3 | 96% |
| Midjourney 6 | 85% |
| Stable Diffusion XL | 83% |
| Stable Diffusion 2.1 | 82% |
| Stable Diffusion 3 | 73% |

## Interpretation

Adding a small amount of diverse cross-generator training data substantially improved generalization to the held-out Defactify evaluation set.

ROC-AUC improved from 0.6174 to 0.8790.

This indicates that the mixed-data model is substantially better at separating real and synthetic images from generators and domains different from the original CIFAKE training distribution.

The remaining false positives and false negatives demonstrate that the detector should be treated as a likelihood assessment rather than an absolute determination of authenticity.