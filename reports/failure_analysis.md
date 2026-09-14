# SignalScope — Failure Analysis

Honest account of known failure modes. Categories follow the problem
statement's template.

## 1. False positives (real → AI)

**What failed?** In the Defactify cross-generator evaluation the mixed model
flagged 27 of 100 real images as AI (FPR 0.270); the CIFAKE-only baseline was
worse (FPR 1.0 on Defactify real images).

**Why?** The model learned CIFAKE-style "synthetic look" cues and over-applied
them to a different real-image domain. Domain shift in the real class is as
harmful as unseen generators in the fake class.

**Mitigation attempted.** Adding a diverse real-image sample (Defactify real
images) during training reduced FPR from 1.0 → 0.270.

**Did it help?** Yes, substantially — but 27% is still too high for production,
so the verdict language stays probabilistic.

## 2. False negatives (AI → real)

**What failed?** Midjourney 6 was the hardest unseen generator (85% detection
vs 96% for DALL-E 3), and Stable Diffusion 3 (73%) also leaked through.

**Why?** These generators were absent or under-represented in the 1-epoch /
5-epoch baselines; high-quality diffusion outputs leave weaker learned artifacts.

**Mitigation attempted.** Mixed training with samples of all five generators.

**Did it help?** Yes (per-generator detection rates improved). Remaining gap is
why unseen-generator AUC is the primary optimisation target.

## 3. Diffuse / unreliable explanations

**What failed?** On several demo images the Grad-CAM attention was diffuse, so
the explanation reports spread-out attention instead of a crisp region.

**Why?** EfficientNet-B0's final feature map is 7×7; global texture cues spread
activation broadly.

**Mitigation attempted.** The explanation module detects low concentration and
explicitly downgrades its own certainty ("this explanation is weaker").

**Did it help?** Honesty, not accuracy — which is the intended behaviour.

## 4. Degraded-image failures

**What failed?** Robustness varies by degradation (JPEG, resize, blur, noise,
crop, screenshot); see `reports/robustness_results.csv`.

**Why?** Downscaling and heavy JPEG remove high-frequency evidence the model
relies on.

**Mitigation attempted.** Stronger augmentation (JPEG/blur/noise) during
training; robustness benchmark to measure, not hide, the drop.

**Did it help?** Pending dataset-local validation; direction documented in
`config.yaml`.

**Measured (per-image demo, 2026-09-14, `reports/robustness_demo_results.csv`).**
The ensemble held its verdict across all 13 degradations on 6 of 7 demo images.
One honest failure surfaced: `ai_landscape` (a photorealistic generated scene)
sits at P(AI) ≈ 0.51 with the ensemble — near the decision boundary — and its
binary label flipped under 6 of 13 degradations. The real-photo adaptation
member of the ensemble pulls this image toward "real" because it resembles the
real-photo adaptation set. This is the known cost of the real-photo fix and is
why the demo script uses `ai_ceramic_mug` (P(AI) = 0.9987, 0 flips) as the
primary AI example.

## 5. Low-confidence / inconclusive cases

**What failed?** Predictions near the decision boundary produce weak verdicts.

**Why?** Genuine ambiguity — the model is near its threshold.

**Mitigation attempted.** A dedicated `Inconclusive` state with reduced
confidence, rather than forcing a confident-looking label.

**Did it help?** This is a responsible-use feature, not an accuracy fix.

## 6. Provenance limitations

**What failed?** C2PA detection is a presence check, not cryptographic
verification; EXIF is frequently stripped.

**Why?** Full C2PA verification needs the official `c2pa` library + trust store
(trade-off: project fragility vs depth).

**Mitigation attempted.** Keep provenance separate from the visual verdict;
"not detected" never maps to "AI-generated".

**Did it help?** Prevents false provenance claims.
