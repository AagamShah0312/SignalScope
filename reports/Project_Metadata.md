# SignalScope — Project Metadata

*Maintained by Member 6. Update this file every time a new model is trained or a new version is
committed. Keep the most recent entry at the top of the log.*

## Current version snapshot

| Field | Value |
|---|---|
| Model name | SignalScope-EffNetB0 |
| Model version | v0.1 |
| Backbone | EfficientNet-B0 (transfer learning) |
| Dataset used | CIFAKE-style real/synthetic set (train/val) + held-out organizer test set |
| Additional public data (if any) | — |
| Training date | 2026-09-14 |
| Overall AUC | — |
| Unseen-generator-split AUC | — |
| Macro-F1 | — |
| Accuracy @ threshold | — |
| False-positive rate @ threshold | — |

## Version log

Add a new row every time metrics change or a new checkpoint is committed.

| Version | Date | Dataset | Overall AUC | Unseen-split AUC | Macro-F1 | Commit hash | Notes |
|---|---|---|---|---|---|---|---|
| v0.1 | 2026-09-14 | CIFAKE-style train/val | — | — | — | — | Initial baseline |

## Notes on fields

- **Model version**: bump this (v0.1 → v0.2 → ...) any time the architecture, training data, or
  calibration method changes materially.
- **Dataset used**: always list the core organizer-provided dataset plus any additional public
  datasets (e.g. GenImage), with citation, per the submission contract's data rules.
- **Training date**: the date the checkpoint being reported was actually trained, not the date
  this file was edited.
- **AUC values**: report both overall held-out AUC and the unseen-generator-split AUC separately —
  the unseen-split AUC is the primary ranking metric for this challenge.
- **Commit hash**: the exact commit the reported numbers correspond to, so results are
  reproducible from the README.