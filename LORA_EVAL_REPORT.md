# LoRA Evaluation Report — Phase E

**Model:** Prithvi-EO-2.0-100M-TL + LoRA (r=8, alpha=16)
**Checkpoint:** `prithvi_lora_best.pt` (epoch 2, val acc 71.1%)
**Dataset:** labels_augmented.csv — 5,600 rows, group-aware val split (seed=42, 20%)
**Val set:** 1,120 rows across 140 unique (zone, date) groups
**Date:** 2026-07-22

---

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| dust | 0.000 | 0.000 | 0.000 | 48 |
| crop_burning_smoke | 0.858 | 0.939 | 0.896 | 424 |
| industrial_haze | 0.722 | 0.793 | 0.755 | 376 |
| traffic_heavy | 0.452 | 0.183 | 0.260 | 104 |
| clear | 0.403 | 0.482 | 0.439 | 168 |
| **accuracy** | | | **0.711** | **1120** |
| macro avg | 0.487 | 0.479 | 0.470 | 1120 |
| weighted avg | 0.669 | 0.711 | 0.683 | 1120 |

---

## Per-Class Analysis

### crop_burning_smoke — F1: 0.896 (strong)
Dominant class (424/1120 val samples, 37.9%). Model learned it well — high precision and recall. The 71.1% overall accuracy is largely driven by this class.

### industrial_haze — F1: 0.755 (solid)
Second largest class (376/1120, 33.6%). Decent performance; some confusion likely with crop_burning_smoke given spectral similarity.

### clear — F1: 0.439 (mediocre)
168 val samples (15.0%). Partially learned but precision is low — model over-predicts clear in ambiguous cases.

### traffic_heavy — F1: 0.260 (poor)
104 val samples (9.3%). Very low recall (0.183) — model misses ~82% of traffic_heavy instances. Likely confused with industrial_haze given similar haze signatures.

### dust — F1: 0.000 (failed)
48 val samples (4.3%). Zero predicted samples — model never outputs this class. Severely underrepresented relative to crop_burning_smoke and industrial_haze.

---

## Notes

- Overall accuracy (71.1%) is inflated by the two dominant classes (crop_burning_smoke + industrial_haze = 71.4% of val set). Macro F1 of 0.470 is the more honest summary metric.
- **Dust failure** is the clearest gap: 48 val samples with 0 recall means the model has no dust signal at all at LoRA r=8 expressivity. Weighted cross-entropy loss (upweighting dust ~9x) or a larger LoRA rank would be the next levers.
- **Traffic heavy underperformance**: 104 samples with recall 0.183. Class imbalance (traffic_heavy is 880/5600 = 15.7% of dataset) combined with haze confusion is the likely cause.
- Labels are CAAQMS heuristic (rule-based, not lab-verified). These numbers reflect agreement with the labelling rule, not ground-truth performance.
