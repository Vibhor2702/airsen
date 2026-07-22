# AirSentinel

Satellite-based air pollution source attribution and enforcement zone ranking for Delhi.

Full write-up and results: [docs/SUBMISSION_SUMMARY.md](docs/SUBMISSION_SUMMARY.md)

---

## Repo Structure

```
/data-pipeline/       Colab satellite-pull notebook, raw VAHAN CSVs, CAAQMS scraper
/model/               Prithvi LoRA training, label building, augmentation scripts
/vehicle-emissions/   Vehicle emissions pipeline (VAHAN → emission index) + methodology doc
/enforcement/         Enforcement zone ranker — combines satellite attribution + vehicle index
/dashboard/           dashboard.html — the final fused output (open this in a browser)
/docs/                LORA_EVAL_REPORT.md and other reference docs
/team-pipeline/       Teammate's forecasting pipeline, fusion layer (fuse.py), dashboard source
```

---

## Final Model

**Prithvi-EO-2.0-100M-TL + LoRA (r=8)** — checkpoint: `prithvi_lora_best.pt`

- Val accuracy: **71.1%** (5-class, heuristic CAAQMS labels)
- Training set: 700 real S2 images × 8 D4 augmentations = 5,600 rows
- Per-class: crop_burning_smoke F1=0.896, industrial_haze F1=0.755, clear F1=0.439, traffic_heavy F1=0.260, dust F1=0.000
- Full classification report: [docs/LORA_EVAL_REPORT.md](docs/LORA_EVAL_REPORT.md)

---

## Enforcement Zone Ranking (2026-07-16)

| Rank | Zone | Source | Composite Score |
|------|------|--------|----------------|
| 1 | RK Puram | industrial_haze | 0.708 |
| 2 | Dwarka | dust | 0.572 |
| 3 | Rohini | dust | 0.493 |
| 4 | Jahangirpuri | traffic_heavy | 0.474 |
| 5 | Ashok Vihar | dust | 0.470 |

Full ranking: [enforcement/enforcement_ranking.csv](enforcement/enforcement_ranking.csv)

Score = `attribution_confidence × source_weight + 0.3 × vehicle_emission_load_index`
Source weights: traffic_heavy=1.0, industrial_haze=0.9, dust=0.4, crop_burning_smoke=0.3, clear=0.0

---

## Running the Pipeline

```bash
# 1. Rebuild enforcement ranking (from repo root)
.venv/Scripts/python.exe enforcement/enforcement_ranker.py

# 2. Regenerate dashboard
cd P2/airsentinel-master/shared
PYTHONPATH=src python -m shared.pipeline
```

---

## Honesty Badges

Every output row carries explicit provenance tags:
- `model_version`: "700-image LoRA r=8 -- final"
- `data_provenance`: "correlation-based (heuristic CAAQMS labels, not ground truth)"
- `land_use_note`: "land-use signal absent from this repo -- scoring uses only satellite + vehicle data"
- Vehicle emissions: DEMO DATA flag (real totals, split evenly across zones — not per-zone verified)
