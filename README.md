# AirSentinel

Satellite-based air pollution source attribution and enforcement zone ranking for Delhi.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vibhor2702/airsen/blob/master/data-pipeline/AirSentinel_Satellite_Pull_OPTIMIZED.ipynb)

Full write-up and results: [docs/SUBMISSION_SUMMARY.md](docs/SUBMISSION_SUMMARY.md)

---

## Repo Structure

```
/data-pipeline/       Colab satellite-pull notebook, raw VAHAN CSVs, CAAQMS scraper
/model/               Prithvi LoRA training, label building, augmentation scripts
/vehicle-emissions/   Vehicle emissions pipeline (VAHAN → emission index) + methodology doc
/enforcement/         Enforcement zone ranker — combines satellite attribution + vehicle index
/dashboard/           index.html — the final fused output (open in browser or via Cloudflare Pages)
/docs/                LORA_EVAL_REPORT.md and other reference docs
/team-pipeline/       Teammate's forecasting pipeline, fusion layer (fuse.py), dashboard source
```

---

## Final Model

**Prithvi-EO-2.0-100M-TL + LoRA (r=8)** — checkpoint: `prithvi_lora_best.pt`

- Val accuracy: **78.9%** (5-class, heuristic CAAQMS labels)
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
cd team-pipeline/shared
PYTHONPATH=src python -m shared.pipeline
```

---

## Architecture

```
Sentinel-2 imagery (GEE)          CAAQMS ground sensors         VAHAN vehicle registry
        │                                  │                              │
        ▼                                  ▼                              ▼
  sort_drive_folder.py            CAAQMS scraper               category_mapper.py
  build_labels.py (heuristic)     AQI computation              BS6 emission factors
  augment_labels.py (D4 ×8)       time-series forecaster       emission load index
        │                                  │                              │
        ▼                                  │                              │
  Prithvi-EO-2.0-100M-TL                  │                              │
  + LoRA r=8 fine-tune                    │                              │
  → per-zone source classification        │                              │
  (dust / traffic / industrial / crop)    │                              │
        │                                  │                              │
        └──────────────────────────────────┴──────────────────────────────┘
                                           │
                                           ▼
                               enforcement_ranker.py
                         score = conf × source_weight + 0.3 × vei
                                           │
                                           ▼
                              enforcement_ranking.csv
                                           │
                                           ▼
                                  dashboard/index.html
                         (GRAP stage mapping · zone ranking · AQI forecast)
```

**Data flows:**
- **Satellite track** — GEE exports 13-zone Sentinel-2 tifs → heuristic labels from CAAQMS rules → D4 augmentation (700 → 5,600 images) → Prithvi LoRA fine-tune → per-zone attribution confidence
- **Forecasting track** — CAAQMS hourly PM2.5/PM10/NO2 → daily panel → 24/48/72h AQI forecast → GRAP stage lookup
- **Vehicle track** — VAHAN per-RTO registration counts + official BS6 emission factors → vehicle emission load index per zone
- **Fusion** — `fuse.py` joins all three signals → `enforcement_ranker.py` scores and ranks → `index.html` renders the live dashboard

---

## Live Dashboard

[airsentinal.pages.dev](https://airsentinal.pages.dev/)
