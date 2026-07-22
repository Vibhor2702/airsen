# AirSentinel — Competition Submission Summary

## What the system does

AirSentinel is an air-quality enforcement decision-support tool for Delhi's 13 designated
hotspot zones. It ingests three independent data streams — satellite imagery (Sentinel-2),
ground sensor readings (CAAQMS), and vehicle registration records (VAHAN) — fuses them into
a per-zone ranked enforcement recommendation, and renders a live dashboard with GRAP
(Graded Response Action Plan) stage mapping. The pipeline runs end-to-end from raw data to
HTML dashboard: a Prithvi-EO-2.0 vision model classifies each zone's dominant pollution
source (dust, crop-burning smoke, traffic, industrial haze) from satellite imagery; a
physics-formula model computes a vehicle emission load index from real per-RTO registration
counts and official BS6 emission factors; an AQI forecaster predicts next-24-hour AQI; and
an enforcement ranker combines all three signals into a ranked list with a one-line
human-readable reason per zone, where every output row is explicitly tagged with its data
quality and model maturity.

---

## What is real, what is heuristic, and what is demo

| Component | Status | Detail |
|---|---|---|
| Sentinel-2 satellite images | **Real** | 247 original images (13 zones x 19 dates, Oct-Nov 2025 and Jun-Jul 2026) pulled from Google Earth Engine |
| CAAQMS ground-sensor readings | **Real** | Hourly PM2.5, PM10, NO2, CO, O3, SO2 from Delhi DPCC stations, scraped directly from the CAAQMS portal |
| VAHAN vehicle registration counts | **Real** | Per-RTO counts exported directly from vahan.parivahan.gov.in on 2026-07-21; 10 RTOs covering all 13 zones |
| BS6 emission factors (g/km) | **Real** | Official MoRTH/CMVR Schedule VI regulatory limits (GSR 889(E)); NOx 60 mg/km petrol, 80 mg/km diesel, PM 4.5 mg/km all categories; verified 2026-07-22 |
| Pollution source labels for model training | **Heuristic** | Rule-based from CAAQMS daily readings (e.g. high PM2.5 + low wind -> dust); not lab-verified ground truth. See `labels.py` for every rule. |
| Prithvi-EO model accuracy (73.9% val) | **Against heuristic labels only** | Measured on held-out heuristic-labeled images, NOT against Supersite or independent ground truth |
| AQI forecasts | **Real model, real data** | CAAQMS-trained time-series forecaster; forecasts for 24/48/72h horizon |
| Vehicle emission load index | **Real formula, demo provenance** | Registration counts and emission factors are real; average daily distance figure (30 km/day) reuses the two-wheeler urban-mobility estimate for cars, which has no independent car-specific citation — flagged as DEMO in `distance_estimates.csv` |
| RTO-to-zone mapping | **Reasoned approximation** | No official zone-RTO correspondence exists; nearest-RTO assignment documented in `rto_mapping.py` with reasoning per zone |
| Enforcement ranking (composite score) | **Placeholder** | Built from current 247-image model; every output row tagged `model_version: "247-image placeholder -- not final"` |
| Land-use signal | **Absent** | No land-use CSV or API is integrated; noted explicitly in ranker output and dashboard |

---

## How to run it

The project has three independent tracks. Run them in order.

### Prerequisites

```
# All tracks: Python 3.12, CUDA GPU (training only)
# Satellite track: activate the airsen venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
source .venv/bin/activate    # macOS/Linux

# Forecasting track: uses its own venv in P2/airsentinel-master/forecasting/
cd P2/airsentinel-master/forecasting
.venv\Scripts\Activate.ps1
```

### Track 1 — Satellite (Person 2)

```bash
# 1. Sort Drive folder (moves root-level tifs into zone subfolders)
python sort_drive_folder.py

# 2. Build labels (joins S2 files against CAAQMS heuristic rules)
python build_labels.py

# 3. Augment (7 D4 transforms per image, ~8x expansion)
python augment_labels.py

# 4. Train Phase 2 (Prithvi fine-tuning, ~10 epochs, ~30 min on RTX 3070)
.venv/Scripts/python.exe train_prithvi.py
# Output: prithvi_airsen_augmented.pt

# 5. (Optional) Pre-compute backbone feature cache for faster retraining
.venv/Scripts/python.exe cache_backbone_features.py

# 6. Fusion training (Prithvi + NO2 late fusion)
.venv/Scripts/python.exe train_fused.py
# Output: prithvi_fused_best.pt

# 7. Run enforcement ranker (produces enforcement_ranking.csv + attribution.csv)
.venv/Scripts/python.exe enforcement_ranker.py
# Data dir: G:\My Drive\AirSentinel_Satellite_Images
# Requires: prithvi_airsen_augmented.pt, labels.csv, vehicle_emission_index.csv
```

### Track 2 — Forecasting (Person 1)

```bash
cd P2/airsentinel-master/forecasting
# Activate forecasting venv, then:
python -m airsentinel.pipeline
# Output: outputs/forecasts.csv, data/processed/airsentinel_daily_panel.csv
```

### Track 3 — Vehicle Emissions (Person 2)

```bash
# From project root — produces real VAHAN-mapped input:
python category_mapper.py

# Then run the pipeline (uses its own venv or system Python + pandas):
cd P2/airsentinel-master/vehicle_emissions
PYTHONPATH=src python -c "
import sys; sys.path.insert(0,'../forecasting/src')
from vehicle_emissions import pipeline; pipeline.run()
"
# Output: outputs/vehicle_emission_index.csv
```

### Final step — Fusion + Dashboard

```bash
cd P2/airsentinel-master/shared
python -c "
import sys
sys.path.insert(0,'src')
sys.path.insert(0,'../forecasting/src')
sys.path.insert(0,'../vehicle_emissions/src')
from shared import dashboard
dashboard.build()
"
# Output: outputs/dashboard.html, outputs/fused_zone_state.csv
```

---

## Known limitations

These are stated plainly. A judge who finds an unstated limitation is worse than one who reads it here.

1. **Small dataset — 247 images, 19 dates.** The Prithvi model is fine-tuned on 13 zones × 19 dates. The expected 36 gap-fill dates (Dec 2025 – May 2026) were not available by submission time; the Drive folder shows 0 files for that period. The 73.9% validation accuracy reflects this constraint.

2. **Labels are heuristic, not ground truth.** Every pollution source label used for training was generated by rule-based logic from CAAQMS sensor readings, not by manual expert annotation or Supersite measurements. "73.9% accuracy" means 73.9% agreement with these rules, not 73.9% against any independent standard.

3. **No Supersite or independent label set.** The only available ground signal is CAAQMS hourly data. No cross-validation against physically sampled source apportionment was possible.

4. **Enforcement ranking is a placeholder.** The current ranker uses the 247-image model. Every output row carries the tag `model_version: "247-image placeholder -- not final"`. The scoring formula (attribution_confidence × source_weight + 0.3 × vehicle_index) is calibrated by domain judgment, not by any optimisation or ground-truth validation.

5. **No land-use signal.** The design plan specifies land-use data as a third enforcement signal; no such data was sourced or ingested. The ranker explicitly notes this on every output row and in the dashboard.

6. **Vehicle emission distance estimates pending verification.** The 30 km/day average daily distance for cars reuses the two-wheeler urban-mobility figure (range 27–33 km/day from an Indian mobility study); no independent car-specific citation was found. This row remains tagged DEMO in `distance_estimates.csv`. The vehicle emission load index values are directionally correct but should be treated as indicative, not precise.

7. **RTO-to-zone mapping is a geographic approximation.** Delhi's 13 hotspot zones do not correspond to VAHAN's RTO boundaries. The mapping in `rto_mapping.py` is nearest-neighbour by geography, fully documented and editable, but not an official correspondence.

8. **Bawana and Punjabi Bagh absent from AQI forecasts.** The forecasting track's `forecasts.csv` covers 11 of 13 zones. Bawana and Punjabi Bagh are in the canonical zone list and have satellite/vehicle data, but no AQI forecast is available — they do not appear in the fused dashboard output. This is flagged, not routed around.

9. **GRAP edge case (AQI between 200 and 201).** The GRAP stage lookup in `grap.py` has a 1-unit gap between the "Below GRAP" ceiling (200) and the "Stage I" floor (201). A fractional AQI between 200 and 201 would fall through to "Stage IV — Severe Plus" by the loop's fallback. In practice AQI is reported as an integer so this never triggers, but it is a latent bug.

---

## What each team member built

**Person 1 (Forecasting track):**
- CAAQMS data scraper (`airsentinel/scraper.py`) — live hourly readings from DPCC portal
- AQI computation and daily panel construction (`airsentinel/aqi.py`, `airsentinel/labels.py`)
- Time-series AQI forecaster (`airsentinel/forecast.py`) — 24/48/72h horizon
- GRAP stage mapper (`shared/grap.py`) — deterministic lookup against official CAQM thresholds
- Shared fusion layer (`shared/fuse.py`) and dashboard renderer (`shared/dashboard.py`)
- Zone urgency ranking by forecast AQI (`shared/zone_ranking.py`)
- Vehicle emissions module design and architecture (`vehicle_emissions/` — full module structure, loaders, index formula, RTO mapping)

**Person 2 (Satellite + Enforcement track):**
- Sentinel-2 image pipeline: GEE export notebook, Drive sort (`sort_drive_folder.py`), label builder (`build_labels.py`), D4 augmentation (`augment_labels.py`)
- CAAQMS heuristic label rules (`labels.py`) and label table delivery to forecasting track
- Phase 2 Prithvi-EO-2.0-100M fine-tuning (`train_prithvi.py`) — 73.9% val accuracy on heuristic labels
- NO2 sensor fusion: labels_fused.csv pipeline, late-fusion architecture (`train_fused.py`)
- Backbone feature cache (`cache_backbone_features.py`) for training speedup
- VAHAN category mapper (`category_mapper.py`) — maps real VAHAN export to emission formula
- Enforcement Zone Ranker (`enforcement_ranker.py`) — Prithvi inference + vehicle index + honesty tags
- Phase 4 wiring: zone name normalisation fix, fuse.py/dashboard.py honesty tag integration
