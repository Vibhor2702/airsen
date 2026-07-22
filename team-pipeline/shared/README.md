# Shared — Fusion Layer & Dashboard (AirSentinel)

*Folder naming note: the overall project is "AirSentinel"; each track lives in its own
folder named for its role — `forecasting/` (Person 1, Python package still importable as
`airsentinel`), `vehicle_emissions/`, `shared/` (this one).*

Combines each AirSentinel track's independent output into one zone-keyed view, and renders
the themed operations dashboard from the design plan mockup (slide 7). This is the
"03 · FUSION LAYER" and "04 · OUTPUT LAYER" from the system architecture (slide 5),
scaffolded ahead of full satellite/enforcement integration so wiring the remaining pieces in
is a data drop, not a rewrite.

## What's real right now vs. what's scaffolded

| Component | Status | Source |
|---|---|---|
| Forecasting Engine | **Live** | `../forecasting` — `outputs/forecasts.csv` |
| CAAQMS current conditions | **Live** | `../forecasting` — `data/processed/airsentinel_daily_panel.csv` |
| Vehicle Emission Load Model | **Live on demo data tonight** (amber "DEMO DATA" badge, not green) | `../vehicle_emissions` — `outputs/vehicle_emission_index.csv`, real-world-informed placeholder pending real per-RTO VAHAN counts — see that module's README "Demo data" |
| GRAP Stage Mapper | **Live** | `grap.py` — official CAQM thresholds, deterministic |
| Zone urgency ranking | **Live** | `zone_ranking.py` — forecast-driven, not the full enforcement ranker (see its docstring for the scope distinction) |
| Alert Generator | **Live** | `alerts.py` — templated from real values |
| Dashboard | **Live**, renders whatever is actually available, distinguishes real vs. demo data visibly | `dashboard.py` |
| Satellite Attribution Engine | **Pending** — awaiting teammate's Prithvi output | see "Satellite attribution contract" below |
| Cross-validation vs SAFAR/Supersite | **Access confirmed, not enough data tonight** — Supersite is a DPCC station (same scraper, real historical data too sparse); SAFAR's site is currently down (expired TLS cert) | `cross_validation.py` — ready to run the moment either has enough real, overlapping data |
| Enforcement Zone Ranker (land-use/industry) | **Not built here** — teammate's track | out of scope for this module by design |

Nothing here fabricates a number for a component that isn't live yet — see each module's
docstring for exactly how the "pending" state is represented and why.

## Setup

```powershell
cd "C:\Users\a955032\OneDrive - ATOS\Desktop\airsentinel\shared"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Run the forecasting pipeline first, and the vehicle emissions pipeline if you want that live
too (`python -m vehicle_emissions.build_demo_data` then `python -m vehicle_emissions.pipeline`
for tonight's real-world-informed demo data — see that module's README), then:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m shared.pipeline --horizon 24
```

Output:
- `outputs/fused_zone_state.csv` — one row per zone: latest actual AQI, forecast AQI, GRAP
  stage, vehicle emission index (or `pending`), satellite source guess (or `pending`),
  urgency rank.
- `outputs/dashboard.html` — open directly in a browser. Renders whatever is real; shows a
  visible "pending" badge for anything a sibling module hasn't produced yet.

## Satellite attribution contract (for when the teammate's module is ready)

Point `shared/src/shared/config.py::SATELLITE_ATTRIBUTION_CSV` at the real output file (or
override it — see that file), with columns:

```
zone, source_guess, confidence
```

`fuse.py` will pick it up automatically the next time the pipeline runs — no code changes
needed on this side. `source_guess` should use the same category vocabulary as the
forecasting module's heuristic labels (`dust`, `crop_burning_smoke`, `industrial_haze`,
`traffic_heavy`, `clear` — see `../forecasting/src/airsentinel/labels.py`) so the dashboard's
badge legend stays consistent across both the CAAQMS-based heuristic and the real satellite
model, but this isn't enforced in code — coordinate the exact vocabulary with your teammate.

## Why nothing here is a fitted model

Every function in this package is a deterministic formula, lookup table, or string
template — GRAP thresholds are an official lookup (config.py, cited), urgency ranking is a
sort, alerts are string formatting. There is no training data anywhere in this layer, so
there is no overfitting risk by construction — the fusion layer's job is to combine outputs
from the tracks that do the modelling (forecasting, and eventually satellite attribution),
not to model anything itself.
