# AirSentinel — Project Status & Gap Analysis vs. Design Plan

*Cross-references `AirSentinel_design_plan.pptx`'s system architecture (slide 5) and data
flow (slide 6) against what's actually built across the three modules in this folder. Written
2026-07-21.*

## Folder layout

```
airsentinel/                (root — was "light_pollution/" until this pass)
├── forecasting/           Forecasting & CAAQMS data pipeline (Person 1 track) — LIVE
│                           (folder renamed from "airsentinel/"; the Python package inside
│                            is still importable as `airsentinel` — see its README)
├── vehicle_emissions/     Vehicle Emission Load Model ("tailpipe pollution") — live on demo data
├── shared/                 Fusion layer + dashboard + alerts — LIVE, degrades gracefully
├── AirSentinel_MasterDoc.pptx   Pitch deck
├── README.md                    Entry point — start here
└── PROJECT_STATUS.md            This file — full gap analysis & audit
```

Each module is an independent Python project (own `.venv`, own `requirements.txt`) so the
two-person split can keep working on separate tracks without touching each other's
environments. `vehicle_emissions` and `shared` both consume `forecasting`'s outputs by file
(CSV) rather than by importing its runtime, except for the zone registry — `vehicle_emissions`
installs `forecasting` in editable mode specifically so the 13-zone list is never retyped
(see `vehicle_emissions/README.md` "Compatibility").

## Component-by-component status (design plan slide 5: system architecture)

### 01 · Data Sources
| Source | Status |
|---|---|
| CPCB/DPCC CAAQMS stations | **Live** — `forecasting` scrapes this directly from dpccairdata.com |
| IMD weather data | **Substituted** — `forecasting` uses each CAAQMS station's own met sensors (temp/humidity/wind) instead of a separate IMD feed; reasonable proxy, not identical to IMD, documented in `forecasting`'s honesty notes |
| Sentinel-2 / Sentinel-5P imagery | **Teammate's track** — satellite/Prithvi, not in this folder |
| VAHAN vehicle registrations | **Demo data live tonight, real mapping ready** — `vehicle_emissions` ships a documented RTO→zone mapping (`rto_mapping.py`, real live Delhi RTO list) plus a real-world-informed demo dataset (`build_demo_data.py`: real Delhi vehicle totals + official BS6 limits + a published distance study, clearly flagged, not random) so the pipeline runs end-to-end tonight; real per-RTO VAHAN counts still needed to replace the demo |
| Delhi Supersite data | **Same site as DPCC, confirmed working, data sparse** — it's a station on dpccairdata.com, not a separate site; the existing scraper pulls it unchanged, but real historical data was only found on one isolated date (station appears to have gone quiet ~March 2025) — see `shared/cross_validation.py` |
| SAFAR forecasts | **Real site, currently down** — safar.tropmet.res.in's TLS certificate is expired as of this session (`certificate has expired`); a site-side outage, not a scraping difficulty |

### 02 · Model Layer
| Module | Status |
|---|---|
| Forecasting Engine (24-72h per zone) | **Live** — `forecasting`, gradient-boosted trees (not the deck's "graph+LSTM" — see `forecasting/src/airsentinel/forecast.py` docstring for why), beats persistence baseline at all horizons |
| Satellite Attribution Engine (fine-tuned Prithvi) | **Teammate's track** |
| Vehicle Emission Load Model (VAHAN-based math model) | **Live tonight on demo data** — `vehicle_emissions`, deterministic formula; running on real-world-informed demo data pending real per-RTO VAHAN counts (see below) |
| Enforcement Zone Ranker (hotspot + emission + land use) | **Not built** — explicitly out of scope for this session; `shared/zone_ranking.py` provides a narrower forecast-only urgency ranking as a stopgap, clearly scoped as *not* a replacement (see its docstring) |

### 03 · Fusion Layer
| Component | Status |
|---|---|
| GRAP Stage Mapper (forecast AQI → CAQM stage I-IV) | **Built this session** — `shared/grap.py`, official CAQM thresholds verified live against caqm.nic.in |
| Cross-validation check vs. Supersite & SAFAR | **Scaffolded, real access confirmed, not enough data to run tonight** — Supersite's scraper access is confirmed working (same DPCC contract) but its real historical data is too sparse for a meaningful backtest; SAFAR's site is currently down (expired cert). `shared/cross_validation.py`'s comparison function is ready the moment either has enough real, overlapping data |

### 04 · Output Layer
| Component | Status |
|---|---|
| Official Dashboard | **Built this session** — `shared/dashboard.py`, static themed HTML matching the deck's mockup (slide 7), renders only real data, shows explicit "pending" badges for anything not yet live |
| Citizen/Official Alert Generator | **Built this session** — `shared/alerts.py`, templated from real computed values (matches deck slide 8) |
| API/export to MCD, DPCC, traffic police | **Not built** — no real endpoint exists to integrate with; out of scope |

## What was deliberately NOT built, and why

- **Live VAHAN vehicle-count scraping.** Checked this session: vahan.parivahan.gov.in is a
  session-bound JSF/PrimeFaces app with no clean public API for the actual registration
  *numbers* (unlike DPCC, which had a scrapable HTML form). What WAS pulled live this
  session is VAHAN's real, current Delhi RTO/office list (13 vehicle-registering offices,
  DL1–DL13), used to build the documented RTO→zone mapping in `vehicle_emissions/rto_mapping.py`
  — see that module's README for the full reasoning. The actual per-RTO vehicle counts still
  need a real export from you (VAHAN's tabular/comparison views support manual export;
  there was no reliable way to automate that step within this session).
- **ARAI/CPCB precise per-category-per-pollutant emission-factor table baked into code.**
  Real official sources were located (see `vehicle_emissions/README.md`) but the
  PDF/table-extraction tools available this session couldn't verify the exact, full g/km
  table with confidence. Where a real, independently-corroborated figure *was* verifiable —
  official BS6 regulatory NOx/PM limits, which are simpler, better-documented public facts
  than a full emission-factor table — it's used in tonight's demo dataset, clearly labeled
  as a regulatory limit, not a full ARAI/CPCB real-world emission-factor table.
- **Full SAFAR/Supersite scraper integration into the live pipeline.** Access is confirmed
  for Supersite (same DPCC contract) and blocked for SAFAR (site down, expired cert) — see
  the Data Sources table above. Neither has enough real, current, overlapping data to run a
  meaningful cross-validation backtest tonight.
- **The full Enforcement Zone Ranker** (hotspot + emission + land-use cross-check) — this is
  explicitly the satellite/enforcement track's Days 8-9 task per the team brief, not
  duplicated here.

## Tonight's demo data — what's real vs. labeled assumption

`vehicle_emissions` needed *something* runnable tonight, ahead of real per-RTO VAHAN exports.
`vehicle_emissions/src/vehicle_emissions/build_demo_data.py` generates a dataset that is
**not random** — every number is either a real cited fact or an explicitly labeled
allocation assumption, never presented as more precise than it is:

| Input | What's real | What's a labeled assumption |
|---|---|---|
| Vehicle counts | Delhi-wide totals: 10,370,047 two-wheelers, 3,311,579 cars (both cited, real) | Split **evenly** across the 13 zones (no real per-zone data yet); cars assumed 70%/30% petrol/diesel (commonly-cited rough order, not Delhi-verified) |
| Emission factors | Official BS6 regulatory NOx/PM limits (India's mandated standard since 1 Apr 2020) — real, government-set, corroborated across independent sources this session | — (used as-is) |
| Distance | Published Indian-cities urban-mobility study, 27-33 km/day for two-wheelers (midpoint used) | Reused for cars too — no car-specific citation found this session |

This propagates all the way to the UI: `outputs/vehicle_emission_index.csv` carries a
`data_provenance` column, and `shared/outputs/dashboard.html` shows an amber "DEMO DATA"
chip (not the green "live" used for verified real data) plus a banner explaining exactly
what's real vs. assumed — the demo is designed to never look more precise than it is. Because
the zone split is even, every zone currently shows the same index (1.00) — that's an honest
signal that real zone-to-zone variation needs real per-RTO data, not a bug.

## Hardcoding & overfitting — session-wide guarantees

- **No fabricated data anywhere in `vehicle_emissions` or `shared`.** Every emission factor,
  distance estimate, and vehicle count must come from a real, cited input file — the modules
  fail loudly with instructions rather than substitute a plausible-looking default. Verified
  this session with a `random|fake|dummy|mock|synthetic` grep across both new modules (clean).
- **No fitted models in `vehicle_emissions` or `shared`.** Both are deterministic formulas,
  lookups, and string templates — verified this session with a `sklearn|.fit(|.predict(` grep
  (only hit: a metric *function*, `mean_absolute_error`, in `cross_validation.py`, not a
  trained model). Zero overfitting risk in either module by construction.
- **`forecasting`'s model** (the one place in this project that does fit anything) has its
  own overfitting safeguards — CV-tuned regularization and persistence-anchored shrinkage —
  documented in `forecasting/SESSION_SUMMARY.md` §7-9.
- **Official constants that ARE hardcoded** (CPCB AQI breakpoints, GRAP stage thresholds) are
  treated the same way throughout this project: cited to a specific, verifiable source, never
  presented as if self-evident.

## Running the full pipeline, in order

```powershell
# 1. Forecasting (live CAAQMS scrape + model)
cd "C:\Users\a955032\OneDrive - ATOS\Desktop\airsentinel\forecasting"
$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m airsentinel.pipeline

# 2. Vehicle emissions — demo data tonight (real-world-informed, not random; swap for real
#    VAHAN exports later, see vehicle_emissions/README.md "Demo data")
cd "C:\Users\a955032\OneDrive - ATOS\Desktop\airsentinel\vehicle_emissions"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m vehicle_emissions.build_demo_data
.\.venv\Scripts\python.exe -m vehicle_emissions.pipeline

# 3. Fusion + dashboard (works with just step 1 done; picks up step 2 automatically if present)
cd "C:\Users\a955032\OneDrive - ATOS\Desktop\airsentinel\shared"
$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m shared.pipeline --horizon 24
# open shared/outputs/dashboard.html in a browser
```
