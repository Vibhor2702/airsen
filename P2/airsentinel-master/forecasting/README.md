# AirSentinel — Forecasting & Data Pipeline (Person 1 track)

*This folder is `forecasting/` (renamed from `airsentinel/` to match its role — see
`../PROJECT_STATUS.md`); the Python package inside is unchanged, still `src/airsentinel/`,
still `import airsentinel`. "AirSentinel" is the overall project brand shared by the sibling
`../vehicle_emissions` and `../shared` modules.*

Part of **AirSentinel** (ET AI Hackathon · Delhi pilot). This repo is the **forecasting /
data** half of the two-person split. It:

1. **Scrapes** DPCC CAAQMS station data (real-time + historical) for Delhi's **13 pollution
   hotspot zones** from [dpccairdata.com](https://www.dpccairdata.com).
2. Aggregates it to a clean **daily panel** and computes the **CPCB National AQI** per zone.
3. Produces the **teammate delivery table** the satellite/enforcement track (Person 2)
   needs to replace its placeholder labels — in the exact agreed schema.
4. Trains a **per-zone AQI forecasting model** (24 / 48 / 72 h ahead) and benchmarks it
   against a persistence baseline ("tomorrow = today"), the baseline the plan calls out.

## Project layout

```
forecasting/                (Python package inside is still named "airsentinel")
├── src/airsentinel/
│   ├── config.py       # every tunable constant, one place (scrape timing, CV/model knobs,
│   │                   #   CPCB aggregation windows) — see comments for what's overridable
│   │                   #   via env var vs. fixed methodology
│   ├── stations.py     # 13 zones -> DPCC station tokens; parameter/endpoint registry
│   ├── scraper.py      # DPCC "Advance Search" scraper (parses inline Highcharts series)
│   ├── aqi.py          # CPCB National AQI (sub-indices + overall AQI)
│   ├── labels.py        # optional heuristic source_category labels (clearly not ground truth)
│   ├── forecast.py     # feature building, model, persistence baseline, evaluation
│   ├── viz.py           # output charts
│   └── pipeline.py     # end-to-end orchestration + CLI
├── data/
│   ├── raw/             # per-(zone, param) live-scrape audit trail, tagged with date window
│   ├── processed/       # hourly_tidy.csv, airsentinel_daily_panel.csv
│   └── teammate_delivery/
│       ├── caaqms_readings.csv          # THE deliverable: zone,date,PM2.5,PM10,NO2,SO2,CO,O3
│       ├── caaqms_heuristic_labels.csv  # optional candidate source_category labels
│       └── DATA_HANDOFF.md              # answers Person 2's 5 questions + data dictionary
├── outputs/
│   ├── forecasts.csv          # 24/48/72h AQI forecast per zone
│   ├── metrics.json           # model vs baseline MAE/RMSE/R²/accuracy/precision/recall/F1
│   ├── model_vs_baseline.png  # chart
│   └── forecast_example.png   # chart
├── requirements.txt
├── SESSION_SUMMARY.md   # full build history, current metrics, run instructions
└── README.md
```

All tunable behaviour (scrape delay, live-window length, model regularization search space,
CV settings, CPCB aggregation windows) lives in `config.py` with inline documentation of why
each value is what it is — nothing is a bare magic number in application code. A few are
overridable via environment variable without touching code:
`AIRSENTINEL_WINDOW_DAYS`, `AIRSENTINEL_SCRAPE_DELAY`, `AIRSENTINEL_SCRAPE_TIMEOUT`,
`AIRSENTINEL_SCRAPE_RETRIES`.

## Setup

```powershell
# Python 3.12
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
$env:PYTHONPATH = "src"
# live run: scrapes the trailing 45 days up to right now (no hardcoded dates) + builds + forecasts
.\.venv\Scripts\python.exe -m airsentinel.pipeline
# explicit date range instead of the live default
.\.venv\Scripts\python.exe -m airsentinel.pipeline --start 2026-06-01 --end 2026-07-20
# dev-only: rebuild from the last live scrape without hitting the site again
.\.venv\Scripts\python.exe -m airsentinel.pipeline --skip-scrape
```

See [SESSION_SUMMARY.md](SESSION_SUMMARY.md) for the full session history, current model
performance (accuracy/precision/recall/F1 + MAE/RMSE/R²), and a walkthrough of how the live
scrape works.

## How the scrape works

DPCC's "Advance Search" pages render each requested series into an inline **Highcharts**
config in the returned HTML. The scraper POSTs the same form the UI posts and parses the
`categories` (timestamps) and series `data` (values) arrays back out.

Contract (reverse-engineered from the live site):

- `parameters` — the metric code (e.g. `PM25`, `NO2`); met vs gas metrics use two different
  endpoints (`AallAdvanceSearchMet.php` / `AallAdvanceSearchCconc.php`).
- `fDate` / `eDate` — `YYYY-MM-DD HH:MM`; the window must be **strictly less than 7 days**
  (the UI hint says "not more than 7 days", but a full 7-day span returns no data). The
  scraper chunks any range into 6-day windows.
- `duration` — aggregation bucket in hours (`1` = hourly; we pull hourly and aggregate
  ourselves for full control over CPCB averaging periods).
- `submit=Search` — required; any other value returns the blank form.

## Honesty notes (carried from the AirSentinel deck)

- The forecast is **gradient-boosted trees** (`HistGradientBoostingRegressor`) on lagged
  readings, momentum/volatility, weather, and calendar features — **not** the "graph+LSTM"
  the original deck sketches. This is a deliberate choice, not a shortcut: with ~45 daily
  points per zone, an LSTM has nowhere near the sequence length it needs to avoid
  overfitting, while a tree ensemble is far more sample-efficient on small tabular data. See
  `forecast.py`'s module docstring for the full reasoning, and `SESSION_SUMMARY.md` §5/§9 for
  current numbers.
- The model predicts a **residual correction to persistence** (not raw AQI), shrunk by a
  cross-validation-tuned weight that can never be selected to underperform the persistence
  baseline out-of-fold — see `forecast.py` for the mechanism and `SESSION_SUMMARY.md` §8 for
  the overfitting fix history.
- AQI at daily granularity uses daily-mean concentrations (with 8-hour-max for CO/O₃ per
  CPCB); this is a reasonable approximation of the official station-hour methodology, not
  a claim of identical computation.
