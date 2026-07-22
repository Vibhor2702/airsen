# AirSentinel — Data Handoff (Forecasting/Data track → Satellite/Enforcement track)

Hi — here's the CAAQMS data for the labels table, plus answers to your five questions in
the same order you asked them. Everything below is pulled live from DPCC's 26-station
network for the **13 hotspot zones**, scraped and aggregated by the pipeline in this repo.

## Files in this folder

| File | What it is |
|---|---|
| **`caaqms_readings.csv`** | **Primary deliverable** — raw pollutant readings, one row per zone/date. |
| `caaqms_heuristic_labels.csv` | *Optional* — heuristic `source_category` guesses (candidate labels, **not** ground truth). See Q1. |
| `DATA_HANDOFF.md` | This note. |

---

## Your five questions, answered

### 1. Source breakdown, or just raw pollution numbers?
**Raw pollution numbers.** DPCC's CAAQMS stations report *concentrations* (PM2.5, PM10,
NO2, SO2, CO, O3) — they do **not** do source apportionment. So the primary table is the
raw-readings format from your request, not the `source_category` format.

That said, I know your model needs a *cause* to train against, so I've **also** included
`caaqms_heuristic_labels.csv`: a `source_category` column derived from the pollutant mix by
a simple, transparent rule (e.g. high PM10 + low PM2.5/PM10 ratio → `dust`; SO2 elevated →
`industrial_haze`; NO2+CO elevated → `traffic_heavy`). **Treat these as weak candidate
labels to replace your random placeholders, not as truth** — real source labels should come
from the Delhi Supersite lab, which is the cross-check our plan already names. The rule is
documented in `src/airsentinel/labels.py`. Distribution over the 648 rows: dust 476,
traffic_heavy 59, industrial_haze 58, clear 49, crop_burning_smoke 6 (crop-burning is rare
here because the window is June–July, not the Oct–Nov burning season).

### 2. Exported as a simple table?
Yes — plain CSV. `caaqms_readings.csv` has exactly the columns you specified:

```
zone, date, PM2.5, PM10, NO2, SO2, CO, O3
```

### 3. Do zone names match exactly?
**Yes — all 13, exact spelling from your list.** DPCC's station names differ from the
hotspot names, so I mapped them explicitly (no silent guessing):

| Your zone name | DPCC station used |
|---|---|
| Anand Vihar | Anand Vihar |
| Mundka | Mundka |
| Wazirpur | Wazirpur |
| Jahangirpuri | Jahangirpuri |
| RK Puram | R.K. Puram |
| Rohini | Rohini (Sector 16) |
| Punjabi Bagh | Punjabi Bagh |
| Okhla | Okhla Phase-2 |
| Bawana | Pooth Khurd, Bawana |
| Vivek Vihar | Vivek Vihar |
| Narela | Narela |
| Ashok Vihar | Ashok Vihar |
| Dwarka | Dwarka, Sector 8 |

If you'd rather join on a different key, the mapping lives in `src/airsentinel/stations.py`.

### 4. Same date range?
**2026-06-01 → 2026-07-20, daily** (50 consecutive dates). That sits inside your
June–July 2026 satellite window, so every label has a same-date image to pair with. If you
shift your satellite pulls, tell me the new window and I'll re-run — it's one command
(`--start`/`--end`).

### 5. How many zone/date combinations?
**648 rows = 13 zones × ~50 dates.** That's well past the "13 zones × 5–10 dates" you said
would be enough — this is a full batch, not a test batch. (648 rather than 650 because a
couple of zone-days had no valid PM reading and were dropped.)

---

## Data dictionary — `caaqms_readings.csv`

| Column | Unit | Meaning / aggregation |
|---|---|---|
| `zone` | — | Hotspot zone name (matches your 13 exactly). |
| `date` | `YYYY-MM-DD` | Calendar date (IST). |
| `PM2.5` | µg/m³ | Daily mean of hourly values. |
| `PM10` | µg/m³ | Daily mean of hourly values. |
| `NO2` | µg/m³ | Daily mean of hourly values. |
| `SO2` | µg/m³ | Daily mean of hourly values. |
| `CO` | **mg/m³** | Daily max of the 8-hour rolling mean (CPCB convention). Note the unit is mg/m³, not µg/m³. |
| `O3` | µg/m³ | Daily max of the 8-hour rolling mean (CPCB convention). |

Notes:
- Source: DPCC real-time CAAQMS (`dpccairdata.com`), 5-minute data aggregated to hourly by
  the site, then to daily here.
- A small number of `SO2`/`CO`/`O3` cells are blank where the station didn't report enough
  hours that day — left empty rather than imputed. `PM2.5`/`PM10` are complete.
- The richer processed panel (adds weather + a computed CPCB AQI + dominant pollutant per
  zone/day) is in `data/processed/airsentinel_daily_panel.csv` if useful for the dashboard.

## What I'm doing with the same data (so our handoff format stays aligned)
On my side this feeds a per-zone **AQI forecast (24/48/72h)** that beats a persistence
baseline (see `outputs/`). When we wire the dashboard, my forecast output is
`outputs/forecasts.csv` → `zone, forecast_from, horizon_hours, target_date, predicted_aqi`.
Let's lock that as the forecast handoff schema the same way we've locked this one.
