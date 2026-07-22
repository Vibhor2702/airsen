"""
AirSentinel end-to-end pipeline (Person 1 — Forecasting & Data).

    scrape hourly CAAQMS  ->  daily panel + CPCB AQI  ->  teammate delivery CSV
                                                       ->  forecasting model + metrics

Run (defaults to a live pull: the trailing DEFAULT_WINDOW_DAYS days up to right now —
there is no hardcoded date; the window is computed from the clock at run time):
    python -m airsentinel.pipeline
    python -m airsentinel.pipeline --start 2026-06-01 --end 2026-07-20   # explicit range
    python -m airsentinel.pipeline --skip-scrape        # rebuild from previously-scraped cache
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config
from .aqi import category, compute_aqi
from .forecast import forecast_latest, train_and_evaluate
from .labels import add_labels
from .scraper import DpccScraper
from .stations import ALL_PARAMS, DELIVERY_COLUMNS, POLLUTANTS, WEATHER, ZONES

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DELIVERY = ROOT / "data" / "teammate_delivery"
OUTPUTS = ROOT / "outputs"
for _d in (RAW, PROCESSED, DELIVERY, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)

# Aggregation convention per pollutant (official CPCB averaging periods — see config.py).
MEAN_POLLUTANTS = config.CPCB_MEAN_POLLUTANTS
MAX8H_POLLUTANTS = config.CPCB_MAX8H_POLLUTANTS

# Default lookback for a live run, in days (config.py; override via AIRSENTINEL_WINDOW_DAYS).
# Not a fixed calendar date — the actual window is [now - DEFAULT_WINDOW_DAYS, now], computed
# fresh every run so the pipeline always pulls whatever is currently live on DPCC rather than
# replaying a stale hardcoded date range.
DEFAULT_WINDOW_DAYS = config.PIPELINE_DEFAULT_WINDOW_DAYS


# --------------------------------------------------------------------------- scrape ----
def scrape_all(start: datetime, end: datetime, delay: float) -> pd.DataFrame:
    """Scrape hourly series for every zone x param, live, directly from dpccairdata.com.

    Every call hits the live site for the requested window — there is no silent reuse of a
    previous run's cache here, so the data this produces is always current for [start, end].
    Each (zone, param) response is still written to data/raw/ afterwards purely as an audit
    trail (so you can see exactly what was scraped), tagged with the date window in the
    filename so a different window can never be mistaken for this one.
    """
    scraper = DpccScraper(delay=delay)
    frames = []
    total = len(ZONES) * len(ALL_PARAMS)
    i = 0
    window_tag = f"{start:%Y%m%d}_{end:%Y%m%d}"
    for zone, station in ZONES.items():
        for param in ALL_PARAMS:
            i += 1
            try:
                df = scraper.fetch_series(station, param, start, end, duration=1)
            except Exception as e:  # noqa: BLE001 - keep going on isolated failures
                print(f"[{i:3d}/{total}] FAILED  {zone:12s} {param.column:10s}: {e}")
                continue
            audit = RAW / f"{zone.replace(' ', '_')}__{param.column.replace('.', '')}__{window_tag}.csv"
            df.to_csv(audit, index=False)
            print(f"[{i:3d}/{total}] live    {zone:12s} {param.column:10s} ({len(df)} rows)")
            if not df.empty:
                frames.append(df)
    if not frames:
        raise RuntimeError(
            "Live scrape returned no data for any zone/param — check network access to "
            "dpccairdata.com and the requested date window before trusting downstream output."
        )
    tidy = pd.concat(frames, ignore_index=True)
    tidy.to_csv(PROCESSED / "hourly_tidy.csv", index=False)
    return tidy


# ------------------------------------------------------------------- daily aggregation --
def _rolling_8h_max(series: pd.Series) -> float:
    """Daily max of the 8-hour rolling mean (official CPCB convention for CO and O3)."""
    if series.dropna().empty:
        return float("nan")
    roll = series.rolling(config.CPCB_8H_WINDOW_HOURS, min_periods=config.CPCB_8H_MIN_PERIODS).mean()
    return float(roll.max()) if roll.notna().any() else float(series.max())


def build_daily_panel(tidy: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly tidy data to one row per (zone, date) with pollutants, weather, AQI."""
    df = tidy.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    wide = df.pivot_table(index=["zone", "date", "timestamp"], columns="param",
                          values="value", aggfunc="first").reset_index()

    records = []
    for (zone, date), grp in wide.groupby(["zone", "date"]):
        grp = grp.sort_values("timestamp")
        row: dict = {"zone": zone, "date": date}
        for pol in MEAN_POLLUTANTS:
            row[pol] = round(float(grp[pol].mean()), 1) if pol in grp else float("nan")
        for pol in MAX8H_POLLUTANTS:
            row[pol] = round(_rolling_8h_max(grp[pol]), 2) if pol in grp else float("nan")
        for w in [p.column for p in WEATHER]:
            row[w] = round(float(grp[w].mean()), 1) if w in grp else float("nan")
        aqi, dom = compute_aqi({p: row.get(p) for p in [x.column for x in POLLUTANTS]})
        row["AQI"] = aqi
        row["AQI_category"] = category(aqi)
        row["dominant_pollutant"] = dom
        records.append(row)

    panel = pd.DataFrame(records).sort_values(["zone", "date"]).reset_index(drop=True)
    panel.to_csv(PROCESSED / "airsentinel_daily_panel.csv", index=False)
    return panel


# ----------------------------------------------------------------- teammate delivery ----
def write_delivery(panel: pd.DataFrame) -> tuple[Path, Path]:
    """Write the deliverables for Person 2.

    Primary: the exact raw table asked for -> zone, date, PM2.5, PM10, NO2, SO2, CO, O3.
    Optional: heuristic source_category labels (candidate labels, clearly not ground truth).
    """
    out = panel[DELIVERY_COLUMNS].copy()
    out = out.dropna(subset=["PM2.5", "PM10"], how="all")  # keep rows with at least some PM
    out = out.sort_values(["zone", "date"])
    raw_path = DELIVERY / "caaqms_readings.csv"
    out.to_csv(raw_path, index=False)

    labelled = add_labels(panel)
    lab = labelled[["zone", "date", "source_category"]].sort_values(["zone", "date"])
    lab_path = DELIVERY / "caaqms_heuristic_labels.csv"
    lab.to_csv(lab_path, index=False)
    return raw_path, lab_path


# ----------------------------------------------------------------------- forecasting ----
def run_forecast(panel: pd.DataFrame) -> dict:
    valid = panel.dropna(subset=["AQI"]).copy()
    models, metrics, feats = train_and_evaluate(valid)
    if models:
        fc = forecast_latest(models, feats)
        fc.to_csv(OUTPUTS / "forecasts.csv", index=False)
    with open(OUTPUTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ------------------------------------------------------------------------------- CLI ----
def main() -> None:
    ap = argparse.ArgumentParser(description="AirSentinel forecasting & data pipeline")
    ap.add_argument("--start", default=None,
                    help=f"YYYY-MM-DD; default = today minus {DEFAULT_WINDOW_DAYS} days (live)")
    ap.add_argument("--end", default=None,
                    help="YYYY-MM-DD; default = right now (live)")
    ap.add_argument("--delay", type=float, default=config.SCRAPE_DELAY_SECONDS,
                    help="seconds between requests")
    ap.add_argument("--skip-scrape", action="store_true",
                    help="dev-only: rebuild from a previously-scraped data/processed/hourly_tidy.csv "
                         "instead of hitting the live site again")
    args = ap.parse_args()

    if args.skip_scrape:
        tidy = pd.read_csv(PROCESSED / "hourly_tidy.csv", parse_dates=["timestamp"])
        print(f"Loaded previously-scraped hourly data (real DPCC data, not re-fetched "
              f"live this run): {len(tidy)} rows")
    else:
        end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
        start = (datetime.strptime(args.start, "%Y-%m-%d") if args.start
                  else end - timedelta(days=DEFAULT_WINDOW_DAYS))
        print(f"Scraping LIVE DPCC CAAQMS data {start:%Y-%m-%d} -> {end:%Y-%m-%d} for "
              f"{len(ZONES)} zones x {len(ALL_PARAMS)} params (hourly)...")
        tidy = scrape_all(start, end, args.delay)

    print("\nAggregating to daily panel + computing CPCB AQI...")
    panel = build_daily_panel(tidy)
    print(f"  panel: {len(panel)} zone-days, "
          f"{panel['AQI'].notna().sum()} with valid AQI, "
          f"{panel['zone'].nunique()} zones, "
          f"{panel['date'].min()} -> {panel['date'].max()}")

    raw_path, lab_path = write_delivery(panel)
    n_rows = len(pd.read_csv(raw_path))
    print(f"\nTeammate delivery (raw table):        {raw_path} ({n_rows} rows)")
    print(f"Teammate delivery (heuristic labels): {lab_path}")

    print("\nTraining forecasting model (24/48/72h) vs persistence baseline...")
    print("  (hyperparameters per horizon selected by walk-forward CV on the training "
          "period only)")
    metrics = run_forecast(panel)
    for h, m in metrics.items():
        print(f"  {m['horizon_hours']}h: MAE {m['model_mae']:.1f} vs baseline "
              f"{m['baseline_mae']:.1f} ({m['mae_improvement_pct']:+.1f}% better)  "
              f"R2={m['model_r2']:.2f}  "
              f"AQI-category accuracy={m['model_category_accuracy']:.0%} "
              f"(baseline {m['baseline_category_accuracy']:.0%})  [test n={m['n_test']}]")
        print(f"       overfit check: train MAE {m['train_mae_delta_target']:.1f} vs "
              f"CV MAE {m['cv_mae_delta_target']:.1f} (delta target; gap "
              f"{m['overfit_gap_pct']:+.1f}%)  params={m['best_params']}  "
              f"alpha(shrinkage)={m['alpha_shrinkage']:.2f}")

    if metrics:
        from .viz import generate_all
        made = generate_all(panel, metrics, OUTPUTS)
        print("\nCharts written: " + ", ".join(p.name for p in made))

    print("\nDone. Outputs in data/ and outputs/.")


if __name__ == "__main__":
    main()
