"""
Cross-validation check vs. SAFAR / Delhi Supersite — the "03 · FUSION LAYER" component from
the design plan (slide 5: "vs. Supersite & SAFAR, flags disagreement") and the Person 1
Days 8-9 task from the team brief ("Compare against SAFAR's real forecasts... this becomes
the strongest 'we beat a real government tool' slide").

Status, checked live this session:

- **Delhi Supersite is NOT a separate site — it's a station on DPCC itself**
  (`dpccairdata.com`, station "Supersite(Rouse Avenue)", same POST contract as every other
  CAAQMS station forecasting/scraper.py already handles). Confirmed: the same
  `DpccScraper` class successfully pulled real hourly NO2 data from it. The catch —
  real data is extremely sparse via this endpoint: probing several windows around its
  last-known-active period found genuine readings on only a single day
  (2025-03-27); the station appears to have gone quiet (one of its three DPCC page
  variants is explicitly labelled "under construction"). Not enough overlapping real
  data for a meaningful backtest tonight, but the access path is confirmed working the
  moment DPCC populates more history — no new reverse-engineering needed, just a
  station-code addition.
- **SAFAR** (safar.tropmet.res.in) is a real, separate government site, but its TLS
  certificate is currently expired — it refused a secure connection when checked this
  session (`certificate has expired`). That's a site-side outage, not a scraping
  difficulty; revisit once SAFAR's certificate is renewed.

Rather than skip the comparison entirely or fabricate a "we beat SAFAR" number, this module
provides the comparison *function*, ready to run the moment either feed has enough real,
overlapping data — wire in a real CSV and call compare(), don't call it with placeholder
numbers.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error


def compare(
    our_forecast: pd.DataFrame,
    external: pd.DataFrame,
    disagreement_threshold_aqi: float = 50.0,
) -> pd.DataFrame:
    """Compare our forecast to an external reference (SAFAR or Delhi Supersite), row-matched
    on (zone, target_date).

    our_forecast: zone, target_date, predicted_aqi (airsentinel's outputs/forecasts.csv)
    external: zone, target_date, reference_aqi, source (a real SAFAR/Supersite export — not
              provided by this codebase; see module docstring)

    Returns one row per matched (zone, target_date) with both values, the absolute
    difference, and a disagreement flag — never fabricates rows for dates/zones that don't
    have a real external match.
    """
    required = {"zone", "target_date", "reference_aqi", "source"}
    missing = required - set(external.columns)
    if missing:
        raise ValueError(
            f"external comparison data is missing required columns: {missing}. This function "
            f"does not accept incomplete or placeholder comparison data — see module "
            f"docstring for what a real SAFAR/Supersite export needs to provide."
        )

    merged = our_forecast.merge(
        external, on=["zone", "target_date"], how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise ValueError(
            "No (zone, target_date) rows matched between our_forecast and external — check "
            "that the external data actually covers the same zones/dates as our forecast."
        )

    merged["abs_diff_aqi"] = (merged["predicted_aqi"] - merged["reference_aqi"]).abs()
    merged["disagreement_flag"] = merged["abs_diff_aqi"] > disagreement_threshold_aqi
    return merged[[
        "zone", "target_date", "predicted_aqi", "reference_aqi", "source",
        "abs_diff_aqi", "disagreement_flag",
    ]]


def summary(comparison: pd.DataFrame) -> dict:
    """Aggregate metrics for a comparison result — real numbers only, computed from
    whatever real external data was actually supplied to compare()."""
    return {
        "n_compared": int(len(comparison)),
        "mae_vs_external": round(
            float(mean_absolute_error(comparison["reference_aqi"], comparison["predicted_aqi"])), 2
        ),
        "n_disagreements": int(comparison["disagreement_flag"].sum()),
        "sources": sorted(comparison["source"].unique().tolist()),
    }
