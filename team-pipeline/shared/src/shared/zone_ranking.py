"""
Forecast-driven zone urgency ranking — feeds the dashboard's "Top zones needing attention"
panel (design plan slide 7).

Scope note: this is deliberately NOT the design plan's full "Enforcement Zone Ranker"
(slide 5: "hotspot + emission + land use", owned by the satellite/enforcement track,
Days 8-9 per the team brief). This ranks by forecast urgency only — predicted AQI, with
vehicle emission load as a tiebreaker when available — so the dashboard has something real
to show today. When the teammate's actual enforcement ranker lands (with land-use/industry
cross-checks), it supersedes this for enforcement decisions; this stays useful as the
"what's the forecast telling us" view.
"""

from __future__ import annotations

import pandas as pd

from . import grap


def rank(
    forecast_df: pd.DataFrame,
    vehicle_index_df: pd.DataFrame | None = None,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Rank zones by forecast urgency at a given horizon.

    forecast_df: airsentinel's outputs/forecasts.csv (zone, horizon_hours, predicted_aqi, ...)
    vehicle_index_df: optional vehicle_emissions output (zone, vehicle_emission_load_index)
    """
    sub = forecast_df[forecast_df["horizon_hours"] == horizon_hours].copy()
    if sub.empty:
        raise ValueError(f"No forecast rows found for horizon_hours={horizon_hours}")

    sub[["grap_stage", "grap_label"]] = sub["predicted_aqi"].apply(
        lambda v: pd.Series(grap.stage(v))
    )

    if vehicle_index_df is not None and not vehicle_index_df.empty:
        sub = sub.merge(
            vehicle_index_df[["zone", "vehicle_emission_load_index"]], on="zone", how="left"
        )
        tiebreak_col = "vehicle_emission_load_index"
    else:
        sub["vehicle_emission_load_index"] = pd.NA
        tiebreak_col = "predicted_aqi"  # no-op tiebreak if vehicle data isn't available yet

    sub = sub.sort_values(["predicted_aqi", tiebreak_col], ascending=[False, False])
    sub["urgency_rank"] = range(1, len(sub) + 1)
    return sub[[
        "urgency_rank", "zone", "predicted_aqi", "grap_stage", "grap_label",
        "vehicle_emission_load_index", "forecast_from", "target_date",
    ]].reset_index(drop=True)
