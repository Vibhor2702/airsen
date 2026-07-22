"""
Fusion layer — combines each track's independently-produced output into one zone-keyed
table, joined only on `zone` (and `date`/`target_date` where relevant). This is the "03 ·
FUSION LAYER" from the design plan (slide 5), built ahead of full satellite/enforcement
integration so wiring in the remaining pieces is a data drop, not a rewrite.

Every source is optional except the forecasting output — if a sibling module (vehicle
emissions, satellite attribution) hasn't produced real output yet, its columns are filled
with an explicit "pending" marker, never a fabricated number or silently dropped row. Check
`data_completeness` in the returned summary to see exactly what's real in a given run.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config, grap, zone_ranking

PENDING = "pending — module not yet run"


def _load_optional(path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def fuse(horizon_hours: int = 24) -> tuple[pd.DataFrame, dict]:
    if not config.AIRSENTINEL_FORECASTS_CSV.exists():
        raise FileNotFoundError(
            f"No forecasts found at {config.AIRSENTINEL_FORECASTS_CSV} — run the "
            f"forecasting pipeline first (`python -m airsentinel.pipeline` from ../forecasting)."
        )
    forecasts = pd.read_csv(config.AIRSENTINEL_FORECASTS_CSV)
    panel = _load_optional(config.AIRSENTINEL_PANEL_CSV, "daily panel")
    vehicle_idx = _load_optional(config.VEHICLE_EMISSION_INDEX_CSV, "vehicle emissions")
    satellite = _load_optional(config.SATELLITE_ATTRIBUTION_CSV, "satellite attribution")

    ranked = zone_ranking.rank(forecasts, vehicle_idx, horizon_hours=horizon_hours)

    # Most recent real (not forecast) reading per zone, for "current conditions" context.
    if panel is not None:
        latest = panel.sort_values("date").groupby("zone").tail(1)
        latest = latest[["zone", "date", "AQI", "AQI_category", "dominant_pollutant"]].rename(
            columns={"date": "latest_actual_date", "AQI": "latest_actual_aqi",
                     "AQI_category": "latest_actual_category",
                     "dominant_pollutant": "latest_actual_dominant_pollutant"}
        )
        fused = ranked.merge(latest, on="zone", how="left")
    else:
        fused = ranked.copy()
        for c in ["latest_actual_date", "latest_actual_aqi", "latest_actual_category",
                  "latest_actual_dominant_pollutant"]:
            fused[c] = PENDING

    if satellite is not None:
        # Required schema: zone, source_guess, confidence.
        # Optional honesty columns (added by enforcement_ranker.py): model_version,
        # data_provenance, land_use_note — pulled through when present so the dashboard
        # can render them; silently set to PENDING if absent (backwards-compatible).
        expected = {"zone", "source_guess", "confidence"}
        if not expected.issubset(satellite.columns):
            raise ValueError(
                f"satellite attribution output at {config.SATELLITE_ATTRIBUTION_CSV} is "
                f"missing expected columns {expected - set(satellite.columns)} — check the "
                f"schema with your teammate before fusing."
            )
        sat_cols = ["zone", "source_guess", "confidence"]
        honesty_cols = ["model_version", "data_provenance", "land_use_note"]
        sat_cols += [c for c in honesty_cols if c in satellite.columns]
        rename_map = {"source_guess": "satellite_source_guess", "confidence": "satellite_confidence"}
        rename_map.update({c: f"satellite_{c}" for c in honesty_cols if c in satellite.columns})
        fused = fused.merge(
            satellite[sat_cols].rename(columns=rename_map),
            on="zone", how="left",
        )
        fused["satellite_source_guess"] = fused["satellite_source_guess"].fillna(PENDING)
        fused["satellite_confidence"] = fused["satellite_confidence"].fillna(PENDING)
        for c in honesty_cols:
            col = f"satellite_{c}"
            if col not in fused.columns:
                fused[col] = PENDING
            else:
                fused[col] = fused[col].fillna(PENDING)
    else:
        fused["satellite_source_guess"] = PENDING
        fused["satellite_confidence"] = PENDING
        for c in ["model_version", "data_provenance", "land_use_note"]:
            fused[f"satellite_{c}"] = PENDING

    vehicle_provenance = None
    if vehicle_idx is None:
        fused["vehicle_emission_load_index"] = PENDING
    elif "data_provenance" in vehicle_idx.columns:
        # Propagate the demo-vs-real flag onto the fused rows too, not just the completeness
        # summary, so anyone reading fused_zone_state.csv directly sees it as well.
        fused = fused.merge(
            vehicle_idx[["zone", "data_provenance"]].rename(
                columns={"data_provenance": "vehicle_emission_data_provenance"}
            ),
            on="zone", how="left",
        )
        vehicle_provenance = vehicle_idx["data_provenance"].iloc[0] if len(vehicle_idx) else None

    fused["fused_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if vehicle_idx is None:
        vehicle_status = PENDING
    elif vehicle_provenance and "demo" in str(vehicle_provenance).lower():
        vehicle_status = "live — DEMO DATA (real-world-informed, not verified precision)"
    else:
        vehicle_status = "live"

    completeness = {
        "forecasting": "live" if not forecasts.empty else PENDING,
        "caaqms_current_conditions": "live" if panel is not None else PENDING,
        "vehicle_emissions": vehicle_status,
        "satellite_attribution": "live" if satellite is not None else PENDING,
    }

    out_path = config.OUTPUTS_DIR / "fused_zone_state.csv"
    fused.to_csv(out_path, index=False)
    return fused, completeness
