"""
Vehicle Emission Load Index — deterministic formula, matching the AirSentinel design plan
exactly (slide 4):

    vehicle count (by type & fuel, per RTO)
    x emission factor (g/km, ARAI/CPCB tables)
    x avg. distance (published transport surveys)
    = Vehicle Emission Load Index per zone

This is arithmetic over real inputs, not a fitted model — there is no training data, no
parameters estimated from data, and therefore no overfitting risk by construction. The only
"tuning" is the (documented, not hidden) choice of how to combine per-category loads into
one per-zone number, which is a sum followed by a relative (0-1) normalization — exactly
what the design plan's dashboard mockup shows ("Vehicle Emission Load Index: 0.81").

Honesty flag (carried verbatim from the design plan, slide 9): this is a modelled estimate
from registration data, not a live sensor reading — a legitimate, standard methodology (the
same one national emission inventories use), but not the same claim as "we measured a
tailpipe."
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config

DEMO_TAG = "[DEMO DATA"  # matches build_demo_data.py's DEMO_TAG prefix


def compute(
    registrations: pd.DataFrame,
    emission_factors: pd.DataFrame,
    distances: pd.DataFrame,
    known_zones: set[str] | None = None,
) -> pd.DataFrame:
    """Join the three real input tables and compute a relative Vehicle Emission Load Index
    per zone.

    Rows in `registrations` whose (vehicle_category, fuel_type) has no matching emission
    factor or distance estimate are excluded from that zone's load and reported in
    `coverage_note` — never silently assumed to be zero or imputed.
    """
    if known_zones is not None:
        unknown = set(registrations["zone"].unique()) - known_zones
        if unknown:
            raise ValueError(
                f"vehicle_registrations.csv has zone names not in the AirSentinel 13-zone "
                f"list: {sorted(unknown)}. Zone names must match exactly (same spelling) as "
                f"airsentinel.stations.ZONES, or the merge with the forecasting output will "
                f"silently fail — see registrations.py."
            )

    # Emission factors may have multiple pollutants per (vehicle_category, fuel_type); sum
    # them to one total g/km per vehicle for the load calculation, but keep the per-pollutant
    # detail available via `emission_factors` for anyone who wants pollutant-level breakdown.
    ef_total = (
        emission_factors.groupby(config.JOIN_KEYS)["emission_factor_g_per_km"]
        .sum()
        .rename("total_emission_factor_g_per_km")
        .reset_index()
    )
    dist = distances.set_index("vehicle_category")["avg_daily_distance_km"]

    merged = registrations.merge(ef_total, on=config.JOIN_KEYS, how="left")
    merged["avg_daily_distance_km"] = merged["vehicle_category"].map(dist)

    merged["matched"] = merged["total_emission_factor_g_per_km"].notna() & merged["avg_daily_distance_km"].notna()
    merged["load_g_per_day"] = (
        merged["vehicle_count"]
        * merged["total_emission_factor_g_per_km"]
        * merged["avg_daily_distance_km"]
    ).where(merged["matched"], 0.0)

    rows = []
    for zone, grp in merged.groupby("zone"):
        n_included = int(grp["matched"].sum())
        n_total = int(len(grp))
        raw_load = float(grp["load_g_per_day"].sum())
        unmatched = grp.loc[~grp["matched"], config.JOIN_KEYS].drop_duplicates()
        coverage_note = (
            f"{n_included}/{n_total} vehicle_category+fuel_type combinations matched an "
            f"emission factor and distance estimate."
        )
        if not unmatched.empty:
            missing = ", ".join(f"{r.vehicle_category}/{r.fuel_type}" for r in unmatched.itertuples())
            coverage_note += f" Excluded (no matching factor/distance data): {missing}."
        rows.append({
            "zone": zone,
            "vehicle_emission_load_raw_g_per_day": raw_load,
            "n_vehicle_categories_included": n_included,
            "coverage_note": coverage_note,
        })

    out = pd.DataFrame(rows)
    max_raw = out["vehicle_emission_load_raw_g_per_day"].max()
    out["vehicle_emission_load_index"] = (
        (out["vehicle_emission_load_raw_g_per_day"] / max_raw).round(3) if max_raw and max_raw > 0 else 0.0
    )
    out["data_asof"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Honesty propagation: if ANY input row is tagged as demo data (build_demo_data.py),
    # the whole output is marked "demo" — never silently presented as verified real data.
    all_citations = pd.concat([
        registrations.get("source_citation", pd.Series(dtype=str)),
        emission_factors.get("source_citation", pd.Series(dtype=str)),
        distances.get("source_citation", pd.Series(dtype=str)),
    ])
    is_demo = all_citations.astype(str).str.contains(DEMO_TAG, regex=False).any()
    out["data_provenance"] = "demo — real-world-informed, not verified precision" if is_demo else "real, cited"

    return out[config.OUTPUT_COLUMNS]
