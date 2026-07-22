"""
Loader for real average-daily-distance-per-vehicle-category estimates, from published
transport surveys (e.g. Delhi's Comprehensive Mobility Plan, Economic Survey of Delhi
transport chapter, or a CSE/TERI mobility study) — never invented here.

No distance figure is hardcoded in this codebase — same reasoning as emission_factors.py.
"""

from __future__ import annotations

import pandas as pd

from . import config


def write_template() -> None:
    pd.DataFrame(columns=config.DISTANCE_ESTIMATES_COLUMNS).to_csv(
        config.TEMPLATES_DIR / "distance_estimates_template.csv", index=False
    )


def load() -> pd.DataFrame:
    if not config.DISTANCE_ESTIMATES_CSV.exists():
        raise FileNotFoundError(
            f"No distance estimate table found at {config.DISTANCE_ESTIMATES_CSV}.\n"
            f"This module does not invent average-distance figures — see README.md for "
            f"published transport survey sources, then fill in the template at "
            f"{config.TEMPLATES_DIR / 'distance_estimates_template.csv'} "
            f"and save it to {config.DISTANCE_ESTIMATES_CSV}."
        )
    df = pd.read_csv(config.DISTANCE_ESTIMATES_CSV)
    missing_cols = set(config.DISTANCE_ESTIMATES_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"distance_estimates.csv is missing required columns: {missing_cols}")
    if df["source_citation"].isna().any() or (df["source_citation"].astype(str).str.strip() == "").any():
        raise ValueError("Every row in distance_estimates.csv must cite its source survey.")
    if (df["avg_daily_distance_km"] <= 0).any():
        raise ValueError("distance_estimates.csv has non-positive avg_daily_distance_km values.")
    return df
