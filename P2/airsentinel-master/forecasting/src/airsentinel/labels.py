"""
OPTIONAL heuristic source-category labels.

Person 2's data request asks whether we can supply a *source breakdown*
(dust / crop_burning_smoke / industrial_haze / traffic_heavy / clear) or only raw pollutant
numbers. CAAQMS gives raw numbers, so the primary deliverable (``caaqms_readings.csv``) is
the raw table. This module additionally derives a **heuristic** source_category from the
pollutant mix, purely as candidate labels to replace the random placeholders — it is NOT
source apportionment and must never be presented as ground truth.

The rules are deliberately simple and transparent (chemical-marker logic):
    - clear             : AQI in Good/Satisfactory range (<= 100)
    - dust              : coarse-dominated — high PM10 with a low PM2.5/PM10 ratio
    - industrial_haze   : SO2 elevated (industrial combustion marker)
    - traffic_heavy     : NO2 + CO elevated together (vehicular combustion markers)
    - crop_burning_smoke: fine-dominated smoke — high PM2.5 with high PM2.5/PM10 ratio
                          (rare in a June–July window; crop-burning season is Oct–Nov)

Ground-truth source labels should come from the Delhi Supersite lab (IIT-Kanpur/DPCC),
which is the cross-check the AirSentinel plan already names for the satellite attribution.
"""

from __future__ import annotations

import math

import pandas as pd

SOURCE_CATEGORIES = ["dust", "crop_burning_smoke", "industrial_haze", "traffic_heavy", "clear"]


def _nz(x) -> float:
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def classify(row: pd.Series) -> str:
    aqi = _nz(row.get("AQI"))
    pm25, pm10 = _nz(row.get("PM2.5")), _nz(row.get("PM10"))
    no2, so2, co = _nz(row.get("NO2")), _nz(row.get("SO2")), _nz(row.get("CO"))
    ratio = pm25 / pm10 if pm10 > 0 else 0.0

    if aqi and aqi <= 100:
        return "clear"
    # Fine-particle smoke: high PM2.5 and fine-dominated.
    if pm25 >= 90 and ratio >= 0.5:
        return "crop_burning_smoke"
    # Industrial: SO2 is the distinguishing marker in this pollutant set.
    if so2 >= 25:
        return "industrial_haze"
    # Vehicular: combustion gases elevated together.
    if no2 >= 40 and co >= 1.2:
        return "traffic_heavy"
    # Coarse/dust dominated.
    if pm10 >= 150 and ratio <= 0.45:
        return "dust"
    # Fall back on the dominant marker.
    if no2 >= 30 and co >= 1.0:
        return "traffic_heavy"
    if pm10 >= pm25 * 2:
        return "dust"
    return "industrial_haze" if so2 >= 15 else "dust"


def add_labels(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["source_category"] = out.apply(classify, axis=1)
    return out
