"""
CPCB (India) National Air Quality Index.

Reference: Central Pollution Control Board, National AQI. Sub-indices are computed by
piecewise-linear interpolation between the published breakpoints; the overall AQI is the
maximum sub-index across available pollutants. CPCB requires at least three pollutants,
with at least one of PM2.5 / PM10, for a valid AQI — enforced by ``compute_aqi``.

Averaging periods per CPCB: PM2.5, PM10, NO2, SO2 use 24-hour averages; CO and O3 use the
daily maximum 8-hour average. See ``build_daily_panel`` for how the hourly scrape is
aggregated to those conventions.
"""

from __future__ import annotations

import math

from . import config

# (C_lo, C_hi, I_lo, I_hi) bands per pollutant, in the pollutant's native unit.
_BREAKPOINTS: dict[str, list[tuple[float, float, float, float]]] = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
              (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "PM10":  [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
              (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "NO2":   [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
              (181, 280, 201, 300), (281, 400, 301, 400), (401, 600, 401, 500)],
    "SO2":   [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
              (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)],
    "CO":    [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200),   # mg/m3
              (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    "O3":    [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
              (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
}

AQI_CATEGORIES = [
    (50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
    (300, "Poor"), (400, "Very Poor"), (config.AQI_MAX, "Severe"),
]


def sub_index(pollutant: str, conc: float | None) -> float | None:
    """Piecewise-linear CPCB sub-index for one pollutant concentration."""
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return None
    bands = _BREAKPOINTS.get(pollutant)
    if bands is None:
        return None
    if conc <= 0:
        return float(config.AQI_MIN)
    for c_lo, c_hi, i_lo, i_hi in bands:
        if c_lo <= conc <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (conc - c_lo) + i_lo)
    # Above the top band -> cap at the CPCB maximum.
    return float(config.AQI_MAX)


def category(aqi: float | None) -> str:
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return "Unknown"
    for hi, name in AQI_CATEGORIES:
        if aqi <= hi:
            return name
    return "Severe"


def compute_aqi(concentrations: dict[str, float]) -> tuple[float | None, str | None]:
    """Overall AQI = max sub-index. Requires >=3 pollutants incl. one of PM2.5/PM10.

    Returns (aqi, dominant_pollutant). Returns (None, None) if the CPCB minimum-data
    rule is not met.
    """
    subs: dict[str, float] = {}
    for pol, conc in concentrations.items():
        si = sub_index(pol, conc)
        if si is not None:
            subs[pol] = si
    has_pm = any(p in subs for p in ("PM2.5", "PM10"))
    if len(subs) < 3 or not has_pm:
        return None, None
    dominant = max(subs, key=subs.get)
    return float(subs[dominant]), dominant
