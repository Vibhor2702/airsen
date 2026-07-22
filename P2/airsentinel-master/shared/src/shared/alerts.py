"""
Citizen / Official Alert Generator — templated text from real computed values only (design
plan slide 8). Pure string formatting over numbers already computed elsewhere in the
pipeline (forecast AQI, GRAP stage, dominant pollutant) — no new data, no fitted parameters.

The advisory bullets below are deliberately generic/conservative: general public-health
guidance (avoid outdoor activity when air quality is poor) plus the one GRAP action we could
verify with a citable current source (construction/demolition restrictions apply from Stage
III). We do NOT enumerate the full, frequently-revised CAQM restriction list per stage here
— that changes over time and belongs in a real citation-backed reference table (same
treatment as vehicle_emissions' emission factors), not invented prose. Wire in the official
GRAP action list from caqm.nic.in before using this for anything beyond a demo.
"""

from __future__ import annotations


def generate(
    zone: str,
    aqi: float,
    grap_stage_label: str,
    grap_stage_num: int,
    horizon_hours: int,
    dominant_source: str | None = None,
) -> str:
    source_line = (
        f" Likely major contributor: {dominant_source}."
        if dominant_source and dominant_source not in ("clear", "Unknown", None)
        else ""
    )
    bullets = [
        "Children, elderly & those with respiratory or heart conditions: avoid outdoor activity",
        "Use public transport where possible; avoid unnecessary vehicle use in this zone",
    ]
    if grap_stage_num >= 3:
        bullets.append(f"Construction/demolition restricted under GRAP {grap_stage_label}")

    advisory = "\n".join(f"  - {b}" for b in bullets)

    return (
        f"AIR QUALITY ALERT\n"
        f"{zone}, Delhi\n"
        f"{round(aqi)}\n"
        f"{grap_stage_label}\n"
        f"Expected to stay in this range for the next {horizon_hours} hours.{source_line}\n\n"
        f"Advisory for residents\n{advisory}\n\n"
        f"Follows the Graded Response Action Plan (GRAP), "
        f"Commission for Air Quality Management (CAQM)."
    )
