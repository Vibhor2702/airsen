"""
GRAP Stage Mapper — the "03 · FUSION LAYER" component from the design plan (slide 5):
"forecast AQI -> CAQM stage I-IV." Deterministic lookup against the official CAQM
thresholds (config.GRAP_STAGES, cited there) — not a model, so there is nothing to overfit.
"""

from __future__ import annotations

from . import config


def stage(aqi: float | None) -> tuple[int, str]:
    """Map an AQI value to its GRAP stage. Returns (stage_number, stage_label)."""
    if aqi is None:
        return (-1, "Unknown")
    for lo, hi, num, label in config.GRAP_STAGES:
        if lo <= aqi <= hi:
            return (num, label)
    return (4, "Stage IV — Severe Plus")  # anything above the top band
