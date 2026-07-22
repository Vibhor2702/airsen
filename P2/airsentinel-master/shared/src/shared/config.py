"""Paths and cited official constants for the fusion/dashboard layer."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Sibling module output locations — read-only from here, each module owns writing its own.
# "forecasting" is the folder name (renamed from "airsentinel" to match Person 1's role);
# the Python package inside it is still importable as `airsentinel` (see its pyproject.toml)
# — "AirSentinel" is the overall project brand, "forecasting" is just this module's folder.
AIRSENTINEL_FORECASTS_CSV = ROOT.parent / "forecasting" / "outputs" / "forecasts.csv"
AIRSENTINEL_PANEL_CSV = ROOT.parent / "forecasting" / "data" / "processed" / "airsentinel_daily_panel.csv"
VEHICLE_EMISSION_INDEX_CSV = ROOT.parent / "vehicle_emissions" / "outputs" / "vehicle_emission_index.csv"
# Not produced by any module yet — the satellite/Prithvi track's attribution output. Once
# your teammate has real predictions, point this at the real file (or override via env var)
# and fuse.py will pick it up automatically; until then, fuse.py reports it as pending.
SATELLITE_ATTRIBUTION_CSV = ROOT.parent / "satellite_attribution" / "outputs" / "attribution.csv"

# GRAP (Graded Response Action Plan) AQI stage thresholds — official CAQM classification,
# verified current for 2026 against caqm.nic.in GRAP order documents (not invented; see
# shared/README.md for the citation). (AQI_lo, AQI_hi_inclusive, stage_number, stage_label).
GRAP_STAGES = [
    (0, 200, 0, "Below GRAP"),      # GRAP itself only triggers from Poor upward
    (201, 300, 1, "Stage I — Poor"),
    (301, 400, 2, "Stage II — Very Poor"),
    (401, 450, 3, "Stage III — Severe"),
    (451, 1000, 4, "Stage IV — Severe Plus"),
]

# Dominant-source badge codes used on the dashboard map (design plan slide 7: "V vehicular,
# D dust, I industrial, C crop-burning"). Purely a display convention, not a data source.
SOURCE_BADGES = {
    "traffic_heavy": "V", "dust": "D", "industrial_haze": "I",
    "crop_burning_smoke": "C", "clear": "-",
}
