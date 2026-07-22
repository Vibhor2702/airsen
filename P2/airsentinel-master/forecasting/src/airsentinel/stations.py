"""
Zone / station registry and parameter definitions for the DPCC CAAQMS scraper.

The 13 Delhi pollution "hotspot" zones are the canonical zone names agreed with the
satellite/enforcement track (Person 2). The spelling of ``ZONES`` keys MUST match the
zone list in the Teammate Data Request exactly, because both tracks merge on this column.

Each zone maps to a DPCC monitoring station. ``st_code`` is the base64 token the DPCC
site uses in ``?stName=`` — it encodes an *internal* station id (e.g. "RohiniSector16",
"PoothKhurdBawana", "DwarkaSectro8" — note the site's own typo), so the tokens are stored
verbatim as captured from the live site rather than recomputed from the display name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    zone: str          # canonical hotspot zone name (matches teammate's list exactly)
    dpcc_name: str     # station name as shown on dpccairdata.com
    st_code: str       # base64 stName token used by the site


# --- The 13 hotspot zones -> DPCC station, in the teammate's exact spelling -------------
ZONES: dict[str, Station] = {
    "Anand Vihar":  Station("Anand Vihar",  "Anand Vihar",          "QW5hbmRWaWhhcg=="),
    "Mundka":       Station("Mundka",       "Mundka",               "TXVuZGth"),
    "Wazirpur":     Station("Wazirpur",     "Wazirpur",             "V2F6aXJwdXI="),
    "Jahangirpuri": Station("Jahangirpuri", "Jahangirpuri",         "SmFoYW5naXJwdXJp"),
    "RK Puram":     Station("RK Puram",     "R.K. Puram",           "UktQdXJhbQ=="),
    "Rohini":       Station("Rohini",       "Rohini",               "Um9oaW5pU2VjdG9yMTY="),
    "Punjabi Bagh": Station("Punjabi Bagh", "Punjabi Bagh",         "UHVuamFiaUJhZ2g="),
    "Okhla":        Station("Okhla",        "Okhla Phase-2",        "T2tobGFQaGFzZTI="),
    "Bawana":       Station("Bawana",       "Pooth Khurd, Bawana",  "UG9vdGhLaHVyZEJhd2FuYQ=="),
    "Vivek Vihar":  Station("Vivek Vihar",  "Vivek Vihar",          "Vml2ZWtWaWhhcg=="),
    "Narela":       Station("Narela",       "Narela",               "TmFyZWxh"),
    "Ashok Vihar":  Station("Ashok Vihar",  "Ashok Vihar",          "QXNob2tWaWhhcg=="),
    "Dwarka":       Station("Dwarka",       "Dwarka, Sector 8",     "RHdhcmthU2VjdHJvOA=="),
}


# --- Parameters -------------------------------------------------------------------------
# endpoint: "met"  -> AallAdvanceSearchMet.php  (particulate + meteorological)
#           "gas"  -> AallAdvanceSearchCconc.php (gas concentrations)
# code:     the value posted in the `parameters` field.
# unit:     native unit reported by the site (kept as-is; documented in the data dictionary).
@dataclass(frozen=True)
class Param:
    column: str        # canonical column name used across the pipeline
    endpoint: str      # "met" or "gas"
    code: str          # DPCC `parameters` form value
    unit: str


# Pollutants the teammate deliverable requires (zone, date, PM2.5, PM10, NO2, SO2, CO, O3)
POLLUTANTS: list[Param] = [
    Param("PM2.5", "met", "PM25", "ug/m3"),
    Param("PM10",  "met", "PM10", "ug/m3"),
    Param("NO2",   "gas", "NO2",  "ug/m3"),
    Param("SO2",   "gas", "SO2",  "ug/m3"),
    Param("CO",    "gas", "CO",   "mg/m3"),
    Param("O3",    "gas", "O3",   "ug/m3"),
]

# Weather features used by the forecasting model (not part of the teammate table).
WEATHER: list[Param] = [
    Param("temp",       "met", "AT1", "degC"),
    Param("humidity",   "met", "RH",  "%"),
    Param("wind_speed", "met", "WS",  "m/s"),
]

ALL_PARAMS: list[Param] = POLLUTANTS + WEATHER

ENDPOINTS = {
    "met": "https://www.dpccairdata.com/dpccairdata/display/AallAdvanceSearchMet.php",
    "gas": "https://www.dpccairdata.com/dpccairdata/display/AallAdvanceSearchCconc.php",
}

# The exact column order for the teammate deliverable CSV.
DELIVERY_COLUMNS = ["zone", "date", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# Delhi Supersite — NOT one of the 13 forecasting zones (it's a reference/ground-truth lab,
# not a hotspot), but confirmed live on DPCC itself (same site, same POST contract, station
# name "Supersite(Rouse Avenue)") — kept here so shared/cross_validation.py or a future
# script can pull it without re-discovering the station code. Verified this session: the
# scraper works against it, but real data was only found on isolated dates (station appears
# to have gone quiet after ~March 2025) — see shared/cross_validation.py's docstring.
SUPERSITE = Station("Supersite", "Supersite(Rouse Avenue)", "U3VwZXJzaXRl")
