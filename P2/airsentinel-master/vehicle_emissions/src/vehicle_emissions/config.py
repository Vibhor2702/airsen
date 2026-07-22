"""
Single source of truth for every tunable constant in this module — same policy as
`airsentinel.config`: nothing here is fabricated data, and nothing that IS a genuine
external fact (an emission factor, a vehicle count) lives here as a Python literal. This
file only holds file paths, schema/column names, and the (documented, cited) formula
parameters that are structural, not numeric facts about vehicles.

Why there is no built-in emission-factor or distance table in this codebase
-----------------------------------------------------------------------------
The AirSentinel design plan specifies the Vehicle Emission Load Index as:

    vehicle count (by type & fuel, per RTO)  x  emission factor (g/km, ARAI/CPCB tables)
    x  avg. distance (published transport surveys)  =  Vehicle Emission Load Index

Two of those three inputs are specific numeric facts (emission factors, average distances)
that come from official technical reports, not from anything we can verify ourselves. A
web search this session located real, existing official sources:

  - CPCB's own emission-factor PDF: https://cpcb.nic.in/displaypdf.php?id=RW1pc3Npb25fRmFjdG9yc19WZWhpY2xlcy5wZGY=
  - CPCB vehicular exhaust page:    https://cpcb.nic.in/vehicular-exhaust/
  - ARAI "Emission Factors for Indian In-Use Vehicles" (report, via img.jari.or.jp mirror)
  - Lok Sabha reply (9 Aug 2021) on vehicle-wise emission factors, on data.gov.in

but the PDF/table-rendering tools available in this environment could not reliably extract
the exact g/km figures from those documents to verify them. Rather than guess plausible-
looking numbers and present them as "the ARAI/CPCB table," this module requires the real
table as an input file (see registrations.py / emission_factors.py / distance_estimates.py)
— pull the actual numbers from the links above (or your own team's sourced copy) into the
provided CSV templates. This is slower than hardcoding, but it means every number this
module ever outputs is traceable to a document you can point to, which is the whole point
of the "what we are not claiming" honesty the rest of AirSentinel already commits to.

Vehicle registration counts have a related but different treatment: Delhi's 13 hotspot
zones (Anand Vihar, Wazirpur, ...) are not the same thing as VAHAN's RTO boundaries, and no
official zone->RTO mapping is published. This module DOES ship a nearest-RTO approximation
(`rto_mapping.py`) — built from the real, current list of Delhi RTOs pulled live from
vahan.parivahan.gov.in — so registrations can be supplied at VAHAN's actual export
granularity (per RTO) rather than requiring you to do the zone aggregation by hand. It's
still an approximation, but it's fully visible and editable in one place rather than hidden
inside a scraper, with the reasoning for each assignment documented inline.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
RAW_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = ROOT / "outputs"
for _d in (TEMPLATES_DIR, RAW_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Real input files (you supply these — see README.md "Getting real data" for exact steps
# and the citation links above). Overridable via env var so a teammate can point this at a
# shared drive location instead of copying files locally.
#
# Two ways to supply registration counts:
#   - REGISTRATIONS_BY_RTO_CSV (recommended): raw counts per real VAHAN RTO, at VAHAN's
#     actual export granularity. Aggregated to zones via rto_mapping.py's documented
#     nearest-RTO approximation (registrations.load_by_rto()).
#   - REGISTRATIONS_CSV: counts already aggregated to zones yourself, if you'd rather do
#     that mapping by hand (registrations.load()).
REGISTRATIONS_CSV = Path(os.environ.get(
    "VEHICLE_REGISTRATIONS_CSV", RAW_DIR / "vehicle_registrations.csv"
))
REGISTRATIONS_BY_RTO_CSV = Path(os.environ.get(
    "VEHICLE_REGISTRATIONS_BY_RTO_CSV", RAW_DIR / "vehicle_registrations_by_rto.csv"
))
EMISSION_FACTORS_CSV = Path(os.environ.get(
    "VEHICLE_EMISSION_FACTORS_CSV", RAW_DIR / "emission_factors.csv"
))
DISTANCE_ESTIMATES_CSV = Path(os.environ.get(
    "VEHICLE_DISTANCE_ESTIMATES_CSV", RAW_DIR / "distance_estimates.csv"
))

# Required schema for each input — enforced by the loaders in registrations.py /
# emission_factors.py / distance_estimates.py so a malformed or incomplete real-data file
# fails loudly instead of silently producing a wrong index.
REGISTRATIONS_COLUMNS = ["zone", "vehicle_category", "fuel_type", "vehicle_count", "data_period", "source_citation"]
REGISTRATIONS_BY_RTO_COLUMNS = ["rto_code", "vehicle_category", "fuel_type", "vehicle_count", "data_period", "source_citation"]
EMISSION_FACTORS_COLUMNS = ["vehicle_category", "fuel_type", "pollutant", "emission_factor_g_per_km", "bs_norm", "source_citation"]
DISTANCE_ESTIMATES_COLUMNS = ["vehicle_category", "avg_daily_distance_km", "source_citation"]

# The join key across the three tables — a (vehicle_category, fuel_type) combination must
# appear in all three for it to contribute to a zone's index (see index.py). Rows that don't
# join are reported, not silently dropped.
JOIN_KEYS = ["vehicle_category", "fuel_type"]

# Output schema, matching the (zone, ...) key convention the forecasting module already uses
# so the two are trivially joinable — see airsentinel's data/teammate_delivery/caaqms_readings.csv.
OUTPUT_COLUMNS = [
    "zone", "vehicle_emission_load_raw_g_per_day", "vehicle_emission_load_index",
    "n_vehicle_categories_included", "data_asof", "coverage_note", "data_provenance",
]
