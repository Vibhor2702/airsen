"""
Demo data generator — real-world-informed reference dataset for tonight's demo.

This is a deliberately SEPARATE, clearly-labeled path from the "real verified data" flow
in registrations.py / emission_factors.py / distance_estimates.py. It exists because a
working demo was needed before real VAHAN per-RTO exports and a verified ARAI/CPCB table
could be sourced. It is NOT random/fabricated data — every figure below is either:

  (a) a REAL, cited public figure (vehicle registration totals, official BS6 regulatory
      emission limits, a published urban-mobility distance study), used as-is; or
  (b) an explicit, labeled ALLOCATION ASSUMPTION where no granular real data exists yet
      (e.g. splitting a real Delhi-wide total evenly across the 13 zones, or a documented
      petrol/diesel fleet-mix assumption) — never presented as more precise than it is.

Every row's source_citation says which of (a) or (b) it is. Nothing here is a random number.

Real citations used (checked this session):
  - Delhi two-wheeler registrations: 10,370,047 (motorcycles + scooters), Business Standard,
    "Delhi's EV transition by the numbers", citing VAHAN aggregate data, 2025-2026.
  - Delhi car/jeep registrations: 3,311,579 (2019-20), historical VAHAN-sourced figure
    widely reported; the most recent breakdown available via general search this session.
  - BS6 (Bharat Stage VI) regulatory NOx/PM limits — official, India's mandated standard
    since 1 April 2020, corroborated across multiple independent sources this session:
      * Petrol passenger car / two-wheeler: NOx 60 mg/km = 0.06 g/km, PM 4.5 mg/km = 0.0045 g/km
      * Diesel passenger car:               NOx 80 mg/km = 0.08 g/km, PM 4.5 mg/km = 0.0045 g/km
  - Average daily distance, two-wheelers, Indian cities: ~27-33 km/day (midpoint 30 km/day
    used here), from a published Indian urban-mobility study cited in general search results.

Explicit assumptions (not citations — labeled as such in every row that uses them):
  - Petrol/diesel split of Delhi's car fleet: assumed 70% petrol / 30% diesel (a commonly
    cited rough order for India's post-BS6 passenger fleet; NOT a Delhi-specific verified
    figure — replace with a real SIAM/VAHAN fuel-type breakdown when available).
  - Car average daily distance: reused the two-wheeler urban-mobility figure (30 km/day) in
    the absence of a car-specific citation found this session — an approximation, not a
    distinct real number for cars.
  - Zone allocation: the real Delhi-wide totals above are split EVENLY across the 13
    AirSentinel zones. This is a deliberate placeholder, not a claim about which zones
    actually have more vehicles — replace with real per-RTO VAHAN exports (via
    registrations.load_by_rto()) as soon as they're available.

Run:
    python -m vehicle_emissions.build_demo_data
writes the three files load_by_rto()/load()/emission_factors.load()/distance_estimates.load()
expect, so `python -m vehicle_emissions.pipeline` runs end-to-end immediately after.
"""

from __future__ import annotations

import sys

import pandas as pd

from . import config

DEMO_TAG = "[DEMO DATA — see build_demo_data.py for full citations]"

# --- real Delhi-wide vehicle totals, split with a LABELED assumption ---------------------
TWO_WHEELER_TOTAL = 10_370_047  # real, cited: Business Standard / VAHAN aggregate
CAR_TOTAL = 3_311_579           # real, cited: VAHAN-sourced 2019-20 registration figure
CAR_PETROL_SHARE = 0.70         # ASSUMPTION, not cited — see module docstring
CAR_DIESEL_SHARE = 0.30         # ASSUMPTION, not cited — see module docstring


def _known_zones() -> list[str]:
    try:
        from airsentinel.stations import ZONES
    except ImportError as e:
        raise ImportError(
            "Could not import airsentinel.stations — install the forecasting module in "
            "editable mode first: `pip install -e ../forecasting` from this module's venv."
        ) from e
    return sorted(ZONES.keys())


def build_registrations_by_zone() -> pd.DataFrame:
    zones = _known_zones()
    n = len(zones)
    rows = []
    per_zone_2w = TWO_WHEELER_TOTAL / n
    per_zone_car_petrol = (CAR_TOTAL * CAR_PETROL_SHARE) / n
    per_zone_car_diesel = (CAR_TOTAL * CAR_DIESEL_SHARE) / n
    for zone in zones:
        rows.append({
            "zone": zone, "vehicle_category": "two_wheeler", "fuel_type": "petrol",
            "vehicle_count": round(per_zone_2w), "data_period": "2025-2026 (Delhi-wide total)",
            "source_citation": f"{DEMO_TAG} Real Delhi-wide total 10,370,047 (Business Standard/"
                                f"VAHAN), split EVENLY across 13 zones (assumption, not a real "
                                f"per-zone count).",
        })
        rows.append({
            "zone": zone, "vehicle_category": "car", "fuel_type": "petrol",
            "vehicle_count": round(per_zone_car_petrol), "data_period": "2019-20 (Delhi-wide total)",
            "source_citation": f"{DEMO_TAG} Real Delhi-wide car total 3,311,579 (VAHAN, 2019-20) "
                                f"x assumed 70% petrol share, split evenly across 13 zones.",
        })
        rows.append({
            "zone": zone, "vehicle_category": "car", "fuel_type": "diesel",
            "vehicle_count": round(per_zone_car_diesel), "data_period": "2019-20 (Delhi-wide total)",
            "source_citation": f"{DEMO_TAG} Real Delhi-wide car total 3,311,579 (VAHAN, 2019-20) "
                                f"x assumed 30% diesel share, split evenly across 13 zones.",
        })
    return pd.DataFrame(rows, columns=config.REGISTRATIONS_COLUMNS)


def build_emission_factors() -> pd.DataFrame:
    # Official BS6 regulatory limits (India, mandated 1 Apr 2020) — real, not estimated.
    rows = [
        ("two_wheeler", "petrol", "NOx", 0.060, "BS6",
         f"{DEMO_TAG} Official BS6 NOx limit, petrol two-wheeler: 60 mg/km."),
        ("two_wheeler", "petrol", "PM", 0.0045, "BS6",
         f"{DEMO_TAG} Official BS6 PM limit (shared across BS6 categories): 4.5 mg/km."),
        ("car", "petrol", "NOx", 0.060, "BS6",
         f"{DEMO_TAG} Official BS6 NOx limit, petrol passenger car: 60 mg/km."),
        ("car", "petrol", "PM", 0.0045, "BS6",
         f"{DEMO_TAG} Official BS6 PM limit: 4.5 mg/km."),
        ("car", "diesel", "NOx", 0.080, "BS6",
         f"{DEMO_TAG} Official BS6 NOx limit, diesel passenger car: 80 mg/km."),
        ("car", "diesel", "PM", 0.0045, "BS6",
         f"{DEMO_TAG} Official BS6 PM limit: 4.5 mg/km."),
    ]
    return pd.DataFrame(rows, columns=config.EMISSION_FACTORS_COLUMNS)


def build_distance_estimates() -> pd.DataFrame:
    rows = [
        ("two_wheeler", 30.0,
         f"{DEMO_TAG} Real cited range 27-33 km/day, Indian-cities urban mobility study; "
         f"midpoint used."),
        ("car", 30.0,
         f"{DEMO_TAG} No car-specific citation found this session — reused the two-wheeler "
         f"urban-mobility figure as an approximation, NOT a distinct real car number."),
    ]
    return pd.DataFrame(rows, columns=config.DISTANCE_ESTIMATES_COLUMNS)


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    regs = build_registrations_by_zone()
    efs = build_emission_factors()
    dists = build_distance_estimates()

    regs.to_csv(config.REGISTRATIONS_CSV, index=False)
    efs.to_csv(config.EMISSION_FACTORS_CSV, index=False)
    dists.to_csv(config.DISTANCE_ESTIMATES_CSV, index=False)

    print(f"Demo data written (real-world-informed, NOT random — see this file's docstring "
          f"for every citation):")
    print(f"  {config.REGISTRATIONS_CSV}  ({len(regs)} rows)")
    print(f"  {config.EMISSION_FACTORS_CSV}  ({len(efs)} rows)")
    print(f"  {config.DISTANCE_ESTIMATES_CSV}  ({len(dists)} rows)")
    print("\nRun `python -m vehicle_emissions.pipeline` next.")
    print("Replace these before using this for anything beyond tonight's demo — see README.md.")


if __name__ == "__main__":
    sys.exit(main())
