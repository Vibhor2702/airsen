"""
Vehicle Emission Load Model — end-to-end pipeline.

    load real registrations + emission factors + distances
        -> validate zone names against AirSentinel's canonical 13-zone list
        -> compute the Vehicle Emission Load Index (index.py, pure arithmetic)
        -> write outputs/vehicle_emission_index.csv (zone-keyed, joins directly with the
           forecasting module's output on `zone`)

Run:
    python -m vehicle_emissions.pipeline

This will fail loudly (not fabricate output) until you've filled in the three real-data
templates in data/templates/ and saved them to data/raw/ — see README.md.

Compatibility with the forecasting track
-----------------------------------------
The `../forecasting` module is installed in editable mode as a dependency of this package
(see README.md "Setup": `pip install -e ../forecasting`), so the zone list is imported
directly from `airsentinel.stations.ZONES` (the Python package name, unchanged even though
its folder was renamed to `forecasting`) rather than retyped here — the two modules cannot
drift out of sync on zone names, which is exactly the kind of silent-merge-failure risk the
teammate data handoff (../forecasting/data/teammate_delivery/DATA_HANDOFF.md, question 3)
already flags for the satellite track.
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import config, distance_estimates, emission_factors, index, registrations


def _known_zones() -> set[str]:
    try:
        from airsentinel.stations import ZONES
    except ImportError as e:
        raise ImportError(
            "Could not import airsentinel.stations — install the forecasting module in "
            "editable mode first: `pip install -e ../forecasting` from this module's venv. "
            "This is required so the two modules share one canonical zone list instead of "
            "two hardcoded copies."
        ) from e
    return set(ZONES.keys())


def write_templates() -> None:
    registrations.write_template()
    emission_factors.write_template()
    distance_estimates.write_template()


def _load_registrations() -> pd.DataFrame:
    """Prefer RTO-level real data (VAHAN's actual export granularity, auto-aggregated to
    zones via rto_mapping.py) over hand-aggregated zone-level data, if both are absent falls
    through to load()'s error message."""
    if config.REGISTRATIONS_BY_RTO_CSV.exists():
        return registrations.load_by_rto()
    return registrations.load()


def run() -> pd.DataFrame:
    known_zones = _known_zones()
    regs = _load_registrations()
    efs = emission_factors.load()
    dists = distance_estimates.load()
    result = index.compute(regs, efs, dists, known_zones=known_zones)
    out_path = config.OUTPUTS_DIR / "vehicle_emission_index.csv"
    result.to_csv(out_path, index=False)
    return result, out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Vehicle Emission Load Model pipeline")
    ap.add_argument("--write-templates", action="store_true",
                    help="(re)write the empty CSV templates in data/templates/ and exit")
    args = ap.parse_args()

    if args.write_templates:
        write_templates()
        print(f"Templates written to {config.TEMPLATES_DIR}")
        return

    print("Loading real vehicle registration / emission-factor / distance data...")
    result, out_path = run()
    print(f"\nComputed Vehicle Emission Load Index for {len(result)} zones:")
    for _, r in result.sort_values("vehicle_emission_load_index", ascending=False).iterrows():
        print(f"  {r['zone']:14s} index={r['vehicle_emission_load_index']:.3f}  "
              f"({r['coverage_note']})")
    print(f"\nWritten to {out_path}")
    print("This is a modelled estimate from registration data, not a live sensor reading — "
          "see README.md 'What this module is not claiming'.")


if __name__ == "__main__":
    main()
