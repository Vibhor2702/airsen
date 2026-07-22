"""
Loader for real vehicle registration counts by zone / vehicle category / fuel type.

This is deliberately a *loader*, not a scraper or a data generator: VAHAN's dashboard
(vahan.parivahan.gov.in) is a session-bound JSF application with no clean public API, and —
more fundamentally — Delhi's 13 AirSentinel hotspot zones are not the same thing as VAHAN's
RTO (Regional Transport Office) boundaries, and no official zone<->RTO mapping is published.
Approximating that mapping is a real judgment call, and it needs to be made visibly by a
person filling in the template below, with the mapping documented in `source_citation` —
not silently guessed at inside library code.

Getting real data (see README.md for the full walkthrough):
    1. https://vahan.parivahan.gov.in/vahan4dashboard/ -> Tabular Summary / Comparison View
    2. Filter State = Delhi(16), pick the RTO(s) nearest each hotspot zone
    3. Export vehicle counts by category/fuel type for the RTOs you picked
    4. Fill data/raw/vehicle_registrations.csv using the template in data/templates/
       (one row per zone x vehicle_category x fuel_type), noting your RTO->zone mapping
       and the export date in `source_citation`
"""

from __future__ import annotations

import pandas as pd

from . import config, rto_mapping


def write_template() -> None:
    """Write empty, header-only CSV templates — no example numeric rows, so there is no
    risk of a placeholder being mistaken for real data."""
    pd.DataFrame(columns=config.REGISTRATIONS_COLUMNS).to_csv(
        config.TEMPLATES_DIR / "vehicle_registrations_template.csv", index=False
    )
    pd.DataFrame(columns=config.REGISTRATIONS_BY_RTO_COLUMNS).to_csv(
        config.TEMPLATES_DIR / "vehicle_registrations_by_rto_template.csv", index=False
    )


def load_by_rto() -> pd.DataFrame:
    """Load real RTO-level registration counts (VAHAN's actual export granularity — see
    rto_mapping.py) and aggregate them up to AirSentinel zones via the documented,
    inspectable nearest-RTO mapping.

    When an RTO is the nearest match for more than one zone (e.g. DL8 -> Wazirpur AND Ashok
    Vihar), its count is split evenly across those zones rather than attributed in full to
    each — attributing in full would double-count the same real vehicles across nearby
    zones, which would misrepresent the underlying VAHAN data even though every input number
    is real.
    """
    if not config.REGISTRATIONS_BY_RTO_CSV.exists():
        raise FileNotFoundError(
            f"No RTO-level registration data found at {config.REGISTRATIONS_BY_RTO_CSV}.\n"
            f"This is the recommended path (matches VAHAN's actual export granularity) — "
            f"see README.md 'Getting real data'. Fill in the template at "
            f"{config.TEMPLATES_DIR / 'vehicle_registrations_by_rto_template.csv'} "
            f"and save it to {config.REGISTRATIONS_BY_RTO_CSV}. If you already have "
            f"zone-level counts instead, use load() with vehicle_registrations.csv."
        )
    df = pd.read_csv(config.REGISTRATIONS_BY_RTO_CSV)
    missing_cols = set(config.REGISTRATIONS_BY_RTO_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"vehicle_registrations_by_rto.csv is missing required columns: {missing_cols}")
    if df["source_citation"].isna().any() or (df["source_citation"].astype(str).str.strip() == "").any():
        raise ValueError("Every row in vehicle_registrations_by_rto.csv must have a non-empty source_citation.")
    if (df["vehicle_count"] < 0).any():
        raise ValueError("vehicle_registrations_by_rto.csv has negative vehicle_count values.")

    unknown_rtos = set(df["rto_code"]) - rto_mapping.known_rto_codes()
    if unknown_rtos:
        raise ValueError(
            f"vehicle_registrations_by_rto.csv has rto_code values not in rto_mapping.py's "
            f"known Delhi RTOs: {sorted(unknown_rtos)}. Check for typos, or add the RTO to "
            f"rto_mapping.py if VAHAN has added a new office since 2026-07-21."
        )

    rows = []
    for _, r in df.iterrows():
        zones = rto_mapping.zone_for_rto_code(r["rto_code"])
        if not zones:
            continue  # a real RTO with no nearest-zone match (see rto_mapping.UNUSED_RTOS)
        split_count = r["vehicle_count"] / len(zones)
        for zone in zones:
            rows.append({
                "zone": zone,
                "vehicle_category": r["vehicle_category"],
                "fuel_type": r["fuel_type"],
                "vehicle_count": split_count,
                "data_period": r["data_period"],
                "source_citation": (
                    f"{r['source_citation']} [via RTO {r['rto_code']}"
                    + (f", split across {len(zones)} zones" if len(zones) > 1 else "") + "]"
                ),
            })
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    """Load and validate real registration counts. Raises with clear next steps if the
    real-data file doesn't exist yet or is incomplete — never fabricates a fallback."""
    if not config.REGISTRATIONS_CSV.exists():
        raise FileNotFoundError(
            f"No registration data found at {config.REGISTRATIONS_CSV}.\n"
            f"This module does not fabricate vehicle counts — see README.md 'Getting real "
            f"data' for how to source them from VAHAN, then fill in the template at "
            f"{config.TEMPLATES_DIR / 'vehicle_registrations_template.csv'} "
            f"and save it to {config.REGISTRATIONS_CSV}."
        )
    df = pd.read_csv(config.REGISTRATIONS_CSV)
    missing_cols = set(config.REGISTRATIONS_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"vehicle_registrations.csv is missing required columns: {missing_cols}")
    if df["source_citation"].isna().any() or (df["source_citation"].astype(str).str.strip() == "").any():
        raise ValueError(
            "Every row in vehicle_registrations.csv must have a non-empty source_citation "
            "(where/when this count came from, and your RTO->zone mapping if approximated) "
            "— provenance is required, not optional, for real-data inputs to this module."
        )
    if (df["vehicle_count"] < 0).any():
        raise ValueError("vehicle_registrations.csv has negative vehicle_count values.")
    return df
