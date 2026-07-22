"""
VAHAN RTO -> AirSentinel zone mapping.

Delhi's 13 AirSentinel hotspot zones are not the same thing as VAHAN's RTO jurisdictions —
no official mapping between them exists. This module provides a **transparent, documented
geographic approximation** (each zone assigned to whichever real Delhi RTO is nearest to
it), built from real data, not fabricated: the RTO names/codes below were pulled live from
vahan.parivahan.gov.in on 2026-07-21 (State = Delhi(16); the site returned 13 real
vehicle-registering offices, DL1-DL13, plus several fitness-testing centers and specialised
units — DL51/52/53, DL201-207 — which are excluded here since they don't register vehicles
and so have no meaningful "nearest zone").

This is a REASONED APPROXIMATION, not a surveyed fact — the `reasoning` column below is
general knowledge of Delhi geography (which RTO office a zone is administratively/physically
closest to), not a verified distance calculation. Two zones (Wazirpur, Dwarka) happen to
share a name with their real RTO, so those two are exact. The rest are nearest-neighbour
judgment calls, visible here for anyone to correct. If your team has better local knowledge
of the actual RTO boundaries, edit this table directly — it's the single place this
assumption lives.

Registration counts pulled per RTO (real VAHAN exports) are aggregated up to zones through
this table by `registrations.load_by_rto()` — the *counts* are always real; only the
*zone assignment* is approximated, and that approximation is fully visible here rather than
hidden inside a scraper or baked into a hardcoded per-zone number.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rto:
    code: str            # VAHAN RTO code, e.g. "DL8"
    vahan_name: str       # exact name as shown on vahan.parivahan.gov.in
    zone: str             # nearest AirSentinel hotspot zone (see module docstring)
    reasoning: str        # why this RTO was assigned to this zone


# Source: vahan.parivahan.gov.in/vahan4dashboard, State=Delhi(16), accessed 2026-07-21.
# Excludes non-registering offices: AUTO UNIT HQ (DL53), TAXI UNIT HQ (DL52),
# RAJPUR ROAD/VIU BURARI (DL51), and all *FITNESS CENTER offices (DL201-207).
RTOS: list[Rto] = [
    Rto("DL8", "CENTRAL NORTH (WAZIRPUR)", "Wazirpur",
        "Exact match — same locality"),
    Rto("DL9", "SOUTH WEST (DWARKA)", "Dwarka",
        "Exact match — same locality"),
    Rto("DL11", "OUTER NORTH (ROHINI-I)", "Rohini",
        "Exact match — Rohini-named RTO in the zone"),
    Rto("DL13", "NORTH WEST (ROHINI-II)", "Rohini",
        "Exact match — second Rohini-named RTO in the zone (both map to Rohini)"),
    Rto("DL13", "NORTH WEST (ROHINI-II)", "Jahangirpuri",
        "Same North West Delhi district as Jahangirpuri"),
    Rto("DL8", "CENTRAL NORTH (WAZIRPUR)", "Ashok Vihar",
        "Ashok Vihar borders Wazirpur, same North Delhi belt"),
    Rto("DL7", "EAST (MAYUR VIHAR)", "Anand Vihar",
        "Mayur Vihar is the nearest registering RTO to Anand Vihar, both East Delhi"),
    Rto("DL7", "EAST (MAYUR VIHAR)", "Vivek Vihar",
        "Adjacent to Anand Vihar, same East Delhi belt"),
    Rto("DL11", "OUTER NORTH (ROHINI-I)", "Bawana",
        "North West Delhi, nearest listed Outer-North RTO"),
    Rto("DL12", "NORTH (BURARI)", "Narela",
        "Northernmost registering RTO, closest to far-north Narela"),
    Rto("DL4", "WEST (HARI NAGAR)", "Mundka",
        "West Delhi district, nearest listed West RTO (no dedicated Outer-West office)"),
    Rto("DL10", "CENTRAL (RAJA GARDEN)", "Punjabi Bagh",
        "Raja Garden is directly adjacent to Punjabi Bagh"),
    Rto("DL3", "SOUTH (LADO SARAI)", "RK Puram",
        "South Delhi district"),
    Rto("DL6", "SOUTH EAST (SARAI KALE KHAN)", "Okhla",
        "South East Delhi district"),
]

# RTOs that exist in Delhi's real registering-office list but aren't the nearest match for
# any of the 13 zones (kept here for completeness/audit — not used in aggregation).
UNUSED_RTOS = ["DL1 (OLD DELHI - MALL ROAD)", "DL2 (NEW DELHI - JAM NAGAR HOUSE)",
               "DL5 (NORTH EAST - LONI)"]


def zone_for_rto_code(rto_code: str) -> list[str]:
    """An RTO can be the nearest match for more than one zone (e.g. DL8 -> Wazirpur AND
    Ashok Vihar) — returns every zone this RTO's registrations should be attributed to.
    Callers that want a 1:1 split should divide counts accordingly; see registrations.py.
    """
    return [r.zone for r in RTOS if r.code == rto_code]


def known_rto_codes() -> set[str]:
    return {r.code for r in RTOS}
