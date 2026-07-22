# category_mapper.py — VAHAN category name -> vehicle_emissions pipeline input
#
# Reads the two real VAHAN CSV files from the project root:
#   vehicle_registrations_by_rto_category.csv  (per-RTO counts by vehicle class, fuel="ALL")
#   vehicle_registrations_by_rto.csv            (per-RTO counts by fuel type, category="ALL")
#
# Maps VAHAN vehicle classes -> the emission formula's category names (car, two_wheeler),
# derives per-RTO petrol/diesel splits for Motor Car from the fuel-type file, and writes
# a properly formatted vehicle_registrations_by_rto.csv into the vehicle_emissions module's
# data/raw/ directory so the pipeline can consume real data.
#
# Every mapping decision is documented inline. Categories that cannot be defensibly mapped
# (e-rickshaws, adapted vehicles, off-road equipment) are left out; the pipeline's
# coverage_note column will report excluded counts explicitly.
#
# Usage: python category_mapper.py
#
# Output: P2/airsentinel-master/vehicle_emissions/data/raw/vehicle_registrations_by_rto.csv

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
CATEGORY_CSV = ROOT / "vehicle_registrations_by_rto_category.csv"
FUEL_CSV     = ROOT / "vehicle_registrations_by_rto.csv"
OUT_PATH     = ROOT / "P2/airsentinel-master/vehicle_emissions/data/raw/vehicle_registrations_by_rto.csv"

# ---------------------------------------------------------------------------
# Category mapping — VAHAN class name -> emission formula category
# ---------------------------------------------------------------------------
# Unmapped categories are left out; the pipeline reports them in coverage_note.
# Fuel type for two_wheelers is hardcoded to petrol: diesel 2-wheelers have never
# been commercially sold in India; BS6 CMVR Schedule VI L1/L2/L3 norms cover petrol only.
#
# Each entry: VAHAN_name -> (emission_category, fuel_type | None)
#   fuel_type = None means "derive from per-RTO fuel file" (used for car only)
#   fuel_type = "petrol" means assign directly without fuel-file lookup

CATEGORY_MAP: dict[str, tuple[str, str | None]] = {
    # Direct mappings (defensible 1:1 correspondence)
    "Motor Car":                    ("car",         None),      # fuel split derived from fuel file
    "M-Cycle/Scooter":              ("two_wheeler", "petrol"),  # standard 2-wheeler class
    "M-Cycle/Scooter-With Side Car":("two_wheeler", "petrol"),  # sidecar = cargo add-on; engine is same 2W
    "Moped":                        ("two_wheeler", "petrol"),  # ARAI/CPCB L1 category; same norms as M-Cycle
    "Motorised Cycle (CC > 25cc)":  ("two_wheeler", "petrol"),  # CMVR L-category; BS6 2W limits apply

    # Intentionally excluded — reason logged here and in coverage_note:
    # "e-Rickshaw(P)"              : battery electric (CMVR L5e/L6e); zero tailpipe NOx/PM; no exhaust EF
    # "e-Rickshaw with Cart (G)"   : battery electric, goods variant; same reason
    # "Adapted Vehicle"            : undefined body type; no defensible single EF
    # "Fork Lift"                  : off-road industrial; not subject to on-road emission norms
    # "Vintage Motor Vehicle"      : pre-BS norm vehicle; BS6 EF doesn't apply; count = 1
}

# Fuel types classified as petrol-type for the per-RTO split ratio.
# These share the BS6 petrol emission norm regardless of ethanol blend or CNG bi-fuel option:
#   PETROL(E20) vehicles use the same petrol BS6 norm; PETROL/CNG bi-fuel vehicles are petrol-primary.
PETROL_FUEL_TYPES = {
    "PETROL", "PETROL/CNG", "PETROL(E20)", "PETROL(E20)/CNG",
    "PETROL/HYBRID", "PETROL(E20)/HYBRID", "PETROL/HYBRID/CNG",
    "PETROL(E20)/HYBRID/CNG", "FLEX-FUEL(ETHANOL)",
}
DIESEL_FUEL_TYPES = {"DIESEL", "DIESEL/HYBRID"}
# Excluded from split denominator (no tailpipe NOx/PM EF applies to any of these):
#   CNG ONLY, PURE EV, STRONG HYBRID EV, ELECTRIC(BOV), PLUG-IN HYBRID EV


def compute_fuel_fractions(fuel_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Per-RTO petrol_fraction and diesel_fraction, computed from the fuel-type file
    (vehicle_category = ALL).

    Denominator = petrol-type + diesel-type only (excludes EVs, CNG-only, strong hybrids —
    none of which have an applicable tailpipe NOx/PM emission factor).

    This is a cross-category approximation: the fraction is computed across all vehicle
    types at the RTO, not Motor Car-specific. Since Delhi 2-wheelers are almost entirely
    petrol, this biases petrol fraction slightly upward relative to a car-only split.
    That approximation is documented in the output's source_citation column.

    Returns dict[rto_code -> (petrol_frac, diesel_frac)].
    If an RTO has no petrol or diesel vehicles at all (unlikely), both fractions default to 0.5.
    """
    fracs: dict[str, tuple[float, float]] = {}
    for rto_code, grp in fuel_df.groupby("rto_code"):
        petrol_total = grp.loc[grp["fuel_type"].isin(PETROL_FUEL_TYPES), "vehicle_count"].sum()
        diesel_total = grp.loc[grp["fuel_type"].isin(DIESEL_FUEL_TYPES), "vehicle_count"].sum()
        denom = petrol_total + diesel_total
        if denom == 0:
            fracs[rto_code] = (0.5, 0.5)
        else:
            fracs[rto_code] = (petrol_total / denom, diesel_total / denom)
    return fracs


def main() -> None:
    if not CATEGORY_CSV.exists():
        raise FileNotFoundError(f"Category CSV not found: {CATEGORY_CSV}")
    if not FUEL_CSV.exists():
        raise FileNotFoundError(f"Fuel CSV not found: {FUEL_CSV}")

    cat_df  = pd.read_csv(CATEGORY_CSV)
    fuel_df = pd.read_csv(FUEL_CSV)

    print(f"Category file: {len(cat_df)} rows, RTOs: {sorted(cat_df['rto_code'].unique())}")
    print(f"Fuel file    : {len(fuel_df)} rows, RTOs: {sorted(fuel_df['rto_code'].unique())}")

    fuel_fracs = compute_fuel_fractions(fuel_df)
    print(f"\nPer-RTO petrol/diesel split (from fuel file, cross-category approximation):")
    for rto, (pf, df_) in sorted(fuel_fracs.items()):
        print(f"  {rto}: petrol={pf:.3f}  diesel={df_:.3f}")

    # -- Map categories, build output rows --------------------------------------
    out_rows: list[dict] = []
    unmapped_summary: dict[str, int] = {}

    for _, row in cat_df.iterrows():
        vahan_cat = row["vehicle_category"]
        rto       = row["rto_code"]
        count     = row["vehicle_count"]
        period    = row["data_period"]
        orig_cite = row["source_citation"]

        if vahan_cat not in CATEGORY_MAP:
            unmapped_summary[vahan_cat] = unmapped_summary.get(vahan_cat, 0) + int(count)
            continue

        emi_cat, fuel_type = CATEGORY_MAP[vahan_cat]

        if fuel_type is not None:
            # Two-wheelers: direct petrol assignment
            out_rows.append({
                "rto_code":         rto,
                "vehicle_category": emi_cat,
                "fuel_type":        fuel_type,
                "vehicle_count":    count,
                "data_period":      period,
                "source_citation": (
                    f"{orig_cite} [mapped from VAHAN '{vahan_cat}' -> {emi_cat}/{fuel_type}; "
                    f"petrol assigned: diesel 2-wheelers not commercially sold in India, "
                    f"BS6 CMVR Schedule VI covers L1/L2/L3 petrol engines only]"
                ),
            })
        else:
            # Motor Car: split by per-RTO petrol/diesel fraction
            pf, df_ = fuel_fracs.get(rto, (0.5, 0.5))
            approx_note = (
                f"fuel split derived from per-RTO all-vehicle petrol/diesel ratio "
                f"({pf:.3f} petrol / {df_:.3f} diesel) — cross-category approximation, "
                f"2-wheeler-dominated RTO biases petrol fraction upward"
            )
            petrol_count = count * pf
            diesel_count = count * df_

            out_rows.append({
                "rto_code":         rto,
                "vehicle_category": emi_cat,
                "fuel_type":        "petrol",
                "vehicle_count":    petrol_count,
                "data_period":      period,
                "source_citation": (
                    f"{orig_cite} [mapped from VAHAN '{vahan_cat}' -> {emi_cat}/petrol; {approx_note}]"
                ),
            })
            out_rows.append({
                "rto_code":         rto,
                "vehicle_category": emi_cat,
                "fuel_type":        "diesel",
                "vehicle_count":    diesel_count,
                "data_period":      period,
                "source_citation": (
                    f"{orig_cite} [mapped from VAHAN '{vahan_cat}' -> {emi_cat}/diesel; {approx_note}]"
                ),
            })

    out_df = pd.DataFrame(out_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    # -- Summary ----------------------------------------------------------------
    print(f"\n=== Category mapping summary ===")
    print(f"Output rows written: {len(out_df)}  ->  {OUT_PATH}")
    print(f"\nMapped categories:")
    for vahan_cat, (emi_cat, ft) in CATEGORY_MAP.items():
        sub = cat_df[cat_df["vehicle_category"] == vahan_cat]["vehicle_count"].sum()
        fuel_label = ft if ft else "petrol+diesel (split)"
        print(f"  '{vahan_cat}' -> {emi_cat}/{fuel_label}  (total count across all RTOs: {int(sub)})")

    print(f"\nUnmapped categories (excluded — see plan comment):")
    for cat, total in sorted(unmapped_summary.items(), key=lambda x: -x[1]):
        print(f"  '{cat}': {total} total across all RTOs — will appear in pipeline coverage_note")

    print(f"\nPer emission-category totals in output:")
    for (ec, ft), grp in out_df.groupby(["vehicle_category", "fuel_type"]):
        print(f"  {ec}/{ft}: {grp['vehicle_count'].sum():.0f}")


if __name__ == "__main__":
    main()
