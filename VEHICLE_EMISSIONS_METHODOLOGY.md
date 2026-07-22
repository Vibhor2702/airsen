# Vehicle Emissions Methodology

This document formally describes the logic in `category_mapper.py` — how real VAHAN
registration data is converted into per-zone emission load indices for the AirSentinel
Enforcement Zone Ranker.  Written to be readable without reading the code.

---

## 1. Source files

| File | Content | Scope |
|---|---|---|
| `vehicle_registrations_by_rto_category.csv` | Per-RTO vehicle counts by VAHAN class name (`fuel_type = ALL`) | Project root |
| `vehicle_registrations_by_rto.csv` (source) | Per-RTO vehicle counts by fuel type (`vehicle_category = ALL`) | Project root |
| `P2/.../vehicle_emissions/data/raw/vehicle_registrations_by_rto.csv` (output) | Mapped, split rows consumed by the pipeline | Module input |

---

## 2. Category mapping decisions

Each VAHAN class name is mapped to one of the two emission formula categories (`car`,
`two_wheeler`) with an explicit fuel assignment.  Every decision is logged below.

| VAHAN class | → Emission category | Fuel | Reasoning |
|---|---|---|---|
| Motor Car | `car` | petrol + diesel *(split — see §3)* | Direct 1:1: VAHAN "Motor Car" = CPCB/ARAI passenger car emission category |
| M-Cycle/Scooter | `two_wheeler` | petrol | Standard 2-wheeler class = CPCB L2/L3 two-wheeler emission category |
| M-Cycle/Scooter-With Side Car | `two_wheeler` | petrol | Sidecar is a cargo attachment; powertrain is the same single-engine two-wheeler |
| Moped | `two_wheeler` | petrol | ARAI/CPCB L1/L2 category; same BS6 two-wheeler emission norm as M-Cycle |
| Motorised Cycle (CC > 25cc) | `two_wheeler` | petrol | CMVR L-category; BS6 two-wheeler limits apply |

**Why all two-wheelers are assigned petrol:** Diesel two-wheelers have never been
commercially sold in India.  BS6 CMVR Schedule VI covers only L1/L2/L3 petrol engines
in this vehicle class.  This is a fleet fact, not an assumption.

---

## 3. Fuel-split derivation for Motor Car

VAHAN's category file reports Motor Car counts with `fuel_type = ALL` — no per-car fuel
breakdown is given.  The split is derived from the fuel-type file instead.

**Formula per RTO:**

```
petrol_fraction(rto) = petrol_count(rto) / (petrol_count(rto) + diesel_count(rto))
diesel_fraction(rto) = 1 - petrol_fraction(rto)
```

Where:

- **petrol_count** = sum of vehicles with fuel types in:
  `PETROL`, `PETROL/CNG`, `PETROL(E20)`, `PETROL(E20)/CNG`, `PETROL/HYBRID`,
  `PETROL(E20)/HYBRID`, `PETROL/HYBRID/CNG`, `PETROL(E20)/HYBRID/CNG`,
  `FLEX-FUEL(ETHANOL)`
  *(E20 and flex-fuel vehicles run on petrol; same BS6 petrol norm applies regardless
  of ethanol blend)*

- **diesel_count** = sum of vehicles with fuel types in:
  `DIESEL`, `DIESEL/HYBRID`

- **Excluded from denominator** (no tailpipe NOx/PM emission factor applies):
  `CNG ONLY`, `PURE EV`, `STRONG HYBRID EV`, `ELECTRIC(BOV)`, `PLUG-IN HYBRID EV`

**Known approximation:** The split is computed across all vehicle types at the RTO
(category = ALL), not car-specific.  Since Delhi two-wheelers are overwhelmingly petrol,
this biases the petrol fraction slightly upward relative to a cars-only split.  The
approximation is documented in every affected `source_citation` row in the output CSV.

**Per-RTO petrol fractions (from VAHAN data, computed 2026-07-21):**

| RTO | Petrol fraction | Diesel fraction |
|---|---|---|
| DL3 (Lado Sarai / RK Puram) | 0.989 | 0.011 |
| DL4 (Janakpuri / Dwarka) | 0.984 | 0.016 |
| DL6 (Shahdara / Vivek Vihar) | 0.984 | 0.016 |
| DL7 (Mayur Vihar / Anand Vihar / Vivek Vihar) | 0.975 | 0.025 |
| DL8 (Rohini) | 0.976 | 0.024 |
| DL9 (South West / Dwarka) | 0.986 | 0.014 |
| DL10 (Mundka / Wazirpur / Ashok Vihar) | 0.965 | 0.035 |
| DL11 (Narela / Bawana / Jahangirpuri) | 0.993 | 0.007 |
| DL12 (Okhla / Punjabi Bagh) | 0.971 | 0.029 |
| DL13 (Rohini / Narela) | 0.994 | 0.006 |

---

## 4. Excluded categories — full reasoning

These VAHAN categories have real vehicle counts but are deliberately not mapped to any
emission formula category.  They do not appear in the pipeline's `coverage_note` because
they are pre-filtered in `category_mapper.py` before any data reaches the pipeline.
This is intentional: the pipeline's `coverage_note` reports unmatched rows it received,
not categories pre-filtered upstream.

| VAHAN category | Count (all RTOs) | Why excluded |
|---|---|---|
| e-Rickshaw(P) | ~1,155 | Battery electric (CMVR L5e/L6e with electric powertrain per Delhi EV Policy 2020). Zero tailpipe NOx/PM emissions by construction. No ARAI/CPCB exhaust emission factor exists — not a sourcing gap, a physical fact. |
| e-Rickshaw with Cart (G) | ~705 | Same — battery electric, goods variant. Zero tailpipe emissions. |
| Adapted Vehicle | ~110 | Undefined body type; could be any powertrain. No defensible single emission factor can be assigned without knowing the individual vehicles. |
| Fork Lift | 1 | Off-road industrial equipment. Not subject to on-road CMVR emission norms. BS6 factors do not apply. |
| Vintage Motor Vehicle | 1 | Pre-BS norm vehicle. BS6 emission factors don't apply; no reliable pre-BS factor can be cited without the specific vehicle's original certification. |

**Total excluded count: ~1,972 vehicles** across all RTOs (~1.2% of total registered
vehicles in the VAHAN extract).

---

## 5. Output schema

The output CSV (`vehicle_registrations_by_rto.csv`) has one row per (RTO, category,
fuel_type) combination.  For Motor Car, each RTO produces two rows (petrol and diesel
with split counts).  For two-wheelers, each RTO produces one row (petrol only).

Every row's `source_citation` column embeds:
- The original VAHAN source citation
- The mapping decision (which VAHAN class was mapped to which emission category)
- For car rows: the fuel split fraction and the cross-category approximation caveat

---

## 6. What is not in this pipeline

- **CNG-only vehicles** (auto-rickshaws, buses): excluded because the emission formula
  uses BS6 petrol/diesel factors only.  CNG emission factors would require different
  constants (lower NOx/PM per km than diesel, but different from petrol).
- **Electric cars**: excluded; zero tailpipe emissions.
- **Heavy commercial vehicles (trucks, buses)**: not present in the VAHAN extract used.
  Delhi's major emission sources include HDV — this is a known gap in the current model.
- **Land use signal**: absent from this repo entirely.  The ranker notes this explicitly
  in every output row's `land_use_note` column.
