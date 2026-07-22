# Vehicle Emission Load Model — "tailpipe pollution" (AirSentinel)

Implements the **Vehicle Emission Load Model** from the AirSentinel design plan (slides 3–4,
6, 9): a modelled estimate of tailpipe pollution per zone, calculated the same way national
emission inventories are built — not sensed, not invented.

```
vehicle count (by type & fuel, per RTO)
  x emission factor (g/km, ARAI/CPCB tables)
  x avg. distance (published transport surveys)
  = Vehicle Emission Load Index per zone
```

## Demo data (for tonight's demo — read this first)

`python -m vehicle_emissions.build_demo_data` generates a **real-world-informed, not
random** dataset so the pipeline runs end-to-end tonight, ahead of the real VAHAN/ARAI
sourcing below. Every number is either a **real cited figure** or an **explicitly labeled
assumption** — see `src/vehicle_emissions/build_demo_data.py`'s docstring for every citation:

- Real: Delhi's actual two-wheeler (10,370,047) and car (3,311,579) registration totals,
  and official BS6 regulatory NOx/PM emission limits (India's mandated standard since
  1 Apr 2020: petrol 60 mg/km NOx, diesel 80 mg/km NOx, 4.5 mg/km PM — all categories), and
  a published Indian-cities urban-mobility distance figure (27-33 km/day for two-wheelers).
- Labeled assumption, not cited: the Delhi-wide totals are split **evenly** across the 13
  zones (we don't have real per-zone counts yet), and cars are assumed 70%/30% petrol/diesel
  (a rough commonly-cited order, not a verified Delhi figure).

The dashboard and every output CSV carry this forward visibly — a `data_provenance` column,
an amber "DEMO DATA" chip (not green "live"), and a banner explaining exactly what's real vs.
assumed. This is deliberate: **the demo should never look more precise than it is.**

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m vehicle_emissions.build_demo_data
.\.venv\Scripts\python.exe -m vehicle_emissions.pipeline
```

Replace with real per-RTO VAHAN data (below) before using this for anything beyond a demo —
the even-zone-split means every zone currently shows the identical index (1.00), which
honestly reflects that we don't have real zone-to-zone variation yet, not a bug.

## Why this module needs real data files, not defaults

Two of the three inputs above (emission factors, average distances) are specific numeric
facts from official technical reports. This session located real, official sources for them
(links below) but could not extract exact verified figures with the tools available — so
rather than guess plausible-looking numbers and label them "ARAI/CPCB data," **this module
requires the real tables as input files**. Every number it ever outputs is traceable to a
document you (or your teammate) can point to. See `src/vehicle_emissions/config.py` for the
full reasoning.

The same applies to vehicle registration counts: Delhi's 13 AirSentinel hotspot zones are
not the same thing as VAHAN's RTO boundaries, and no official zone↔RTO mapping is published.

## The RTO → zone mapping

Rather than leave this mapping as a manual judgment call for whoever fills in the template,
`src/vehicle_emissions/rto_mapping.py` ships a **documented, inspectable nearest-RTO
approximation** — built from the real, current list of Delhi's 13 vehicle-registering RTOs,
pulled live from vahan.parivahan.gov.in this session (State = `Delhi(16)`). Two zones happen
to share a name with their RTO (Wazirpur, Dwarka — exact matches); the rest are geographic
nearest-neighbour judgment calls, each with its reasoning written inline in that file. When
one RTO is the nearest match for more than one zone (e.g. `DL8 CENTRAL NORTH (WAZIRPUR)` is
nearest to both Wazirpur and Ashok Vihar), its registration count is **split evenly** across
those zones rather than counted in full for each — counting in full would double the same
real vehicles across nearby zones.

This is still an approximation of real RTO office locations to real zone locations — not
a survey, not verified against actual road distances — and it's meant to be corrected: if
your team has better local knowledge of the real catchment areas, edit `rto_mapping.py`
directly, it's the single place this assumption lives.

**Until you supply real data, running the pipeline fails loudly with instructions — it never
produces a fabricated chart or number.** This is intentional.

## Getting real data

1. **Vehicle registrations** (recommended: RTO-level) —
   [vahan.parivahan.gov.in/vahan4dashboard](https://vahan.parivahan.gov.in/vahan4dashboard/)
   → *Tabular Summary* or *Comparison View* → filter State = `Delhi(16)` → export vehicle
   counts by category & fuel type **per RTO** (DL1–DL13; ignore the DL5x/DL20x
   fitness-centre and specialised-unit codes, they don't register vehicles) — this matches
   VAHAN's actual export granularity, so no manual zone-mapping is needed on your end; the
   pipeline aggregates it via `rto_mapping.py` automatically. (If you'd rather do the zone
   aggregation yourself, `vehicle_registrations.csv`/`registrations.load()` still works as a
   fallback.)
2. **Emission factors** — pull g/km figures from:
   - [CPCB emission-factor PDF](https://cpcb.nic.in/displaypdf.php?id=RW1pc3Npb25fRmFjdG9yc19WZWhpY2xlcy5wZGY=)
   - [CPCB vehicular exhaust page](https://cpcb.nic.in/vehicular-exhaust/)
   - ARAI "Emission Factors for Indian In-Use Vehicles" report
   - Lok Sabha reply, 9 Aug 2021, "Vehicle-wise Emission Factors..." (data.gov.in)
3. **Average distances** — pull from a published transport survey, e.g. Delhi's
   Comprehensive Mobility Plan or the Economic Survey of Delhi's transport chapter.

Then fill in the templates in `data/templates/` (regenerate them any time with
`python -m vehicle_emissions.pipeline --write-templates`) and save the filled versions to
`data/raw/vehicle_registrations_by_rto.csv` (recommended) or `vehicle_registrations.csv`,
plus `data/raw/emission_factors.csv` and `data/raw/distance_estimates.csv`.

## Setup

```powershell
cd "C:\Users\a955032\OneDrive - ATOS\Desktop\airsentinel\vehicle_emissions"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# installs the forecasting module's zone registry in editable mode — see "Compatibility" below
.\.venv\Scripts\python.exe -m pip install -e ..\forecasting
```

## Run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m vehicle_emissions.pipeline --write-templates   # first time
# ... fill in the three CSVs in data/raw/ with real, sourced data ...
.\.venv\Scripts\python.exe -m vehicle_emissions.pipeline
```

Output: `outputs/vehicle_emission_index.csv` —
`zone, vehicle_emission_load_raw_g_per_day, vehicle_emission_load_index, n_vehicle_categories_included, data_asof, coverage_note, data_provenance`.
`data_provenance` says `"demo — real-world-informed, not verified precision"` or `"real, cited"`
depending on whether any input row came from `build_demo_data.py` — computed automatically,
not something you have to remember to set.
`vehicle_emission_load_index` is relative (0–1, current zone's load ÷ the highest zone's
load), matching the design plan's dashboard mockup (slide 7: "Vehicle Emission Load Index:
0.81").

## Compatibility with the forecasting track

The `../forecasting` module (Python package still importable as `airsentinel`, per the
overall project brand) is installed in editable mode as this module's only project-specific
dependency — the 13-zone list is imported directly from `airsentinel.stations.ZONES`
(`pipeline.py::_known_zones`) rather than retyped, so the two modules cannot drift out of
sync on zone spelling. Output is keyed on `zone` alone (not `zone, date`, since registration
data doesn't change daily) so it joins trivially onto the forecasting module's
`zone, date, ...` output — see `../shared/README.md` for how the two are combined.

## What this module is not claiming

Carried verbatim from the design plan (slide 9): **the Vehicle Emission Load Index is a
modelled estimate from registration data, not a live sensor reading** — a legitimate,
standard methodology (the same one national emission inventories use), but not the same
claim as "we measured a tailpipe." No overfitting risk: this is a deterministic formula
(sum + relative normalization) over real inputs, with no fitted parameters at all.
