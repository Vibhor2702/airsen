# AirSentinel — Session Brief: Prithvi Fine-Tuning + Enforcement Ranker + Dashboard Integration

This file is for Copilot, attached alongside: (1) two VAHAN CSVs, (2) my teammate's repo as an extracted folder. Read this fully before writing any code — several things below correct assumptions from earlier in this project, and skipping this will cause rework.

---

## 1. What's new in the Google Drive folder since you last saw it

The satellite data pull was substantially reworked. If any earlier code (yours or mine) assumed the old flat structure, it's now wrong:

- **Per-zone subfolders.** Drive folder `AirSentinel_Satellite_Images/` now contains 13 subfolders, one per zone (`Anand_Vihar/`, `Mundka/`, ... `Dwarka/`) — files are **not** flat in the root folder anymore.
- **6-band Sentinel-2 images, not 3.** Every `_S2.tif` file now has 6 bands in this exact order: `B2, B3, B4, B8A, B11, B12` (Blue, Green, Red, Narrow NIR, SWIR1, SWIR2) — matching Prithvi-EO-2.0-100M-TL's expected input exactly, confirmed against that model's own documentation. Old 3-band files (RGB only) may still exist somewhere and should be treated as stale/unusable for fine-tuning if found.
- **19 dates per zone now, not 10.** Two batches: 10 original dates (5 days apart, June–July 2026), plus 9 historical dates added afterward — weekly through Oct 1 to Nov 26, **2025** (Delhi's actual last real crop-burning season). The historical batch exists specifically because `crop_burning_smoke` had zero real examples in the 2026-only data — that's a season problem, not a volume problem, so more 2026 dates would not have fixed it.
- **Pull is complete: 247 photo (`_S2`) tasks and 234 NO2 (`_NO2`) tasks started** (13 zones × 19 dates, minus the small number of NO2 gaps already expected from Sentinel-5P's coverage limits). Confirm actual landed file counts before Phase 1 rather than trusting this number blindly — exports finish on a delay, and the count above is tasks *started*, not files confirmed present in Drive.
- **Verified coordinates, not guessed.** Zone coordinates now match the real, official CPCB CAAQMS/DPCC monitoring station locations (source: CPCB's national station list). The previous guessed coordinates were off by up to 6.5km for some zones (Rohini was worst) — if any earlier-pulled images or analysis used the old coordinates, they're centered on the wrong spot and should not be trusted.
- **Filenames encode zone + date + layer:** `{Zone_Name}_{YYYY-MM-DD}_S2.tif` and `{Zone_Name}_{YYYY-MM-DD}_NO2.tif`, inside `{Zone_Name}/`.

**Sort the Drive folder yourself before Phase 1 — don't just work around the mess, actually fix it.** Google Drive sync lag means a recurring, small number of files still land outside their zone subfolder even after the Colab notebook's own sort cell runs. Rather than repeatedly re-running that flaky cell by hand: write and run your own one-time local sort script, against the local Drive-for-Desktop-synced folder path (not Colab), that does a full recursive scan of the entire `AirSentinel_Satellite_Images/` tree for every `.tif` file, works out its correct zone from the filename, and moves any misplaced file into the right subfolder. This should be part of your Phase 1 setup, run once, with the results (how many files found, how many were already correct, how many moved) logged in `SESSION_LOG.md` — don't just silently fix it and move on.

---

## 2. The two attached VAHAN CSVs — what they are and are NOT

Both are **real** government vehicle-registration data, pulled directly from `vahan.parivahan.gov.in`'s dashboard, for the 10 real Delhi RTOs that matter to our 13 zones (per my teammate's own `rto_mapping.py` — see below).

- **`vehicle_registrations_by_rto.csv`** — RTO × **Fuel type** (Petrol/Diesel/CNG/Electric/Other split, real 2026 counts). `vehicle_category` is `"ALL"` in every row — this export has no vehicle-class breakdown.
- **`vehicle_registrations_by_rto_category.csv`** — RTO × **Vehicle class** (real VAHAN categories: Motor Car, M-Cycle/Scooter, e-Rickshaw, etc.). `fuel_type` is `"ALL"` in every row — this export has no fuel breakdown.

**Neither file has both dimensions crossed** (the dashboard tool only exports two dimensions at a time) — this was a deliberate, documented tradeoff, not an oversight. Don't try to silently merge these into a fake combined table with invented numbers to fill the gap.

**Before using either file, check which one the emission-factor formula in `vehicle_emissions/` actually needs.** Vehicle category likely matters more for accuracy (a bus vs. a scooter differs by orders of magnitude; BS6 limit tables are organized primarily by category), but fuel type is not negligible either (diesel genuinely emits more NOx/PM than petrol or CNG under BS6). Open the actual formula code and confirm: does it key emission limits by category alone, or by category *and* fuel together? Report back which file(s) are actually consumed before assuming one is unnecessary — don't guess, and don't silently drop the fuel CSV without checking first.

**Both files already match the exact schema my teammate's `registrations.py` expects** (`rto_code, vehicle_category, fuel_type, vehicle_count, data_period, source_citation`) and **already pass that file's own validation checks** (verified this directly against their actual validation code before sending — not just assumed). They're meant to replace the demo data in `vehicle_emissions/`, not be re-processed with new logic. **Do not re-invent the RTO→zone aggregation — `vehicle_emissions/registrations.py::load_by_rto()` and `rto_mapping.py` already do this correctly; call them, don't duplicate their logic.**

---

## 3. My teammate's repo (attached as extracted folder) — what's real there, what isn't

Read `PROJECT_STATUS.md` and `README.md` at the repo root first — they're detailed and honest about what's real vs. demo. Do not re-verify claims that file already verified (e.g. GRAP thresholds, CPCB AQI breakpoints) — trust its citations, they were checked against live official sources.

**Key things already built and working, treat as stable dependencies, don't rebuild:**
- `forecasting/` — live CAAQMS scraper + AQI forecast model (gradient-boosted trees), beats a persistence baseline
- `shared/grap.py` — GRAP stage mapper, verified against caqm.nic.in
- `shared/dashboard.py` — themed HTML dashboard, renders only real data, shows "pending" badges for anything not live
- `shared/alerts.py` — alert card generator
- `vehicle_emissions/` — real formulas (BS6 limits, real distance study), currently running on **demo data** (even zone split) — this is what the two attached CSVs are meant to replace

**Explicitly NOT built — this is our job, not duplicated work:**
- **Enforcement Zone Ranker** (satellite attribution + vehicle emissions + land-use → ranked "zones needing attention" list) — their own status doc explicitly scopes this out as *"the satellite/enforcement track's Days 8-9 task"*. This is the main new thing to build.
- Prithvi fine-tuning itself — teammate's track never touched this, it's ours entirely.

**Known real gaps to work around, not silently paper over:**
- SAFAR (safar.tropmet.res.in) was down as of last check — expired TLS certificate, a site-side outage. Don't block on it; note it if still down.
- Delhi Supersite has real access but sparse historical data (mostly one isolated date found so far) — use what exists, don't fabricate a fuller comparison.
- **No CAAQMS "dominant pollutant source category" label exists anywhere in the repo or elsewhere.** Their forecasting module scrapes real pollutant *concentrations* (PM2.5, PM10, NO2, etc.) for AQI forecasting — that is a different thing from the `dust / crop_burning_smoke / industrial_haze / traffic_heavy / clear` category Prithvi needs to predict. See Phase 1 below for how to handle this honestly.

---

## 4. Three hard rules for all of this work — non-negotiable, apply to every phase

1. **No hardcoding.** Every number that isn't a cited, verifiable constant (GRAP thresholds, BS6 limits, official coordinates — the kind of thing already cited elsewhere in this project) must come from a real data file, not be typed in as a literal. If real data isn't available for something, the code should fail loudly with clear instructions (the pattern `registrations.py` already uses — follow it), never silently substitute a plausible-looking placeholder.
2. **Everything ethical and explainable.** Every derived value — especially anything not directly measured (heuristic labels, aggregated/split counts, model confidence scores) — must be traceable to *why* it has that value, in a comment or a `source`/`label_source`/`data_provenance`-style column, matching the pattern already used throughout this project (see `vehicle_emissions`' `data_provenance` flag and the amber "DEMO DATA" dashboard badge as the reference example). No black-box numbers.
3. **Phases, not one giant pass.** Do not attempt to write all of this in one uninterrupted session. Stop at the end of each phase below, report what was actually done vs. planned, and wait for confirmation before starting the next phase.

---

## 5. Keep a running log — `SESSION_LOG.md`, updated every phase

Create `SESSION_LOG.md` at the repo root if it doesn't exist. Follow the same tone and structure as the existing `run_log.md` / `PROJECT_STATUS.md` in this project (problem hit → fix applied, real numbers not vague summaries, explicit "what's real vs. what's assumed" sections where relevant). After each phase below, append a dated entry — don't rewrite history, append to it.

---

## 6. The phases

### Phase 1 — Real (or honestly-labeled heuristic) labels, replacing the random placeholder
- **First, confirm whether the CAAQMS scraper can reach historical dates (Oct–Nov 2025), not just current/live data.** The new historical image batch is only useful if there's matching pollution-reading data for those same dates — check this before assuming the crop-burning-season images will come with usable labels, and report the answer in `SESSION_LOG.md` either way.
- Check `forecasting/`'s scraped CAAQMS output for what raw pollutant data is actually available per zone/date (PM2.5, PM10, NO2, SO2, CO, wind — whatever it captures).
- If a genuine source-category signal exists somewhere (Supersite's sparse real data), use it and label those rows `label_source = 'Supersite'`.
- For everything else, derive a **rule-based heuristic** label from the raw concentration numbers (e.g. high PM10:PM2.5 ratio → dust, high NO2+CO → traffic_heavy, high PM2.5 + Oct–Jan → crop_burning_smoke, elevated SO2 → industrial_haze, everything low → clear). Label these rows clearly: `label_source = 'CAAQMS_heuristic — rule-based, not lab-verified'`. Document the exact rule used per category in `SESSION_LOG.md`, not just in a code comment.
- Do **not** present heuristic labels as equivalent to real source apportionment anywhere — not in variable names, not in dashboard text, not in the demo script.
- Replace the random-placeholder rows in the labels table (currently random, seed 42, clearly flagged) with these new labels wherever real zone/date coverage allows. Where it doesn't, leave the placeholder flag intact rather than guessing.

### Phase 2 — Fine-tune Prithvi on the new labels
- Model: `ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL`, via TerraTorch, as already set up.
- Use the 6-band images from the newly-organized Drive folders (verify each file actually has 6 bands before using it — a stray old 3-band file would break this silently).
- This is now a real attempt, not a plumbing check — report real accuracy/loss numbers honestly in `SESSION_LOG.md`, including how many samples were heuristic-labeled vs. placeholder vs. Supersite-verified, since that context matters for interpreting the result.
- Save the checkpoint; note its exact filename/path in the log.

### Phase 3 — Enforcement Zone Ranker
- Combine: Prithvi's per-zone attribution output (Phase 2) + the two VAHAN CSVs (via `vehicle_emissions`'s own `registrations.py`/`rto_mapping.py` — do not reimplement) + any usable land-use signal already available in the repo, if none exists, note that gap rather than fabricating one.
- Output a ranked "zones needing attention" list with a one-line, human-readable reason per zone (per the original design plan).
- Label this clearly as a correlation-based estimate, not proof of a legal pollution source — same standard as everywhere else in this project.

### Phase 4 — Integrate with the existing dashboard/fusion layer
- Wire the ranker's output into `shared/`'s already-built dashboard and GRAP stage mapper — do not rebuild these, they're already live.
- Confirm the dashboard's existing "pending"/demo badges correctly reflect the new real vs. heuristic vs. demo status of everything now flowing into it.

### Phase 5 — Wrap-up
- Final `SESSION_LOG.md` entry summarizing what's real, what's heuristic, what's still demo data, and what's genuinely not built, in the same style as `PROJECT_STATUS.md`.
- Flag anything that still needs a human decision (not a code fix) before the demo.
