# AirSentinel — Session Summary

*Everything built in this session, in order, plus current model performance and how to
reproduce it yourself. Written 2026-07-21.*

---

## 1. What this session started from

You provided three source documents for the **AirSentinel** project (ET AI Hackathon, Delhi
pilot):

- `AirSentinel_Teammate_Data_Request-1.docx` — your teammate (satellite/enforcement track,
  fine-tuning the **Prithvi** satellite model) asking you (**forecasting/data track**) for a
  real CAAQMS pollution table to replace the random placeholder labels he'd been using to
  test his training pipeline.
- `AirSentinel_design_plan.pptx` — the system architecture: 13 Delhi hotspot zones, GRAP
  staging, SAFAR benchmark, a fusion layer, a dashboard.
- `Teammate_Brief_PPT.pptx` — the combined team plan, confirming the two-person split:
  **you own forecasting + the CPCB/IMD data pipeline**; he owns satellite attribution +
  enforcement ranking.

You also gave a DPCC (Delhi Pollution Control Committee) live air-quality data URL and asked
for a scraper covering all 13 zones, a forecasting model, a proper Python environment, and
your teammate's data handoff.

## 2. What was built, step by step

1. **Read all three source documents** (docx/pptx parsed via OOXML) to extract the exact
   requirements: the 13 zone names (verbatim spelling), the exact CSV schema your teammate
   needs (`zone, date, PM2.5, PM10, NO2, SO2, CO, O3`), and the confirmation that you are
   "Person 1."
2. **Set up a real Python environment** — Python was not actually installed (only the
   Windows Store stub existed). Installed **Python 3.12.10**, created an isolated
   **`.venv`** inside `airsentinel/`, pinned dependencies in `requirements.txt`.
3. **Reverse-engineered the DPCC scraping contract.** The site has no public API; its
   "Advance Search" pages render each requested series into an inline **Highcharts** config
   embedded in the HTML response. Found the exact POST contract by driving the real form in
   a browser and inspecting the network request:
   - `parameters` = metric code (`PM25`, `NO2`, …), two endpoints for met vs. gas metrics
   - `fDate`/`eDate` = `YYYY-MM-DD HH:MM`, and — the one non-obvious gotcha — the window
     must be **strictly less than 7 days** (the UI's own hint says "not more than 7 days,"
     but a full 7-day span silently returns "Record Not found"). Fixed by chunking any
     range into 6-day windows.
   - `submit=Search` is required verbatim, or the server ignores the query.
4. **Built the package** (`src/airsentinel/`):
   - `stations.py` — the 13 hotspot-zone → DPCC-station mapping, in your teammate's exact
     spelling.
   - `scraper.py` — the live scraper (POSTs the form, parses the Highcharts `categories`/
     `data` arrays back into a tidy DataFrame).
   - `aqi.py` — the official **CPCB National AQI** formula (piecewise-linear sub-indices per
     pollutant, overall AQI = max sub-index, CPCB's ≥3-pollutants-incl.-PM rule enforced).
   - `forecast.py` — the forecasting model (see §4).
   - `labels.py` — an **optional**, clearly-flagged heuristic that guesses a
     `source_category` (dust/traffic/industrial/etc.) from the pollutant mix, for your
     teammate's Q1 (see `data/teammate_delivery/DATA_HANDOFF.md`) — never presented as
     ground truth.
   - `pipeline.py` — orchestrates scrape → aggregate → AQI → deliverable → model → charts.
   - `viz.py` — the two output charts.
5. **Ran the full pipeline live** against `dpccairdata.com` and produced your teammate's
   deliverable (`data/teammate_delivery/caaqms_readings.csv`) plus a handoff note answering
   his five questions directly.
6. **Follow-up pass 1**: removed everything unrelated to AirSentinel from the folder,
   audited the codebase for hardcoded/fake data, fixed a real issue found during that audit
   (§4), and added classification-style scores (accuracy/precision/recall/F1) alongside the
   regression metrics.
7. **Follow-up pass 2**: you asked what the baseline model is, why the deck says LSTM when
   we're not using one, to specifically improve 24h accuracy, and to confirm no overfitting
   + compatibility with the Prithvi track. Answered directly and made three real model
   changes — see §5a/§5b.
8. **Follow-up pass 3** (this message's ask): "fix all hardcoding and overfitting," push the
   repo to GitHub. Centralized every remaining tunable constant into `config.py`, widened
   the model's regularization search space (which materially cut the 72h overfitting gap —
   see §9), and set up git + GitHub. Details in §9–§11.

## 3. Cleanup performed this pass

`light_pollution/` also contained an unrelated, pre-existing Android app project ("Roshni" —
a phone light-exposure/sleep tracker: `app/`, `.idea/`, `.gradle/`, `build.gradle.kts`,
`settings.gradle.kts`, `gradle.properties`, `local.properties`, its own `README.md` and
`.gitignore`). **You confirmed removing it.** It was not under git version control anywhere
on disk, so this was a permanent deletion — flagging that here in case that matters later.
`light_pollution/` now contains only `airsentinel/` (and `.claude/`, this session's harness
config).

*(Historical note: this describes the folder structure as of this pass. A later pass
renamed the project root from `light_pollution/` to `airsentinel/` and this module's folder
from `airsentinel/` to `forecasting/` — see §13. The narrative above is left as-written for
accuracy about what was true at the time.)*

## 4. Hardcoding audit — what was found and fixed

Searched the whole codebase for fabricated/simulated data (`random`, `fake`, `mock`,
`dummy`, `synthetic`, `np.random.*`, etc.) — **none found**. Every number in
`data/` and `outputs/` comes from a live HTTP response from `dpccairdata.com`, computed
through documented, official formulas (CPCB AQI breakpoints) or a model trained on that
scraped data. There is no synthetic/placeholder data generation anywhere in the pipeline.

One real hardcoding issue *was* found and fixed:

- **Before:** `pipeline.py`'s CLI defaulted to a fixed, literal date range
  (`--start 2026-06-01 --end 2026-07-20`) — a snapshot from when it was first written, not a
  "live" window. Worse, `scrape_all()` silently reused any pre-existing file in `data/raw/`
  even on a normal run, so a second run could serve **stale** data from a previous scrape
  without saying so.
- **After:** the default window is now computed **at run time** from the system clock —
  `[now − 45 days, now]`, no literal date anywhere. `scrape_all()` no longer has a hidden
  reuse path: every non-`--skip-scrape` run hits the live site fresh for every zone/param,
  every time. Each response is still written to `data/raw/` afterwards, but purely as a
  timestamped **audit trail** (filenames now carry the date window,
  e.g. `Anand_Vihar__PM25__20260606_20260721.csv`), not as a cache that could serve stale
  numbers silently. `--skip-scrape` still exists as an explicit, clearly-labelled opt-out for
  fast local iteration on already-scraped data — it will never run by accident.
- Re-ran the full pipeline against the live site after the fix to regenerate every file in
  `data/` and `outputs/` from scratch, confirming the fix works end-to-end (see §5 for the
  resulting numbers, freshly measured on this live pull).

Everything else that looks "hardcoded" is intentionally-fixed **configuration**, not data:
the CPCB AQI breakpoints (`aqi.py`) are the official published government constants (like
hardcoding π would be silly to avoid), and the 13 zone→station name mappings
(`stations.py`) are a one-time mapping requested explicitly in your teammate's document,
required because DPCC's own station names don't match your hotspot names.

## 5. Model — baseline, architecture, and why not LSTM

**Baseline model:** naive **persistence** — "AQI 24/48/72h from now = AQI right now." This
is the standard weak benchmark the plan calls out; the real model has to beat it at every
horizon to be worth showing.

**Real model:** gradient-boosted trees (`sklearn.ensemble.HistGradientBoostingRegressor`),
one per horizon — **not an LSTM.** The original deck sketches "graph+LSTM," but that was
never implemented, and this was a deliberate choice rather than an oversight, worth stating
plainly: with ~45 daily points per zone (a few hundred rows total across all 13 zones), an
LSTM has nowhere near the sequence length it needs to learn temporal dynamics without
overfitting — RNNs are data-hungry and typically want thousands of steps per sequence to
generalize. A tree ensemble on hand-built lag/momentum/volatility features is far more
sample-efficient on small tabular data like this, trains in seconds on a laptop CPU with no
GPU, and is trivial to regularize and inspect. This matches the plan's own instruction to
"start simple, get it end-to-end, then add complexity" (§Person 1, Days 3–7). If the live
history grows into thousands of rows per zone, an LSTM or temporal transformer becomes a
reasonable upgrade — not before, and not for a hackathon demo.

## 5a. This pass's model changes (aimed at 24h specifically)

You asked to improve 24h accuracy and confirm no overfitting. Three real changes, in order:

1. **Richer short-term features**: added 3-day AQI volatility (rolling std), and PM2.5/PM10/
   NO2 day-over-day trend (not just same-day level) — the original feature set only had AQI
   momentum, not pollutant-level momentum.
2. **Residual (delta) target**: the model now predicts `AQI(t+h) − AQI(t)` — a *correction*
   to persistence — instead of the raw future AQI. This is the main lever for 24h: at short
   horizons persistence is already a strong, low-noise reference point, so asking the model
   to learn a small correction is a much easier, less noisy learning problem than
   reconstructing the whole value from scratch, and it degrades gracefully toward
   persistence itself when the model has nothing useful to add.
3. **CV-tuned shrinkage (the actual overfitting guard)**: hyperparameters (tree depth, leaf
   size, L2) and a shrinkage weight **alpha ∈ [0, 1]** are chosen per horizon by
   **walk-forward time-series cross-validation on the training period only** — the held-out
   test tail is never touched during model selection. Final prediction =
   `AQI(t) + alpha × predicted_delta`. Because **alpha = 0 reproduces pure persistence
   exactly**, and alpha is chosen to minimize out-of-fold error, model selection can never
   settle on a configuration that beats persistence in-sample but loses to it out-of-fold —
   alpha is free to shrink toward 0 wherever the model isn't adding real signal, which is
   exactly the 24h regime where persistence is hardest to beat.

**Overfitting check, reported per horizon:** in-sample train MAE (on the delta target) vs.
out-of-fold CV MAE on the same training period. A small gap means the model generalizes; a
large one flags memorization. This is now printed on every run and stored in
`outputs/metrics.json` as `overfit_gap_pct`.

## 5b. Model performance history (superseded — see §9 for current numbers)

The first iteration of the delta+shrinkage model (§5a) measured 24h/48h/72h MAE improvements
of +4.0%/+23.4%/+30.5% with a 72h train-vs-CV overfit gap of +33.4%. That overfit gap is what
§9 fixes. **§9 has the current, correct numbers — this subsection is kept only as a record of
what changed and why.**

All numbers regenerate fresh every time you run the pipeline — see §6.

## 6. How to run this yourself

### One-time setup
```powershell
cd "C:\Users\a955032\OneDrive - ATOS\Desktop\airsentinel\forecasting"
python -m venv .venv                                    # if .venv doesn't already exist
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run the full live pipeline
```powershell
cd "C:\Users\a955032\OneDrive - ATOS\Desktop\airsentinel\forecasting"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m airsentinel.pipeline
```
This scrapes the **live** trailing-45-day window from `dpccairdata.com` for all 13 zones
(~10 minutes — 117 live HTTP requests, ~0.25s apart, polite to the site), then:
- writes `data/processed/airsentinel_daily_panel.csv` (full panel: pollutants + weather + AQI)
- writes `data/teammate_delivery/caaqms_readings.csv` (your teammate's exact requested schema)
- writes `data/teammate_delivery/caaqms_heuristic_labels.csv` (optional candidate source labels)
- trains the forecasting model and writes `outputs/forecasts.csv`, `outputs/metrics.json`
- writes `outputs/model_vs_baseline.png` and `outputs/forecast_example.png`

### Useful variations
```powershell
# Explicit date range instead of the live default:
.\.venv\Scripts\python.exe -m airsentinel.pipeline --start 2026-06-01 --end 2026-07-20

# Re-run just the modelling/aggregation from the last live scrape, without hitting the site again
# (fast — use this while iterating on the model, not for a "final" run before demo day):
.\.venv\Scripts\python.exe -m airsentinel.pipeline --skip-scrape
```

### Where to look at results
- `outputs/metrics.json` — the numbers in the table above, regenerated fresh each run.
- `outputs/model_vs_baseline.png` — bar chart, model vs. persistence MAE per horizon.
- `outputs/forecast_example.png` — one zone's observed AQI history + its 72h forecast.
- `data/teammate_delivery/DATA_HANDOFF.md` — the note for your teammate, answering his five
  questions.

## 7. Hardcoding audit, pass 2 — `config.py`

You asked to "fix all hardcoding" a second time, more thoroughly. Pass 1 (§4) fixed the one
data-correctness bug (a stale literal date range). This pass went further and centralized
every remaining tunable *constant* — scattered magic numbers that were correct but hard to
find and change — into a single new module, **`src/airsentinel/config.py`**:

- **Scraper**: request delay, timeout, retry count, the 6-day window cap (with the reasoning
  for why it's 6 and not 7, documented inline).
- **Live-window default**: the 45-day lookback (now overridable via
  `AIRSENTINEL_WINDOW_DAYS` env var without touching code).
- **CPCB aggregation constants**: the 8-hour rolling window and its 6-hour minimum-data
  requirement for CO/O3, the mean-vs-max8h pollutant lists — all cited back to the official
  CPCB averaging-period convention, not invented.
- **AQI bounds** (0–500): used consistently now in `aqi.py`'s category cap and every clip
  call in `forecast.py`, instead of the literal `500` appearing independently in three
  places (a classic hardcoding smell — same constant, multiple copies, one risks going stale
  if only one is ever updated).
- **Model/CV settings**: the regularization search grid, CV fold count, shrinkage grid step,
  test-holdout size, and the fixed (non-searched) HistGBM settings — all named and explained
  in one place instead of buried as inline literals in `forecast.py`'s function bodies.

`scraper.py`, `aqi.py`, `forecast.py`, and `pipeline.py` now import from `config.py` rather
than repeating literals. Nothing about the actual scraping/modelling behaviour changed from
this refactor alone — re-running produced byte-identical results — **except** for one
deliberate, called-out change: the regularization search grid was *widened*, which is a real
methodology change and is covered in §8, not a pure refactor.

What's *not* moved into `config.py`, on purpose: the CPCB AQI breakpoint table (`aqi.py`)
and the 13 zone→station mappings (`stations.py`). Both are look-up **data**, not tunable
knobs — moving a lookup table into a file called "config" wouldn't make it more correct, and
would bury the fact that they're externally-sourced facts (a government breakpoint table, a
site's internal station identifiers) rather than something you'd ever want to "tune."

## 8. Overfitting fix — widened the regularization search space

The 72h horizon's train-vs-CV MAE gap (§5b) was +33.4% — the model fit its own training data
noticeably better than it generalized to held-out folds, a real overfitting signal even
after the CV-based model selection in §5a. The cause: the regularization grid CV was
searching over wasn't conservative enough in its most-restrained corner — the most
regularized option available was `max_depth=2, min_samples_leaf=35, l2=3.0`, which still let
72h's tree ensemble fit training noise.

**Fix:** widened `config.MODEL_PARAM_GRID_*` to include genuinely conservative options —
`max_depth=1` (decision stumps), `min_samples_leaf` up to 50, `l2_regularization` up to 5.0
— alongside the original range. This does **not** force a more conservative model; CV still
picks whichever combination generalizes best. It just makes sure strong regularization is on
the table for CV to choose *if and when it actually wins*, which for 72h (and 24h) it did.

**Result**, same live data, same test split:

| Horizon | Overfit gap before | Overfit gap after | Params CV chose |
|---|---|---|---|
| 24h | +11.7% | **+3.3%** | depth=1, leaf=15, l2=5.0 |
| 48h | +12.8% | +16.6% (slightly worse, still moderate) | depth=3, leaf=15, l2=5.0 |
| 72h | **+33.4%** | **+14.9%** | depth=1, leaf=15, l2=5.0 |

24h and 72h both improved substantially (CV picked decision stumps for both once stumps were
available); 48h moved slightly the other way but stayed in the same "modest, expected" range
as before — CV chose a deeper tree there because it's what generalized best over the folds,
which is the whole point of letting cross-validation decide rather than hand-picking a single
"safe" setting for every horizon. Full current numbers in §9.

## 9. Current model performance (measured live, this run)

**Data window:** 2026-06-06 → 2026-07-21 (the trailing 45 days as of today — this updates
automatically every run, it is not fixed). 13 zones × 45 days ≈ **597 zone-days**, mean AQI
**209** (CPCB "Poor") — a genuinely volatile, mostly-bad-air period, which is the honest
context for the error numbers below.

| Horizon | Model MAE | Baseline MAE | MAE improvement | Model R² | Baseline R² | Alpha (shrinkage) |
|---|---|---|---|---|---|---|
| 24h | 96.2 | 101.2 | **+4.9%** | **+0.03** | −0.36 | 0.75 |
| 48h | 100.0 | 130.2 | **+23.2%** | −0.15 | −0.91 | 1.00 |
| 72h | 100.3 | 153.4 | **+34.6%** | −0.16 | −1.45 | 0.80 |

24h R² is now **positive** for the first time (+0.03) — the model finally, if narrowly, beats
"always guess the training-period average" on the hardest horizon, not just persistence.

**Overfitting check (train MAE vs. out-of-fold CV MAE, on the delta target):**

| Horizon | Train MAE | CV MAE | Gap |
|---|---|---|---|
| 24h | 93.6 | 96.7 | **+3.3%** |
| 48h | 85.3 | 102.2 | +16.6% |
| 72h | 86.1 | 101.2 | **+14.9%** (was +33.4%) |

All three horizons are now in a defensible range for ~400 training rows; none stand out as
alarming the way 72h did before §8's fix.

**"Accuracy, precision, and other scores"** — AQI itself is a continuous number, so
accuracy/precision/recall don't directly apply to it. To answer that honestly, the model is
also scored as a **classifier**: bucket predicted and actual AQI into the six CPCB categories
(Good/Satisfactory/Moderate/Poor/Very Poor/Severe) and score category-match on the same
held-out test data:

| Horizon | Model accuracy | Model precision (weighted) | Model recall (weighted) | Model F1 (weighted) | Baseline accuracy |
|---|---|---|---|---|---|
| 24h | 30.1% | 43.9% | 30.1% | 27.9% | **48.0%** |
| 48h | 32.0% | 34.0% | 32.0% | 28.7% | 33.6% |
| 72h | **35.7%** | 35.8% | 35.7% | 32.4% | 24.6% |

Persistence still edges the model on exact-category accuracy at 24h specifically, even though
the model has lower average numerical error (MAE) there too. Same honest explanation as
before: persistence trivially gets the category right whenever AQI doesn't cross a category
boundary overnight, while the model's numerically-closer predictions sometimes land just
across one. **Say this proactively in the demo**, consistent with the "what we are not
claiming" posture your deck already commits to (slide 9).

All of these numbers regenerate fresh every time you run the pipeline — see §6.

## 10. Compatibility with your teammate's Prithvi track

No risk of interference, by construction: the forecasting model only ever reads
`data/processed/airsentinel_daily_panel.csv` (CAAQMS + weather) — it shares no code,
weights, features, or training data with Prithvi fine-tuning. The two tracks touch at
exactly one surface: the **(`zone`, `date`) join key**, which is identical across
`data/teammate_delivery/caaqms_readings.csv` (already handed off) and
`outputs/forecasts.csv` (`zone, forecast_from, horizon_hours, target_date, predicted_aqi`).
Nothing in this pass's changes touched that schema, so his fine-tuning pipeline and any
fusion-layer code already written against the delivered CSV keeps working unmodified.

## 11. GitHub repository

Pushed to a new **public** GitHub repo via `gh` CLI (installed this pass), authenticated
through the device-code browser flow you completed yourself — no credentials were ever
typed or handled on your behalf. `.gitignore` excludes only `.venv/`, `__pycache__/`, and
editor/OS cruft; **all data your teammate needs is committed**, including:
`data/teammate_delivery/` (the deliverable + handoff note), `data/processed/` (the full
daily panel), `data/raw/` (the per-zone/param scraped audit trail), and `outputs/`
(forecasts, metrics, charts) — nothing in `data/` or `outputs/` is gitignored. Repo URL and
final push confirmation are in the chat where this was run.

## 12. Known limitations (say these before a judge finds them)

- ~400–430 training rows per horizon is a small-data regime for a 13-zone, ~45-day live
  window; both model and baseline show negative R² on the held-out tail at 48h/72h, which
  reflects how volatile this particular stretch of Delhi AQI was (swings between ~100 and
  ~500 within days), not a modelling error — but it does mean confidence in any single
  forecast is limited. More history (once further back can be scraped) will help most.
- Persistence still edges the model on exact CPCB-category accuracy at 24h, even though the
  model has lower average numerical error (MAE) at that same horizon (§9) — flag this
  proactively rather than letting a judge find it.
- The `source_category` heuristic labels are a simple rule over pollutant thresholds, not
  real source apportionment — clearly labelled as such everywhere they appear, and the
  primary deliverable to your teammate is the raw pollutant table, not these labels.
- The scraper depends on `dpccairdata.com`'s current HTML structure (regex-parsed from an
  inline Highcharts config, §3) — if DPCC changes their site layout, `scraper.py`'s parsing
  will need updating. There's no official API contract to depend on instead.

## 13. Project renamed; sibling modules added (follow-up pass)

Two things changed in a later pass, both outside this module's own code:

1. **Folder rename.** The project root moved from `light_pollution/` to `airsentinel/`, and
   this module's folder moved from `airsentinel/` to `forecasting/` — matching Person 1's
   role name. The Python package inside is **unchanged**: still `src/airsentinel/`, still
   imported as `import airsentinel` / `from airsentinel.stations import ZONES`. "AirSentinel"
   is the overall project brand; "forecasting" is just this module's folder label. All
   absolute-path examples in this document and `README.md` were updated to the new location;
   nothing in the actual pipeline code referenced an absolute path in the first place (every
   module resolves its own paths relative to `Path(__file__)`), so the rename itself required
   no code changes here — only doc updates.
2. **Sibling modules added**: `../vehicle_emissions` (the Vehicle Emission Load Model /
   "tailpipe pollution," design plan slide 4) and `../shared` (fusion layer + themed
   dashboard). Both consume this module's output by file (`outputs/forecasts.csv`,
   `data/processed/airsentinel_daily_panel.csv`) — nothing in `forecasting/` needed to change
   for those to work. `vehicle_emissions` additionally installs this module in editable mode
   (`pip install -e ../forecasting`) purely to import `airsentinel.stations.ZONES` directly
   rather than retyping the 13 zone names a second time. See `../PROJECT_STATUS.md` for the
   full picture across all three modules.
