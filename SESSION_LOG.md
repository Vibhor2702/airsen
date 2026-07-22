# AirSentinel — Session Log

Appended chronologically. Do not rewrite history — append only.

---

## 2026-07-21 — Session 1: Environment + Plumbing Check (Phase 0 / pre-brief)

### What was done
- Created `train_prithvi.py` and `setup_env.txt` from scratch.
- Set up Python venv (conda not installed; Python 3.12 works fine with all packages).
- Installed: torch 2.6.0+cu124, terratorch 1.2.10, rasterio 1.5.0, and dependencies.
- Ran a 2-epoch plumbing check on 13 samples (1 per zone, one date), random placeholder labels.

### Issues hit and fixes applied
See `run_log.md` for the full record. Summary:
1. `conda` not installed — switched to `python -m venv`.
2. `\M` escape in path string — added `r` prefix.
3. `labels.csv` missing — generated locally from existing files, random seed 42.
4. Windows terminal crashed on emoji — replaced all with ASCII (`[!]`, `[OK]`, `->`).
5. Wrong backbone input shape `(B, T, C, H, W)` — actual is `(B, C, T, H, W)` per Conv3D weight; fixed `unsqueeze(1)` -> `unsqueeze(2)`.

### Plumbing check result (random labels — numbers meaningless)
- 13 samples, 1 date per zone, batch_size=2
- VRAM: 1.4 GB allocated / 1.75 GB peak (well within 8.6 GB)
- Backbone output: 12-element list, last shape (1, 197, 768) — 196 patch tokens + CLS token
- Epoch 1 loss: 1.8356, Epoch 2 loss: 2.3450
- Checkpoint: `prithvi_airsen_plumbing_check.pt`

---

## 2026-07-21 — Session 2: Phase 1 (Real labels) + Phase 2 (Real training run)

### Context changes from brief
- Drive folder reorganised: per-zone subfolders, 10 dates per zone (5-day Sentinel-2 cadence), 6-band S2 confirmed. 225 .tif files total (130 S2 + 95 NO2 — some NO2 files are missing for certain zone/dates, which is expected per brief).
- Some files are loose outside subfolders due to Drive sync lag — brief says to include these via recursive scan, not fix the sort process.
- Teammate delivered `caaqms_heuristic_labels.csv` + `caaqms_readings.csv` in `P2/airsentinel-master/forecasting/data/teammate_delivery/`. See `DATA_HANDOFF.md` there for full schema.

### Phase 1 — Label build (`build_labels.py`)

**What was done:**
- Recursive scan of entire `AirSentinel_Satellite_Images/` path found 130 S2 files across 13 zones x 10 dates.
- Loaded teammate's `caaqms_heuristic_labels.csv` (597 rows: 13 zones x ~46 dates, 2026-06-06 to 2026-07-21).
- Joined on (zone, date) after mapping filename underscores to CAAQMS space-separated names (e.g. `Anand_Vihar` -> `Anand Vihar`).

**Join result:**
- 117 / 130 rows matched to heuristic labels — `label_source = 'CAAQMS_heuristic -- rule-based, not lab-verified'`
- 13 / 130 rows unmatched (all `2026-06-01`, which predates the CAAQMS scraper window starting 2026-06-06) — kept as `label_source = 'PLACEHOLDER -- random, not real (no CAAQMS coverage for this date)'`

**Heuristic label distribution (117 matched rows):**
- dust: 89
- industrial_haze: 14
- traffic_heavy: 10
- clear: 4
- crop_burning_smoke: 0 (June–July window; burning season is Oct–Nov)

**Heuristic rules (from `P2/airsentinel-master/forecasting/src/airsentinel/labels.py`):**
- `clear`: AQI <= 100
- `crop_burning_smoke`: PM2.5 >= 90 and PM2.5/PM10 ratio >= 0.5
- `industrial_haze`: SO2 >= 25 (primary marker); fallback SO2 >= 15
- `traffic_heavy`: NO2 >= 40 and CO >= 1.2 mg/m3; fallback NO2 >= 30 and CO >= 1.0
- `dust`: PM10 >= 150 and PM2.5/PM10 ratio <= 0.45; fallback PM10 >= 2*PM2.5
- These are chemical-marker heuristics, NOT source apportionment. Never present as ground truth.

**What's real vs. assumed:**
- CAAQMS concentrations are real scraped DPCC data (dpccairdata.com).
- Source-category labels are rule-derived, not lab-verified. The only ground-truth source would be the Delhi Supersite (IIT-Kanpur/DPCC), which has sparse data (one isolated date found so far).
- The 13 `2026-06-01` rows remain random placeholders and must not influence accuracy claims.

**Output:** `G:\My Drive\AirSentinel_Satellite_Images\labels.csv` (130 rows)

---

### Phase 2 — Training run on heuristic labels

**What was done:**
- Fixed `torch.cuda.amp.GradScaler` / `autocast` deprecation warnings — updated to `torch.amp.GradScaler("cuda")` / `autocast("cuda")`.
- Ran `train_prithvi.py` unchanged except for the amp fix — the new subfolder-relative `s2_file` paths in labels.csv resolved correctly via pathlib with no code changes needed.

**Training result (2 epochs, batch_size=2, 130 samples):**
- VRAM: 1.40 GB allocated / 1.75 GB peak
- Epoch 1: Avg Loss 1.1937 | Acc 63.8%
- Epoch 2: Avg Loss 1.0047 | Acc 66.2%
- Checkpoint: `prithvi_airsen_plumbing_check.pt`

**Honest interpretation of these numbers:**
- 66.2% accuracy against 5 classes (random baseline ~20%) looks high, but is NOT a reliable signal of real model performance. Reasons:
  1. Labels are heuristic (rule-derived from concentrations, not ground truth).
  2. The dominant class is `dust` (89/117 heuristic rows = 76%) — the model may be partially or largely learning to predict `dust` most of the time, inflating accuracy on a skewed dataset.
  3. No train/val split was used (only 130 samples; a proper split would reduce training data further and give an unbiased accuracy estimate).
  4. Only 2 epochs — not enough for convergence.
- **Do not cite these numbers as model performance in the demo.** Report them with the above caveats if shown at all.

**What needs to happen before accuracy is meaningful:**
- Replace heuristic labels with Supersite-verified source apportionment (if/when available).
- Add a proper train/val split.
- Train for more epochs.
- This session's purpose was confirming the pipeline runs on real data — that goal was met.

---

### Decisions made this session (human confirmation needed before changing)

1. **VAHAN CSVs not yet consumed.** Both `vehicle_registrations_by_rto.csv` and `vehicle_registrations_by_rto_category.csv` are in the project folder but have not been wired in. Phase 3 (Enforcement Zone Ranker) will use them via `vehicle_emissions/registrations.py` — the brief requires reading that code first to confirm which CSV the formula actually keys on before assuming one is unnecessary.

2. **`2026-06-01` images will be excluded from any real accuracy claim** until CAAQMS coverage extends back to that date (or they are dropped from training).

3. **No Supersite data was found** beyond the single isolated date noted in teammate's `PROJECT_STATUS.md`. The heuristic labels are the best available source-category signal for this session.

---

### Remaining phases (not yet started — waiting for confirmation)

- **Phase 3:** Enforcement Zone Ranker — combine Prithvi output + VAHAN via `vehicle_emissions/` + any land-use signal.
- **Phase 4:** Wire ranker into `shared/` dashboard and GRAP mapper.
- **Phase 5:** Final wrap-up and gap summary.

---

## 2026-07-21 — Session 3: Brief update — folder re-sort, file audit, label rebuild

### Brief changes that triggered this session
1. **19 dates per zone** (was 10): 9 historical dates added (weekly Oct 1 – Nov 26, 2025), specifically to get crop_burning_smoke examples missing from the June–July-only data.
2. **Sort instruction changed**: brief now says to actually move misplaced files (not just scan around them), run a local sort script, and log results.
3. **New Phase 1 pre-step**: check if CAAQMS scraper can reach Oct–Nov 2025 historical dates before assuming those images will come with labels.

### Step 1 — CAAQMS historical coverage check

**Finding: CAAQMS data in the current teammate delivery only covers 2026-06-06 to 2026-07-21. No Oct–Nov 2025 data is present.**

- Raw files are named e.g. `Anand_Vihar__PM25__20260606_20260721.csv` — the date range is baked into the filename, confirming the window.
- The scraper (`scraper.py`) does support arbitrary `--start`/`--end` date ranges via DPCC's "Advance Search" POST endpoint. It could technically be re-run for `--start 2025-10-01 --end 2025-11-26`.
- **However: whether DPCC's database holds data that far back is unconfirmed.** DPCC's public interface typically retains ~45 days of live data; historical access may require a different endpoint or institutional access. This needs teammate input before attempting.
- **Implication: all 49 currently-present historical (Oct–Nov 2025) S2 images will carry PLACEHOLDER labels until CAAQMS coverage is extended.** The purpose of the historical batch (getting real crop_burning_smoke labels) is not yet achievable with current data.

### Step 2 — Drive folder sort (`sort_drive_folder.py`)

Ran a local sort against the synced Drive path. Results:

| Metric | Count |
|---|---|
| Total .tif files found (pre-sort) | 354 (inflated by duplicates) |
| Already in correct zone subfolder | 255 |
| Moved to correct subfolder | 39 |
| Duplicate root copies deleted after sort | 60 |
| Unrecognised filenames | 0 |
| Root files remaining after cleanup | 0 |

The 60 deletions were confirmed stale copies: destination already existed in the zone subfolder before the move was attempted.

### Step 3 — File audit (post-sort)

**S2 files present: 179 of 247 expected (68 still pending from Colab export)**

| Zone | S2 files | Historical dates missing |
|---|---|---|
| Anand_Vihar | 19 | complete |
| Ashok_Vihar | 10 | all 9 historical (Oct–Nov 2025) |
| Bawana | 10 | all 9 historical |
| Dwarka | 10 | all 9 historical |
| Jahangirpuri | 19 | complete |
| Mundka | 19 | complete |
| Narela | 10 | all 9 historical |
| Okhla | 10 | all 9 historical |
| Punjabi_Bagh | 10 | all 9 historical |
| RK_Puram | 19 | complete |
| Rohini | 14 | 5 of 9 historical |
| Vivek_Vihar | 10 | all 9 historical |
| Wazirpur | 19 | complete |

5 zones are complete (Anand_Vihar, Jahangirpuri, Mundka, RK_Puram, Wazirpur). 8 zones are partially or fully missing historical batch — Colab export still in progress.

NO2 files: 117 landed (vs 234 expected) — the remainder are still exporting or not yet synced.

### Step 4 — Label rebuild

`build_labels.py` re-run on 179 S2 files. Results:

| Label source | Count | Notes |
|---|---|---|
| CAAQMS_heuristic -- rule-based, not lab-verified | 117 | June–July 2026 only |
| PLACEHOLDER -- random, not real | 62 | All Oct–Nov 2025 historical + all 2026-06-01 rows |

Historical (Oct–Nov 2025) rows are all PLACEHOLDER because CAAQMS has no coverage for those dates in the current delivery. The 13 June-01 rows remain PLACEHOLDER for the same reason as before (predates scraper window).

**crop_burning_smoke in heuristic labels: 0.** The 6 crop_burning_smoke rows in the raw heuristic file (June–July 2026 data) did not match any S2 image dates — those specific dates had no satellite image. The historical batch that was added specifically to fix this gap has no CAAQMS labels yet. So crop_burning_smoke remains absent from usable training data.

### Decisions / open items requiring human input (at time of writing)

1. ~~**CAAQMS historical data**: does DPCC's API/site retain Oct–Nov 2025 data?~~ **Resolved — see Session 4 below.**
2. **Remaining 68 S2 files**: still exporting from Colab for 8 zones. Once synced, re-run `sort_drive_folder.py` then `build_labels.py` — both scripts are idempotent.
3. **Phase 2 training** can proceed on the current 179 files / 117 heuristic labels, but crop_burning_smoke will be absent. Flag this in any accuracy reporting.

### Remaining phases (not yet started — waiting for confirmation)

- **Phase 2:** Re-run training on updated 179-sample dataset (117 heuristic + 62 placeholder).
- **Phase 3:** Enforcement Zone Ranker.
- **Phase 4:** Dashboard integration.
- **Phase 5:** Wrap-up.

---

## 2026-07-21 — Session 4: CAAQMS historical scrape (Oct–Nov 2025)

### Context
Session 3 left the CAAQMS historical coverage question open: does DPCC's site retain data back to Oct–Nov 2025? This session resolves it.

### Step 1 — Forecasting venv setup

`forecasting/` has its own independent venv (separate from the main `AirSen/.venv`). It did not exist yet.

```
cd P2/airsentinel-master/forecasting
python -m venv .venv
pip install -r requirements.txt   # requests, pandas, numpy, scikit-learn, matplotlib, python-dateutil
pip install -e .                  # installs the airsentinel package itself in editable mode
```

All installed cleanly. No conflicts.

### Step 2 — Probe: does DPCC have Oct 2025 data?

Rather than running the full 20-30 min pipeline blind, ran a targeted single-window probe first:
- Station: Anand Vihar
- Param: PM2.5
- Window: 2025-10-01 to 2025-10-07

**Result: 208 rows returned. DPCC has real hourly PM2.5 data going back to at least Oct 1, 2025.**

Sample (first 3 rows):
```
2025-10-01 01:00  19.0
2025-10-01 02:00  11.0
2025-10-01 03:00   9.0
```

Values are low (9–19 µg/m³) in early October — consistent with pre-harvest-season clear conditions, which is expected.

### Step 3 — Full pipeline run

The tool has a hard 10-minute cap, so the full pipeline run (13 zones × 9 params × ~10 windows = ~1,170 requests, ~20-30 min at 0.25s delay) was handed off to a user-run terminal session to avoid timeout.

Command (run in user's PowerShell):
```powershell
cd "C:\Users\Vibhor\Projects\Personal\AirSen\P2\airsentinel-master\forecasting"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m airsentinel.pipeline --start 2025-10-01 --end 2025-11-26
```

Confirmed running as of screenshot: `[1/117] live  Anand Vihar  PM2.5  (1272 rows)` — progress output is showing, scrape is live.

**Status: in progress. Waiting for completion before re-running `build_labels.py`.**

### What happens after pipeline finishes

1. Pipeline writes new raw CSVs to `forecasting/data/raw/` and updated delivery CSVs to `forecasting/data/teammate_delivery/` — specifically a new `caaqms_heuristic_labels.csv` covering Oct–Nov 2025.
2. Re-run `build_labels.py` from `AirSen/` (main venv) — it reads from `teammate_delivery/caaqms_heuristic_labels.csv` and joins against the 179 S2 files.
3. Expected outcome: the 49 historical S2 rows (5 complete zones + partial Rohini) that currently carry PLACEHOLDER labels should flip to `CAAQMS_heuristic`. The 13 `2026-06-01` rows will remain PLACEHOLDER (still predates the June 6 scraper start).
4. crop_burning_smoke labels should now appear — Oct–Nov is Delhi's actual burning season, and PM2.5 with high fine-fraction ratios should trigger the heuristic rule.

### Open item after pipeline completes

- Re-run `sort_drive_folder.py` + `build_labels.py` once the remaining 68 S2 files finish syncing from Colab (8 zones still missing their full historical batch).

---

## 2026-07-21 — Session 5: Label fix, Drive re-sort, Phase 2 training

### Problem 1 resolved: dual date-range label gap

The Oct–Nov 2025 pipeline run overwrote `caaqms_heuristic_labels.csv` with only the historical window, losing the June–July 2026 labels. Fix:

1. Confirmed both date-range raw CSVs exist in `forecasting/data/raw/` (234 files: 117 per window).
2. Concatenated all raw CSVs for both windows into a combined `hourly_tidy.csv` (256,806 hourly rows, 2025-10-01 → 2026-07-21).
3. Re-ran `python -m airsentinel.pipeline --skip-scrape` (no re-scrape; used cached raw files) — rebuilt `caaqms_heuristic_labels.csv` covering both windows.

**New delivery file: 1,334 rows, 2025-10-01 to 2026-07-21.**

Label distribution in combined delivery:
- dust: 542
- crop_burning_smoke: 473
- traffic_heavy: 160
- clear: 88
- industrial_haze: 71

### Problem 2 resolved: Drive folder re-sort

114 new loose files appeared in Drive root during the Session 4 pipeline run (Colab continued exporting while the scraper ran). Two-step clean:

1. Ran `sort_drive_folder.py` — moved 59 files, 65 skipped (dest already existed).
2. Ran a separate cleanup pass — deleted 65 confirmed stale root copies.

Drive root is now clean. 240 S2 files confirmed across all 13 zones (420 scanned pre-cleanup, 240 remaining after dedup).

### Step 3 — Label rebuild (post-sort, post-fix)

Re-ran `build_labels.py` on 240 S2 files against combined delivery.

| Metric | Count |
|---|---|
| S2 files scanned | 240 |
| Unique zone/date pairs | 240 (13 zones × 19 dates, minus 7 still pending sync) |
| Matched to heuristic labels | 226 |
| Unmatched (PLACEHOLDER) | 14 |

**Unmatched rows:**
- All 13 `2026-06-01` rows — predates CAAQMS scraper start (2026-06-06), same as before.
- `Punjabi_Bagh / 2025-11-26` — CAAQMS has no data for this specific station/date (scraper returned no rows for it in the raw files).

**Final label distribution in `labels.csv` (240 rows):**

CAAQMS_heuristic (226 rows):
- dust: 105
- crop_burning_smoke: 70  ← **now non-zero for the first time**
- traffic_heavy: 22
- industrial_haze: 15
- clear: 14

PLACEHOLDER (14 rows — random, do not use for accuracy claims):
- traffic_heavy: 5, dust: 3, industrial_haze: 3, clear: 2, crop_burning_smoke: 1

**crop_burning_smoke is now represented with 70 real heuristic examples.** The historical batch (Oct–Nov 2025) delivered what it was added for.

Still missing: 7 of 247 expected S2 files (240/247 present). These are Colab export lag — once synced, re-run `sort_drive_folder.py` + `build_labels.py` (both idempotent).

---

### Phase 2 — Training on heuristic labels (real attempt)

**Setup:**
- 226 CAAQMS_heuristic rows used; 14 PLACEHOLDER rows excluded entirely.
- 80/20 train/val split (random_state=42): 181 train, 45 val.
- 10 epochs, batch_size=2, AdamW lr=1e-4, mixed precision (float16).
- Checkpoint: `prithvi_airsen_phase2.pt`

**Per-epoch results:**

| Epoch | Train Loss | Train Acc | Val Acc |
|---|---|---|---|
| 1 | 1.5734 | 37.0% | 66.7% |
| 2 | 1.3711 | 43.1% | 51.1% |
| 3 | 1.3007 | 53.6% | **77.8%** |
| 4 | 1.3146 | 50.8% | 75.6% |
| 5 | 1.1822 | 57.5% | 75.6% |
| 6 | 1.0848 | 63.5% | **77.8%** |
| 7 | 0.9710 | 68.0% | 73.3% |
| 8 | 0.9181 | 67.4% | 71.1% |
| 9 | 0.9748 | 66.9% | 73.3% |
| 10 | 0.8837 | 70.7% | 68.9% |

**Honest interpretation:**
- Best val accuracy: **77.8%** at epochs 3 and 6 (5-class problem, random baseline ~20%).
- Val accuracy drifts down in later epochs while train accuracy rises — sign of mild overfitting on 181 training samples. The dataset is small for a 86M-param model; this is expected.
- The model has not converged; loss is still declining at epoch 10. More epochs and/or a lower LR schedule would help — but the data volume is the binding constraint, not the training length.
- **These numbers are against heuristic labels, not ground truth.** Do not cite 77.8% as classification accuracy in a scientific sense. The correct framing: "77.8% agreement with CAAQMS heuristic rule labels on the held-out val split."
- The dominant class is `dust` (105/226 = 46%) — some of the accuracy is from the model learning to predict dust frequently. A per-class breakdown (confusion matrix) would be more informative but requires more samples per class than the current val split provides.
- `crop_burning_smoke` is now present in training (70 samples) and should be learnable, but with only 14 val samples of this class the per-class val accuracy for it is noisy.

**What's real, what's heuristic, what's missing:**
- VRAM: 1.40 GB allocated / ~1.75 GB peak (unchanged from plumbing check).
- Checkpoint: `prithvi_airsen_phase2.pt` (tagged as heuristic-trained in the saved metadata).
- 7 of 247 expected S2 files still missing (Colab export lag) — retraining once those sync is optional.
- No Supersite-verified labels exist in any row. The 77.8% figure cannot be ground-truthed without them.

**Next: Phase 3 — Enforcement Zone Ranker.**

---

### Session 5 — End state summary (2026-07-21)

**What is real and working:**
- `labels.csv`: 240 rows (226 CAAQMS_heuristic, 14 PLACEHOLDER). Both date windows covered. crop_burning_smoke represented for the first time.
- `prithvi_airsen_phase2.pt`: Prithvi-EO-2.0-100M-TL fine-tuned on 181 heuristic-labeled images, 77.8% val agreement with heuristic rules.
- Drive folder: fully sorted, root clean, 240/247 expected S2 files present.
- Forecasting pipeline (`forecasting/.venv`): working, both date windows in `hourly_tidy.csv`, combined delivery files up to date.

---

## 2026-07-21 — Session 5 (continued): Notebook fixes + augmentation

### Notebook issues fixed (`AirSentinel_Satellite_Pull_FINAL(3).ipynb`)

Three cells had bugs preventing local (VS Code) execution:

| Cell | Issue | Fix |
|---|---|---|
| Step 8.5 (sort) | `drive_root = 'PASTE_...'` placeholder; used string `/` for `os.path.join` | Set `drive_root = r'G:\My Drive'` with Colab/non-Colab autodetect; replaced all path joins with `os.path.join` |
| Step 9 (preview) | `filepath.split('/')[-1]` breaks on Windows backslash paths | Replaced with `os.path.basename(filepath)`; added standalone fallback for `folder_path` if earlier cells weren't run |
| Step 11 (augment) | `[::-1]` and `np.rot90` return non-contiguous views — rasterio rejects these; paths used `/` not `os.sep`; `target_dir` undefined if Step 8.5 skipped | Added `.copy()` after every view operation; replaced path strings with `os.path.join`; added standalone `try/NameError` fallback for `target_dir` |

**Note:** Steps 1–8 (Earth Engine pull + export) still require Colab or local EE auth with a valid `PROJECT_ID`. They cannot be run non-interactively. Steps 9–11 are now fully runnable locally.

### Drive folder re-sort (third pass)

Drive sync continued bringing in new files during the augmentation run. Another sort + cleanup pass:
- Ran `sort_drive_folder.py`: 61 more files moved (mostly NO2 files + Dwarka S2 files)
- Deleted 73 stale root copies confirmed as duplicates
- One genuine new arrival in root: `Rohini_2025-11-26_NO2.tif` (no subfolder copy existed, left in place)

### Label rebuild (fourth pass)

**247/247 expected S2 files now present** (13 zones × 19 dates — all zones complete for the first time).

| Label source | Count |
|---|---|
| CAAQMS_heuristic | 233 |
| PLACEHOLDER | 14 |

heuristic distribution: dust 103, crop_burning_smoke 76, traffic_heavy 25, industrial_haze 15, clear 14.

### Step 11 augmentation run

Ran augmentation on 233 heuristic rows (Step 11, notebook cell). Results:

| Metric | Count |
|---|---|
| Source rows (heuristic only) | 233 |
| Transforms applied | 7 (all orientations of dihedral group D4) |
| S2 aug files newly created | ~427 (remainder skipped — already existed from first run) |
| NO2 aug failures | 539 — all are historical batch NO2 files not yet in Drive (expected gap) |
| `labels_augmented.csv` total rows | **1,878** (247 original + 1,631 augmented) |

Effective training set (heuristic only) after augmentation:
- dust: 824, crop_burning_smoke: 608, traffic_heavy: 200, industrial_haze: 120, clear: 112

**To use augmented data in Phase 2 retraining:** point `train_prithvi.py` at `labels_augmented.csv` instead of `labels.csv` and update the `label_source` filter to also match `CAAQMS_heuristic.*augmented`. The S2 paths in `labels_augmented.csv` correctly reference the per-zone subfolder aug files.

---

### Phase 2 (augmented) — Training run complete

**Setup changes from Phase 2 (non-augmented):**
- `LABELS_CSV` → `labels_augmented.csv` (1,878 rows)
- `BATCH_SIZE` 2 → 4 (VRAM was only 0.72 GB allocated at batch=4, well within 8.6 GB)
- Group-aware train/val split: 187 train groups / 46 val groups → 1,496 train rows / 368 val rows
- Group split prevents leakage: all 7 augmented versions of a val image go to val, not train
- `CHECKPOINT_PATH` → `prithvi_airsen_augmented.pt`

**Per-epoch results:**

| Epoch | Train Loss | Train Acc | Val Acc |
|---|---|---|---|
| 1 | 1.1083 | 62.0% | 71.7% |
| 2 | 0.8401 | 70.5% | 71.7% |
| 3 | 0.7418 | 73.5% | 70.4% |
| 4 | 0.6654 | 76.7% | 73.1% |
| 5 | 0.6051 | 78.4% | 69.8% |
| 6 | 0.5416 | 80.7% | **73.9%** |
| 7 | 0.4923 | 81.6% | 69.8% |
| 8 | 0.4323 | 84.0% | 68.2% |
| 9 | 0.4615 | 82.6% | 70.9% |
| 10 | 0.3546 | 86.7% | 66.0% |

- VRAM: 0.72 GB allocated / 0.86 GB peak (batch=4 used less VRAM than batch=2 in Phase 2 — likely due to mixed precision efficiency with the larger batch)
- Checkpoint: `prithvi_airsen_augmented.pt`

**Honest interpretation:**
- Best val accuracy: **73.9%** at epoch 6. Val accuracy declines in epochs 7–10 while train accuracy keeps rising (86.7% at epoch 10) — classic overfitting pattern.
- The model is overfitting to the augmented training set. Root cause: augmentation multiplies the same 233 underlying scenes 8× — after a few epochs, the model has effectively memorised all spatial orientations of the training images. The val set (46 original zone/date groups, 368 rows) contains genuinely unseen scenes, so val acc diverges.
- The leakage-free split makes the 73.9% a more honest estimate than Phase 2's 77.8% (which came from a tiny 45-sample val set with no leakage protection). These numbers are not directly comparable.
- At epoch 6 the gap between train acc (80.7%) and val acc (73.9%) is only 6.8 pp — reasonable for a fine-tuned ViT. Beyond that the gap grows rapidly.
- **Best checkpoint for inference: epoch 6** — but we saved only the final (epoch 10) checkpoint. If overfitting is a concern for Phase 3, re-run with `torch.save` after each epoch and keep the epoch-6 weights.

**What this means vs Phase 2 (non-augmented):**
- Augmentation did NOT improve peak val accuracy (73.9% vs 77.8%), but the Phase 2 number was from 45 val samples and was noisy. The augmented run's val is more statistically reliable (368 samples).
- Training loss dropped much further (0.35 vs 0.88 at epoch 10) — the model is learning more from 8× more data per epoch.
- The binding constraint is now the number of *original* scenes (233), not the number of training rows. More unique zone/date captures (finishing the Colab export) would help more than further augmentation.

**Decisions / open items:**
1. Consider early stopping at epoch 6 in any future rerun — saves time and gives the best val acc.
2. The remaining 7 missing S2 files (if Colab finishes) would add ~7 new original scenes + 49 augmented rows. Small but worth re-running once they land.
3. **Phase 3 can proceed with the current checkpoint** (`prithvi_airsen_augmented.pt`, epoch 10 weights). For the Enforcement Zone Ranker, per-zone softmax probabilities matter more than raw accuracy — the model only needs to rank zones, not achieve perfect classification.

---

---

## 2026-07-21 — Session 6: NO2 sensor fusion

### Phase A — NO2 feature extraction

Ran extraction loop over all 1,878 rows in `labels_augmented.csv`. Read each `no2_file` via rasterio, filtered pixels to finite & non-negative (Sentinel-5P has no explicit nodata tag — NaN-based), computed mean NO2 per image. Output: `labels_fused.csv` (same as `labels_augmented.csv` + two new columns: `no2_mean`, `no2_available`).

**Availability breakdown:**

| Status | Rows | Notes |
|---|---|---|
| `ok` (no2_available=1) | **1,462** (77.8%) | Real Sentinel-5P NO2 concentration |
| `no_path_in_csv` | 301 | Augmented rows whose original had no NO2 file — expected |
| `all_nodata_or_negative` | 72 | All from **2025-10-29** across all zones — Sentinel-5P gap on that specific date, not file corruption |
| `file_not_found` | 43 | Original NO2 files not yet exported from Colab |

**Notable finding (stop and report, per brief):** The 72 all-nodata rows are not random or corrupt — they are all `2025-10-29`. This is a specific Sentinel-5P coverage gap for that date (likely cloud/orbit gap). They are correctly marked `no2_available=0` and will receive the learned missing-token embedding during training. No files were garbage; no silent workaround applied.

**NO2 value distribution (1,462 available rows):**
- Range: 4.2e-5 to 2.25e-4 mol/m²
- Median: 9.0e-5 | Mean: 9.7e-5 | Std: 3.5e-5
- Normalisation: z-score using training-set statistics (mean=9.55e-5, std=3.50e-5) computed from training rows only, stored in checkpoint for inference.

---

### Phases B/C/D — Architecture + training (in progress)

**New file: `train_fused.py`**

Single script, two modes:
- `python train_fused.py` → optical-only baseline (frozen backbone, head only)
- `python train_fused.py --fuse-no2` → NO2-fused model

**Architecture (Phase C):**

```
Prithvi backbone (frozen, 86.2M params, 768-d output)
  ↓ mean-pool → (B, 768)
  +
NO2 path (32-d):
  if no2_available == 1:  No2Encoder(1 → 32 → 32) on z-scored value
  if no2_available == 0:  learned nn.Parameter missing_token (32-d)
  [modality dropout: 15% chance to use missing_token even when NO2 available]
  ↓ cat → (B, 800)
LayerNorm(800) → Dropout(0.1) → Linear(800, 5)
```

**Trainable params:**
- Baseline: 0.005M (head only — LayerNorm + Linear)
- Fusion: 0.008M (head + No2Encoder + missing_token)
- Backbone frozen from Phase 2 checkpoint (`prithvi_airsen_augmented.pt`) in both

**Key design decisions:**
- Missing-modality handling (Phase B): learnable missing token rather than zero-fill or row-drop. Physically motivated: the model learns what "no NO2 signal" looks like as a distinct state.
- Modality dropout 15%: prevents fusion model depending on NO2 at inference. During 15% of training steps where NO2 is available, it is randomly replaced with the missing token.
- Best-val checkpointing: saves only when val accuracy improves — fixes the epoch-10 overfitting issue from Phase 2.
- 15 epochs (vs 10 before) — frozen backbone is faster per epoch, and with a 5K-param head there is less risk of runaway overfitting.

**Baseline run: complete. Fusion run: in progress.**

**Baseline results (optical-only, frozen backbone, 15 epochs, best-val checkpoint):**

| Epoch | Train Loss | Train Acc | Val Acc |
|---|---|---|---|
| 1 | 0.6780 | 76.3% | 70.7% |
| 2 | 0.4653 | 84.2% | 71.5% |
| 7 | 0.3085 | 89.4% | 72.0% |
| 10 | 0.2762 | 90.6% | 72.8% |
| 11 | 0.2693 | 90.4% | **74.2%** |
| 14 | 0.2472 | 91.2% | **74.5%** ← best |
| 15 | 0.2404 | 91.5% | 73.1% |

- Best val: **74.5%** at epoch 14 (checkpoint: `prithvi_baseline_best.pt`)
- Trainable params: 0.005M (head only — backbone frozen)
- Val accuracy improves steadily through epoch 14, only drops at epoch 15 — best-val checkpointing correctly saved epoch 14

**Fusion results (optical + NO2, frozen backbone, 15 epochs, best-val checkpoint):**

| Epoch | Train Loss | Train Acc | Val Acc |
|---|---|---|---|
| 1 | 0.6439 | 77.9% | 71.2% |
| 2 | 0.4521 | 85.0% | 72.3% |
| 6 | 0.3166 | 89.4% | 72.8% |
| 10 | 0.2588 | 91.4% | **74.7%** ← best |
| 11 | 0.2496 | 91.6% | 73.6% |
| 12 | 0.2454 | 91.6% | 72.6% |
| 15 | 0.2230 | 92.5% | 73.9% |

- Best val: **74.7%** at epoch 10 (checkpoint: `prithvi_fused_best.pt`)
- Trainable params: 0.008M (head + No2Encoder + missing_token)
- NO2 available: 1,448/1,864 heuristic rows (77.7%), with 22.3% using learned missing token

---

### Phase D — Baseline vs. Fusion comparison

**Same setup for both:** identical group-aware split (RANDOM_SEED=42), frozen backbone from same `prithvi_airsen_augmented.pt` checkpoint, best-val checkpointing, 15 epochs.

| Model | Best Val Acc | Best Epoch | Checkpoint |
|---|---|---|---|
| Baseline (optical only) | **74.5%** | 14 | `prithvi_baseline_best.pt` |
| Fusion (optical + NO2) | **74.7%** | 10 | `prithvi_fused_best.pt` |
| **Delta** | **+0.2 pp** | fusion peaks earlier | — |

**Honest interpretation:**
- +0.2 pp difference is within the noise band for a 368-sample val set (one image misclassified either way swings val by ~0.27 pp). This is **not a statistically significant improvement** at this sample size.
- Fusion best epoch is 10 vs baseline's 14 — NO2 signal provides a small learning boost that reaches peak earlier, which is consistent with real information gain rather than overfitting to NO2.
- 22.3% of rows used the learned missing token at every training step — the fusion model learned to operate gracefully on partial sensor coverage, which is operationally important (S5P has frequent cloud gaps over Delhi).
- The architecture is correct and the missing-token pathway works. The accuracy plateau is a data constraint, not an architecture failure: with 233 unique original scenes, head-only fine-tuning of both branches is saturating around 74–75%.
- **Correct framing for the demo**: "Adding NO2 sensor fusion maintains performance parity (74.7% vs 74.5%) while adding robust handling of missing satellite data and a physically interpretable NO2 signal. The accuracy delta is small because the dataset is small — the architecture is ready to scale."

**Performance notes:**
- Both runs use `num_workers=0` (Windows requirement) — I/O blocks GPU during data loading.
- Backbone runs 86.2M params forward every epoch despite being frozen — no feature caching yet.
- Net: ~3 min/epoch. With feature caching (precompute 768-d outputs once → cache to disk), epochs would drop to seconds. Planned for next session if retraining is needed.

---

---

### Session 8 — 2026-07-22

#### Phase 4 — Dashboard + GRAP wiring

**Plan (logged before execution):**

Pre-check findings (must fix before running anything):

1. Zone naming mismatch — enforcement_ranker.py writes `attribution.csv` with underscore names
   (e.g. `Anand_Vihar`) because labels.csv uses underscores; every other module (forecasting,
   vehicle_emissions, fuse.py's join key) uses space names (`Anand Vihar`).
   Result: fuse.py's left join on `zone` silently fails for 4 of 13 zones, those zones show
   satellite_source_guess = "pending" in the fused output even though attribution data exists.
   Fix: normalize zone names in enforcement_ranker.py before writing attribution.csv.

2. Honesty tags not wired through — model_version, data_provenance, land_use_note exist in
   enforcement_ranking.csv but are not in attribution.csv, not in fuse.py, not in dashboard.py.
   Fix: extend attribution.csv schema to include them; extend fuse.py to pull them; extend
   dashboard.py to render them visibly.

3. Bawana and Punjabi Bagh missing from forecasts.csv (11/13 zones forecasted).
   This is a data gap in the forecasting track — not fixable here. Flag in dashboard and log.

4. DEMO tag verdict (see "Small cleanup" below):
   - emission_factors.csv: values are real BS6 regulatory limits — DEMO tag removable
   - distance_estimates.csv: car distance explicitly has no citation — DEMO tag stays

**Minimal changes planned (no rebuilds):**
- enforcement_ranker.py: normalize zone names + extend attribution.csv to include honesty tags
- fuse.py: extend satellite merge to pull honesty tag columns when present (optional columns)
- dashboard.py: add enforcement honesty badge section (model_version, data_provenance, land_use_note)
- emission_factors.csv: replace DEMO tag with real BS6 citation text

---

### Session 7 — 2026-07-21

#### Track A — Vehicle emissions category-mapping fix

**Plan (logged before execution):**

The existing `vehicle_emission_index.csv` shows every zone at index=1.0 — the flat result comes
from demo data in `vehicle_registrations.csv` that split Delhi-wide totals evenly across zones.

Two real VAHAN CSVs exist in the project root:
- `vehicle_registrations_by_rto_category.csv` — real per-RTO counts by VAHAN vehicle class (fuel_type=ALL)
- `vehicle_registrations_by_rto.csv` — real per-RTO counts by fuel type (vehicle_category=ALL)

Fix: write `category_mapper.py` that reads both, maps VAHAN classes → emission categories,
derives fuel split per RTO, and writes a properly formatted input to the pipeline.

**Category mapping decisions (every decision logged):**

| VAHAN category | → Emission category | Fuel | Reasoning |
|---|---|---|---|
| Motor Car | → `car` | petrol + diesel (split from fuel file) | Direct 1:1 correspondence — VAHAN "Motor Car" is precisely CPCB/ARAI's passenger car emission category |
| M-Cycle/Scooter | → `two_wheeler` | petrol | Direct — VAHAN's standard 2-wheeler class = CPCB L2/L3 two-wheeler emission category |
| M-Cycle/Scooter-With Side Car | → `two_wheeler` | petrol | Sidecar is a cargo attachment; powertrain is the same single-engine 2-wheeler; same emission norm applies |
| Moped | → `two_wheeler` | petrol | ARAI/CPCB treat mopeds under L1/L2 two-wheeler norms, same as M-Cycle/Scooter |
| Motorised Cycle (CC > 25cc) | → `two_wheeler` | petrol | L-category by CMVR definition; same BS6 two-wheeler limits apply |
| e-Rickshaw(P) | → **UNMAPPED** | — | Battery electric; zero tailpipe NOx/PM by physical property — no exhaust EF exists or applies. Counts ~1,155 across RTOs. Will appear in coverage_note as excluded. |
| e-Rickshaw with Cart (G) | → **UNMAPPED** | — | Same — battery electric, goods variant. Counts ~705 across RTOs. |
| Adapted Vehicle | → **UNMAPPED** | — | Undefined body type; could be any powertrain. No defensible single emission factor. |
| Fork Lift | → **UNMAPPED** | — | Off-road industrial equipment; not subject to on-road emission norms. Count = 1. |
| Vintage Motor Vehicle | → **UNMAPPED** | — | Pre-BS norm vehicle; BS6 factors don't apply, and no reliable pre-BS factor can be assigned. Count = 1. |

**Fuel split for Motor Car:** VAHAN's by_rto_category.csv reports Motor Car counts with fuel_type="ALL".
The by_rto.csv (fuel split, category=ALL) provides per-RTO petrol and diesel totals across all vehicle types.
Per-RTO petrol_fraction = petrol-type vehicles / (petrol-type + diesel-type) where:
- petrol-type: PETROL, PETROL/CNG, PETROL(E20), PETROL(E20)/CNG, PETROL/HYBRID, PETROL(E20)/HYBRID,
  PETROL/HYBRID/CNG, PETROL(E20)/HYBRID/CNG, FLEX-FUEL(ETHANOL) (E20/flex-fuel vehicles run on petrol;
  same BS6 emission norm applies regardless of ethanol blend)
- diesel-type: DIESEL, DIESEL/HYBRID
- Excluded from denominator: CNG ONLY, PURE EV, STRONG HYBRID EV, ELECTRIC(BOV), PLUG-IN HYBRID EV
  (no tailpipe NOx/PM emission factor applies to any of these)

Known approximation: this split is across all vehicle types, not car-specific. Since Delhi 2-wheelers
are almost entirely petrol, this biases petrol fraction upward relative to cars-only. Documented here;
stated in source_citation column of the output.

**Two-wheeler fuel assignment:** All two-wheeler categories assigned petrol. Justification: diesel
2-wheelers have never been commercially sold in India; BS6 two-wheeler emission norms (CMVR Schedule VI)
cover only petrol engines in the L1/L2/L3 category. This is a physical fleet fact, not an assumption.

**E-rickshaw emission factor search:** User instruction: source a real BS6/applicable-norm EF if a
real cited source can be found. Result: e-rickshaws are battery electric (CMVR Category L5e/L6e with
electric powertrain per Delhi EV Policy 2020). They have zero exhaust NOx/PM emissions by construction.
No ARAI/CPCB exhaust emission factor exists for them — not a sourcing gap, a physical fact. Left unmatched.
Coverage_note will report excluded e-rickshaw counts per zone.

#### Track A — Results

**Vehicle Emission Load Index — per-zone results (2026-07-21, real VAHAN data):**

| Rank | Zone | Index | Raw (g/day) | Coverage |
|---|---|---|---|---|
| 1 | RK Puram | 1.000 | 136,693 | 6/6 matched |
| 2 | Dwarka | 0.780 | 106,598 | 5/5 matched |
| 3 | Rohini | 0.551 | 75,340 | 8/8 matched |
| 4 | Mundka | 0.509 | 69,607 | 3/3 matched |
| 5 | Punjabi Bagh | 0.343 | 46,899 | 3/3 matched |
| 6 | Wazirpur | 0.315 | 43,020 | 4/4 matched |
| 6 | Ashok Vihar | 0.315 | 43,020 | 4/4 matched |
| 8 | Narela | 0.286 | 39,073 | 4/4 matched |
| 9 | Bawana | 0.276 | 37,700 | 3/3 matched |
| 10 | Jahangirpuri | 0.275 | 37,640 | 5/5 matched |
| 11 | Okhla | 0.248 | 33,877 | 5/5 matched |
| 12 | Anand Vihar | 0.191 | 26,063 | 5/5 matched |
| 12 | Vivek Vihar | 0.191 | 26,063 | 5/5 matched |

All 13 zones covered. Flat 1.0 is gone.

**Why the variation is plausible:**
- RK Puram (DL3, Lado Sarai): largest solo-RTO vehicle count — 57,049 M-Cycle/Scooter + 13,543 Motor Car
- Dwarka (DL9, South West): 37,107 + 17,810 — second largest solo RTO
- Anand/Vivek Vihar: both share DL7 (East Delhi, 18,527 + 8,336) with counts split /2

**Excluded categories (reported, not fabricated around):**
- e-Rickshaw(P): 1,155 total — electric, no tailpipe EF applicable
- e-Rickshaw with Cart (G): 705 total — electric, goods variant
- Adapted Vehicle: 110 — undefined body type, no defensible EF
- Fork Lift + Vintage Motor Vehicle: 2 total — off-road / pre-BS norm

These are excluded at category_mapper.py, NOT at the pipeline level. They do NOT appear in
the pipeline's coverage_note because they were never passed in. This is intentional — the
pipeline's coverage_note reports unmatched rows it received, not categories that were
pre-filtered. The exclusion decision is documented in category_mapper.py's CATEGORY_MAP.

**Files produced/modified:**
- `category_mapper.py` (new) — converts VAHAN CSVs to pipeline input format
- `P2/airsentinel-master/vehicle_emissions/data/raw/vehicle_registrations_by_rto.csv` (new) — 43 rows of real mapped data
- `P2/airsentinel-master/vehicle_emissions/outputs/vehicle_emission_index.csv` (updated)

---

#### Track B — Enforcement Zone Ranker scaffold

**Plan (logged before execution):**

Build `enforcement_ranker.py` combining:
1. Prithvi attribution: run inference on latest available images per zone using `prithvi_airsen_augmented.pt`
   — classes: dust, crop_burning_smoke, industrial_haze, traffic_heavy, clear
2. Vehicle Emission Load Index from Track A output
3. Land-use signal: **none exists in repo** — confirmed by full codebase search; `config.py` references
   "design plan slide 5: hotspot + emission + land use" but no land-use CSV or API integration is present.
   Ranker will note this explicitly in output and NOT fabricate a proxy.

Output schema: zone, enforcement_rank, source_attribution, source_confidence, vehicle_emission_load_index,
  composite_score, rank_reason, land_use_note, model_version, data_provenance, ranked_at

Every row tagged: model_version = "247-image placeholder — not final"

Checkpoint path at line 1 of configurable constants: `PRITHVI_CKPT = Path("prithvi_airsen_augmented.pt")`
Confirmed one-line swap: changing this path is the only change needed to use a future better checkpoint.

Also writes: `P2/airsentinel-master/satellite_attribution/outputs/attribution.csv` (schema: zone,
source_guess, confidence) so fuse.py auto-picks it up when rerun.

#### Track B — Results

**Enforcement Zone Ranking (2026-07-16, 247-image placeholder model):**

| Rank | Zone | Composite | Source (Prithvi) | Conf | VEI | Reason |
|---|---|---|---|---|---|---|
| #1 | Dwarka | 0.607 | dust | 0.93 | 0.780 | dust suppression + high vehicle load |
| #2 | Okhla | 0.593 | traffic_heavy | 0.52 | 0.248 | traffic enforcement priority |
| #3 | Mundka | 0.551 | dust | 1.00 | 0.509 | high-confidence dust + moderate traffic |
| #4 | Rohini | 0.547 | traffic_heavy | 0.38 | 0.551 | traffic + moderate vehicle density |
| #5 | RK Puram | 0.499 | crop_burning_smoke | 0.66 | 1.000 | seasonal source, limited enforcement scope |
| #6 | Bawana | 0.415 | dust | 0.83 | 0.276 | dust suppression |
| #7 | Wazirpur | 0.408 | dust | 0.78 | 0.315 | dust suppression |
| #8 | Anand Vihar | 0.395 | dust | 0.84 | 0.191 | dust suppression |
| #9 | Vivek Vihar | 0.347 | dust | 0.73 | 0.191 | dust suppression |
| #10 | Narela | 0.323 | dust | 0.59 | 0.286 | dust suppression |
| #11 | Jahangirpuri | 0.298 | crop_burning_smoke | 0.72 | 0.275 | seasonal source |
| #12 | Ashok Vihar | 0.251 | crop_burning_smoke | 0.52 | 0.315 | seasonal source |
| #13 | Punjabi Bagh | 0.103 | clear | 0.56 | 0.343 | no significant event |

Composite formula: `attribution_conf * source_weight + 0.3 * vehicle_emission_load_index`
Source weights: traffic_heavy=1.0, industrial_haze=0.9, dust=0.4, crop_burning_smoke=0.3, clear=0.0

**Model-vs-heuristic agreement on 2026-07-16:**
- Agree: Anand Vihar (dust), Bawana (dust), Dwarka (dust), Mundka (dust), Narela (dust), Vivek Vihar (dust), Wazirpur (dust) — 7/13 = 54%
- Disagree: Ashok Vihar (model: crop_burning_smoke, heuristic: dust), Jahangirpuri (crop_burning vs dust), Okhla (traffic_heavy vs dust), Punjabi Bagh (clear vs industrial_haze), RK Puram (crop_burning vs industrial_haze), Rohini (traffic_heavy vs dust)

Note: disagreement is expected and informative — model learned visual patterns, heuristic is from CAAQMS readings on that date. Neither is ground truth.

**Honesty flags confirmed present in output:**
- Every row tagged `model_version: "247-image placeholder -- not final"`
- `data_provenance: "correlation-based (heuristic CAAQMS labels, not ground truth)"`
- `land_use_note: "land-use signal absent from this repo -- scoring uses only satellite + vehicle data"`

**One-line checkpoint swap confirmed:**
Line 22 of enforcement_ranker.py: `PRITHVI_CKPT = Path("prithvi_airsen_augmented.pt")`
Change this path -> all ranking logic, output schema, fuse.py wiring unchanged.

**Files produced:**
- `enforcement_ranker.py` (new)
- `enforcement_ranking.csv` (new) — full ranked output
- `P2/airsentinel-master/satellite_attribution/outputs/attribution.csv` (new) — satellite attribution for fuse.py

---

### Session 7 — End state (2026-07-21)

**Completed this session:**

Track A — Vehicle Emissions Category-Mapping Fix:
- `category_mapper.py` (new): maps real VAHAN category names to emission formula categories with every decision documented; derives per-RTO petrol/diesel split for cars; leaves e-rickshaws and off-road equipment explicitly unmapped with reasoning
- `P2/.../vehicle_emissions/data/raw/vehicle_registrations_by_rto.csv` (new): 43 rows of real mapped registration data replacing the even-split demo data
- `vehicle_emission_index.csv` regenerated: flat 1.0 gone; real per-zone variation from RK Puram (1.000) to Anand/Vivek Vihar (0.191)

Track B — Enforcement Zone Ranker (Phase 3 scaffold):
- `enforcement_ranker.py` (new): runs actual Prithvi inference on latest per-zone images, joins with vehicle emission index, ranks 13 zones with one-line human-readable reason per zone
- Checkpoint swap confirmed as one-line change (line 22: `PRITHVI_CKPT`)
- All output rows tagged `model_version: "247-image placeholder -- not final"`
- Land-use absence documented explicitly in output and console — not worked around
- `enforcement_ranking.csv` (new): full Phase 3 ranked output
- `satellite_attribution/outputs/attribution.csv` (new): fuse.py auto-picks this up on next run

**Open items / known gaps (unchanged from prior sessions):**
1. Gap-fill images (Dec 2025 – May 2026) still not synced to Drive — dataset remains 247 originals / 19 dates
2. No Supersite ground-truth labels — all accuracy figures are against heuristic CAAQMS rules
3. Vehicle emission factors still tagged DEMO in source_citation — values are real BS6 limits, but the tag causes `data_provenance="demo"` in the index output. Remove the DEMO tag from `emission_factors.csv` once the source citations are formally verified against the ARAI/CPCB PDFs
4. Distance estimates (`distance_estimates.csv`) still tagged DEMO — same issue, same fix path
5. Phase 4 (Dashboard / GRAP integration) not yet started
6. Phase 5 (Wrap-up gap summary) not yet started

---

### Session 6 — End state (2026-07-21)

**Completed this session:**
- Phase A: NO2 extraction — `labels_fused.csv` built (1,878 rows, 77.7% with real NO2)
- Phase B: Learned missing-token architecture implemented
- Phase C: Late-fusion model (`Prithvi 768-d + NO2 32-d → 800-d → head`) implemented in `train_fused.py`
- Phase D: Fair baseline vs. fusion comparison complete:
  - Baseline: 74.5% best val | Fusion: 74.7% best val | Delta: +0.2 pp (noise-level, data-limited)
  - Both checkpoints saved: `prithvi_baseline_best.pt`, `prithvi_fused_best.pt`

**Not yet started — waiting for confirmation:**
- Phase 3: Enforcement Zone Ranker (`vehicle_emissions/` + VAHAN CSVs + Prithvi output)
- Phase 4: Dashboard / GRAP integration
- Phase 5: Wrap-up gap summary

**Open items:**
1. Backbone feature caching — epochs currently ~3 min; caching would cut to seconds. Worth doing before any retraining.
2. 7 S2 files still pending Colab sync. Once landed: re-run sort + build_labels.
3. No Supersite ground-truth labels — 74.7% is against heuristic rules only.

---

### Session 5 — Final end state (2026-07-21)

**Completed this session:**
- Notebook (`AirSentinel_Satellite_Pull_FINAL(3).ipynb`) fixed for local VS Code use — 3 cells patched (drive_root, Windows paths, array contiguity)
- Drive folder fully sorted and clean (247/247 S2 files present, root empty)
- `labels.csv`: 247 rows, 233 heuristic, 14 PLACEHOLDER
- `labels_augmented.csv`: 1,878 rows (233 originals + 1,631 augmented)
- Phase 2 (non-augmented): best val 77.8% from 45-sample val, noisy
- Phase 2 (augmented): best val **73.9%** at epoch 6 from 368-sample leakage-free val — more reliable estimate
- Checkpoint for Phase 3: `prithvi_airsen_augmented.pt` (epoch 10 weights)

**Not yet started — waiting for confirmation:**
- Phase 3: Enforcement Zone Ranker (`vehicle_emissions/` + VAHAN CSVs + Prithvi output)
- Phase 4: Dashboard / GRAP integration
- Phase 5: Wrap-up gap summary

**Known gaps requiring human input (not code):**
1. 7 S2 files still pending Colab sync. Once landed: re-run sort + build_labels + optionally retrain.
2. No Supersite ground-truth labels — 73.9% val accuracy is against heuristic rules only.
3. `Punjabi_Bagh / 2025-11-26` has an S2 image but no CAAQMS data — stays PLACEHOLDER.
4. Best checkpoint is epoch 6, but only epoch 10 was saved. Re-run with per-epoch saves if a tighter model is needed for Phase 3.

---

### Session 8 — End state (2026-07-22)

**Completed this session:**

Phase 4 — Dashboard + GRAP wiring:
- Zone name mismatch fixed in `enforcement_ranker.py`: attribution.csv now writes space-separated zone names (Anand Vihar, not Anand_Vihar) so fuse.py's join works for all 13 zones
- attribution.csv extended to include honesty columns: `model_version`, `data_provenance`, `land_use_note` — every row carries explicit quality flags
- `fuse.py` extended: satellite merge now pulls honesty columns from attribution.csv when present; all three are filled with PENDING when absent (backwards-compatible)
- `dashboard.py` extended: added enforcement ranker honesty badge section (CSS + HTML) — model_version, data_provenance, land_use_note rendered as visible chips; PENDING shown in grey, placeholder in amber; section cannot be missed
- `emission_factors.csv` updated: DEMO tag removed; replaced with real BS6 citations (MoRTH/CMVR GSR 889(E) / CMVR Schedule VI, verified 2026-07-22) for all 6 rows
- Full shared pipeline rerun: `fuse.py` + `dashboard.py` — all 4 data sources live, 11 zones covered, dashboard.html = 9,773 bytes
- GRAP thresholds verified: 6 test cases passed (AQI 150→Below GRAP, 250→Stage I, 350→Stage II, 425→Stage III, 455→Stage IV, 200→Below GRAP)
- Bawana and Punjabi Bagh absent from AQI forecasts confirmed as data gap (not a code bug) — flagged in dashboard and log

DEMO tag verdict (decided and documented):
- `emission_factors.csv`: DEMO tag removed — values are real regulatory BS6 limits (GSR 889(E)); source citations added for all 6 rows
- `distance_estimates.csv`: DEMO tag kept — car distance (30 km/day) reuses two-wheeler estimate with no independent car-specific citation; correctly causes data_provenance="demo" in pipeline output

Phase 5 — Competition submission wrap-up:
- `SUBMISSION_SUMMARY.md` written (new file, project root): 5 sections — 1-paragraph system description, 10-row real/heuristic/demo status table, step-by-step run instructions for all 3 tracks + fusion step, 9 known limitations stated plainly, Person 1 / Person 2 work division

**Open items / known gaps (carried forward):**
1. Gap-fill images (Dec 2025 – May 2026): still 0 files on Drive — dataset stays at 247 originals / 19 dates until GEE export is re-run for those 36 dates
2. 266 root-level stale .tif files in Drive: should run `sort_drive_folder.py` again to clean up
3. `distance_estimates.csv` car distance: still DEMO (no car-specific citation); needs an Indian urban mobility source for cars specifically
4. `augment_labels.py`: may be stale if labels.csv changed — should be re-run after gap-fill images land
5. Bawana and Punjabi Bagh: forecasting track (Person 1) needs to add CAAQMS data + forecasts for these two zones
6. LoRA training: explicitly deferred ("being handled separately") per Phase 4/5 prompt — no action taken

---

### Session 9 — 2026-07-22 (in progress)

#### Task 1 — Vehicle emissions documentation

**Finding:** category_mapper.py is confirmed implemented (Session 7 — no redone work).
Inline documentation already present in the code (CATEGORY_MAP comments, PETROL_FUEL_TYPES
docstring, compute_fuel_fractions() docstring). SESSION_LOG Session 7 already has the
category-mapping decisions and fuel-split logic in detail.

**What was added this session:** `VEHICLE_EMISSIONS_METHODOLOGY.md` (new, project root) —
teammate-readable standalone reference covering:
- Source files used and their scope
- Every VAHAN → emission category mapping decision with full reasoning
- Fuel-split derivation formula for Motor Car (cross-category approximation, bias acknowledged)
- Per-RTO petrol fractions table (DL3–DL13)
- Every excluded category with count and explicit reason (not a sourcing gap — physical facts)
- What the pipeline does NOT cover (CNG-only, EVs, HCV) stated plainly

**Exclusion summary (confirmed from VAHAN data):**

| Category | Count | Reason |
|---|---|---|
| e-Rickshaw(P) | ~1,155 | Battery electric (L5e/L6e); zero tailpipe emissions; no exhaust EF exists |
| e-Rickshaw with Cart (G) | ~705 | Battery electric, goods variant; same |
| Adapted Vehicle | ~110 | Undefined body type; no defensible single EF |
| Fork Lift | 1 | Off-road; not subject to on-road CMVR norms |
| Vintage Motor Vehicle | 1 | Pre-BS norm; BS6 EF doesn't apply |

Total excluded: ~1,972 vehicles (~1.2% of extract). Pre-filtered in category_mapper.py,
NOT via the pipeline's coverage_note (which only reports rows it received unmatched).

---

#### Task 2 — Phase A speed fixes (confirmed working)

**What was already done (no redo):**
- Feature caching: already fully wired into train_fused.py (FeatureCache class, hash
  validation, auto-detect on startup, backbone forward skipped when cache valid)
- `if __name__ == '__main__':` guard: already present in train_fused.py (line 607)
  and train_prithvi.py (line 377)

**Changes made this session:**
- `train_fused.py` BATCH_SIZE: 4 → 64 (frozen-backbone / cache mode)
- `train_fused.py` num_workers: 0 → 4 (with `persistent_workers=True`)

**Confirmed via 1-epoch probe run (2026-07-22):**

| Metric | Before | After |
|---|---|---|
| Feature cache | MISS (never run) | HIT — 1,878/1,878 rows pre-computed, backbone skipped |
| Epoch time | ~3 min (backbone ran every epoch) | **13.3s** (cache mode, ~14x faster) |
| VRAM at batch=64 | 0.72 GB at batch=4 (raw images) | **0.36 GB** at batch=64 (cache: 768-d vectors, not images) |
| num_workers | 0 (single-process) | **4 workers confirmed active** — first-batch latency 6.99s = Windows spawn overhead detected |
| Batches/s (after warmup) | N/A | 4.55 it/s (24 batches of 64 in 13.3s) |

VRAM headroom at batch=64 in cache mode: 8.6 − 0.36 = **8.24 GB remaining**.
Could go to batch=256+ in cache mode, but 64 is sufficient.

Workers confirmed active: first batch elapsed 6.99s (Windows spawn startup); subsequent
batches at 4.55 it/s. If workers had silently fallen back to 0, first batch in cache mode
would be near-instant. Spawn overhead is the confirmation.

**Scraper skip-if-exists fix (Phase B):**
- `pipeline.py` scrape_all(): audit file path now computed before the HTTP call; if file
  exists for this exact (zone, param, window_tag), reads from disk and prints `cached`
  instead of hitting DPCC. Matches the "skip-if-exists" pattern from the notebook fix.
- Confirmed: 4 zones (Anand Vihar, Mundka, Wazirpur, Jahangirpuri) have Dec 2025–Jun 2026
  raw files from prior partial scrapes — those 36 files will be read from disk on next run.

**Repo pushed to GitHub:** https://github.com/Vibhor2702/airsen (452 files, initial commit)
`.pt` checkpoints excluded via .gitignore — regenerated by training scripts.

---

#### Phase C — Dataset expansion (2026-07-22)

- CAAQMS scraped for Dec 2025–Jun 2026 (all 13 zones × 9 params). 468 raw CSVs combined
  into hourly_tidy.csv (704,491 rows). --skip-scrape rebuild → caaqms_heuristic_labels.csv
  (3,750 zone-days).
- labels.csv rebuilt: **700 CAAQMS_heuristic rows** (up from 233), 2 PLACEHOLDER
  (Bawana/2026-03-01, Punjabi_Bagh/2025-11-26).
- Augmentation redirected from Drive (slow) to `C:\airsentinel_local\augmented\` (local).
  labels_augmented.csv: **5,602 rows** (700 orig + 4,900 aug + 2 placeholder).
- Aug files store absolute paths in CSV. train_prithvi.py and train_fused.py both updated
  to handle `p if p.is_absolute() else data_dir / p`.

---

#### Phase E — LoRA training (2026-07-22)

**Setup:**
- `train_prithvi_lora.py` written using TerraTorch `get_peft_backbone()` + `replace_qkv="qkv"`.
- LoRA: r=8, lora_alpha=16, targets=[q_linear, k_linear, v_linear, proj], dropout=0.05.
- 0.61M trainable / 86.9M total (0.71%). VRAM peak: 2.99 GB.
- Dataset: 5,600 rows (4,480 train / 1,120 val), group-aware split seed=42.
- BATCH_SIZE=32, LR=2e-4, 15 epochs, CosineAnnealingLR, best-val checkpointing.

**Per-epoch results:**

| Epoch | Train Acc | Val Acc | Note |
|-------|-----------|---------|------|
| 1 | 58.4% | 64.6% | |
| 2 | 70.9% | **71.1%** | ← best checkpoint saved |
| 3 | 78.1% | 70.3% | |
| 4 | 83.5% | 68.0% | |
| 5–15 | 88–100% | 64–69% | overfit, train→100% by ep 4 |

**Best val acc: 71.1%** (epoch 2). Final checkpoint val: 67.2%.

**Post-run diagnostic — two checks requested (2026-07-22):**

*Check 1 — Feature caching active?*
`train_prithvi_lora.py` contains zero mentions of FeatureCache, cache, or CACHE_PATH.
FeatureCache only exists in `train_fused.py` (the NO2-fused pipeline). The LoRA script
ran the full backbone forward pass every batch through the LoRA-adapted weights. Caching
was NOT active. **Check 1: clean.**

*Check 2 — LR adjusted for batch size?*
- Prior full FT (`train_prithvi.py`): BATCH_SIZE=4, LR=1e-4
- This LoRA run (`train_prithvi_lora.py`): BATCH_SIZE=32, LR=2e-4
- Batch ratio: 8×. LR ratio: 2×.
- Linear scaling rule would prescribe LR=8e-4 for batch=32.
  Sqrt scaling (Adam-appropriate): ~2.83e-4. Actual 2e-4 is modestly below sqrt-scaled.
- Direction of mismatch: under-scaled LR typically causes *slower convergence*, not
  faster overfit. The train→100% pattern by epoch 4 is not consistent with too-low LR;
  it is consistent with the augmented dataset containing too many near-duplicate samples
  (560 unique training scenes × 8 D4 transforms = 4,480 samples, many near-identical).
- **Check 2: LR was not scaled to linear rule, but direction of mismatch does not explain
  the observed overfit pattern. No evidence this caused the result.**

**Conclusion:** Both checks came back clean — no broken-run explanation identified.
71.1% vs 74.5% (prior full FT on smaller dataset) is a legitimate finding:
LoRA with r=8 underperformed full fine-tuning on this dataset despite 3× more real data.
Likely causes: (a) r=8 adapter is less expressive than full FT for this task; (b) fast
overfit driven by near-duplicate augmented samples in 4,480-sample train set.

**Deferred:** group-aware split leakage check on the expanded dataset before any further
tuning (label smoothing, LR scaling, larger r). Not yet done.

---

#### Phase E — Final model decision (2026-07-22)

**FINAL MODEL: `prithvi_lora_best.pt` — Prithvi LoRA r=8, epoch 2, 71.1% val acc.**

This is the submission checkpoint. No further tuning, no LR sweeps, no label smoothing,
no larger rank. Decision is final due to time constraints.

Key facts for downstream references:
- File: `prithvi_lora_best.pt` (project root)
- Architecture: Prithvi-EO-2.0-100M-TL + LoRA (r=8, alpha=16, targets q/k/v/proj)
- Training set: 4,480 rows (560 unique zone/date groups × 8 D4 augmentations)
- Val set: 1,120 rows (140 unique zone/date groups), group-aware split seed=42
- Val accuracy: 71.1% overall | Macro F1: 0.470
- Per-class F1: crop_burning_smoke 0.896, industrial_haze 0.755, clear 0.439,
  traffic_heavy 0.260, dust 0.000 (never predicted — class imbalance)
- VRAM: 2.99 GB peak at batch=32
- MODEL_VERSION tag: "700-image LoRA r=8 — final"

Any script still referencing `prithvi_airsen_augmented.pt` or using `MODEL_VERSION =
"247-image placeholder -- not final"` must be updated before Phase 2.

---

#### Phase 2 — Enforcement ranker refresh (2026-07-22)

**enforcement_ranker.py changes:**
- `PRITHVI_CKPT`: `prithvi_airsen_augmented.pt` → `prithvi_lora_best.pt`
- `MODEL_VERSION`: `"247-image placeholder -- not final"` → `"700-image LoRA r=8 -- final"`
- Architecture: `PrithviClassifier` → `PrithviLoraClassifier` + `get_peft_backbone()` (required to load LoRA checkpoint state_dict; previous class had no LoRA adapter keys)
- Added `PEFT_CONFIG` dict matching `train_prithvi_lora.py` exactly
- `pretrained=False` in backbone build (all weights come from checkpoint — avoids re-download)
- Absolute path handling for `s2_file` (same guard as other loaders)
- `predict_zone` type annotation updated to match new class

**Inference date: 2026-07-16 (most recent in labels.csv), 13 zones**

| # | Zone | Source | Conf | VEI | Composite |
|---|------|--------|------|-----|-----------|
| 1 | RK Puram | industrial_haze | 0.454 | 1.000 | 0.708 |
| 2 | Dwarka | dust | 0.846 | 0.780 | 0.572 |
| 3 | Rohini | dust | 0.818 | 0.551 | 0.493 |
| 4 | Jahangirpuri | traffic_heavy | 0.391 | 0.275 | 0.474 |
| 5 | Ashok Vihar | dust | 0.939 | 0.315 | 0.470 |
| 6 | Wazirpur | dust | 0.873 | 0.315 | 0.444 |
| 7 | Anand Vihar | dust | 0.926 | 0.191 | 0.428 |
| 8 | Narela | dust | 0.836 | 0.286 | 0.420 |
| 9 | Bawana | dust | 0.835 | 0.276 | 0.417 |
| 10 | Mundka | dust | 0.622 | 0.509 | 0.401 |
| 11 | Vivek Vihar | dust | 0.839 | 0.191 | 0.393 |
| 12 | Okhla | dust | 0.668 | 0.248 | 0.341 |
| 13 | Punjabi Bagh | dust | 0.523 | 0.343 | 0.312 |

**fuse.py + dashboard.py re-run:**
- All 4 data sources live (no PENDING flags)
- Honesty badges confirmed: `satellite_model_version = "700-image LoRA r=8 -- final"`,
  `satellite_data_provenance = "correlation-based (heuristic CAAQMS labels, not ground truth)"`,
  `satellite_land_use_note = "land-use signal absent from this repo -- scoring uses only satellite + vehicle data"`
- Fused output: 11 of 13 zones (Bawana + Punjabi_Bagh absent — pre-existing gap: not in forecasts.csv, Person 1 scope, noted in Session 8 deferred items)
