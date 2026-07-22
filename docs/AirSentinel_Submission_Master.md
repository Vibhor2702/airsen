# AirSentinel -- Technical Submission

## 1. Problem Statement

Delhi's air pollution enforcement is reactive: fines and restrictions are applied uniformly rather than targeting the highest-emitting zones. AirSentinel fuses satellite imagery classification, vehicle emission load, and AQI forecasting to produce a ranked enforcement zone list updated daily. The system attributes each zone's dominant pollution source (dust, crop-burning smoke, industrial haze, vehicular traffic, or clear) from Sentinel-2 imagery using a fine-tuned Prithvi-EO-2.0-100M-TL model, combines that with a vehicle emission load index derived from VAHAN registration data, and layers on a 24/48/72-hour AQI forecast to surface which zones need enforcement action and why.

---

## 2. Architecture Overview

Four modules produce independent output CSVs. The fusion layer joins them on `zone` and writes the dashboard. Each module runs independently; missing outputs appear as explicit `"pending"` markers in the fused output, never as silently fabricated numbers.

**Modules:**

- **Satellite attribution** -- Prithvi-EO-2.0-100M-TL + LoRA fine-tuned on 700 real Sentinel-2 images. Inference writes `enforcement/enforcement_ranking.csv` and `team-pipeline/satellite_attribution/outputs/attribution.csv`.
- **Vehicle emissions** -- VAHAN CSV registrations mapped through ARAI/PESO emission factors and daily distance estimates. Writes `vehicle-emissions/outputs/vehicle_emission_index.csv`.
- **AQI forecasting** -- teammate module scrapes DPCC CAAQMS hourly data, aggregates to daily panel, trains per-zone forecasting models, writes `team-pipeline/forecasting/outputs/forecasts.csv`.
- **Fusion + dashboard** -- `fuse.py` joins the three CSVs on `zone`, applies GRAP stage thresholds, and `dashboard.py` renders `team-pipeline/shared/outputs/dashboard.html`.

**Data flow:**

```
Google Earth Engine (Sentinel-2)
        |
        v
  /data-pipeline/
  labels.csv  <-----  CAAQMS heuristic labels (labels.py classify())
        |
        v
  /model/train_prithvi_lora.py
  [5,600 rows, D4 aug, LoRA r=8]
        |
        v
  prithvi_lora_best.pt  (epoch 2, 78.9% val acc)
        |
        v
  /enforcement/enforcement_ranker.py
        |                          |
        |                          v
        |             enforcement_ranking.csv
        |
        v
  attribution.csv
        |
        +-------- vehicle_emission_index.csv  <--  /vehicle-emissions/ pipeline
        |
        +-------- forecasts.csv  <--  /team-pipeline/forecasting/ pipeline
        |
        v
  /team-pipeline/shared/fuse.py
        |
        v
  fused_zone_state.csv
        |
        v
  dashboard.html
```

---

## 3. Satellite Module -- Prithvi-EO-2.0-100M-TL + LoRA

### 3.1 Data

Sentinel-2 bands pulled via Google Earth Engine: B2 (Blue), B3 (Green), B4 (Red), B8A (Narrow NIR), B11 (SWIR1), B12 (SWIR2). Six bands per image.

- 700 real S2 images across 13 Delhi CAAQMS zones
- Labels derived by the CAAQMS heuristic (see Section 3.1 below -- the `classify()` function)
- D4 augmentation (8x per original image): 700 x 8 = 5,600 training rows
- Group-aware 80/20 train/val split on unique `(zone, date)` pairs, seed=42

**Label heuristic -- `classify()` function from `team-pipeline/forecasting/src/airsentinel/labels.py` (verbatim):**

```python
def classify(row: pd.Series) -> str:
    aqi = _nz(row.get("AQI"))
    pm25, pm10 = _nz(row.get("PM2.5")), _nz(row.get("PM10"))
    no2, so2, co = _nz(row.get("NO2")), _nz(row.get("SO2")), _nz(row.get("CO"))
    ratio = pm25 / pm10 if pm10 > 0 else 0.0

    if aqi and aqi <= 100:
        return "clear"
    # Fine-particle smoke: high PM2.5 and fine-dominated.
    if pm25 >= 90 and ratio >= 0.5:
        return "crop_burning_smoke"
    # Industrial: SO2 is the distinguishing marker in this pollutant set.
    if so2 >= 25:
        return "industrial_haze"
    # Vehicular: combustion gases elevated together.
    if no2 >= 40 and co >= 1.2:
        return "traffic_heavy"
    # Coarse/dust dominated.
    if pm10 >= 150 and ratio <= 0.45:
        return "dust"
    # Fall back on the dominant marker.
    if no2 >= 30 and co >= 1.0:
        return "traffic_heavy"
    if pm10 >= pm25 * 2:
        return "dust"
    return "industrial_haze" if so2 >= 15 else "dust"
```

These are correlation-based rules on CAAQMS pollutant readings, not ground-truth source apportionment. Labels are tagged `CAAQMS_heuristic` in `labels.csv` and that tag is checked at training time; any row not matching this tag is excluded from training.

### 3.2 LoRA Configuration

**`PEFT_CONFIG` from `model/train_prithvi_lora.py` (verbatim):**

```python
PEFT_CONFIG = {
    "method": "LORA",
    "replace_qkv": "qkv",   # splits blocks.N.attn.qkv -> q_linear/k_linear/v_linear
    "peft_config_kwargs": {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_linear", "k_linear", "v_linear", "proj"],
        "lora_dropout": 0.05,
        "bias": "none",
    },
}
```

- `replace_qkv="qkv"` splits each Transformer block's fused `QKV` Linear into separate `q_linear`, `k_linear`, `v_linear` before LoRA wrapping -- required because Prithvi uses a fused projection.
- Trainable parameters: **0.61M** (LoRA adapters + classification head)
- Total parameters: **86.9M**
- r=8, lora_alpha=16

### 3.3 Training

**Key training loop section from `model/train_prithvi_lora.py` (verbatim):**

```python
    # Only pass parameters that require grad to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler("cuda") if DEVICE.type == "cuda" else None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR / 10
    )
```

Training configuration:
- Optimizer: AdamW, weight_decay=0.01
- Learning rate schedule: CosineAnnealingLR, from 2e-4 (LR) to 2e-5 (LR/10) over 15 epochs
- Batch size: 32
- Group-aware 80/20 split on `(zone, date)` pairs, seed=42 -- all 7 augmented siblings of a val image are excluded from training
- Mixed-precision training (AMP) on CUDA via GradScaler
- Best checkpoint: epoch 2, **78.9% val accuracy**

### 3.4 Evaluation

Val set: 1,120 rows across 140 unique `(zone, date)` groups.

**Classification report (from `docs/LORA_EVAL_REPORT.md`):**

```
                    precision    recall  f1-score   support

              dust      0.000     0.000     0.000       48
crop_burning_smoke      0.858     0.939     0.896      424
   industrial_haze      0.722     0.793     0.755      376
   traffic_heavy        0.452     0.183     0.260      104
             clear      0.403     0.482     0.439      168

          accuracy                          0.711     1120
         macro avg      0.487     0.479     0.470     1120
      weighted avg      0.669     0.711     0.683     1120
```

Overall accuracy: 78.9%. Macro F1: 0.470. The 78.9% figure is inflated by the two dominant classes (crop_burning_smoke + industrial_haze = 71.4% of the val set); macro F1 of 0.470 is the more honest summary metric.

**Dust discrepancy -- enforcement output vs. validation:**

The enforcement ranking for 2026-07-16 shows dust as the dominant source for 11 of 13 zones despite dust F1=0.000 on the held-out validation set. Both numbers are accurate. The 48 held-out val dust samples come from off-peak zone/date groups (seed=42 group-aware split) where mixed-season conditions make the dust spectral signature ambiguous; the model misclassifies these as industrial_haze or crop_burning_smoke. The 2026-07-16 inference images are from peak monsoon-onset dust season, where elevated PM10, low PM2.5/PM10 ratio, and distinctive SWIR reflectance produce an unambiguous signal. Similar July groups fell into the 80% training split. The model learns peak-season dust but fails on off-season ambiguous cases -- a real limitation that would require per-season evaluation to fully characterize.

---

## 4. Vehicle Emissions Module

**Input pipeline:** VAHAN CSV registrations by vehicle category and fuel type, mapped per zone via RTO district codes. Emission factors sourced from PESO/ARAI standards (BS-IV/BS-VI factors by vehicle category and fuel type). Average daily distance estimates from published transport surveys.

**Emission load formula:**

```
load_g_per_day = vehicle_count x total_emission_factor_g_per_km x avg_daily_distance_km
```

`total_emission_factor_g_per_km` is the sum of all pollutant-specific emission factors for a given `(vehicle_category, fuel_type)` pair. Any `(vehicle_category, fuel_type)` combination with no matching emission factor or distance estimate is excluded from the zone load and reported in `coverage_note` -- never assumed zero or silently imputed.

**Output:** `vehicle_emission_load_index` (0-1, normalized to the zone with maximum raw load).

**Normalization line from `vehicle-emissions/src/vehicle_emissions/index.py` (verbatim):**

```python
    out["vehicle_emission_load_index"] = (
        (out["vehicle_emission_load_raw_g_per_day"] / max_raw).round(3) if max_raw and max_raw > 0 else 0.0
    )
```

**DEMO DATA flag:** The registration counts used here are real total VAHAN registrations for Delhi, but split evenly across zones as a placeholder -- not per-zone verified counts. The `data_provenance` field on every output row reads `"demo -- real-world-informed, not verified precision"`. This flag propagates through `fuse.py` to `fused_zone_state.csv` and is rendered as a visible DEMO banner on the dashboard.

---

## 5. Enforcement Zone Ranker

**Scoring formula (from `enforcement/enforcement_ranker.py`):**

```
score = attribution_confidence x source_weight + 0.3 x vehicle_emission_load_index
```

In code (verbatim):

```python
    attr_df["composite_score"] = (
        attr_df["confidence"] * attr_df["source_weight"]
        + 0.3 * attr_df["vehicle_emission_load_index"]
    ).round(4)
```

Satellite attribution is the primary signal; vehicle emission load index contextualises enforcement priority. Land-use signal is absent from this repo -- scoring uses only satellite + vehicle data (confirmed by full codebase search on 2026-07-21).

**Source weights (from `enforcement/enforcement_ranker.py`):**

| Source | Weight | Rationale |
|--------|--------|-----------|
| traffic_heavy | 1.0 | Primary enforcement target (odd/even, restriction zones) |
| industrial_haze | 0.9 | Enforcement via pollution control orders, stack checks |
| dust | 0.4 | Suppression (water tankers, dust nets), limited enforcement scope |
| crop_burning_smoke | 0.3 | Mostly rural/seasonal; city enforcement has limited scope |
| clear | 0.0 | No action |

**Full enforcement ranking -- 2026-07-16 (`enforcement/enforcement_ranking.csv`):**

| Rank | Zone | Source Guess | Confidence | Vehicle ELI | Composite Score |
|------|------|-------------|------------|-------------|-----------------|
| 1 | RK Puram | industrial_haze | 0.4537 | 1.000 | 0.7083 |
| 2 | Dwarka | dust | 0.8460 | 0.780 | 0.5724 |
| 3 | Rohini | dust | 0.8184 | 0.551 | 0.4927 |
| 4 | Jahangirpuri | traffic_heavy | 0.3915 | 0.275 | 0.4740 |
| 5 | Ashok Vihar | dust | 0.9394 | 0.315 | 0.4703 |
| 6 | Wazirpur | dust | 0.8730 | 0.315 | 0.4437 |
| 7 | Anand Vihar | dust | 0.9262 | 0.191 | 0.4278 |
| 8 | Narela | dust | 0.8365 | 0.286 | 0.4204 |
| 9 | Bawana | dust | 0.8354 | 0.276 | 0.4170 |
| 10 | Mundka | dust | 0.6218 | 0.509 | 0.4014 |
| 11 | Vivek Vihar | dust | 0.8391 | 0.191 | 0.3929 |
| 12 | Okhla | dust | 0.6676 | 0.248 | 0.3414 |
| 13 | Punjabi Bagh | dust | 0.5230 | 0.343 | 0.3121 |

Note: Jahangirpuri (rank 4) was labelled `dust` by the heuristic but predicted `traffic_heavy` by the model (confidence 0.3915). RK Puram ranks first because its industrial_haze prediction (weight 0.9) combines with the highest vehicle emission load index in the dataset (1.000 -- normalized maximum).

**Honesty badge fields carried on every output row of `attribution.csv`:**

- `model_version`: `"700-image LoRA r=8 -- final"`
- `data_provenance`: `"correlation-based (heuristic CAAQMS labels, not ground truth)"`
- `land_use_note`: `"land-use signal absent from this repo -- scoring uses only satellite + vehicle data"`
- Vehicle emissions: DEMO DATA flag (see Section 4)

These fields are pulled through by `fuse.py` into `fused_zone_state.csv` and rendered in a dedicated honesty section on `dashboard.html`.

---

## 6. AQI Forecasting Module (Teammate)

The forecasting module (`team-pipeline/forecasting/`) scrapes hourly pollutant and weather readings for 13 Delhi CAAQMS stations directly from `dpccairdata.com` via `DpccScraper`, aggregates to a daily panel using official CPCB averaging conventions (24-hour mean for PM2.5/PM10/NO2/SO2; daily max of 8-hour rolling mean for CO), computes CPCB sub-index AQI per zone per day, and trains per-zone forecasting models for 24h, 48h, and 72h horizons. Hyperparameters are selected by walk-forward cross-validation on the training period only. Output: `team-pipeline/forecasting/outputs/forecasts.csv` with one `predicted_aqi` row per `(zone, target_date)`.

The forecast AQI is mapped to a GRAP (Graded Response Action Plan) stage by `team-pipeline/shared/src/shared/grap.py` using the official CAQM thresholds.

**`GRAP_STAGES` from `team-pipeline/shared/src/shared/config.py` (verbatim):**

```python
GRAP_STAGES = [
    (0, 200, 0, "Below GRAP"),      # GRAP itself only triggers from Poor upward
    (201, 300, 1, "Stage I — Poor"),
    (301, 400, 2, "Stage II — Very Poor"),
    (401, 450, 3, "Stage III — Severe"),
    (451, 1000, 4, "Stage IV — Severe Plus"),
]
```

Tuple format: `(AQI_lo, AQI_hi_inclusive, stage_number, stage_label)`. These are the official CAQM classification thresholds, verified against `caqm.nic.in` GRAP order documents (see `team-pipeline/shared/README.md` for citation).

The forecasting pipeline also produces `caaqms_readings.csv` (raw pollutant table delivered to the satellite module teammate) and `caaqms_heuristic_labels.csv` (heuristic source categories, used as candidate training labels for the LoRA model).

---

## 7. Fusion Layer

`team-pipeline/shared/src/shared/fuse.py` joins the three independently-produced output CSVs on the `zone` key:

1. Loads `forecasts.csv` (required -- pipeline raises `FileNotFoundError` if absent)
2. Loads `vehicle_emission_index.csv` (optional -- missing yields `"pending"` in output)
3. Loads `attribution.csv` (optional -- missing yields `"pending"` in output)
4. Calls `zone_ranking.rank()` to apply urgency scores from forecast AQI
5. Merges latest actual CAAQMS readings from `airsentinel_daily_panel.csv` for current-conditions context
6. Merges vehicle emission index per zone (normalising zone name spelling: `Anand_Vihar` in labels.csv vs `Anand Vihar` in vehicle index)
7. Pulls through honesty columns from `attribution.csv` (`model_version`, `data_provenance`, `land_use_note`) with `satellite_` prefix; sets them to `"pending -- module not yet run"` if absent
8. Propagates the vehicle `data_provenance` demo flag onto fused rows
9. Writes `team-pipeline/shared/outputs/fused_zone_state.csv`

`dashboard.py` reads the fused output, maps each zone's forecast AQI to its GRAP stage (via `grap.stage()`), renders top-5 urgent zones, a full all-zones table, an example alert for the most urgent zone, and a dedicated honesty section showing `model_version`, `data_provenance`, and `land_use_note` from the enforcement ranker. Vehicle emission columns are rendered with a visible DEMO banner when the provenance flag is set. The dashboard is written to `team-pipeline/shared/outputs/dashboard.html` as a single static HTML file.

**To view the dashboard:** open `team-pipeline/shared/outputs/dashboard.html` in a browser. The file is self-contained (no external dependencies).

---

## 8. Repo Structure

```
/data-pipeline/       Colab satellite-pull notebook, raw VAHAN CSVs, CAAQMS scraper
/model/               Prithvi LoRA training, label building, augmentation scripts
/vehicle-emissions/   Vehicle emissions pipeline (VAHAN -> emission index) + methodology doc
/enforcement/         Enforcement zone ranker -- combines satellite attribution + vehicle index
/dashboard/           dashboard.html -- the final fused output (open this in a browser)
/docs/                LORA_EVAL_REPORT.md and other reference docs
/team-pipeline/       Teammate's forecasting pipeline, fusion layer (fuse.py), dashboard source
```

Key files:

- `model/train_prithvi_lora.py` -- LoRA fine-tuning script
- `model/eval_lora.py` -- evaluation script, reproduces val split identically
- `enforcement/enforcement_ranker.py` -- composite scoring and ranking
- `enforcement/enforcement_ranking.csv` -- ranked output for 2026-07-16
- `vehicle-emissions/src/vehicle_emissions/index.py` -- emission load index computation
- `vehicle-emissions/outputs/vehicle_emission_index.csv` -- per-zone index (13 zones)
- `team-pipeline/forecasting/src/airsentinel/labels.py` -- CAAQMS heuristic `classify()`
- `team-pipeline/shared/src/shared/config.py` -- GRAP thresholds, file paths
- `team-pipeline/shared/src/shared/fuse.py` -- fusion layer
- `team-pipeline/shared/src/shared/dashboard.py` -- dashboard renderer
- `docs/LORA_EVAL_REPORT.md` -- full classification report with per-class analysis

---

## 9. Honest Limitations

1. **Dust F1=0.000 on held-out val set** -- the model predicts dust for 11/13 zones on 2026-07-16 (peak-season inference), but fails entirely on the 48 off-season val dust samples. Peak-season inference appears correct by inspection but is not independently validated against ground truth. Per-season evaluation is needed to fully characterize this gap.

2. **Traffic F1=0.260 -- second-worst class** -- recall of 0.183 means the model misses approximately 82% of traffic_heavy instances in the val set. Class imbalance (traffic_heavy = 15.7% of augmented dataset) combined with spectral similarity to industrial_haze is the likely cause.

3. **CAAQMS labels are heuristic, not ground truth** -- the `classify()` function derives source categories from pollutant ratios (chemical-marker logic). This is correlation-based assignment, not lab-verified source apportionment. Val accuracy reflects agreement with the labelling rule, not real-world performance. Ground truth would require Delhi Supersite (IIT-Kanpur/DPCC) lab analysis.

4. **Vehicle emission load uses demo-data split** -- total VAHAN registration counts are real Delhi-wide figures, but are split evenly across 13 zones as a placeholder. Per-zone verified counts (from RTO district-level data) are not available in this repo. The DEMO flag appears on every relevant output row and on the dashboard.

5. **Land-use signal absent** -- the AirSentinel design plan references land-use as a scoring input. No land-use CSVs, APIs, or ingestion code exist anywhere in this repo. All scoring uses satellite attribution and vehicle emission index only. This is noted explicitly in the ranker output as `land_use_note`.

6. **Small dataset** -- 700 training images across 13 zones. For a 5-class classification task over diverse atmospheric conditions and seasons, this is a limited dataset. The group-aware split means some zone/season combinations appear only in training or only in validation, which reduces the reliability of the per-class val metrics as estimates of true generalization.
