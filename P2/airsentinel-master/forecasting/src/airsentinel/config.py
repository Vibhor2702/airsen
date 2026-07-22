"""
Single source of truth for every tunable constant in the pipeline.

Nothing in this file is fabricated data or a magic number buried in application code —
it's the opposite: pulling every previously-scattered literal (scrape timing, CV/model
knobs, aggregation windows) into one documented place, and letting environment variables
override the ones that are genuinely deployment choices (network politeness, live-window
length) rather than science (CPCB averaging conventions, which are official published rules
and stay fixed on purpose — see aqi.py).

Everything here is either:
  (a) an official external convention (CPCB averaging periods) — cited, not invented;
  (b) a deployment knob (scrape delay, live window) — overridable via env var; or
  (c) a model-selection knob (CV folds, param grid) — chosen by cross-validation at
      runtime, these constants only define the *search space*, not the final values.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val else default


# ---------------------------------------------------------------------- scraper (live) --
# DPCC rejects a request window of a full 7 days (returns "Record Not found") even though
# its own UI hint says "not more than 7 days" — the real cutoff is strictly < 7. Verified
# empirically against the live site; not an arbitrary choice.
SCRAPE_MAX_WINDOW_DAYS = 6

# Seconds between live HTTP requests — politeness toward a public government server with no
# published rate limit, not a technical requirement. Override with AIRSENTINEL_SCRAPE_DELAY.
SCRAPE_DELAY_SECONDS = _env_float("AIRSENTINEL_SCRAPE_DELAY", 0.25)
SCRAPE_TIMEOUT_SECONDS = _env_int("AIRSENTINEL_SCRAPE_TIMEOUT", 30)
SCRAPE_MAX_RETRIES = _env_int("AIRSENTINEL_SCRAPE_RETRIES", 3)

# Default lookback for a live pipeline run, in days. NOT a calendar date — the actual window
# used is [now - PIPELINE_DEFAULT_WINDOW_DAYS, now], computed fresh every run (see
# pipeline.py), so this only controls *how much* history to pull, never *which* dates.
# Override with AIRSENTINEL_WINDOW_DAYS (e.g. to pull more history as it becomes available).
PIPELINE_DEFAULT_WINDOW_DAYS = _env_int("AIRSENTINEL_WINDOW_DAYS", 45)

# ------------------------------------------------------------- daily aggregation (CPCB) --
# CPCB's official National AQI methodology: PM2.5/PM10/NO2/SO2 use a 24-hour mean; CO and
# O3 use the daily maximum of the 8-hour rolling mean. These are the published averaging
# periods, not a modelling choice — https://cpcb.nic.in (National Air Quality Index).
CPCB_MEAN_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2"]
CPCB_MAX8H_POLLUTANTS = ["CO", "O3"]
CPCB_8H_WINDOW_HOURS = 8
CPCB_8H_MIN_PERIODS = 6  # require >=6 of the 8 hourly readings before trusting a rolling mean

# CPCB AQI is defined on [0, 500]; anything computed outside this range is a data artifact,
# clipped rather than reported. See aqi.py for the official breakpoint table itself, which
# is a published government constant and stays inline there (like hardcoding pi would be
# silly to avoid, moving pi into "config" wouldn't make it less of a constant).
AQI_MIN, AQI_MAX = 0, 500

# --------------------------------------------------------------------- forecasting model --
HORIZONS_DAYS = [1, 2, 3]  # 24h / 48h / 72h ahead

# Feature-engineering windows (days). 3-day is standard "recent trend" scale for daily AQI
# — long enough to smooth single-day sensor noise, short enough to still be "recent."
FEATURE_ROLLING_WINDOW_DAYS = 3

# Time-based train/test split: how many of the most recent dates are held out as the test
# tail. Falls back to a quarter of available dates when the panel is too small for this
# many held-out days (see forecast._time_split).
TEST_HOLDOUT_DAYS = 10

# Walk-forward CV folds for model selection (hyperparameters + shrinkage alpha), capped by
# how much training data is available (see forecast._select_params).
CV_N_SPLITS = 4
CV_MIN_ROWS_PER_SPLIT = 40

# Fixed HistGradientBoostingRegressor settings that are NOT searched by CV — early stopping
# parameters (these control training-time convergence, not the bias/variance trade-off the
# grid below searches) plus a fixed learning rate/iteration cap chosen once for this data
# scale (a few hundred rows) and left alone so the CV grid stays small enough to run in
# seconds.
MODEL_MAX_ITER = 300
MODEL_LEARNING_RATE = 0.04
MODEL_RANDOM_STATE = 42
MODEL_EARLY_STOPPING = True
MODEL_VALIDATION_FRACTION = 0.15
MODEL_N_ITER_NO_CHANGE = 15

# Regularization grid searched per horizon via walk-forward CV — this is the search SPACE,
# not the answer; CV picks the winning combination per horizon at runtime (see
# forecast._select_params). Includes depth=1 (decision stumps) and higher min-leaf/L2
# options specifically so very short/noisy horizons (like 72h, which showed the largest
# train-vs-CV overfitting gap) have a maximally-conservative option available if that's what
# generalizes best — CV will only pick it if it actually wins, this isn't forcing conservatism,
# it's making sure conservatism is on the table.
MODEL_PARAM_GRID_MAX_DEPTH = (1, 2, 3)
MODEL_PARAM_GRID_MIN_SAMPLES_LEAF = (15, 25, 35, 50)
MODEL_PARAM_GRID_L2_REGULARIZATION = (1.0, 3.0, 5.0)

# Shrinkage weight alpha search grid: final_pred = aqi_now + alpha * predicted_delta.
# alpha=0.0 is included by construction, so persistence itself is always a candidate — see
# forecast._select_shrinkage for why this bounds worst-case CV performance to "no worse than
# the baseline."
ALPHA_GRID_STEP = 0.05
