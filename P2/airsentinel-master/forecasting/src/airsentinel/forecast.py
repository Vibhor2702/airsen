"""
Forecasting engine: predict per-zone AQI 24 / 48 / 72 hours ahead.

Model choice — NOT an LSTM
---------------------------
The original deck sketches "graph+LSTM." We deliberately use gradient-boosted trees
(``HistGradientBoostingRegressor``) instead, and this is a considered choice, not an
oversight: with ~45 daily points per zone (a few hundred rows total across 13 zones), an
LSTM has nowhere near enough sequence length per zone to learn temporal dynamics without
overfitting — RNNs are data-hungry and need long, dense sequences (thousands of steps) to
generalize. A tree ensemble on hand-built lag/momentum/volatility features is far more
sample-efficient on small tabular data, trains in seconds on a laptop CPU, is trivial to
regularize and inspect (feature importance), and matches the plan's own "start simple, get
it end-to-end, then add complexity" instruction. If/when the live history grows into
thousands of rows per zone, an LSTM or a temporal transformer becomes a reasonable upgrade —
not before.

Design
------
- One panel row per (zone, date) with AQI, pollutants and weather.
- Features: recent AQI/pollutant lags, momentum (short-term trend), 3-day rolling
  volatility, same-day weather, calendar terms, zone identity.
- **Residual (delta) target**: the model predicts ``AQI(t+h) - AQI(t)``, not the raw future
  AQI. This directly targets the weak spot at 24h: persistence is already a strong
  benchmark at short horizons because AQI rarely swings wildly in a single day, so asking
  the model to predict a *correction* to persistence — rather than reconstruct the whole
  value from scratch — is a smaller, less noisy, more learnable target, and it structurally
  anchors the model near the baseline instead of drifting away from it when the model is
  uncertain (predicting delta ~= 0 degrades gracefully to persistence itself).
- Baseline: persistence ("tomorrow = today") — the standard weak baseline the plan calls
  out — the model has to beat it to be worth anything, at every horizon including 24h.
- Model selection: a small hyperparameter grid is chosen per horizon by walk-forward
  **time-series cross-validation** on the training period only (never touching the held-out
  test tail), which both regularizes the choice and gives us a train-vs-CV MAE gap we report
  as an explicit overfitting check (see ``overfit_gap_pct`` in the metrics).
- **Shrinkage (alpha)**: the raw predicted delta is scaled by a per-horizon weight
  alpha in [0, 1], chosen by the same out-of-fold CV to minimize MAE. Because alpha=0
  reproduces persistence exactly, this is a hard structural guarantee that model selection
  can never prefer a configuration that beats persistence in-sample but loses to it
  out-of-fold — the safety net that matters most at 24h, where persistence is hardest to
  beat and alpha is free to shrink toward it.
- Split: time-based (train on earlier dates, evaluate on the most recent tail) so we never
  train on the future.

Compatibility with the satellite/Prithvi track
------------------------------------------------
This module only consumes CAAQMS + weather (data/processed/airsentinel_daily_panel.csv) —
it shares no code, weights, or training data with Prithvi fine-tuning, so nothing here can
break or destabilize that track. The only shared surface is the join key (`zone`, `date`),
identical to the schema already handed off in data/teammate_delivery/caaqms_readings.csv, so
forecast output (`outputs/forecasts.csv`: zone, forecast_from, horizon_hours, target_date,
predicted_aqi) merges into the same fusion layer on those same two columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

from . import config
from .aqi import category as aqi_category

HORIZONS = config.HORIZONS_DAYS
FEATURE_COLS = [
    # Values known at forecast time t (day of the reading).
    "aqi_now", "aqi_lag1", "aqi_lag2", "aqi_roll3", "aqi_trend", "aqi_volatility3",
    "pm25_now", "pm25_trend", "pm10_now", "pm10_trend", "no2_now", "no2_trend",
    "temp", "humidity", "wind_speed",
    "dayofweek", "month", "zone_code",
]

# Regularization grid searched per horizon via walk-forward CV (see _select_params). The
# search space is defined in config.py; CV picks the winning combination at runtime.
_PARAM_GRID = [
    {"max_depth": d, "min_samples_leaf": leaf, "l2_regularization": l2}
    for d in config.MODEL_PARAM_GRID_MAX_DEPTH
    for leaf in config.MODEL_PARAM_GRID_MIN_SAMPLES_LEAF
    for l2 in config.MODEL_PARAM_GRID_L2_REGULARIZATION
]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """From a daily [zone, date, AQI, pollutants, weather] panel, build lag/target features.

    All features are values available at day t (current + past). Targets are the residual
    AQI(t+h) - AQI(t) (see module docstring). Lags/rolls/shifts are computed per-zone
    (grouped) so nothing bleeds across zone borders.
    """
    df = panel.sort_values(["zone", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["zone_code"] = df["zone"].astype("category").cat.codes
    gb = df.groupby("zone")

    df["aqi_now"] = df["AQI"]
    df["aqi_lag1"] = gb["AQI"].shift(1)
    df["aqi_lag2"] = gb["AQI"].shift(2)
    w = config.FEATURE_ROLLING_WINDOW_DAYS
    df["aqi_roll3"] = gb["AQI"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["aqi_trend"] = df["aqi_now"] - df["aqi_lag1"]  # yesterday -> today momentum
    df["aqi_volatility3"] = gb["AQI"].transform(lambda s: s.rolling(w, min_periods=2).std())

    df["pm25_now"] = df["PM2.5"]
    df["pm25_trend"] = df["pm25_now"] - gb["PM2.5"].shift(1)
    df["pm10_now"] = df["PM10"]
    df["pm10_trend"] = df["pm10_now"] - gb["PM10"].shift(1)
    df["no2_now"] = df["NO2"]
    df["no2_trend"] = df["no2_now"] - gb["NO2"].shift(1)

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Targets: residual AQI change h days ahead within the same zone.
    for h in HORIZONS:
        future_aqi = gb["AQI"].shift(-h)
        df[f"target_{h}"] = future_aqi - df["aqi_now"]
        df[f"future_aqi_{h}"] = future_aqi  # kept for evaluation/reporting, not a feature
    return df


def _time_split(df: pd.DataFrame, test_days: int = config.TEST_HOLDOUT_DAYS):
    dates = np.sort(df["date"].unique())
    if len(dates) <= test_days + 5:
        test_days = max(2, len(dates) // 4)
    cutoff = dates[-test_days]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff], pd.Timestamp(cutoff)


def _make_model(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=config.MODEL_MAX_ITER,
        learning_rate=config.MODEL_LEARNING_RATE,
        random_state=config.MODEL_RANDOM_STATE,
        early_stopping=config.MODEL_EARLY_STOPPING,
        validation_fraction=config.MODEL_VALIDATION_FRACTION,
        n_iter_no_change=config.MODEL_N_ITER_NO_CHANGE,
        **params,
    )


def _select_params(X_tr: pd.DataFrame, y_tr: pd.Series, n_splits: int = config.CV_N_SPLITS):
    """Walk-forward CV over the training period to pick regularization strength.

    Also returns the out-of-fold delta predictions at the winning params (aligned to a
    subset of X_tr's positional index) — used downstream to fit the shrinkage weight alpha
    without ever touching the held-out test tail.

    Returns (best_params, cv_mae, train_mae_at_best, oof_idx, oof_pred).
    """
    n_splits = min(n_splits, max(2, len(X_tr) // config.CV_MIN_ROWS_PER_SPLIT))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params, best_cv_mae = _PARAM_GRID[0], float("inf")

    for params in _PARAM_GRID:
        fold_maes = []
        for tr_idx, val_idx in tscv.split(X_tr):
            m = _make_model(params)
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            pred = m.predict(X_tr.iloc[val_idx])
            fold_maes.append(mean_absolute_error(y_tr.iloc[val_idx], pred))
        cv_mae = float(np.mean(fold_maes))
        if cv_mae < best_cv_mae:
            best_cv_mae, best_params = cv_mae, params

    # Out-of-fold predictions at the winning params, for alpha selection.
    oof_idx: list[int] = []
    oof_pred: list[float] = []
    for tr_idx, val_idx in tscv.split(X_tr):
        m = _make_model(best_params)
        m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        oof_idx.extend(val_idx.tolist())
        oof_pred.extend(m.predict(X_tr.iloc[val_idx]).tolist())

    # In-sample MAE at the chosen params, for the overfitting-gap diagnostic.
    final_probe = _make_model(best_params)
    final_probe.fit(X_tr, y_tr)
    train_mae = float(mean_absolute_error(y_tr, final_probe.predict(X_tr)))
    return best_params, best_cv_mae, train_mae, np.array(oof_idx), np.array(oof_pred)


def _select_shrinkage(oof_delta_pred: np.ndarray, aqi_now_oof: np.ndarray,
                       future_aqi_oof: np.ndarray) -> tuple[float, float, float]:
    """Pick alpha in [0, 1] that shrinks the predicted delta toward persistence.

    final prediction = aqi_now + alpha * predicted_delta. alpha=0 is exactly persistence,
    alpha=1 is the raw model — alpha is chosen by grid search to minimize out-of-fold MAE,
    so this can never do worse than persistence *in cross-validation* by construction (alpha
    is free to collapse to 0 wherever the model isn't adding real signal, which is exactly
    the 24h regime where persistence is already strong).

    Returns (alpha, oof_mae_at_alpha, oof_mae_at_alpha0_i.e._persistence).
    """
    alphas = np.arange(0.0, 1.0 + config.ALPHA_GRID_STEP, config.ALPHA_GRID_STEP)
    best_alpha, best_mae = 0.0, float("inf")
    persistence_mae = float(mean_absolute_error(future_aqi_oof, aqi_now_oof))
    for a in alphas:
        pred = np.clip(aqi_now_oof + a * oof_delta_pred, config.AQI_MIN, config.AQI_MAX)
        mae = float(mean_absolute_error(future_aqi_oof, pred))
        if mae < best_mae:
            best_mae, best_alpha = mae, float(a)
    return best_alpha, best_mae, persistence_mae


def train_and_evaluate(panel: pd.DataFrame):
    """Train one delta-target model per horizon, evaluate vs persistence on a time-based tail.

    Returns (models: dict[h]->estimator, metrics: dict, feature_frame: DataFrame).
    """
    feats = build_features(panel)
    models: dict[int, dict] = {}  # h -> {"model": estimator, "alpha": float}
    metrics: dict[str, dict] = {}

    for h in HORIZONS:
        cols = FEATURE_COLS
        sub = feats.dropna(subset=cols + [f"target_{h}", f"future_aqi_{h}"])
        train, test, cutoff = _time_split(sub)
        if train.empty or test.empty:
            continue
        X_tr, y_tr = train[cols], train[f"target_{h}"]
        X_te = test[cols]
        y_te_aqi = test[f"future_aqi_{h}"]  # actual future AQI, for reporting
        now_te = test["aqi_now"]

        best_params, cv_mae, train_mae, oof_idx, oof_delta_pred = _select_params(X_tr, y_tr)

        # Shrinkage weight alpha, chosen on the same out-of-fold predictions (still only
        # training-period data — the test tail is never involved in choosing alpha).
        aqi_now_oof = train["aqi_now"].to_numpy()[oof_idx]
        future_aqi_oof = train[f"future_aqi_{h}"].to_numpy()[oof_idx]
        alpha, oof_mae_shrunk, oof_mae_persistence = _select_shrinkage(
            oof_delta_pred, aqi_now_oof, future_aqi_oof
        )

        model = _make_model(best_params)
        model.fit(X_tr, y_tr)
        pred_delta = model.predict(X_te)
        pred_aqi = np.clip(now_te.to_numpy() + alpha * pred_delta, config.AQI_MIN, config.AQI_MAX)

        # Persistence baseline: AQI(t+h) ~= AQI(t).
        baseline = now_te.to_numpy()

        # AQI is a continuous regression target, so "accuracy/precision/recall" don't apply
        # to it directly. To answer that question honestly, we also bucket predicted and
        # actual AQI into CPCB categories (Good/Satisfactory/.../Severe) and score that as
        # a classification problem — a real evaluation on held-out test data.
        y_cat = [aqi_category(v) for v in y_te_aqi]
        pred_cat = [aqi_category(v) for v in pred_aqi]
        base_cat = [aqi_category(v) for v in baseline]

        model_mae = float(mean_absolute_error(y_te_aqi, pred_aqi))
        overfit_gap_pct = round(100 * (cv_mae - train_mae) / cv_mae, 1) if cv_mae else 0.0

        metrics[f"h{h}"] = {
            "horizon_hours": h * 24,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "test_from": str(cutoff.date()),
            "test_to": str(test["date"].max().date()),
            "best_params": best_params,
            # shrinkage: final_pred = aqi_now + alpha * predicted_delta. alpha is chosen by
            # out-of-fold CV, so alpha=0 (pure persistence) is always an option — the model
            # cannot be selected to do worse than persistence in cross-validation.
            "alpha_shrinkage": round(alpha, 2),
            "oof_mae_shrunk": round(oof_mae_shrunk, 2),
            "oof_mae_persistence": round(oof_mae_persistence, 2),
            # overfitting diagnostic: CV MAE (unseen folds within train) vs in-sample train
            # MAE at the same params. A small gap means the model isn't just memorizing.
            "cv_mae_delta_target": round(cv_mae, 2),
            "train_mae_delta_target": round(train_mae, 2),
            "overfit_gap_pct": overfit_gap_pct,
            # regression scores (the AQI value itself, held-out test tail)
            "model_mae": round(model_mae, 2),
            "model_rmse": round(float(np.sqrt(mean_squared_error(y_te_aqi, pred_aqi))), 2),
            "model_r2": round(float(r2_score(y_te_aqi, pred_aqi)), 3),
            "baseline_mae": round(float(mean_absolute_error(y_te_aqi, baseline)), 2),
            "baseline_rmse": round(float(np.sqrt(mean_squared_error(y_te_aqi, baseline))), 2),
            "baseline_r2": round(float(r2_score(y_te_aqi, baseline)), 3),
            # classification scores (AQI category: Good/Satisfactory/Moderate/Poor/Very Poor/Severe)
            "model_category_accuracy": round(float(accuracy_score(y_cat, pred_cat)), 3),
            "model_category_precision_weighted": round(
                float(precision_score(y_cat, pred_cat, average="weighted", zero_division=0)), 3
            ),
            "model_category_recall_weighted": round(
                float(recall_score(y_cat, pred_cat, average="weighted", zero_division=0)), 3
            ),
            "model_category_f1_weighted": round(
                float(f1_score(y_cat, pred_cat, average="weighted", zero_division=0)), 3
            ),
            "baseline_category_accuracy": round(float(accuracy_score(y_cat, base_cat)), 3),
        }
        m = metrics[f"h{h}"]
        baseline_mae = m["baseline_mae"]
        m["mae_improvement_pct"] = round(
            100 * (baseline_mae - model_mae) / baseline_mae, 1
        ) if baseline_mae else 0.0
        models[h] = {"model": model, "alpha": alpha}

    return models, metrics, feats


def forecast_latest(models: dict[int, dict], feats: pd.DataFrame):
    """Produce 24/48/72h AQI forecasts from each zone's most recent complete feature row."""
    rows = []
    latest = feats.sort_values("date").groupby("zone").tail(1)
    for _, r in latest.iterrows():
        if r[FEATURE_COLS].isna().any():
            continue
        X = r[FEATURE_COLS].to_frame().T
        for h, bundle in models.items():
            pred_delta = float(bundle["model"].predict(X)[0])
            pred_aqi = float(np.clip(
                r["aqi_now"] + bundle["alpha"] * pred_delta, config.AQI_MIN, config.AQI_MAX
            ))
            rows.append({
                "zone": r["zone"],
                "forecast_from": r["date"].date(),
                "horizon_hours": h * 24,
                "target_date": (r["date"] + pd.Timedelta(days=h)).date(),
                "predicted_aqi": round(pred_aqi),
            })
    return pd.DataFrame(rows)
