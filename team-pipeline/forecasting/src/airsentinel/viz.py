"""Visualisations for the forecasting track: model-vs-baseline and a per-zone forecast."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_AQI_BANDS = [
    (0, 50, "#2ecc71", "Good"), (50, 100, "#a3d977", "Satisfactory"),
    (100, 200, "#f1c40f", "Moderate"), (200, 300, "#e67e22", "Poor"),
    (300, 400, "#e74c3c", "Very Poor"), (400, 500, "#8e44ad", "Severe"),
]


def plot_model_vs_baseline(metrics: dict, out: Path) -> None:
    horizons = [m["horizon_hours"] for m in metrics.values()]
    model = [m["model_mae"] for m in metrics.values()]
    base = [m["baseline_mae"] for m in metrics.values()]
    x = range(len(horizons))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - w / 2 for i in x], base, w, label="Persistence baseline", color="#bdc3c7")
    ax.bar([i + w / 2 for i in x], model, w, label="AirSentinel model", color="#2980b9")
    for i, (b, m) in enumerate(zip(base, model)):
        imp = 100 * (b - m) / b if b else 0
        ax.text(i, max(b, m) + 2, f"{imp:+.0f}%", ha="center", fontsize=10, fontweight="bold",
                color="#27ae60" if imp >= 0 else "#c0392b")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_ylabel("Mean Absolute Error (AQI points)")
    ax.set_title("Forecast accuracy vs persistence baseline (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_zone_forecast(panel: pd.DataFrame, forecasts: pd.DataFrame, zone: str, out: Path) -> None:
    p = panel[panel["zone"] == zone].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date")
    fc = forecasts[forecasts["zone"] == zone].copy()
    fc["target_date"] = pd.to_datetime(fc["target_date"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for lo, hi, color, _ in _AQI_BANDS:
        ax.axhspan(lo, hi, color=color, alpha=0.12)
    ax.plot(p["date"], p["AQI"], color="#2c3e50", lw=1.6, marker="o", ms=3, label="Observed AQI")
    if not fc.empty:
        f0 = p.iloc[-1]
        xs = [f0["date"]] + list(fc.sort_values("target_date")["target_date"])
        ys = [f0["AQI"]] + list(fc.sort_values("target_date")["predicted_aqi"])
        ax.plot(xs, ys, color="#e74c3c", lw=2, marker="s", ms=6, ls="--",
                label="Forecast (24/48/72h)")
    ax.set_title(f"AirSentinel — {zone}: observed AQI + 72h forecast")
    ax.set_ylabel("CPCB AQI")
    ax.set_ylim(0, max(320, p["AQI"].max() * 1.1))
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def generate_all(panel: pd.DataFrame, metrics: dict, outputs: Path) -> list[Path]:
    made = []
    p1 = outputs / "model_vs_baseline.png"
    plot_model_vs_baseline(metrics, p1)
    made.append(p1)
    fc_path = outputs / "forecasts.csv"
    if fc_path.exists():
        fc = pd.read_csv(fc_path)
        # Pick the zone with the highest mean AQI as the headline example.
        zone = panel.groupby("zone")["AQI"].mean().idxmax()
        p2 = outputs / "forecast_example.png"
        plot_zone_forecast(panel, fc, zone, p2)
        made.append(p2)
    return made
