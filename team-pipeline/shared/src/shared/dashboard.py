"""
Official Dashboard — the "04 · OUTPUT LAYER" component from the design plan (slide 5),
styled to match the deck's own mockup (slide 7: city-wide AQI, zones-by-stage counts, top
zones needing attention, per-zone GRAP stage + dominant-source badge).

Renders a single static HTML file from `fuse.fuse()`'s real output only — every number in
the page traces back to a real CSV written by a real pipeline run. Anything a sibling module
hasn't produced yet renders as a visible "pending" badge, never a placeholder number
dressed up as real.
"""

from __future__ import annotations

from html import escape

import pandas as pd

from . import alerts, config, grap

_STAGE_COLORS = {
    0: ("#e8f5e9", "#2e7d32"),  # below GRAP -> green
    1: ("#fff8e1", "#f9a825"),  # Stage I -> amber
    2: ("#ffe0b2", "#ef6c00"),  # Stage II -> orange
    3: ("#ffcdd2", "#c62828"),  # Stage III -> red
    4: ("#e1bee7", "#6a1b9a"),  # Stage IV -> deep purple
    -1: ("#eceff1", "#455a64"),  # unknown -> grey
}


def _stage_badge(stage_num: int, label: str) -> str:
    bg, fg = _STAGE_COLORS.get(stage_num, _STAGE_COLORS[-1])
    return f'<span class="badge" style="background:{bg};color:{fg}">{escape(label)}</span>'


def render(fused: pd.DataFrame, completeness: dict, horizon_hours: int = 24) -> str:
    n_zones = len(fused)
    stage_counts = fused["grap_stage"].value_counts().to_dict()
    avg_aqi = fused["predicted_aqi"].mean()
    vehicle_is_demo = "demo" in str(completeness.get("vehicle_emissions", "")).lower()
    demo_banner = (
        '<div class="demo-banner">⚠ Vehicle Emission Load Index below is DEMO DATA — '
        'real Delhi-wide vehicle totals and real official BS6 emission limits, but split '
        'EVENLY across zones as a placeholder (not real per-zone counts). '
        'See vehicle_emissions/src/vehicle_emissions/build_demo_data.py for every citation. '
        'Not random — but not verified per-zone precision either.</div>'
        if vehicle_is_demo else ""
    )

    def _chip_class(v: str) -> str:
        if "demo" in v.lower():
            return "demo-chip"
        if v == "live":
            return "live"
        return "pending-chip"

    completeness_html = "".join(
        f'<span class="chip {_chip_class(v)}">{escape(k)}: {escape(v)}</span>'
        for k, v in completeness.items()
    )

    stage_summary = "".join(
        f'<div class="stat"><div class="stat-num">{stage_counts.get(n, 0)}</div>'
        f'<div class="stat-label">{escape(lbl)}</div></div>'
        for n, lbl in [(1, "Stage I"), (2, "Stage II"), (3, "Stage III"), (4, "Stage IV")]
    )

    def _veh_cell(veh) -> str:
        if not isinstance(veh, (int, float)):
            return "<span class='pending'>pending</span>"
        suffix = " <span class='demo-tag'>(demo)</span>" if vehicle_is_demo else ""
        return f"{veh:.2f}{suffix}"

    top5 = fused.sort_values("urgency_rank").head(5)
    top5_rows = ""
    for _, r in top5.iterrows():
        veh = r.get("vehicle_emission_load_index", "pending")
        sat = r.get("satellite_source_guess", "pending")
        top5_rows += (
            f'<tr>'
            f'<td>#{int(r["urgency_rank"])}</td>'
            f'<td class="zone-name">{escape(str(r["zone"]))}</td>'
            f'<td class="aqi-num">{round(r["predicted_aqi"])}</td>'
            f'<td>{_stage_badge(int(r["grap_stage"]), r["grap_label"])}</td>'
            f'<td>{_veh_cell(veh)}</td>'
            f'<td>{escape(str(sat)) if "pending" not in str(sat).lower() else "<span class=\'pending\'>pending</span>"}</td>'
            f'</tr>'
        )

    all_rows = ""
    for _, r in fused.sort_values("urgency_rank").iterrows():
        veh = r.get("vehicle_emission_load_index", "pending")
        actual = r.get("latest_actual_aqi", "pending")
        actual_str = f"{actual:.0f}" if isinstance(actual, (int, float)) else "<span class='pending'>pending</span>"
        all_rows += (
            f'<tr>'
            f'<td>{escape(str(r["zone"]))}</td>'
            f'<td>{actual_str}</td>'
            f'<td class="aqi-num">{round(r["predicted_aqi"])}</td>'
            f'<td>{_stage_badge(int(r["grap_stage"]), r["grap_label"])}</td>'
            f'<td>{_veh_cell(veh)}</td>'
            f'</tr>'
        )

    # -- Enforcement ranker honesty section ----------------------------------------
    # model_version, data_provenance, land_use_note must be visible to anyone reading
    # the dashboard — not silently dropped. Pull from the first non-pending fused row;
    # all rows share the same values (they come from a single ranker run).
    PENDING = "pending \u2014 module not yet run"

    def _first_non_pending(col: str) -> str:
        if col not in fused.columns:
            return PENDING
        vals = fused[col][fused[col] != PENDING]
        return str(vals.iloc[0]) if not vals.empty else PENDING
    sat_model_version   = _first_non_pending("satellite_model_version")
    sat_data_provenance = _first_non_pending("satellite_data_provenance")
    sat_land_use_note   = _first_non_pending("satellite_land_use_note")

    def _honesty_chip(label: str, value: str) -> str:
        is_pending = "pending" in value.lower()
        is_placeholder = "placeholder" in value.lower()
        cls = "pending-chip" if is_pending else ("demo-chip" if is_placeholder else "live")
        return (f'<div class="honesty-row"><span class="honesty-label">{escape(label)}</span>'
                f'<span class="chip {cls}">{escape(value)}</span></div>')

    honesty_html = (
        '<section class="honesty-section">'
        '<h2>Enforcement ranker — data quality flags</h2>'
        '<p style="font-size:0.82rem;color:#888;margin-bottom:0.75rem">'
        'These tags are present on every row of enforcement_ranking.csv and must be '
        'visible here so they cannot be mistaken for final or ground-truth output.</p>'
        + _honesty_chip("model_version", sat_model_version)
        + _honesty_chip("data_provenance", sat_data_provenance)
        + _honesty_chip("land_use_note", sat_land_use_note)
        + '</section>'
    )

    # One example alert for the single most urgent zone, matching deck slide 8.
    top = fused.sort_values("urgency_rank").iloc[0]
    dominant = top.get("latest_actual_dominant_pollutant", None)
    alert_text = alerts.generate(
        zone=str(top["zone"]), aqi=float(top["predicted_aqi"]),
        grap_stage_label=str(top["grap_label"]), grap_stage_num=int(top["grap_stage"]),
        horizon_hours=horizon_hours, dominant_source=dominant if isinstance(dominant, str) else None,
    ).replace("\n", "<br>")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AirSentinel — Delhi Operations Dashboard</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem;
          background: #fafafa; color: #1a1a1a; }}
  @media (prefers-color-scheme: dark) {{ body {{ background: #121212; color: #eee; }} }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #666; margin-bottom: 1.5rem; }}
  .completeness {{ margin-bottom: 1.5rem; }}
  .chip {{ display: inline-block; padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.75rem;
           margin-right: 0.4rem; }}
  .chip.live {{ background: #e8f5e9; color: #2e7d32; }}
  .chip.pending-chip {{ background: #eceff1; color: #455a64; }}
  .chip.demo-chip {{ background: #fff8e1; color: #e6a700; font-weight: 600; }}
  .demo-banner {{ background: #fff8e1; border: 1px solid #f9a825; color: #7a5900; padding: 0.75rem 1rem;
                   border-radius: 8px; font-size: 0.85rem; margin-bottom: 1.5rem; line-height: 1.4; }}
  .demo-tag {{ font-size: 0.75rem; color: #e6a700; font-style: italic; }}
  .top-stats {{ display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .stat {{ background: white; border-radius: 12px; padding: 1rem 1.5rem; min-width: 100px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  @media (prefers-color-scheme: dark) {{ .stat {{ background: #1e1e1e; }} }}
  .stat-num {{ font-size: 1.8rem; font-weight: 700; }}
  .stat-label {{ font-size: 0.8rem; color: #888; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px;
           overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
  @media (prefers-color-scheme: dark) {{ table {{ background: #1e1e1e; }} }}
  th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #eee; }}
  @media (prefers-color-scheme: dark) {{ th, td {{ border-color: #333; }} }}
  th {{ font-size: 0.75rem; text-transform: uppercase; color: #888; }}
  .zone-name {{ font-weight: 600; }}
  .aqi-num {{ font-weight: 700; font-size: 1.1rem; }}
  .badge {{ padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }}
  .pending {{ color: #999; font-style: italic; font-size: 0.85rem; }}
  .honesty-section {{ background: #f3f4f6; border: 1px solid #d1d5db; border-radius: 10px;
                      padding: 1rem 1.25rem; margin-bottom: 2rem; }}
  @media (prefers-color-scheme: dark) {{ .honesty-section {{ background: #1a1a2e; border-color: #374151; }} }}
  .honesty-row {{ display: flex; align-items: flex-start; gap: 0.75rem; margin-bottom: 0.5rem; flex-wrap: wrap; }}
  .honesty-label {{ font-size: 0.78rem; font-weight: 600; color: #555; min-width: 140px;
                    padding-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.03em; }}
  .alert-card {{ background: white; border-radius: 12px; padding: 1.5rem; max-width: 420px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 0.9rem; line-height: 1.6; }}
  @media (prefers-color-scheme: dark) {{ .alert-card {{ background: #1e1e1e; }} }}
  section {{ margin-bottom: 2rem; }}
  h2 {{ font-size: 1.1rem; }}
</style>
</head>
<body>
<h1>AirSentinel — Delhi Operations</h1>
<div class="subtitle">Forecast horizon: {horizon_hours}h · {n_zones} zones</div>

<div class="completeness">{completeness_html}</div>
{demo_banner}
<div class="top-stats">
  <div class="stat"><div class="stat-num">{avg_aqi:.0f}</div><div class="stat-label">CITY-WIDE AVG AQI (forecast)</div></div>
  {stage_summary}
</div>

{honesty_html}

<section>
  <h2>Top zones needing attention</h2>
  <table>
    <tr><th>Rank</th><th>Zone</th><th>Forecast AQI</th><th>GRAP Stage</th><th>Vehicle load index</th><th>Satellite source guess</th></tr>
    {top5_rows}
  </table>
</section>

<section>
  <h2>Example alert (most urgent zone)</h2>
  <div class="alert-card">{alert_text}</div>
</section>

<section>
  <h2>All zones</h2>
  <table>
    <tr><th>Zone</th><th>Latest actual AQI</th><th>Forecast AQI</th><th>GRAP Stage</th><th>Vehicle load index</th></tr>
    {all_rows}
  </table>
</section>

</body>
</html>"""


def build(horizon_hours: int = 24) -> tuple:
    from . import fuse
    fused, completeness = fuse.fuse(horizon_hours=horizon_hours)
    html = render(fused, completeness, horizon_hours=horizon_hours)
    out_path = config.OUTPUTS_DIR / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path, completeness
