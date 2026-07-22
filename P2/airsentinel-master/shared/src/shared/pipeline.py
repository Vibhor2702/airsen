"""
Fusion + dashboard pipeline. Run after airsentinel (and optionally vehicle_emissions) have
produced their outputs.

    python -m shared.pipeline
"""

from __future__ import annotations

import argparse
import json

from . import dashboard


def main() -> None:
    ap = argparse.ArgumentParser(description="AirSentinel fusion + dashboard pipeline")
    ap.add_argument("--horizon", type=int, default=24, choices=[24, 48, 72],
                    help="forecast horizon (hours) to rank/display")
    args = ap.parse_args()

    out_path, completeness = dashboard.build(horizon_hours=args.horizon)
    print("Data completeness this run:")
    print(json.dumps(completeness, indent=2))
    print(f"\nFused table: {out_path.parent / 'fused_zone_state.csv'}")
    print(f"Dashboard:   {out_path}")


if __name__ == "__main__":
    main()
