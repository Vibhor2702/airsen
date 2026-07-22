"""
Loader for real ARAI/CPCB vehicle emission-factor tables (g/km, by vehicle category, fuel
type, and Bharat Stage emission norm).

No emission factor is hardcoded in this codebase — see config.py's module docstring for why.
Real, official sources located this session (pull the actual figures from these into the
template rather than any number quoted elsewhere):

    - CPCB emission-factor PDF: https://cpcb.nic.in/displaypdf.php?id=RW1pc3Npb25fRmFjdG9yc19WZWhpY2xlcy5wZGY=
    - CPCB vehicular exhaust page: https://cpcb.nic.in/vehicular-exhaust/
    - ARAI "Emission Factors for Indian In-Use Vehicles" report
    - Lok Sabha reply, 9 Aug 2021, "Vehicle-wise Emission Factors for PM, NOx, HC, CO and
      CO2" (data.gov.in Open Government Data Platform)
"""

from __future__ import annotations

import pandas as pd

from . import config


def write_template() -> None:
    pd.DataFrame(columns=config.EMISSION_FACTORS_COLUMNS).to_csv(
        config.TEMPLATES_DIR / "emission_factors_template.csv", index=False
    )


def load() -> pd.DataFrame:
    if not config.EMISSION_FACTORS_CSV.exists():
        raise FileNotFoundError(
            f"No emission factor table found at {config.EMISSION_FACTORS_CSV}.\n"
            f"This module does not hardcode emission factors — see this module's docstring "
            f"and README.md for the real ARAI/CPCB sources to pull them from, then fill in "
            f"the template at {config.TEMPLATES_DIR / 'emission_factors_template.csv'} "
            f"and save it to {config.EMISSION_FACTORS_CSV}."
        )
    df = pd.read_csv(config.EMISSION_FACTORS_CSV)
    missing_cols = set(config.EMISSION_FACTORS_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"emission_factors.csv is missing required columns: {missing_cols}")
    if df["source_citation"].isna().any() or (df["source_citation"].astype(str).str.strip() == "").any():
        raise ValueError(
            "Every row in emission_factors.csv must cite its source report (and page/table "
            "number, ideally) in source_citation."
        )
    if (df["emission_factor_g_per_km"] < 0).any():
        raise ValueError("emission_factors.csv has negative emission_factor_g_per_km values.")
    return df
