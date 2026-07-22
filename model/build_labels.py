# build_labels.py — Phase 1: Build real (heuristic) labels for Prithvi training
#
# Replaces the random placeholder labels.csv with heuristic labels derived from
# teammate's CAAQMS data wherever coverage exists. Keeps explicit PLACEHOLDER flag
# where it doesn't (2026-06-01, which predates the CAAQMS scraper window).
#
# Heuristic rules are documented in:
#   P2/airsentinel-master/forecasting/src/airsentinel/labels.py
# They are NOT source apportionment — never present these as ground truth.
#
# Run: python build_labels.py
# Output: writes labels.csv into the Google Drive folder (DATA_DIR).

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"G:\My Drive\AirSentinel_Satellite_Images")

HEURISTIC_CSV = Path(
    r"P2\airsentinel-master\forecasting\data\teammate_delivery\caaqms_heuristic_labels.csv"
)

RNG_SEED = 42  # only used for the few rows that genuinely have no CAAQMS coverage
CLASSES = ["dust", "crop_burning_smoke", "industrial_haze", "traffic_heavy", "clear"]

# Zone name in filenames uses underscores; CAAQMS data uses spaces.
# Mapping covers every case that differs.
FILENAME_TO_CAAQMS_ZONE = {
    "Anand_Vihar":   "Anand Vihar",
    "Ashok_Vihar":   "Ashok Vihar",
    "Bawana":        "Bawana",
    "Dwarka":        "Dwarka",
    "Jahangirpuri":  "Jahangirpuri",
    "Mundka":        "Mundka",
    "Narela":        "Narela",
    "Okhla":         "Okhla",
    "Punjabi_Bagh":  "Punjabi Bagh",
    "RK_Puram":      "RK Puram",
    "Rohini":        "Rohini",
    "Vivek_Vihar":   "Vivek Vihar",
    "Wazirpur":      "Wazirpur",
}

# ---------------------------------------------------------------------------
# Step 1: Recursive scan for all S2 .tif files (handles loose files from Drive
# sync lag, per the brief: don't skip a file just because it's misplaced).
# ---------------------------------------------------------------------------
print("Scanning for S2 files recursively...")
s2_files = sorted(DATA_DIR.rglob("*_S2.tif"))
print(f"  Found {len(s2_files)} S2 files across all subfolders and root")

rows = []
parse_errors = []
for f in s2_files:
    # Filename format: {Zone_Name}_{YYYY-MM-DD}_S2.tif
    # Zone names can contain underscores (e.g. Anand_Vihar), so we split from the right.
    stem = f.stem  # e.g. Anand_Vihar_2026-06-01_S2
    parts = stem.rsplit("_", 2)  # ['Anand_Vihar', '2026-06-01', 'S2']
    if len(parts) != 3 or parts[2] != "S2":
        parse_errors.append(f.name)
        continue
    zone_filename, date, _ = parts
    no2_name = f.name.replace("_S2.tif", "_NO2.tif")
    # Store path relative to DATA_DIR so the CSV is portable (not machine-specific).
    s2_rel = f.relative_to(DATA_DIR).as_posix()
    no2_rel = (f.parent / no2_name).relative_to(DATA_DIR).as_posix()
    rows.append({
        "zone":          zone_filename,
        "date":          date,
        "s2_file":       s2_rel,
        "no2_file":      no2_rel,
        "caaqms_zone":   FILENAME_TO_CAAQMS_ZONE.get(zone_filename),
    })

if parse_errors:
    print(f"  WARNING: could not parse {len(parse_errors)} filename(s): {parse_errors}")

df = pd.DataFrame(rows)
print(f"  Parsed: {len(df)} zone/date pairs | {df['zone'].nunique()} zones | {df['date'].nunique()} unique dates")

# ---------------------------------------------------------------------------
# Step 2: Load heuristic labels from teammate's delivery.
# ---------------------------------------------------------------------------
print("\nLoading heuristic labels...")
heuristic = pd.read_csv(HEURISTIC_CSV)
print(f"  {len(heuristic)} rows | zones: {sorted(heuristic['zone'].unique())}")
print(f"  Dates: {heuristic['date'].min()} to {heuristic['date'].max()}")
print(f"  Label distribution (heuristic, not ground truth):")
for cat, count in heuristic["source_category"].value_counts().items():
    print(f"    {cat}: {count}")

# ---------------------------------------------------------------------------
# Step 3: Join — match each S2 image to a heuristic label by (caaqms_zone, date).
# ---------------------------------------------------------------------------
df = df.merge(
    heuristic.rename(columns={"zone": "caaqms_zone", "source_category": "dominant_pollutant_heuristic"}),
    on=["caaqms_zone", "date"],
    how="left",
)

matched = df["dominant_pollutant_heuristic"].notna().sum()
unmatched = df["dominant_pollutant_heuristic"].isna().sum()
print(f"\nJoin result: {matched} matched | {unmatched} unmatched (no CAAQMS coverage)")

if unmatched > 0:
    print("  Unmatched zone/date pairs (will keep PLACEHOLDER label):")
    for _, r in df[df["dominant_pollutant_heuristic"].isna()].iterrows():
        print(f"    {r['zone']} / {r['date']}")

# ---------------------------------------------------------------------------
# Step 4: Assign labels and provenance.
#   - Matched rows: heuristic label, clearly flagged as such.
#   - Unmatched rows: random placeholder, explicitly flagged.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)

def assign_label(row):
    if pd.notna(row["dominant_pollutant_heuristic"]):
        return row["dominant_pollutant_heuristic"], "CAAQMS_heuristic -- rule-based, not lab-verified"
    else:
        return rng.choice(CLASSES), "PLACEHOLDER -- random, not real (no CAAQMS coverage for this date)"

df[["dominant_pollutant", "label_source"]] = df.apply(
    assign_label, axis=1, result_type="expand"
)

# ---------------------------------------------------------------------------
# Step 5: Final output table (drop the working columns, keep the spec columns).
# ---------------------------------------------------------------------------
out = df[["zone", "date", "s2_file", "no2_file", "dominant_pollutant", "label_source"]].copy()
out = out.sort_values(["zone", "date"]).reset_index(drop=True)

out_path = DATA_DIR / "labels.csv"
out.to_csv(out_path, index=False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\nWrote {len(out)} rows to {out_path}")
print("\nFinal label distribution:")
for src in out["label_source"].unique():
    sub = out[out["label_source"] == src]
    print(f"\n  [{src}]")
    for cat, count in sub["dominant_pollutant"].value_counts().items():
        print(f"    {cat}: {count}")

print("\n[OK] Phase 1 label build complete.")
print("     Heuristic labels are NOT ground truth -- see labels.py for rules.")
print("     Rows with PLACEHOLDER flag have no CAAQMS coverage and must not be")
print("     used to claim accuracy of the trained model.")
