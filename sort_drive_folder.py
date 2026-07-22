# sort_drive_folder.py
# One-time local sort: moves any .tif file sitting outside its correct zone subfolder
# into the right place. Works from the locally-synced Drive path, not Colab.
# Results are printed for SESSION_LOG.md.

import shutil
from pathlib import Path

DATA_DIR = Path(r"G:\My Drive\AirSentinel_Satellite_Images")

KNOWN_ZONES = {
    "Anand_Vihar", "Ashok_Vihar", "Bawana", "Dwarka", "Jahangirpuri",
    "Mundka", "Narela", "Okhla", "Punjabi_Bagh", "RK_Puram",
    "Rohini", "Vivek_Vihar", "Wazirpur",
}

# Ensure all zone subfolders exist
for zone in KNOWN_ZONES:
    (DATA_DIR / zone).mkdir(exist_ok=True)

all_tifs = list(DATA_DIR.rglob("*.tif"))
print(f"Total .tif files found (recursive): {len(all_tifs)}")

already_correct = 0
moved = 0
unrecognised = []

for f in sorted(all_tifs):
    # Parse zone from filename: {Zone}_{YYYY-MM-DD}_{Layer}.tif
    # Zone can contain underscores (e.g. Anand_Vihar), so match against known list.
    stem = f.stem  # e.g. Anand_Vihar_2026-06-01_S2
    zone = None
    for z in KNOWN_ZONES:
        if stem.startswith(z + "_"):
            zone = z
            break

    if zone is None:
        unrecognised.append(f.name)
        continue

    expected_parent = DATA_DIR / zone
    if f.parent == expected_parent:
        already_correct += 1
    else:
        dest = expected_parent / f.name
        if dest.exists():
            # Duplicate — skip rather than overwrite
            print(f"  SKIP (dest exists): {f.relative_to(DATA_DIR)} -> already at {dest.relative_to(DATA_DIR)}")
        else:
            shutil.move(str(f), str(dest))
            print(f"  MOVED: {f.relative_to(DATA_DIR)} -> {dest.relative_to(DATA_DIR)}")
            moved += 1

print()
print(f"Results:")
print(f"  Total .tif files scanned : {len(all_tifs)}")
print(f"  Already in correct folder: {already_correct}")
print(f"  Moved to correct folder  : {moved}")
if unrecognised:
    print(f"  Unrecognised filenames   : {unrecognised}")
else:
    print(f"  Unrecognised filenames   : none")
