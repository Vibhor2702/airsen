# augment_labels.py — AirSentinel S2 augmentation + labels_augmented.csv rebuild
#
# Applies 7 geometric transforms (D4 dihedral group) to every heuristic-labeled
# original S2 image, writes augmented .tif files to the same zone subfolder, and
# builds labels_augmented.csv combining original + augmented rows.
#
# Idempotent: skips any augmented file that already exists on disk.
# NO2 files: augmented in parallel with S2 where the NO2 file exists; if it
#   doesn't exist the aug row's no2_file field is left as the expected path
#   (labels_fused.py's Phase A handles NO2 availability separately).
#
# Usage: python augment_labels.py
# Output: G:\My Drive\AirSentinel_Satellite_Images\labels_augmented.csv

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

DATA_DIR   = Path(r"G:\My Drive\AirSentinel_Satellite_Images")
LABELS_CSV = DATA_DIR / "labels.csv"
OUT_CSV    = DATA_DIR / "labels_augmented.csv"

# 7 transforms (+ identity makes 8 total = full D4 symmetry group)
TRANSFORMS = {
    "rot90":          lambda a: np.rot90(a, k=1, axes=(1, 2)).copy(),
    "rot180":         lambda a: np.rot90(a, k=2, axes=(1, 2)).copy(),
    "rot270":         lambda a: np.rot90(a, k=3, axes=(1, 2)).copy(),
    "flipH":          lambda a: a[:, :, ::-1].copy(),
    "flipH_rot90":    lambda a: np.rot90(a[:, :, ::-1].copy(), k=1, axes=(1, 2)).copy(),
    "flipH_rot180":   lambda a: np.rot90(a[:, :, ::-1].copy(), k=2, axes=(1, 2)).copy(),
    "flipH_rot270":   lambda a: np.rot90(a[:, :, ::-1].copy(), k=3, axes=(1, 2)).copy(),
}


def augment_tif(src_path: Path, dst_path: Path, transform_fn) -> bool:
    """
    Applies transform_fn to src_path, writes result to dst_path.
    Returns True if written, False if skipped (dst already exists).
    """
    if dst_path.exists():
        return False

    with rasterio.open(src_path) as src:
        data    = src.read()            # (bands, H, W)
        profile = src.profile.copy()
        profile.update(dtype=data.dtype)

    augmented = transform_fn(data)     # (bands, H, W) — .copy() prevents non-contiguous views
    dst_path.parent.mkdir(exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(augmented)
    return True


def main():
    df = pd.read_csv(LABELS_CSV)
    heuristic = df[df["label_source"].str.startswith("CAAQMS_heuristic")].copy()
    placeholder = df[~df["label_source"].str.startswith("CAAQMS_heuristic")].copy()

    print(f"Labels CSV    : {LABELS_CSV}")
    print(f"Heuristic rows: {len(heuristic)}  (will augment)")
    print(f"Placeholder   : {len(placeholder)} (kept unchanged, not augmented)")

    aug_rows   = []
    new_files  = 0
    skipped    = 0
    s2_missing = 0
    no2_errors = 0

    for _, row in heuristic.iterrows():
        s2_src = DATA_DIR / row["s2_file"]
        if not s2_src.exists():
            s2_missing += 1
            continue

        no2_src_path = DATA_DIR / row["no2_file"] if pd.notna(row.get("no2_file")) else None
        has_no2 = no2_src_path is not None and no2_src_path.exists()

        for aug_name, fn in TRANSFORMS.items():
            # Build output paths
            s2_stem = s2_src.stem                          # e.g. Anand_Vihar_2025-10-01_S2
            s2_aug_name  = f"{s2_stem}_aug-{aug_name}.tif"
            s2_aug_path  = s2_src.parent / s2_aug_name
            s2_aug_rel   = s2_aug_path.relative_to(DATA_DIR).as_posix()

            no2_aug_rel = None
            if no2_src_path is not None:
                no2_stem     = no2_src_path.stem
                no2_aug_name = f"{no2_stem}_aug-{aug_name}.tif"
                no2_aug_path = s2_src.parent / no2_aug_name  # same zone folder
                no2_aug_rel  = no2_aug_path.relative_to(DATA_DIR).as_posix()

            # Write S2 aug file
            try:
                if not s2_aug_path.exists():
                    augment_tif(s2_src, s2_aug_path, fn)
                    new_files += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  [!] S2 aug failed: {s2_aug_path.name}: {e}")
                continue

            # Write NO2 aug file (best-effort, don't block the row)
            if has_no2 and no2_aug_rel is not None:
                no2_aug_path_obj = DATA_DIR / no2_aug_rel
                if not no2_aug_path_obj.exists():
                    try:
                        augment_tif(no2_src_path, no2_aug_path_obj, fn)
                    except Exception as e:
                        no2_errors += 1

            # Build aug row
            aug_row = row.to_dict()
            aug_row["s2_file"]     = s2_aug_rel
            aug_row["no2_file"]    = no2_aug_rel if no2_aug_rel else row.get("no2_file", "")
            aug_row["label_source"] = aug_row["label_source"].rstrip() + " -- augmented"
            aug_rows.append(aug_row)

    aug_df = pd.DataFrame(aug_rows)
    combined = pd.concat([heuristic, aug_df, placeholder], ignore_index=True)
    combined = combined.sort_values(["zone", "date"]).reset_index(drop=True)
    combined.to_csv(OUT_CSV, index=False)

    # -- Summary --
    print(f"\nAugmentation complete:")
    print(f"  New aug files written  : {new_files}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  S2 source missing      : {s2_missing}")
    print(f"  NO2 aug write errors   : {no2_errors}")
    print(f"\nlabels_augmented.csv : {len(combined)} total rows")
    print(f"  Original heuristic   : {len(heuristic)}")
    print(f"  Augmented            : {len(aug_df)}")
    print(f"  Placeholder          : {len(placeholder)}")
    print(f"\nHeuristic label distribution (original + aug combined):")
    h_combined = combined[combined["label_source"].str.startswith("CAAQMS_heuristic")]
    for cat, cnt in h_combined["dominant_pollutant"].value_counts().items():
        print(f"  {cat}: {cnt}")
    print(f"\nWrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
