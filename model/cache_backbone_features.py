# cache_backbone_features.py — AirSentinel backbone feature pre-computation
#
# One-time script. Runs every S2 image through the frozen Prithvi backbone
# (with Phase 2 fine-tuned weights) exactly once, and saves the resulting
# 768-d mean-pooled feature vectors to a single cache file on disk.
#
# Cache invalidation: the cache file is tagged with the SHA-256 of the
# backbone checkpoint. If the checkpoint changes (e.g. after re-training the
# backbone), re-run this script — it will detect the mismatch and regenerate.
#
# Usage:
#   python cache_backbone_features.py
#   python cache_backbone_features.py --labels-csv G:\...\labels_fused.csv
#   python cache_backbone_features.py --force       # force regen even if valid
#
# After this runs, train_fused.py will automatically detect and use the cache
# when the backbone is frozen. Epochs drop from ~3 min to ~10-15 s.
# --------------------------------------------------------------------------

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")

DATA_DIR      = Path(r"G:\My Drive\AirSentinel_Satellite_Images")
BACKBONE_CKPT = Path("prithvi_airsen_augmented.pt")   # Phase 2 fine-tuned checkpoint
CACHE_PATH    = Path("backbone_features_cache.pt")
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hash of a file. Used to detect checkpoint changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_s2_image(tif_path: Path, target_size: int = IMG_SIZE) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(tif_path) as src:
        data = src.read(
            out_shape=(src.count, target_size, target_size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    return np.clip(data / 10000.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Backbone loader (same logic as train_fused.py)
# ---------------------------------------------------------------------------

def load_backbone(ckpt_path: Path) -> nn.Module:
    from terratorch.registry import BACKBONE_REGISTRY
    print("Loading Prithvi-EO-2.0-100M-TL backbone...")
    backbone = BACKBONE_REGISTRY.build("prithvi_eo_v2_100_tl", pretrained=True)
    backbone.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"]
    bb_sd = {k.replace("backbone.", "", 1): v
             for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(bb_sd, strict=False)
    print(f"Loaded Phase 2 weights from {ckpt_path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")

    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def extract_feature(backbone: nn.Module, img_np: np.ndarray) -> np.ndarray:
    """Run one (6, H, W) image through the backbone, return (768,) feature."""
    # Add batch and time dims: (1, 6, 1, H, W) — Prithvi input format
    x = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(DEVICE)
    with torch.no_grad():
        raw = backbone(x)
    feat = raw[-1] if isinstance(raw, (list, tuple)) else raw
    if feat.dim() == 3:
        feat = feat.mean(dim=1)     # (1, num_patches, 768) -> (1, 768)
    elif feat.dim() == 4:
        feat = feat.mean(dim=[2, 3])
    return feat.squeeze(0).cpu().numpy()   # (768,)


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------

def save_cache(ckpt_hash: str, features: dict) -> None:
    """
    Saves cache to disk as a single .pt file.
    Format:
      {
        "checkpoint_hash": str,         # SHA-256 of BACKBONE_CKPT
        "checkpoint_path": str,         # which checkpoint produced these features
        "backbone_dim": int,            # sanity check (768 for Prithvi-100M)
        "img_size": int,                # what image size was used (224)
        "features": {s2_file_str: np.ndarray shape (768,)}
      }
    """
    # Verify dim consistency
    dims = {v.shape[0] for v in features.values() if v is not None}
    backbone_dim = dims.pop() if len(dims) == 1 else -1

    torch.save({
        "checkpoint_hash": ckpt_hash,
        "checkpoint_path": str(BACKBONE_CKPT),
        "backbone_dim": backbone_dim,
        "img_size": IMG_SIZE,
        "features": features,
    }, CACHE_PATH)

    size_mb = CACHE_PATH.stat().st_size / 1e6
    print(f"\n[OK] Cache saved: {CACHE_PATH.resolve()}")
    print(f"     Entries   : {len(features)}")
    print(f"     Feat dim  : {backbone_dim}")
    print(f"     Size      : {size_mb:.1f} MB")
    print(f"     Ckpt hash : {ckpt_hash[:16]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-compute frozen Prithvi backbone features for AirSentinel images."
    )
    ap.add_argument(
        "--labels-csv", type=Path,
        default=DATA_DIR / "labels_fused.csv",
        help="CSV with s2_file column (default: labels_fused.csv). "
             "Pass labels_augmented.csv if labels_fused.csv is not yet built."
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Regenerate all features even if the cache appears valid."
    )
    args = ap.parse_args()

    print(f"\nDevice : {DEVICE}\n")

    # -- Validate inputs --------------------------------------------------------
    if not BACKBONE_CKPT.exists():
        print(f"[ERROR] Backbone checkpoint not found: {BACKBONE_CKPT}")
        print("        Run train_prithvi.py (Phase 2) to produce it first.")
        sys.exit(1)

    if not args.labels_csv.exists():
        print(f"[ERROR] Labels CSV not found: {args.labels_csv}")
        sys.exit(1)

    # -- Collect unique s2_file paths from CSV ----------------------------------
    df = pd.read_csv(args.labels_csv)
    # Use only rows that have a valid s2_file
    all_s2 = (df["s2_file"]
              .dropna()
              .unique()
              .tolist())
    print(f"Labels CSV        : {args.labels_csv}")
    print(f"Unique S2 images  : {len(all_s2)}")

    # -- Check existing cache ---------------------------------------------------
    print(f"\nHashing checkpoint {BACKBONE_CKPT} (may take a second)...")
    ckpt_hash = sha256_file(BACKBONE_CKPT)
    print(f"Checkpoint hash   : {ckpt_hash[:32]}...")

    existing_features: dict = {}

    if CACHE_PATH.exists() and not args.force:
        try:
            cached = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
            cached_hash = cached.get("checkpoint_hash", "")
            if cached_hash == ckpt_hash:
                existing_features = cached.get("features", {})
                already = sum(1 for s in all_s2 if s in existing_features)
                print(f"\nExisting cache    : {len(existing_features)} total entries")
                print(f"Already covered   : {already}/{len(all_s2)} needed images")

                if already == len(all_s2):
                    size_mb = CACHE_PATH.stat().st_size / 1e6
                    print(f"\n[OK] Cache is complete and valid.")
                    print(f"     File : {CACHE_PATH.resolve()}")
                    print(f"     Size : {size_mb:.1f} MB")
                    print("     Re-run with --force to regenerate.")
                    return
            else:
                print(f"\n[!] Cache exists but checkpoint hash has changed.")
                print(f"    Cached hash : {cached_hash[:32]}...")
                print(f"    Current     : {ckpt_hash[:32]}...")
                print("    Regenerating (old features discarded).")
                existing_features = {}
        except Exception as e:
            print(f"\n[!] Could not read existing cache ({e}). Regenerating.")
            existing_features = {}
    elif args.force:
        print("\n[force] Ignoring existing cache — regenerating from scratch.")

    # -- Determine what still needs computing -----------------------------------
    to_compute = [s for s in all_s2 if s not in existing_features]
    print(f"\nTo compute : {len(to_compute)} images  "
          f"({len(existing_features)} already done from partial cache)\n")

    if not to_compute:
        save_cache(ckpt_hash, existing_features)
        return

    # -- Load backbone ----------------------------------------------------------
    backbone = load_backbone(BACKBONE_CKPT).to(DEVICE)

    if DEVICE.type == "cuda":
        print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # -- Extract features -------------------------------------------------------
    t0 = time.time()
    skipped: list[str] = []

    pbar = tqdm(to_compute, desc="Extracting backbone features", unit="img")
    for s2_rel in pbar:
        full_path = DATA_DIR / s2_rel
        if not full_path.exists():
            skipped.append(s2_rel)
            pbar.set_postfix(skipped=len(skipped))
            continue
        try:
            img = load_s2_image(full_path)
            feat = extract_feature(backbone, img)
            existing_features[s2_rel] = feat
        except Exception as e:
            skipped.append(s2_rel)
            pbar.set_postfix(skipped=len(skipped), err=str(e)[:25])

    elapsed = time.time() - t0
    computed = len(to_compute) - len(skipped)
    ms_per_img = elapsed / max(1, computed) * 1000

    print(f"\nExtracted : {computed} features in {elapsed:.1f}s "
          f"({ms_per_img:.0f} ms/image)")
    if skipped:
        print(f"Skipped   : {len(skipped)} (file not found or read error)")
        for s in skipped[:5]:
            print(f"  {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    # -- Save -------------------------------------------------------------------
    save_cache(ckpt_hash, existing_features)

    print(f"\nNext steps:")
    print(f"  train_fused.py will now automatically detect and use this cache.")
    print(f"  Expected epoch time: seconds (down from ~3 min with full backbone).")


if __name__ == "__main__":
    main()
