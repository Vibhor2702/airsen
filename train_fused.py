# train_fused.py — AirSentinel NO2-fused Prithvi classifier (Phases B/C/D)
#
# Runs two comparable training jobs from one script:
#
#   python train_fused.py              -> optical-only baseline (frozen backbone)
#   python train_fused.py --fuse-no2  -> late-fusion model with NO2 sensor input
#
# Both jobs:
#   - Load labels_fused.csv (has no2_mean + no2_available columns from Phase A)
#   - Use the same leakage-free group-aware train/val split (RANDOM_SEED=42, VAL_FRACTION=0.2)
#   - Load the Phase 2 checkpoint backbone (already fine-tuned) and freeze it
#   - Save best-val checkpoint only (not final-epoch) -- fixes the epoch-10 overfitting issue
#
# Feature caching (speed optimisation):
#   When the backbone is frozen, train_fused.py automatically detects and uses
#   backbone_features_cache.pt if present and valid (same checkpoint SHA-256).
#   Run cache_backbone_features.py once to build it. Epochs drop from ~3 min to seconds.
#   Cache is bypassed entirely if --unfreeze-backbone is set (features would be stale).
#
# Architecture (fusion mode):
#   Prithvi backbone (frozen, 768-d output)
#     ↓
#   mean-pool -> 768-d feature
#     ↓  concat
#   NO2 encoder output (32-d)            <- real value OR learned missing token
#     ↓
#   LayerNorm(800) -> Dropout(0.1) -> Linear(800, 5)
#
# Missing-modality handling (Phase B):
#   - When no2_available == 1: encode z-scored NO2 value through 2-layer MLP
#   - When no2_available == 0: use a learned nn.Parameter missing-token vector
#   - Modality dropout: during training, 15% of available rows randomly substitute
#     the missing token (prevents the model depending on NO2 always being present)
#
# Backbone un-freeze (future LoRA / full fine-tuning):
#   Pass --unfreeze-backbone to train backbone params as well.
#   Cache is automatically skipped — cached features are only valid for a frozen backbone.
# ------------------------------------------------------------------------------

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR      = Path(r"G:\My Drive\AirSentinel_Satellite_Images")
LABELS_CSV    = DATA_DIR / "labels_fused.csv"
BACKBONE_CKPT = Path("prithvi_airsen_augmented.pt")   # Phase 2 checkpoint to load from
CACHE_PATH    = Path("backbone_features_cache.pt")     # pre-computed feature cache

CLASSES    = ["dust", "crop_burning_smoke", "industrial_haze", "traffic_heavy", "clear"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE       = 224
BATCH_SIZE     = 64    # raised: frozen backbone + cache mode uses negligible VRAM; confirm at runtime
NUM_EPOCHS     = 15     # more epochs now that backbone is frozen (faster per epoch)
LR             = 3e-4   # higher LR ok for head-only training
RANDOM_SEED    = 42
VAL_FRACTION   = 0.2

NO2_EMBED_DIM     = 32     # dimension of NO2 encoder output
MODALITY_DROPOUT  = 0.15   # probability of replacing real NO2 with missing token during train


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hash of a file. Used to detect checkpoint changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class FeatureCache:
    """
    Loads backbone features pre-computed by cache_backbone_features.py.

    Validates that the cache was produced by the exact same checkpoint (via
    SHA-256). If the checkpoint has changed, raises RuntimeError immediately
    rather than silently using stale features — stale features produce wrong
    results without any visible error.

    If no cache file exists, falls back to full image loading gracefully.
    The caller is responsible for deciding whether to proceed without cache.
    """

    def __init__(self, cache_path: Path, ckpt_path: Path):
        self.valid = False
        self._features: dict = {}

        if not cache_path.exists():
            print(f"[cache] No cache at {cache_path}. "
                  "Run cache_backbone_features.py for faster training.")
            return

        if not ckpt_path.exists():
            print(f"[cache] Backbone checkpoint not found at {ckpt_path} — "
                  "skipping cache.")
            return

        print(f"[cache] Hashing {ckpt_path.name} for validation...")
        t0 = time.time()
        ckpt_hash = _sha256_file(ckpt_path)
        print(f"[cache] Hash computed in {time.time()-t0:.1f}s: {ckpt_hash[:16]}...")

        try:
            # weights_only=False: cache contains numpy arrays, not just tensors
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[cache] Could not read cache file ({e}). "
                  "Falling back to full image pipeline.")
            return

        cached_hash = cached.get("checkpoint_hash", "")
        if ckpt_hash != cached_hash:
            # Hard error: silently using stale features would corrupt training
            raise RuntimeError(
                f"\n[cache] STALE CACHE DETECTED — refusing to use it.\n"
                f"  Cache checkpoint hash : {cached_hash[:32]}...\n"
                f"  Current checkpoint    : {ckpt_hash[:32]}...\n"
                f"\n"
                f"  The backbone checkpoint has changed since the cache was built.\n"
                f"  Cached 768-d features are no longer valid for this backbone.\n"
                f"\n"
                f"  Fix: python cache_backbone_features.py\n"
                f"  (Re-runs extraction with the current checkpoint, ~5-10 min once)"
            )

        self._features = cached.get("features", {})
        self.valid = True
        bb_dim = cached.get("backbone_dim", "?")
        print(f"[cache] Loaded {len(self._features)} features "
              f"(dim={bb_dim}, ckpt={ckpt_hash[:12]}...)")

    def get(self, s2_file: str) -> torch.Tensor | None:
        """Return pre-computed 768-d feature tensor, or None if not cached."""
        if not self.valid:
            return None
        arr = self._features.get(s2_file)
        if arr is None:
            return None
        return torch.tensor(arr, dtype=torch.float32)

    def covers(self, s2_files: list[str]) -> bool:
        """True if every file in the list has a cached feature."""
        if not self.valid:
            return False
        return all(f in self._features for f in s2_files)

    def coverage_report(self, s2_files: list[str]) -> tuple[int, int]:
        """Returns (n_covered, n_total)."""
        covered = sum(1 for f in s2_files if f in self._features)
        return covered, len(s2_files)


# ---------------------------------------------------------------------------
# Section 1 — Data loading
# ---------------------------------------------------------------------------

def load_and_split(labels_csv: Path) -> tuple:
    """
    Loads labels_fused.csv, filters to heuristic rows, applies group-aware
    train/val split (same logic as train_prithvi.py — splits on original
    zone/date groups so augmented siblings never leak across splits).

    Also computes NO2 normalisation statistics from training rows only and
    returns them so they can be stored in the checkpoint.
    """
    df = pd.read_csv(labels_csv)

    required = {"zone", "date", "s2_file", "dominant_pollutant",
                "label_source", "no2_mean", "no2_available"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"labels_fused.csv is missing columns: {missing_cols}\n"
            "Run Phase A (train_fused.py's Phase A extraction) first."
        )

    heuristic = df[df["label_source"].str.startswith("CAAQMS_heuristic")].copy()
    placeholder_count = len(df) - len(heuristic)

    print(f"Labels loaded: {len(df)} total rows")
    print(f"  Heuristic (will use): {len(heuristic)}")
    print(f"  Placeholder (excluded): {placeholder_count}")

    # NO2 availability summary
    h_avail = heuristic["no2_available"].sum()
    print(f"  NO2 available in heuristic rows: {h_avail}/{len(heuristic)} "
          f"({h_avail/len(heuristic)*100:.1f}%)")
    print(f"  Label distribution:")
    for cat, cnt in heuristic["dominant_pollutant"].value_counts().items():
        print(f"    {cat}: {cnt}")

    # Group-aware split on original (zone, date) — same as train_prithvi.py
    groups = (heuristic[["zone", "date"]]
              .drop_duplicates()
              .sample(frac=1, random_state=RANDOM_SEED)
              .reset_index(drop=True))
    n_val_groups = max(1, int(len(groups) * VAL_FRACTION))
    val_pairs = set(zip(groups.iloc[:n_val_groups]["zone"],
                        groups.iloc[:n_val_groups]["date"]))

    is_val = heuristic.apply(lambda r: (r["zone"], r["date"]) in val_pairs, axis=1)
    train_df = heuristic[~is_val].copy()
    val_df   = heuristic[is_val].copy()

    print(f"  Split: {len(groups)-n_val_groups} train groups / {n_val_groups} val groups "
          f"-> {len(train_df)} train rows / {len(val_df)} val rows")

    # NO2 normalisation: z-score from training rows only (available only)
    train_no2_vals = train_df.loc[train_df["no2_available"] == 1, "no2_mean"]
    no2_mean_stat = float(train_no2_vals.mean())
    no2_std_stat  = float(train_no2_vals.std())
    print(f"  NO2 norm stats (train available rows): "
          f"mean={no2_mean_stat:.4e}  std={no2_std_stat:.4e}")

    return train_df, val_df, no2_mean_stat, no2_std_stat


def load_s2_image(tif_path: Path, target_size: int = IMG_SIZE) -> np.ndarray:
    """Loads 6-band S2 GeoTIFF, normalises DN->=[0,1], resizes to (6, H, W)."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(tif_path) as src:
        n_bands = src.count
        if n_bands != 6:
            raise ValueError(
                f"Expected 6 bands in {tif_path.name}, got {n_bands}."
            )
        data = src.read(
            out_shape=(n_bands, target_size, target_size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    return np.clip(data / 10000.0, 0.0, 1.0)


class AirSentinelFusedDataset(Dataset):
    """
    Returns (data, label, no2_norm, no2_available) per row.

    data: either a (6, H, W) image tensor (full pipeline) or a (768,) feature
          tensor (cache mode). The calling code tells the model which it is via
          `precomputed_feats` — both paths produce the same result because the
          backbone output is deterministic when frozen.

    Cache mode is activated when feature_cache is provided AND covers all rows
    in this dataset. If any row is missing from the cache, the entire dataset
    falls back to full image loading (no silent per-item mixing).

    no2_norm: z-scored NO2 mean if available; 0.0 placeholder if not
              (the model ignores this value when no2_available == 0 by
              substituting the learned missing-token embedding instead).
    no2_available: int 0 or 1 — passed to the model so it knows which token to use.
    """

    def __init__(self, df: pd.DataFrame, data_dir: Path,
                 no2_mean_stat: float, no2_std_stat: float,
                 feature_cache: FeatureCache | None = None,
                 use_cache: bool = False):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.no2_mean_stat = no2_mean_stat
        self.no2_std_stat  = no2_std_stat
        self.feature_cache = feature_cache
        # use_cache=True means all items are guaranteed to be in cache;
        # __getitem__ will never fall back to image loading in this mode.
        self.use_cache = use_cache

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label_t = torch.tensor(CLASS_TO_IDX[row["dominant_pollutant"]], dtype=torch.long)

        avail = int(row["no2_available"])
        if avail == 1:
            raw = float(row["no2_mean"])
            no2_norm = (raw - self.no2_mean_stat) / (self.no2_std_stat + 1e-8)
        else:
            no2_norm = 0.0  # ignored by model; missing token used instead

        no2_t   = torch.tensor([[no2_norm]], dtype=torch.float32)  # (1, 1) for Linear
        avail_t = torch.tensor(avail, dtype=torch.long)

        if self.use_cache:
            # Cache mode: return pre-computed 768-d feature
            feat = self.feature_cache.get(row["s2_file"])
            # feat should never be None here — covers() was validated upfront
            return feat, label_t, no2_t, avail_t

        # Full pipeline: load raw image
        p = Path(row["s2_file"])
        tif_path = p if p.is_absolute() else self.data_dir / p
        image = load_s2_image(tif_path)
        return (
            torch.tensor(image, dtype=torch.float32),
            label_t,
            no2_t,
            avail_t,
        )


# ---------------------------------------------------------------------------
# Section 2 — Model
# ---------------------------------------------------------------------------

def load_backbone():
    """Loads Prithvi backbone from HuggingFace (cached after first run)."""
    try:
        from terratorch.registry import BACKBONE_REGISTRY
    except ImportError:
        raise ImportError("terratorch not found. Run: pip install 'terratorch>=1.1'")

    print("Loading Prithvi-EO-2.0-100M-TL backbone...")
    backbone = BACKBONE_REGISTRY.build("prithvi_eo_v2_100_tl", pretrained=True)
    backbone.eval()
    return backbone


class No2Encoder(nn.Module):
    """
    Tiny 2-layer MLP: scalar NO2 (z-scored) -> NO2_EMBED_DIM vector.
    Input shape: (B, 1, 1) — the extra dim makes it easy to pass from the dataset.
    """
    def __init__(self, embed_dim: int = NO2_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 1) — squeeze last dim
        return self.net(x.squeeze(-1))  # (B, 1) -> (B, embed_dim)


class FusedClassifier(nn.Module):
    """
    Optical-only baseline or NO2-fused classifier.

    fusion=False: backbone -> 768-d pool -> head(768, num_classes)
    fusion=True:  backbone -> 768-d pool
                  NO2 encoder OR missing token -> 32-d
                  cat(768+32=800) -> head(800, num_classes)

    forward() accepts precomputed_feats=True to skip the backbone when using
    the feature cache. The 768-d input in that case must come from the same
    backbone checkpoint; the cache hash check in FeatureCache enforces this.

    IMPORTANT: if the backbone is un-frozen (--unfreeze-backbone), do NOT use
    precomputed_feats=True — the backbone is changing so cached features are
    stale from step 1. The main block enforces this; this forward() does not
    need to re-check it.
    """

    def __init__(self, backbone: nn.Module, num_classes: int,
                 backbone_dim: int, fusion: bool):
        super().__init__()
        self.backbone = backbone
        self.fusion   = fusion

        if fusion:
            self.no2_encoder   = No2Encoder(NO2_EMBED_DIM)
            self.missing_token = nn.Parameter(torch.zeros(NO2_EMBED_DIM))
            head_in = backbone_dim + NO2_EMBED_DIM
        else:
            head_in = backbone_dim

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(0.1),
            nn.Linear(head_in, num_classes),
        )

    def forward(self, x: torch.Tensor,
                no2_val: torch.Tensor,
                no2_avail: torch.Tensor,
                precomputed_feats: bool = False) -> torch.Tensor:
        """
        x: (B, 768) if precomputed_feats=True (cached features)
           (B, C, H, W) if precomputed_feats=False (raw image, backbone runs)
        """
        if precomputed_feats:
            feats = x   # already (B, backbone_dim)
        else:
            raw   = self.backbone(x.unsqueeze(2))     # (B, C, H, W) -> (B, C, 1, H, W)
            out   = raw[-1] if isinstance(raw, (list, tuple)) else raw
            if out.dim() == 3:
                feats = out.mean(dim=1)           # (B, num_patches, dim) -> (B, dim)
            elif out.dim() == 4:
                feats = out.mean(dim=[2, 3])
            else:
                feats = out

        if not self.fusion:
            return self.head(feats)

        # Build NO2 embedding per sample in the batch
        B = feats.size(0)
        no2_embed = torch.zeros(B, NO2_EMBED_DIM, device=feats.device)

        for i in range(B):
            use_missing = (no2_avail[i].item() == 0)

            # Modality dropout: randomly drop during training
            if (self.training and not use_missing
                    and torch.rand(1).item() < MODALITY_DROPOUT):
                use_missing = True

            if use_missing:
                no2_embed[i] = self.missing_token
            else:
                no2_embed[i] = self.no2_encoder(no2_val[i : i + 1])[0]

        combined = torch.cat([feats, no2_embed], dim=1)  # (B, dim+32)
        return self.head(combined)


def build_model(backbone: nn.Module, fusion: bool,
                freeze_backbone: bool = True) -> tuple:
    """
    Wraps backbone in FusedClassifier.

    freeze_backbone=True  (default): backbone params frozen, cache usable.
    freeze_backbone=False: backbone params are trainable (e.g. LoRA or full
                           fine-tuning). Cache MUST NOT be used in this case
                           because the backbone changes every step.

    Returns (model, backbone_dim).
    """
    # Probe backbone output dim
    dummy = torch.zeros(1, 6, 1, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        raw = backbone(dummy)
    feat = raw[-1] if isinstance(raw, (list, tuple)) else raw
    backbone_dim = feat.shape[-1]
    print(f"Backbone output dim: {backbone_dim}")

    model = FusedClassifier(backbone, NUM_CLASSES, backbone_dim, fusion=fusion)

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        print("[!] Backbone NOT frozen — all backbone params are trainable. "
              "Feature cache will be skipped for this run.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    mode_str  = "fusion" if fusion else "baseline"
    frz_str   = "frozen" if freeze_backbone else "UNFROZEN"
    print(f"Model: {total:.1f}M total params | {trainable:.3f}M trainable "
          f"({mode_str}, backbone {frz_str})")
    return model, backbone_dim


# ---------------------------------------------------------------------------
# Section 3 — Training loop with best-val checkpointing
# ---------------------------------------------------------------------------

def run_training(model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, ckpt_path: Path,
                 use_cache: bool = False) -> dict:
    """
    Trains for NUM_EPOCHS. Saves checkpoint whenever val accuracy improves
    (best-val, not final-epoch). Returns epoch results dict.

    use_cache: if True, batches contain (feature_768d, label, no2, avail)
               and the backbone is skipped in model.forward.
               if False, batches contain (image_6xHxW, label, no2, avail)
               and the full pipeline runs normally.
    """
    model.to(DEVICE)
    # Only optimise parameters that require grad (backbone is frozen when use_cache)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler("cuda")

    best_val_acc = 0.0
    results = []

    cache_mode_str = "CACHE (backbone skipped)" if use_cache else "FULL PIPELINE (backbone runs)"
    print(f"\nTraining: {NUM_EPOCHS} epochs | batch={BATCH_SIZE} | device={DEVICE}")
    print(f"Data mode: {cache_mode_str}")
    print(f"Best-val checkpoint -> {ckpt_path}\n")

    epoch_times: list[float] = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t_epoch = time.time()

        # -- Train ---------------------------------------------------------------
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{NUM_EPOCHS} [train]", leave=False)
        for batch_idx, (data, labels, no2_val, no2_avail) in enumerate(pbar):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            no2_val   = no2_val.to(DEVICE)
            no2_avail = no2_avail.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(data, no2_val, no2_avail,
                               precomputed_feats=use_cache)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item()
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

            if batch_idx == 0 and epoch == 1 and DEVICE.type == "cuda":
                used = torch.cuda.memory_allocated() / 1e9
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"\n  [VRAM] Allocated: {used:.2f} GB  Peak: {peak:.2f} GB")

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = run_loss / len(train_loader)
        train_acc  = 100.0 * correct / total

        # -- Validate ------------------------------------------------------------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for data, labels, no2_val, no2_avail in val_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                no2_val   = no2_val.to(DEVICE)
                no2_avail = no2_avail.to(DEVICE)
                with autocast("cuda"):
                    logits = model(data, no2_val, no2_avail,
                                   precomputed_feats=use_cache)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc    = 100.0 * val_correct / val_total
        epoch_time = time.time() - t_epoch
        epoch_times.append(epoch_time)

        marker = " <- best" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch:2d} [{epoch_time:5.1f}s] "
              f"Loss: {train_loss:.4f} | Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}%{marker}")

        results.append({"epoch": epoch,
                        "epoch_time_s": round(epoch_time, 1),
                        "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 1),
                        "val_acc": round(val_acc, 1)})

        # Best-val checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "img_size": IMG_SIZE,
                "best_val_acc": best_val_acc,
                "best_epoch": epoch,
                "used_feature_cache": use_cache,
            }, ckpt_path)

    avg_epoch_s = sum(epoch_times) / len(epoch_times)
    print(f"\nBest val acc: {best_val_acc:.1f}% — checkpoint: {ckpt_path}")
    print(f"Avg epoch time: {avg_epoch_s:.1f}s  "
          f"(mode: {'cache' if use_cache else 'full pipeline'})")
    return {"epochs": results, "best_val_acc": best_val_acc,
            "avg_epoch_time_s": round(avg_epoch_s, 1),
            "used_cache": use_cache}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuse-no2", action="store_true",
                    help="Enable NO2 sensor fusion (Phase C). "
                         "Without this flag, runs optical-only baseline.")
    ap.add_argument("--unfreeze-backbone", action="store_true",
                    help="Allow backbone params to be trained (e.g. LoRA / full fine-tune). "
                         "DISABLES feature cache — cached features are only valid for a "
                         "frozen backbone. Do not use with --fuse-no2 unless you also "
                         "retrain the NO2 encoder from scratch.")
    args = ap.parse_args()

    fusion          = args.fuse_no2
    freeze_backbone = not args.unfreeze_backbone

    mode = "FUSION (optical + NO2)" if fusion else "BASELINE (optical only)"
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    if not freeze_backbone:
        print(f"  Backbone: UNFROZEN (feature cache disabled)")
    print(f"{'='*60}\n")

    ckpt_path = Path("prithvi_fused_best.pt" if fusion else "prithvi_baseline_best.pt")

    # -- Load data ---------------------------------------------------------------
    train_df, val_df, no2_mean_stat, no2_std_stat = load_and_split(LABELS_CSV)

    # -- Feature cache setup (only when backbone is frozen) ----------------------
    use_cache = False
    feature_cache = None

    if freeze_backbone:
        feature_cache = FeatureCache(CACHE_PATH, BACKBONE_CKPT)
        if feature_cache.valid:
            all_s2 = pd.concat([train_df, val_df])["s2_file"].tolist()
            covered, total_needed = feature_cache.coverage_report(all_s2)
            if covered == total_needed:
                use_cache = True
                print(f"[cache] Full coverage ({covered}/{total_needed}) — "
                      "backbone forward will be SKIPPED during training\n")
            else:
                missing_n = total_needed - covered
                print(f"[cache] Partial coverage ({covered}/{total_needed}). "
                      f"{missing_n} items missing.\n"
                      f"        Run cache_backbone_features.py to complete the cache.\n"
                      f"        Falling back to full image pipeline for this run.\n")
                feature_cache = None   # don't use partial cache
    else:
        print("[!] --unfreeze-backbone set: feature cache is DISABLED for this run.\n"
              "    Reason: backbone weights change every step, so cached features\n"
              "    from the start of training would be wrong by step 2.\n")

    # -- Build datasets ----------------------------------------------------------
    train_ds = AirSentinelFusedDataset(
        train_df, DATA_DIR, no2_mean_stat, no2_std_stat,
        feature_cache=feature_cache, use_cache=use_cache)
    val_ds   = AirSentinelFusedDataset(
        val_df,   DATA_DIR, no2_mean_stat, no2_std_stat,
        feature_cache=feature_cache, use_cache=use_cache)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=(DEVICE.type == "cuda"),
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=(DEVICE.type == "cuda"),
                              persistent_workers=True)

    print(f"Train: {len(train_ds)} rows | Val: {len(val_ds)} rows")

    # -- Build model -------------------------------------------------------------
    backbone = load_backbone()

    # Load Phase 2 fine-tuned weights into backbone where keys match
    if BACKBONE_CKPT.exists():
        ckpt = torch.load(BACKBONE_CKPT, map_location="cpu", weights_only=True)
        sd = ckpt["model_state_dict"]
        # Keys in Phase 2 checkpoint are prefixed "backbone.*"
        bb_sd = {k.replace("backbone.", "", 1): v
                 for k, v in sd.items() if k.startswith("backbone.")}
        missing, unexpected = backbone.load_state_dict(bb_sd, strict=False)
        print(f"Loaded Phase 2 backbone weights from {BACKBONE_CKPT} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"[!] Phase 2 checkpoint not found at {BACKBONE_CKPT} — "
              f"using fresh pretrained weights")

    model, _ = build_model(backbone, fusion=fusion, freeze_backbone=freeze_backbone)

    # -- Train -------------------------------------------------------------------
    run_results = run_training(model, train_loader, val_loader, ckpt_path,
                               use_cache=use_cache)

    # -- Persist NO2 stats into checkpoint so inference can use same norm --------
    if fusion:
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        saved["no2_mean_stat"]    = no2_mean_stat
        saved["no2_std_stat"]     = no2_std_stat
        saved["no2_embed_dim"]    = NO2_EMBED_DIM
        saved["modality_dropout"] = MODALITY_DROPOUT
        torch.save(saved, ckpt_path)

    # -- Save epoch log ----------------------------------------------------------
    log_path = Path(f"train_log_{'fused' if fusion else 'baseline'}.json")
    with open(log_path, "w") as f:
        json.dump(run_results, f, indent=2)
    print(f"Epoch log -> {log_path}")

    print(f"\n{'='*60}")
    print(f"  {mode} complete.")
    print(f"  Best val acc : {run_results['best_val_acc']:.1f}%")
    print(f"  Avg epoch    : {run_results['avg_epoch_time_s']:.1f}s "
          f"({'cache' if use_cache else 'full pipeline'})")
    print(f"  Checkpoint   : {ckpt_path.resolve()}")
    print(f"{'='*60}")
