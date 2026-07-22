# train_prithvi.py — AirSentinel Prithvi Fine-Tuning Pipeline (Phase 2 — Real Training)
#
# GOAL: real fine-tuning attempt on CAAQMS heuristic labels.
# Placeholder rows (label_source contains 'PLACEHOLDER') are excluded from training.
# A train/val split is applied so accuracy is measured on held-out data.
# Accuracy numbers ARE meaningful relative to baseline, but should be reported with
# the caveat that labels are heuristic (rule-based), not lab-verified ground truth.
#
# BEFORE RUNNING: activate the 'airsen' venv (.venv\Scripts\Activate.ps1 on Windows).
#
# NOTE on TerraTorch's built-in ClassificationTask:
#   TerraTorch has a ClassificationTask wrapper, but it's YAML-config-driven
#   and more complex than needed here. We use a custom classification head directly
#   on the backbone — simpler and equally correct.
# ------------------------------------------------------------------------------

# %% -- SECTION 1: Imports & GPU check -----------------------------------------
import os
import sys
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

if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print()
    print("WARNING: No GPU detected. Training will run on CPU — very slow.")
    print("If you have CUDA installed, check: https://pytorch.org/get-started/locally/")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% -- SECTION 2: Configuration -----------------------------------------------
# [!]  SET THIS before running. Point it at the local folder that Google Drive
#     for Desktop syncs to. Ask your OS where it landed — do NOT use a Colab path
#     like /content/drive/... (that path only exists inside Colab).
#
#     Common Windows path:  r"G:\My Drive\AirSentinel_Satellite_Images"
#     Common Mac path:      os.path.expanduser("~/Google Drive/My Drive/AirSentinel_Satellite_Images")
DATA_DIR = Path(r"G:\My Drive\AirSentinel_Satellite_Images")
LABELS_CSV = DATA_DIR / "labels_augmented.csv"   # use augmented set (1878 rows, 8x expansion)

CLASSES = ["dust", "crop_burning_smoke", "industrial_haze", "traffic_heavy", "clear"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE = 224      # Resize all S2 images to this square size (ViT expected input)
BATCH_SIZE = 4      # Raised from 2 — Phase 2 used only 1.4 GB of 8.6 GB VRAM at batch=2
NUM_EPOCHS = 10
LR = 1e-4
CHECKPOINT_PATH = Path("prithvi_airsen_augmented.pt")
VAL_FRACTION = 0.2  # 20% of original (zone, date) groups held out — see load_labels_table
RANDOM_SEED = 42

# Prithvi-EO-2.0-100M uses a ViT-Base backbone -> embed_dim=768.
# If the probe in Section 4 reports a different value, it auto-corrects.
PRITHVI_EMBED_DIM = 768

# %% -- SECTION 3: Data loading ------------------------------------------------

def load_labels_table(labels_csv: Path) -> tuple:
    """
    Loads labels_augmented.csv, excludes placeholder rows, then splits by original
    (zone, date) group — NOT by row. This prevents data leakage: augmented versions
    of a training image cannot appear in val, and vice versa.

    Split logic:
      1. Find unique (zone, date) pairs among heuristic rows (233 originals).
      2. Randomly assign 20% of those pairs to val.
      3. All rows (original + 7 augmented versions) for each pair go to the same split.
    """
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Labels CSV not found at:\n  {labels_csv}\n\n"
            "Make sure build_labels.py + augmentation (Step 11) have been run."
        )
    df = pd.read_csv(labels_csv)

    required = {"zone", "date", "s2_file", "dominant_pollutant", "label_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV is missing expected columns: {missing}")

    unknown_labels = set(df["dominant_pollutant"]) - set(CLASSES)
    if unknown_labels:
        raise ValueError(
            f"Labels CSV contains unknown class names: {unknown_labels}\n"
            f"Expected one of: {CLASSES}"
        )

    heuristic = df[df["label_source"].str.startswith("CAAQMS_heuristic")].copy()
    placeholder_count = (~df["label_source"].str.startswith("CAAQMS_heuristic")).sum()

    print(f"Labels loaded: {len(df)} total rows")
    print(f"  CAAQMS_heuristic rows (will train/val on): {len(heuristic)}")
    print(f"  PLACEHOLDER rows (excluded from training): {placeholder_count}")
    print(f"  Label distribution (heuristic only):")
    for cat, count in heuristic["dominant_pollutant"].value_counts().items():
        print(f"    {cat}: {count}")

    # Group-aware split: split on unique (zone, date) originals, then propagate to all rows
    # This prevents augmented siblings of val images from leaking into the train set.
    groups = heuristic[["zone", "date"]].drop_duplicates().sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    n_val_groups = max(1, int(len(groups) * VAL_FRACTION))
    val_groups = set(zip(groups.iloc[:n_val_groups]["zone"], groups.iloc[:n_val_groups]["date"]))

    is_val = heuristic.apply(lambda r: (r["zone"], r["date"]) in val_groups, axis=1)
    val_df   = heuristic[is_val].copy()
    train_df = heuristic[~is_val].copy()

    print(f"  Group-aware split: {len(groups) - n_val_groups} train groups / {n_val_groups} val groups")
    print(f"  Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    return train_df, val_df


def load_s2_image(tif_path: Path, target_size: int = IMG_SIZE) -> np.ndarray:
    """
    Loads a 6-band Sentinel-2 GeoTIFF (B2 B3 B4 B8A B11 B12), normalizes from
    16-bit DN values to float [0, 1] by dividing by 10000, and resizes to
    (6, target_size, target_size).

    Raises clearly if the file has the wrong band count (catches old 3-band exports).
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(tif_path) as src:
        n_bands = src.count
        if n_bands != 6:
            raise ValueError(
                f"Expected 6 bands in {tif_path.name}, got {n_bands}.\n"
                "This is probably an old 3-band export. Re-run the Colab notebook "
                "with the updated 6-band export (B2, B3, B4, B8A, B11, B12)."
            )
        data = src.read(
            out_shape=(n_bands, target_size, target_size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    data = np.clip(data / 10000.0, 0.0, 1.0)   # Sentinel-2 surface reflectance scale
    return data  # shape: (6, target_size, target_size)


class AirSentinelDataset(Dataset):
    """
    Loads S2 images and PLACEHOLDER labels for the AirSentinel plumbing check.
    Swap `load_labels_table` output with real CPCB labels once available —
    no other change needed.
    """

    def __init__(self, df: pd.DataFrame, data_dir: Path):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        tif_path = self.data_dir / row["s2_file"]
        image = load_s2_image(tif_path)                              # (6, 224, 224)
        label = CLASS_TO_IDX[row["dominant_pollutant"]]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# %% -- SECTION 4: Model — Prithvi backbone + classification head --------------

def build_model(num_classes: int = NUM_CLASSES, embed_dim: int = PRITHVI_EMBED_DIM) -> nn.Module:
    """
    Loads Prithvi-EO-2.0-100M-TL from Hugging Face via TerraTorch (auto-downloaded
    on first run, cached locally after that), probes the output shape, then attaches
    a custom classification head (LayerNorm -> Dropout -> Linear).

    This is a custom head rather than TerraTorch's ClassificationTask because
    ClassificationTask requires a YAML config file and is more complex than needed
    for this plumbing check.
    """
    try:
        from terratorch.registry import BACKBONE_REGISTRY
    except ImportError:
        raise ImportError(
            "terratorch not found. Run:\n"
            '  pip install "terratorch>=1.1"\n'
            "Make sure you are in the 'airsen' conda environment."
        )

    print("Loading Prithvi-EO-2.0-100M-TL backbone from Hugging Face...")
    print("(First run: downloads ~400 MB of weights. Subsequent runs use cache.)")
    backbone = BACKBONE_REGISTRY.build("prithvi_eo_v2_100_tl", pretrained=True)
    backbone.eval()
    print("Backbone loaded successfully.")

    # -- Probe output shape with a dummy forward pass ---------------------------
    # Prithvi patch_embed is Conv3D with weight [768, 6, 1, 16, 16], so it
    # expects input shape: (batch, channels, time_steps, height, width) = (B, C, T, H, W)
    dummy = torch.zeros(1, 6, 1, IMG_SIZE, IMG_SIZE)  # (B, C, T, H, W)
    with torch.no_grad():
        try:
            raw_out = backbone(dummy)
        except Exception as exc:
            raise RuntimeError(
                f"Backbone failed on dummy input (shape {tuple(dummy.shape)}): {exc}\n"
                "Possible causes:\n"
                "  - Wrong terratorch version (need >=1.1)\n"
                "  - Model key typo in BACKBONE_REGISTRY.build()\n"
                "  - IMG_SIZE not divisible by the model's patch size (try 224)"
            ) from exc

    # Backbone can return a single tensor or a list of tensors (multi-scale).
    # For classification we want a single feature tensor.
    if isinstance(raw_out, (list, tuple)):
        feature_tensor = raw_out[-1]   # last feature map is the most abstract
        print(f"Backbone returned {len(raw_out)}-element list; using last: shape {tuple(feature_tensor.shape)}")
    else:
        feature_tensor = raw_out
        print(f"Backbone output shape: {tuple(feature_tensor.shape)}")

    # Detect actual embed_dim from the output
    actual_dim = feature_tensor.shape[-1]
    if actual_dim != embed_dim:
        print(f"Adjusting embed_dim: {embed_dim} -> {actual_dim} (from actual backbone output)")
        embed_dim = actual_dim

    # -- Build the classifier ---------------------------------------------------
    _embed_dim = embed_dim  # capture for the inner class

    class PrithviClassifier(nn.Module):
        def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, H, W)  <- what the DataLoader gives us
            x = x.unsqueeze(2)  # -> (B, C, 1, H, W)  <- Prithvi wants (B, C, T, H, W)

            raw = self.backbone(x)

            if isinstance(raw, (list, tuple)):
                features = raw[-1]
            else:
                features = raw

            # Pool to (B, embed_dim) regardless of spatial or sequence output format
            if features.dim() == 3:
                # (B, num_patches, embed_dim) — sequence output (most common for ViT)
                features = features.mean(dim=1)
            elif features.dim() == 4:
                # (B, embed_dim, H', W') — spatial feature map
                features = features.mean(dim=[2, 3])
            # elif features.dim() == 2: already (B, embed_dim) — no pooling needed

            return self.head(features)

    model = PrithviClassifier(backbone, _embed_dim, num_classes)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: {total:.1f}M total params | {trainable:.1f}M trainable")
    return model


# %% -- SECTION 5: Training loop -----------------------------------------------

def run_training(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """
    Runs NUM_EPOCHS of training with mixed precision (float16) to keep VRAM low.
    Validation accuracy is measured on the held-out val split after each epoch.

    Labels are CAAQMS heuristic (rule-based, not lab-verified). Accuracy numbers
    are real (not random), but should be reported with that caveat.
    """
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    print(f"\nStarting training: {NUM_EPOCHS} epoch(s) | batch_size={BATCH_SIZE} | device={DEVICE}")
    print("[!]  Labels are CAAQMS heuristic — not lab-verified. Report accuracy with this caveat.\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        # -- Train ----------------------------------------------------------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Print VRAM usage after the very first batch so you can spot OOM early
            if batch_idx == 0 and epoch == 1 and DEVICE.type == "cuda":
                used_gb = torch.cuda.memory_allocated() / 1e9
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"\n  [VRAM after first batch]  Allocated: {used_gb:.2f} GB  |  Peak: {peak_gb:.2f} GB")
                if used_gb > 6.5:
                    print("  [!]  VRAM usage is high. If OOM occurs, set BATCH_SIZE=1 at the top of this file.")

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # -- Validate -------------------------------------------------------------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast("cuda"):
                    logits = model(images)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        print(f"  Epoch {epoch:2d} — Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    print("\nTraining complete. [OK]")


# %% -- SECTION 6: Save checkpoint ---------------------------------------------

def save_checkpoint(model: nn.Module, path: Path) -> None:
    """
    Saves the model state dict locally. The checkpoint is tagged so it's clear
    it was trained on placeholder labels and should not be used for real inference.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "img_size": IMG_SIZE,
            "note": (
                "Phase 2 — trained on CAAQMS heuristic labels (rule-based, not lab-verified). "
                "Val accuracy is against held-out heuristic rows, not ground truth. "
                "Do not cite as ground-truth performance."
            ),
        },
        path,
    )
    print(f"Checkpoint saved -> {path.resolve()}")


# %% -- MAIN -------------------------------------------------------------------
if __name__ == "__main__":

    # -- Guard: DATA_DIR must be set -------------------------------------------
    if str(DATA_DIR) == "REPLACE_WITH_YOUR_LOCAL_DRIVE_SYNCED_FOLDER_PATH":
        print("=" * 65)
        print("ACTION REQUIRED: open this file and set DATA_DIR near the top.")
        print("Set it to the folder where Google Drive for Desktop syncs your")
        print("AirSentinel_Satellite_Images folder. Example (Windows):")
        print(r'  DATA_DIR = Path(r"G:\My Drive\AirSentinel_Satellite_Images")')
        print("=" * 65)
        sys.exit(1)

    if not DATA_DIR.exists():
        print(f"ERROR: DATA_DIR does not exist:\n  {DATA_DIR}")
        print("Check that Google Drive for Desktop is running and has finished syncing.")
        sys.exit(1)

    # -- Load data -------------------------------------------------------------
    train_df, val_df = load_labels_table(LABELS_CSV)

    train_dataset = AirSentinelDataset(train_df, DATA_DIR)
    val_dataset   = AirSentinelDataset(val_df,   DATA_DIR)
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,      # Keep 0 on Windows — avoids multiprocessing pickle issues
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # -- Build model -----------------------------------------------------------
    model = build_model()

    # -- Train -----------------------------------------------------------------
    run_training(model, train_loader, val_loader)

    # -- Save checkpoint -------------------------------------------------------
    save_checkpoint(model, CHECKPOINT_PATH)

    # -- Summary ---------------------------------------------------------------
    print()
    print("=" * 65)
    print("PHASE 2 TRAINING COMPLETE")
    print("=" * 65)
    print("  [OK]  S2 images loaded (6-band confirmed per file)")
    print("  [OK]  Prithvi-EO-2.0-100M-TL backbone loaded")
    print("  [OK]  Trained on augmented CAAQMS heuristic labels (8x expansion, group-aware split)")
    print("  [OK]  Val accuracy measured on held-out original zone/date groups")
    print(f"  [OK]  Checkpoint saved: {CHECKPOINT_PATH.resolve()}")
    print()
    print("Labels are heuristic — do not cite val accuracy as ground-truth performance.")
    print("Next: Phase 3 — Enforcement Zone Ranker.")
