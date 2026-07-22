# train_prithvi_lora.py — AirSentinel Prithvi LoRA Fine-Tuning (Phase E)
#
# Applies LoRA adapters to Prithvi-EO-2.0-100M-TL via TerraTorch's PEFT helpers,
# then fine-tunes on the full 5,600-row augmented dataset.
#
# LoRA config: r=8, lora_alpha=16
#   target_modules = [q_linear, k_linear, v_linear, proj]
#   (replace_qkv="qkv" splits the fused QKV Linear into three before applying LoRA)
#
# Only LoRA adapter weights + classification head are trained.
# Backbone non-adapter weights stay frozen (handled automatically by get_peft_model).
# Best-val-acc checkpoint is saved every epoch it improves.
#
# Usage: python train_prithvi_lora.py
# Requires: peft>=0.10  (pip install peft)

# %% -- SECTION 1: Imports & GPU check -----------------------------------------
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
    print("\nWARNING: No GPU detected. Training will run on CPU — very slow.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% -- SECTION 2: Configuration -----------------------------------------------
DATA_DIR       = Path(r"G:\My Drive\AirSentinel_Satellite_Images")
LABELS_CSV     = DATA_DIR / "labels_augmented.csv"

CLASSES        = ["dust", "crop_burning_smoke", "industrial_haze", "traffic_heavy", "clear"]
NUM_CLASSES    = len(CLASSES)
CLASS_TO_IDX   = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE       = 224
BATCH_SIZE     = 32     # LoRA: only adapter weights in grad graph — lower memory than full FT
NUM_EPOCHS     = 15
LR             = 2e-4   # slightly higher than full FT is fine for LoRA (small param count)
VAL_FRACTION   = 0.2
RANDOM_SEED    = 42

CHECKPOINT_PATH     = Path("prithvi_lora_best.pt")   # best-val checkpoint
FINAL_CHECKPOINT    = Path("prithvi_lora_final.pt")  # end-of-training checkpoint

# LoRA hyperparameters passed to TerraTorch's get_peft_backbone()
PEFT_CONFIG = {
    "method": "LORA",
    "replace_qkv": "qkv",   # splits blocks.N.attn.qkv -> q_linear/k_linear/v_linear
    "peft_config_kwargs": {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_linear", "k_linear", "v_linear", "proj"],
        "lora_dropout": 0.05,
        "bias": "none",
    },
}


# %% -- SECTION 3: Data loading ------------------------------------------------

def load_labels_table(labels_csv: Path) -> tuple:
    """
    Group-aware train/val split on unique (zone, date) originals.
    All 7 augmented siblings of a val image are excluded from train.
    """
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Labels CSV not found:\n  {labels_csv}\n"
            "Run build_labels.py then augment_labels.py first."
        )
    df = pd.read_csv(labels_csv)

    required = {"zone", "date", "s2_file", "dominant_pollutant", "label_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {missing}")

    unknown = set(df["dominant_pollutant"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown class names in CSV: {unknown}")

    heuristic = df[df["label_source"].str.startswith("CAAQMS_heuristic")].copy()
    n_placeholder = len(df) - len(heuristic)

    print(f"Labels loaded : {len(df)} total rows")
    print(f"  Heuristic   : {len(heuristic)} (train + val)")
    print(f"  Placeholder : {n_placeholder} (excluded)")
    print("  Distribution:")
    for cat, cnt in heuristic["dominant_pollutant"].value_counts().items():
        print(f"    {cat}: {cnt}")

    # Split on unique (zone, date) pairs — augmented siblings follow the original
    groups = (
        heuristic[["zone", "date"]]
        .drop_duplicates()
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    n_val = max(1, int(len(groups) * VAL_FRACTION))
    val_pairs = set(zip(groups.iloc[:n_val]["zone"], groups.iloc[:n_val]["date"]))

    is_val   = heuristic.apply(lambda r: (r["zone"], r["date"]) in val_pairs, axis=1)
    val_df   = heuristic[is_val].copy()
    train_df = heuristic[~is_val].copy()

    print(f"  Groups: {len(groups)-n_val} train / {n_val} val")
    print(f"  Rows  : {len(train_df)} train / {len(val_df)} val")
    return train_df, val_df


def load_s2_image(tif_path: Path, target_size: int = IMG_SIZE) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(tif_path) as src:
        if src.count != 6:
            raise ValueError(f"Expected 6 bands in {tif_path.name}, got {src.count}.")
        data = src.read(
            out_shape=(src.count, target_size, target_size),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    return np.clip(data / 10000.0, 0.0, 1.0)   # DN -> [0, 1]


class AirSentinelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: Path):
        self.df       = df.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p        = Path(row["s2_file"])
        tif_path = p if p.is_absolute() else self.data_dir / p
        image    = load_s2_image(tif_path)
        label    = CLASS_TO_IDX[row["dominant_pollutant"]]
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# %% -- SECTION 4: Model — Prithvi + LoRA adapters + classification head -------

def build_lora_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    1. Loads Prithvi-EO-2.0-100M-TL backbone.
    2. Calls get_peft_backbone() which:
         a. replace_qkv("qkv"): splits each blocks.N.attn.qkv fused Linear into
            QKVSep(q_linear, k_linear, v_linear) — required for LoRA to target Q/K/V.
         b. get_peft_model(): wraps backbone with LoRA adapters on target_modules,
            freezes all non-adapter weights automatically.
    3. Attaches a trainable classification head.
    """
    try:
        from terratorch.registry import BACKBONE_REGISTRY
        from terratorch.models.peft_utils import get_peft_backbone
    except ImportError:
        raise ImportError("terratorch not found. pip install terratorch>=1.1")

    try:
        import peft  # noqa: F401
    except ImportError:
        raise ImportError("peft not found. pip install peft>=0.10")

    print("Loading Prithvi-EO-2.0-100M-TL backbone...")
    backbone = BACKBONE_REGISTRY.build("prithvi_eo_v2_100_tl", pretrained=True)
    backbone.eval()
    print("Backbone loaded.")

    print("Applying LoRA adapters (r=8, alpha=16, targets: q/k/v/proj)...")
    backbone = get_peft_backbone(PEFT_CONFIG, backbone)
    backbone.print_trainable_parameters()

    # Probe output dim via dummy forward
    dummy = torch.zeros(1, 6, 1, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        raw = backbone(dummy)
    feat = raw[-1] if isinstance(raw, (list, tuple)) else raw
    if feat.dim() == 3:
        embed_dim = feat.shape[-1]      # (B, patches, dim)
    elif feat.dim() == 4:
        embed_dim = feat.shape[1]       # (B, dim, H, W)
    else:
        embed_dim = feat.shape[-1]
    print(f"Backbone output dim: {embed_dim}")

    class PrithviLoraClassifier(nn.Module):
        def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x   = x.unsqueeze(2)   # (B, C, H, W) -> (B, C, 1, H, W)
            raw = self.backbone(x)
            features = raw[-1] if isinstance(raw, (list, tuple)) else raw
            if features.dim() == 3:
                features = features.mean(dim=1)
            elif features.dim() == 4:
                features = features.mean(dim=[2, 3])
            return self.head(features)

    model = PrithviLoraClassifier(backbone, embed_dim, num_classes)

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: {total:.1f}M total | {trainable:.2f}M trainable ({100*trainable/total:.2f}%)")
    return model


# %% -- SECTION 5: Training loop with best-val checkpointing -------------------

def run_training(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> float:
    model.to(DEVICE)

    # Only pass parameters that require grad to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler("cuda") if DEVICE.type == "cuda" else None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR / 10
    )

    best_val_acc = 0.0
    print(f"\nLoRA training: {NUM_EPOCHS} epochs | batch={BATCH_SIZE} | device={DEVICE}")
    print("[!] Labels are CAAQMS heuristic — not lab-verified.\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        # -- Train ----------------------------------------------------------------
        model.train()
        run_loss, correct, total_samples = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler is not None:
                with autocast("cuda"):
                    logits = model(images)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            run_loss += loss.item()
            preds     = logits.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if batch_idx == 0 and epoch == 1 and DEVICE.type == "cuda":
                used = torch.cuda.memory_allocated() / 1e9
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"\n  [VRAM after first batch]  Alloc: {used:.2f} GB | Peak: {peak:.2f} GB")

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = run_loss / len(train_loader)
        train_acc  = 100.0 * correct / total_samples

        # -- Validate -------------------------------------------------------------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if scaler is not None:
                    with autocast("cuda"):
                        logits = model(images)
                else:
                    logits = model(images)
                preds       = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        marker  = " *** best ***" if val_acc > best_val_acc else ""
        print(
            f"  Epoch {epoch:2d} — Loss: {train_loss:.4f} | "
            f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%{marker}"
        )

        # -- Best-val checkpoint --------------------------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(model, CHECKPOINT_PATH, epoch, val_acc, label="best-val")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.1f}%")
    return best_val_acc


def _save_checkpoint(model: nn.Module, path: Path, epoch: int, val_acc: float, label: str) -> None:
    # Save the full model state — LoRA adapter weights are part of backbone state_dict
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes":          CLASSES,
            "class_to_idx":     CLASS_TO_IDX,
            "img_size":         IMG_SIZE,
            "epoch":            epoch,
            "val_acc":          val_acc,
            "lora_config":      PEFT_CONFIG,
            "note": (
                "Phase E — Prithvi LoRA r=8, trained on CAAQMS heuristic labels. "
                "Val accuracy is against held-out heuristic rows, not ground truth."
            ),
        },
        path,
    )
    print(f"  [{label}] Checkpoint saved -> {path.resolve()}  (epoch {epoch}, val {val_acc:.1f}%)")


# %% -- MAIN -------------------------------------------------------------------
if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"ERROR: DATA_DIR does not exist:\n  {DATA_DIR}")
        sys.exit(1)

    # -- Data ------------------------------------------------------------------
    train_df, val_df = load_labels_table(LABELS_CSV)

    train_ds = AirSentinelDataset(train_df, DATA_DIR)
    val_ds   = AirSentinelDataset(val_df,   DATA_DIR)
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=True,
    )

    # -- Model -----------------------------------------------------------------
    model = build_lora_model()

    # -- Train -----------------------------------------------------------------
    best_val = run_training(model, train_loader, val_loader)

    # -- Final checkpoint ------------------------------------------------------
    _save_checkpoint(model, FINAL_CHECKPOINT, NUM_EPOCHS, best_val, label="final")

    # -- Summary ---------------------------------------------------------------
    print()
    print("=" * 65)
    print("PHASE E — LORA TRAINING COMPLETE")
    print("=" * 65)
    print(f"  LoRA: r=8, alpha=16, targets: q/k/v/proj")
    print(f"  Dataset: 5,600 rows (700 orig + 4,900 aug, group-aware split)")
    print(f"  Best val acc: {best_val:.1f}%")
    print(f"  Best checkpoint : {CHECKPOINT_PATH.resolve()}")
    print(f"  Final checkpoint: {FINAL_CHECKPOINT.resolve()}")
    print()
    print("Labels are CAAQMS heuristic — do not cite val acc as ground truth.")
