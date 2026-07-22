import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import rasterio
from rasterio.enums import Resampling

DATA_DIR     = Path(r"G:\My Drive\AirSentinel_Satellite_Images")
LABELS_CSV   = DATA_DIR / "labels_augmented.csv"
CKPT         = Path("prithvi_lora_best.pt")
CLASSES      = ["dust", "crop_burning_smoke", "industrial_haze", "traffic_heavy", "clear"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE     = 224
BATCH_SIZE   = 32
VAL_FRACTION = 0.2
RANDOM_SEED  = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_s2(path, size=IMG_SIZE):
    with rasterio.open(path) as src:
        data = src.read(out_shape=(src.count, size, size),
                        resampling=Resampling.bilinear).astype(np.float32)
    return np.clip(data / 10000.0, 0.0, 1.0)


class ValDS(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        p   = Path(row["s2_file"])
        img = load_s2(p if p.is_absolute() else DATA_DIR / p)
        lbl = CLASS_TO_IDX[row["dominant_pollutant"]]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)


class PrithviLoraClassifier(nn.Module):
    def __init__(self, backbone, embed_dim=768, num_classes=5):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.1),
                                  nn.Linear(embed_dim, num_classes))
    def forward(self, x):
        x   = x.unsqueeze(2)
        raw = self.backbone(x)
        f   = raw[-1] if isinstance(raw, (list, tuple)) else raw
        if f.dim() == 3:   f = f.mean(1)
        elif f.dim() == 4: f = f.mean([2, 3])
        return self.head(f)


if __name__ == "__main__":
    # ── reproduce val split ──────────────────────────────────────────────────
    df        = pd.read_csv(LABELS_CSV)
    heuristic = df[df["label_source"].str.startswith("CAAQMS_heuristic")].copy()
    groups    = (heuristic[["zone", "date"]].drop_duplicates()
                 .sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True))
    n_val     = max(1, int(len(groups) * VAL_FRACTION))
    val_pairs = set(zip(groups.iloc[:n_val]["zone"], groups.iloc[:n_val]["date"]))
    is_val    = heuristic.apply(lambda r: (r["zone"], r["date"]) in val_pairs, axis=1)
    val_df    = heuristic[is_val].copy().reset_index(drop=True)
    print(f"Val set: {len(val_df)} rows across {n_val} unique (zone, date) groups")

    loader = DataLoader(ValDS(val_df), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    # ── rebuild model + load checkpoint ─────────────────────────────────────
    from terratorch.registry import BACKBONE_REGISTRY
    from terratorch.models.peft_utils import get_peft_backbone

    PEFT_CONFIG = {
        "method": "LORA",
        "replace_qkv": "qkv",
        "peft_config_kwargs": {
            "r": 8, "lora_alpha": 16,
            "target_modules": ["q_linear", "k_linear", "v_linear", "proj"],
            "lora_dropout": 0.05, "bias": "none",
        },
    }

    print("Loading backbone + LoRA structure (pretrained=False — weights from checkpoint)...")
    backbone = BACKBONE_REGISTRY.build("prithvi_eo_v2_100_tl", pretrained=False)
    backbone = get_peft_backbone(PEFT_CONFIG, backbone)

    model = PrithviLoraClassifier(backbone).to(DEVICE)
    ckpt  = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint loaded — epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.1f}%\n")

    # ── inference ────────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            preds       = model(imgs.to(DEVICE)).argmax(1).cpu().tolist()
            all_preds  += preds
            all_labels += lbls.tolist()

    idx_to_cls  = {v: k for k, v in CLASS_TO_IDX.items()}
    pred_names  = [idx_to_cls[p] for p in all_preds]
    label_names = [idx_to_cls[l] for l in all_labels]

    print("--- Classification Report ---")
    print(classification_report(label_names, pred_names, target_names=CLASSES, digits=3))
