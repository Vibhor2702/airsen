# enforcement_ranker.py — AirSentinel Phase 3: Enforcement Zone Ranker
#
# Combines:
#   1. Prithvi-EO source attribution (via current fine-tuned checkpoint)
#   2. Vehicle Emission Load Index (from vehicle_emissions pipeline, VAHAN-sourced)
#   3. Land-use signal: NONE — no land-use data exists in this repo. Noted explicitly.
#
# IMPORTANT — model_version tag on every output row:
#   "247-image placeholder -- not final"
# This is built from the current 247-original-image model. When a better checkpoint
# is available, change PRITHVI_CKPT (line ~22 below) — that is the only change needed.
# All ranking logic, output schema, and downstream integrations stay the same.
#
# Output files:
#   enforcement/enforcement_ranking.csv            — full ranked output for this script
#   P2/airsentinel-master/satellite_attribution/outputs/attribution.csv
#                                                  — satellite-only schema picked up by fuse.py
#
# Labels are CAAQMS heuristic (rule-based, not ground truth). Attribution confidence is
# from the model's softmax output, not a calibrated probability. Report accordingly.
#
# Usage (run from any directory — paths are __file__-relative):
#   .venv/Scripts/python.exe enforcement/enforcement_ranker.py
#   .venv/Scripts/python.exe enforcement/enforcement_ranker.py --date 2026-07-16

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Final checkpoint — Prithvi LoRA r=8, epoch 2, 71.1% val acc, 700-image dataset.
# All paths are __file__-relative so the script runs correctly from any working dir.
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).parent          # = /enforcement/
_REPO_ROOT = _HERE.parent                   # = repo root

PRITHVI_CKPT    = _REPO_ROOT / "prithvi_lora_best.pt"
DATA_DIR        = Path(r"G:\My Drive\AirSentinel_Satellite_Images")
LABELS_CSV      = DATA_DIR / "labels.csv"
VEHICLE_INDEX   = _REPO_ROOT / "vehicle-emissions" / "outputs" / "vehicle_emission_index.csv"
OUT_RANKING     = _HERE / "enforcement_ranking.csv"
OUT_ATTRIBUTION = _REPO_ROOT / "P2" / "airsentinel-master" / "satellite_attribution" / "outputs" / "attribution.csv"

IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_VERSION = "700-image LoRA r=8 -- final"

# Classes must match train_prithvi.py order (stored in checkpoint, verified at load time)
CLASSES    = ["dust", "crop_burning_smoke", "industrial_haze", "traffic_heavy", "clear"]
NUM_CLASSES = len(CLASSES)
EMBED_DIM   = 768  # Prithvi-EO-2.0-100M architectural constant

# ---------------------------------------------------------------------------
# Enforcement priority weights per predicted source class.
# Rationale documented here — not in a config file — because these are
# domain judgements that need human review, not tuned parameters.
#
#   traffic_heavy: 1.0  — primary enforcement target (odd/even, restriction zones)
#   industrial_haze: 0.9 — enforcement via pollution control orders, stack checks
#   dust: 0.4  — suppression (water tankers, dust nets), limited enforcement
#   crop_burning_smoke: 0.3 — mostly rural/seasonal; city enforcement has limited scope
#   clear: 0.0  — no action
#
# Composite score = attribution_conf * source_weight + 0.3 * vehicle_emission_load_index
# (satellite attribution is the primary signal; vehicle index contextualises enforcement)
SOURCE_WEIGHTS = {
    "traffic_heavy":       1.0,
    "industrial_haze":     0.9,
    "dust":                0.4,
    "crop_burning_smoke":  0.3,
    "clear":               0.0,
}


# ---------------------------------------------------------------------------
# Model definition — Prithvi + LoRA adapters, matches train_prithvi_lora.py exactly.
# replace_qkv splits fused QKV -> q_linear/k_linear/v_linear before LoRA wrapping.
# Head: LayerNorm(768) -> Dropout(0.1) -> Linear(768, 5)
# ---------------------------------------------------------------------------

PEFT_CONFIG = {
    "method": "LORA",
    "replace_qkv": "qkv",
    "peft_config_kwargs": {
        "r": 8, "lora_alpha": 16,
        "target_modules": ["q_linear", "k_linear", "v_linear", "proj"],
        "lora_dropout": 0.05, "bias": "none",
    },
}


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
        x = x.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        raw = self.backbone(x)
        feat = raw[-1] if isinstance(raw, (list, tuple)) else raw
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        elif feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])
        return self.head(feat)


def load_model(ckpt_path: Path) -> tuple[PrithviLoraClassifier, list[str]]:
    """
    Loads the final LoRA checkpoint into PrithviLoraClassifier.
    Uses pretrained=False — all weights come from the checkpoint.
    Validates class list matches CLASSES (fail loudly if not).
    """
    from terratorch.registry import BACKBONE_REGISTRY
    from terratorch.models.peft_utils import get_peft_backbone

    print(f"Loading backbone (Prithvi-EO-2.0-100M-TL, pretrained=False — weights from checkpoint) ...")
    backbone = BACKBONE_REGISTRY.build("prithvi_eo_v2_100_tl", pretrained=False)
    backbone = get_peft_backbone(PEFT_CONFIG, backbone)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_classes = ckpt.get("classes", CLASSES)
    if ckpt_classes != CLASSES:
        raise RuntimeError(
            f"Checkpoint class list {ckpt_classes} does not match expected {CLASSES}. "
            "Update CLASSES in this file to match the checkpoint."
        )

    model = PrithviLoraClassifier(backbone, EMBED_DIM, NUM_CLASSES)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"Loaded checkpoint: {ckpt_path}  (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    return model, ckpt_classes


def load_s2_image(tif_path: Path) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(tif_path) as src:
        data = src.read(
            out_shape=(src.count, IMG_SIZE, IMG_SIZE),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    return np.clip(data / 10000.0, 0.0, 1.0)


@torch.no_grad()
def predict_zone(model: PrithviLoraClassifier, img_np: np.ndarray) -> tuple[str, float, dict]:
    """
    Returns (predicted_class, confidence, {class: prob}) for one S2 image.
    confidence is the softmax probability of the top class.
    """
    x = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_idx = int(probs.argmax())
    class_probs = {c: float(probs[i]) for i, c in enumerate(CLASSES)}
    return CLASSES[top_idx], float(probs[top_idx]), class_probs


def build_reason(source: str, confidence: float, vei: float) -> str:
    """One-line human-readable enforcement reason."""
    conf_tag  = f"conf={confidence:.2f}"
    vei_label = ("high" if vei >= 0.65 else "moderate" if vei >= 0.35 else "low")

    reasons = {
        "traffic_heavy": (
            f"traffic-dominated source [{conf_tag}]; "
            f"{vei_label} registered vehicle density (index={vei:.2f}) "
            f"-> enforcement-priority zone"
        ),
        "industrial_haze": (
            f"industrial emission signature [{conf_tag}]; "
            f"{vei_label} vehicle load (index={vei:.2f}) "
            f"-> stack checks + PCO orders recommended"
        ),
        "dust": (
            f"dust source signature [{conf_tag}]; "
            f"{vei_label} vehicle load (index={vei:.2f}) "
            f"-> dust suppression over traffic enforcement"
        ),
        "crop_burning_smoke": (
            f"crop-burning smoke signature [{conf_tag}]; "
            f"{vei_label} vehicle load (index={vei:.2f}) "
            f"-> seasonal/rural source, limited local enforcement scope"
        ),
        "clear": (
            f"no significant pollution event at prediction date [{conf_tag}]; "
            f"vehicle load index={vei:.2f}"
        ),
    }
    return reasons.get(source, f"{source} [{conf_tag}], vehicle index={vei:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="AirSentinel Enforcement Zone Ranker (Phase 3)")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD date to run inference on (default: most recent in labels CSV)")
    ap.add_argument("--labels-csv", type=Path, default=LABELS_CSV,
                    help=f"Labels CSV with s2_file column (default: {LABELS_CSV})")
    ap.add_argument("--vehicle-index", type=Path, default=VEHICLE_INDEX,
                    help="vehicle_emission_index.csv from vehicle_emissions pipeline")
    args = ap.parse_args()

    print(f"Python : {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_VERSION}")
    print()

    # -- Validate inputs -------------------------------------------------------
    if not PRITHVI_CKPT.exists():
        print(f"[ERROR] Checkpoint not found: {PRITHVI_CKPT}")
        print("        Run train_prithvi.py (Phase 2) first.")
        sys.exit(1)

    if not args.labels_csv.exists():
        print(f"[ERROR] Labels CSV not found: {args.labels_csv}")
        sys.exit(1)

    if not args.vehicle_index.exists():
        print(f"[WARN] Vehicle emission index not found at {args.vehicle_index}")
        print(f"       Run category_mapper.py then vehicle_emissions pipeline first.")
        print(f"       Proceeding without vehicle index (scores will be attribution-only).")
        vei_df = None
    else:
        vei_df = pd.read_csv(args.vehicle_index)
        print(f"Vehicle index: {len(vei_df)} zones loaded from {args.vehicle_index}")

    # -- Land-use signal check -------------------------------------------------
    print()
    print("Land-use signal: NONE available in this repo. No land-use CSVs or APIs are present.")
    print("  Confirmed by full codebase search (2026-07-21). Design plan slide 5 references")
    print("  'hotspot + emission + land use' but no data source has been identified or ingested.")
    print("  Ranking uses only satellite attribution + vehicle emission index.")
    print()

    # -- Load labels, find inference date --------------------------------------
    labels = pd.read_csv(args.labels_csv)
    heuristic = labels[labels["label_source"].str.startswith("CAAQMS_heuristic")].copy()

    if args.date:
        inference_date = args.date
    else:
        inference_date = heuristic["date"].max()

    date_rows = heuristic[heuristic["date"] == inference_date].drop_duplicates("zone").copy()
    if date_rows.empty:
        print(f"[ERROR] No heuristic-labeled rows found for date {inference_date}")
        sys.exit(1)

    print(f"Inference date: {inference_date}  ({len(date_rows)} zones with S2 images)")

    # -- Load model ------------------------------------------------------------
    model, _ = load_model(PRITHVI_CKPT)
    model = model.to(DEVICE)

    # -- Run inference per zone ------------------------------------------------
    print(f"\nRunning Prithvi inference on {inference_date} images:")
    attr_rows = []
    skipped   = []

    for _, row in date_rows.iterrows():
        zone    = row["zone"]
        s2_file = row["s2_file"]
        p = Path(s2_file)
        tif_path = p if p.is_absolute() else DATA_DIR / p

        if not tif_path.exists():
            print(f"  [SKIP] {zone}: {tif_path.name} not found")
            skipped.append(zone)
            continue

        try:
            img = load_s2_image(tif_path)
            pred_class, confidence, class_probs = predict_zone(model, img)
            heuristic_class = row["dominant_pollutant"]

            agreement = "(agrees with heuristic label)" if pred_class == heuristic_class else \
                        f"(heuristic was '{heuristic_class}')"

            print(f"  {zone:15s}  -> {pred_class:<22s}  conf={confidence:.3f}  {agreement}")
            attr_rows.append({
                "zone":           zone,
                "inference_date": inference_date,
                "source_guess":   pred_class,
                "confidence":     round(confidence, 4),
                **{f"prob_{c}": round(p, 4) for c, p in class_probs.items()},
                "heuristic_label": heuristic_class,
            })
        except Exception as e:
            print(f"  [ERROR] {zone}: {e}")
            skipped.append(zone)

    if not attr_rows:
        print("[ERROR] No inference results produced. Check S2 image paths.")
        sys.exit(1)

    attr_df = pd.DataFrame(attr_rows)

    # -- Join with vehicle emission index --------------------------------------
    if vei_df is not None:
        # Zone name normalisation: labels.csv uses underscore (Anand_Vihar),
        # vehicle_emission_index.csv uses space (Anand Vihar) — match by normalising
        vei_lookup = vei_df.set_index(
            vei_df["zone"].str.replace("_", " ")
        )["vehicle_emission_load_index"].to_dict()

        attr_df["vehicle_emission_load_index"] = attr_df["zone"].str.replace("_", " ").map(vei_lookup)
        missing_vei = attr_df["vehicle_emission_load_index"].isna().sum()
        if missing_vei:
            print(f"\n[WARN] {missing_vei} zones have no vehicle emission index match.")
        attr_df["vehicle_emission_load_index"] = attr_df["vehicle_emission_load_index"].fillna(0.0)
    else:
        attr_df["vehicle_emission_load_index"] = 0.0

    # -- Composite score + rank ------------------------------------------------
    attr_df["source_weight"] = attr_df["source_guess"].map(SOURCE_WEIGHTS)
    attr_df["composite_score"] = (
        attr_df["confidence"] * attr_df["source_weight"]
        + 0.3 * attr_df["vehicle_emission_load_index"]
    ).round(4)

    attr_df = attr_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    attr_df["enforcement_rank"] = range(1, len(attr_df) + 1)

    attr_df["rank_reason"] = attr_df.apply(
        lambda r: build_reason(r["source_guess"], r["confidence"], r["vehicle_emission_load_index"]),
        axis=1,
    )

    attr_df["land_use_note"] = (
        "land-use signal absent from this repo -- scoring uses only satellite + vehicle data"
    )
    attr_df["model_version"]   = MODEL_VERSION
    attr_df["data_provenance"] = "correlation-based (heuristic CAAQMS labels, not ground truth)"
    attr_df["ranked_at"]       = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # -- Write outputs ---------------------------------------------------------
    col_order = [
        "enforcement_rank", "zone", "inference_date",
        "source_guess", "confidence",
        "vehicle_emission_load_index", "composite_score",
        "rank_reason", "land_use_note",
        "heuristic_label",
        "prob_dust", "prob_crop_burning_smoke", "prob_industrial_haze",
        "prob_traffic_heavy", "prob_clear",
        "model_version", "data_provenance", "ranked_at",
    ]
    out_cols = [c for c in col_order if c in attr_df.columns]
    ranking_out = attr_df[out_cols]
    ranking_out.to_csv(OUT_RANKING, index=False)
    print(f"\nEnforcement ranking written: {OUT_RANKING}")

    # Attribution CSV for fuse.py.
    # Zone names normalised to spaces to match forecasting module's canonical naming
    # (labels.csv uses underscores; every other module uses "Anand Vihar" not "Anand_Vihar").
    # Schema extended with honesty tags so fuse.py can propagate them to the dashboard.
    OUT_ATTRIBUTION.parent.mkdir(parents=True, exist_ok=True)
    attr_out = attr_df[["zone", "source_guess", "confidence",
                         "model_version", "data_provenance", "land_use_note"]].copy()
    attr_out["zone"] = attr_out["zone"].str.replace("_", " ")
    attr_out.to_csv(OUT_ATTRIBUTION, index=False)
    print(f"Satellite attribution written: {OUT_ATTRIBUTION}")

    # -- Print ranked results --------------------------------------------------
    print(f"\n=== Enforcement Zone Ranking ({inference_date}) ===")
    print(f"Model: {MODEL_VERSION}")
    print(f"Land-use: absent (satellite + vehicle only)")
    print()
    for _, r in ranking_out.iterrows():
        vei_str = f"{r['vehicle_emission_load_index']:.3f}" if r["vehicle_emission_load_index"] else "n/a"
        print(f"  #{int(r['enforcement_rank']):<2}  {r['zone'].replace('_',' '):<15s}  "
              f"composite={r['composite_score']:.3f}  "
              f"[{r['source_guess']} conf={r['confidence']:.2f}, vei={vei_str}]")

    print(f"\n    Reason:")
    for _, r in ranking_out.iterrows():
        print(f"  #{int(r['enforcement_rank']):<2}  {r['zone'].replace('_',' '):<15s}: {r['rank_reason']}")

    if skipped:
        print(f"\n[!] Skipped {len(skipped)} zones (S2 file missing or read error): {skipped}")

    print(f"\nNote: '{MODEL_VERSION}' tag on every output row.")
    print(f"Swap checkpoint at line 22 of this script to update to a better model.")
    print(f"No other change needed in ranking logic, output schema, or fuse.py wiring.")


if __name__ == "__main__":
    main()
