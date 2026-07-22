# AirSentinel — Prithvi Pipeline Run Log

A record of every issue hit and fix applied to get `train_prithvi.py` running end-to-end.

---

## What we were trying to do

Run a 2-epoch fine-tuning loop using the **Prithvi-EO-2.0-100M-TL** backbone on 13 Sentinel-2 satellite images of Delhi air-quality zones, as a plumbing check (not real training — labels are random placeholders).

---

## Step 1 — Environment setup

**Problem:** `conda` was not installed on the machine.

Running the original setup command failed:
```
conda : The term 'conda' is not recognized...
```

**Fix:** Skipped conda entirely. Used Python's built-in `venv` with the existing Python 3.12 install (Python 3.12 turned out to be fully compatible with all packages).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then installed all dependencies:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "terratorch>=1.1" rasterio pandas numpy tqdm huggingface_hub
```

**Result:** All packages installed successfully. GPU verified:
- PyTorch 2.6.0+cu124
- CUDA available: True
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8.6 GB VRAM)
- terratorch 1.2.10, rasterio 1.5.0

---

## Step 2 — DATA_DIR path had an invalid escape sequence

**Problem:** The path string in the script was written as:
```python
DATA_DIR = Path("G:\My Drive\AirSentinel_Satellite_Images")
```
Python interpreted `\M` as an escape sequence and raised a `SyntaxWarning`.

**Fix:** Added the `r` raw-string prefix so backslashes are treated literally:
```python
DATA_DIR = Path(r"G:\My Drive\AirSentinel_Satellite_Images")
```

---

## Step 3 — labels.csv was missing

**Problem:** The Colab notebook had not saved `labels.csv` to Google Drive, so the script raised:
```
FileNotFoundError: labels.csv not found at:
  G:\My Drive\AirSentinel_Satellite_Images\labels.csv
```

**What we found:** All 26 `.tif` files (13 zones x S2 + NO2) were already synced to the local Drive folder — only the labels file was missing.

**Fix:** Generated `labels.csv` locally by scanning the existing `.tif` files and assigning random placeholder labels (fixed seed = 42 for reproducibility), matching the same format the Colab notebook would have produced:

| Column | Value |
|---|---|
| zone | e.g. `Anand_Vihar` |
| date | `2026-06-01` |
| s2_file | e.g. `Anand_Vihar_2026-06-01_S2.tif` |
| no2_file | e.g. `Anand_Vihar_2026-06-01_NO2.tif` |
| dominant_pollutant | random choice from 5 classes |
| label_source | `PLACEHOLDER — random, not real` |

Saved to `G:\My Drive\AirSentinel_Satellite_Images\labels.csv`.

---

## Step 4 — Windows terminal blocked Unicode characters

**Problem:** Print statements in the script used emoji characters (`⚠️`, `✓`, `→`, etc.). Windows' default terminal encoding (CP1252) cannot encode these, causing a crash:
```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3
```

**Fix:** Replaced all non-ASCII characters throughout the script with plain ASCII equivalents:

| Original | Replaced with |
|---|---|
| `⚠️` | `[!]` |
| `✓` | `[OK]` |
| `→` | `->` |
| `ℹ️` | `[i]` |
| `─` / `━` | `-` / `=` |

---

## Step 5 — Wrong input shape for Prithvi backbone

**Problem:** The backbone's `patch_embed` is a **Conv3D** layer with weight shape `[768, 6, 1, 16, 16]`. This means it expects input in `(B, C, T, H, W)` order — channels first, then time. The script had the time and channel dimensions swapped.

The dummy probe input was:
```python
dummy = torch.zeros(1, 1, 6, 224, 224)  # (B, T, C, H, W)  <-- wrong
```

Which caused:
```
RuntimeError: expected input to have 6 channels, but got 1 channels instead
```

**Fix 1 — dummy probe input:**
```python
dummy = torch.zeros(1, 6, 1, 224, 224)  # (B, C, T, H, W)  <-- correct
```

**Fix 2 — forward pass inside `PrithviClassifier`:**
```python
# Before (wrong):
x = x.unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)

# After (correct):
x = x.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
```

---

## Final result

```
Python : 3.12.10
PyTorch: 2.6.0+cu124
CUDA available: True
GPU   : NVIDIA GeForce RTX 4060 Laptop GPU
VRAM  : 8.6 GB

Labels loaded: 13 rows
Dataset: 13 samples across 13 zones

Backbone loaded successfully.
Backbone returned 12-element list; using last: shape (1, 197, 768)
Model: 86.2M total params | 86.2M trainable

Epoch 1/2 complete - Avg Loss: 1.8356 | Acc: 23.1%  ([!] random labels, ignore)
Epoch 2/2 complete - Avg Loss: 2.3450 | Acc: 23.1%  ([!] random labels, ignore)

Training loop completed without errors. [OK]
Checkpoint saved -> prithvi_airsen_plumbing_check.pt

PLUMBING CHECK COMPLETE
  [OK]  S2 images loaded (6-band confirmed per file)
  [OK]  Prithvi-EO-2.0-100M-TL backbone loaded
  [OK]  Custom classification head built and probed
  [OK]  Training loop ran without crashing
  [OK]  Checkpoint saved
```

**VRAM usage: 1.4 GB allocated / 1.75 GB peak** — well within the RTX 4060's 8.6 GB.

---

## Notes on the output

- **Backbone output shape `(1, 197, 768)`:** 197 = 196 patch tokens + 1 CLS token from the ViT. The classifier mean-pools all 197 tokens into a single 768-dim vector, then passes it through `LayerNorm -> Dropout -> Linear(768, 5)`.
- **Backbone returns a 12-element list:** one feature tensor per transformer layer. We use the last one (most abstract features).
- **Loss / accuracy are meaningless** — labels were randomly assigned. These numbers should be ignored until real CPCB/CAAQMS labels replace the placeholder CSV.

---

## Next step

Replace `G:\My Drive\AirSentinel_Satellite_Images\labels.csv` with real dominant-pollutant labels from CPCB/CAAQMS data and re-run:
```
python train_prithvi.py
```
No other code changes are needed.
