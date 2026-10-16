# Multispectral Document Binarization using Band Reweighting and Band Dropping

**Start here:** [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) is the current operational
guide, with status, exact commands, recovery instructions and remaining research.
Use the v3 package and revised Colab notebook for this source. The older examples
below are background, not the controlled experiment procedure.

A deep learning pipeline for binarizing multispectral document images using a **ResNet50-based U-Net** with optional **Band Reweighting** (squeeze-excitation-style per-channel attention) and **Band Dropping** (channel dropout). Their effects in this project remain to be measured in controlled experiments.

Designed for the [MSBIN / DIBCO](https://ieeexplore.ieee.org/document/9412513) dataset with 12-band multispectral images of historical documents.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Prepare the Validation Split](#1-prepare-the-validation-split)
  - [2. Train](#2-train)
  - [3. Inference](#3-inference)
  - [4. Evaluate](#4-evaluate)
- [Training Arguments](#training-arguments)
- [Loss Function](#loss-function)
- [Label Format](#label-format)
- [Requirements](#requirements)

---

## Overview

Document binarization — the task of separating foreground text/ink from background — is a foundational step in document image analysis. Multispectral imaging captures documents across multiple wavelength bands (12 bands in MSBIN), providing richer information than standard RGB but also presenting the challenge of how to best combine those bands.

This project tackles that challenge by training a segmentation model that can:
- **Reweight** spectral bands dynamically via a learned squeeze-excitation module (`BandReweight`), so more informative bands receive higher weight during feature extraction.
- **Drop** bands randomly during training (`Dropout2d` over channels), intended to reduce reliance on individual bands; improved generalization has not yet been established here.

---

## Key Features

- **ResNet50 encoder** with optional ImageNet initialization adapted for 12-channel (or 1-channel) input. The training CLI defaults to offline random initialization.
- **U-Net decoder** with bilinear upsampling and skip connections from all four ResNet encoder stages.
- **BandReweight module**: a lightweight MLP applied over global average-pooled channel features, producing per-band sigmoid weights — inspired by Squeeze-and-Excitation (SE) networks.
- **Band Dropping**: `nn.Dropout2d` on the input tensor zeroes out entire spectral bands at random during training, encouraging the network to learn redundant, robust representations.
- **Masked weighted BCE + Dice loss** excludes uncertain regions and artificial padding.
- **Band weight logging**: per-epoch CSV log of learned band weights for interpretability analysis.
- **CPU/GPU transparent**: runs on CPU or CUDA with mixed-precision (`torch.amp`) support when GPU is available.
- **Checkpoint save/resume** with full RNG state for reproducibility.

---

## Architecture

```
Input: (B, 12, H, W)  [or (B, 1, H, W) with --white_only]
    │
    ├─ BandReweight (optional SE-style per-band attention)
    ├─ BandDrop    (optional Dropout2d on channels)
    │
    └─ ResNet50 Encoder
          conv1 → x0: (B, 64,   H/2,  W/2)
          layer1 → x1: (B, 256,  H/4,  W/4)
          layer2 → x2: (B, 512,  H/8,  W/8)
          layer3 → x3: (B, 1024, H/16, W/16)
          layer4 → x4: (B, 2048, H/32, W/32)
          │
    U-Net Decoder (with skip connections)
          dec4: x4 + x3 → (B, 512,  H/16)
          dec3: d4 + x2 → (B, 256,  H/8)
          dec2: d3 + x1 → (B, 128,  H/4)
          dec1: d2 + x0 → (B, 64,   H/2)
          upsample → head_conv → out
          │
Output: (B, 1, H, W)  [logits, binarized via sigmoid > 0.5]
```

---

## Dataset

This project uses the **MSBIN** dataset, a multispectral document binarization benchmark. Each document page is represented as **12 grayscale PNG images** (one per wavelength band, named `<BookId>_<PageId>_0.png` through `<BookId>_<PageId>_11.png`) plus a **color-coded label PNG** in `labels/`.

**Expected directory structure:**

```
<msbin_root>/
    train/
        images/
            BookA_001_0.png
            BookA_001_1.png
            ...
            BookA_001_11.png
            BookA_002_0.png
            ...
        labels/
            BookA_001.png
            BookA_002.png
            ...
    test/
        images/
            ...
        labels/
            ...
```

**Label color coding:**

| Color (BGR)         | Meaning                   | Treated as     |
|---------------------|---------------------------|----------------|
| White `(255,255,255)` | Foreground type 1 (ink)  | Foreground with `--fg_type 1` |
| Gray  `(122,122,122)` | Foreground type 2        | Foreground (with `--fg_type 2`) |
| Blue  `(255,0,0)`    | Uncertain Region (UR)     | **Invalid; excluded** |
| Black `(0,0,0)`      | Background               | Background     |
| Other                | Unexpected color         | Error          |

---

## Project Structure

```
.
├── unet_small.py            # Model definition: BandReweight, UNetSmall (ResNet50 U-Net)
├── msbin_dataset.py         # Dataset: patch extraction, label parsing, band loading
├── train_unet_msbin_cpu.py  # Training script (CPU/GPU)
├── infer_msbin_cpu.py       # Inference script: produces binary output images
├── eval_msbin_val.py        # Evaluation script: computes pixel-level F1, P, R
├── make_val_split.py        # Utility: creates train/val key split text files
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/VIGNESHWARRAN/Multispectral_Document_Binarization_using_Band_Reweighting_and_Dropping.git
cd Multispectral_Document_Binarization_using_Band_Reweighting_and_Dropping

pip install torch torchvision tqdm opencv-python numpy
```

A CUDA-capable GPU is recommended for training but not required — the scripts fall back to CPU automatically.

---

## Usage

### 1. Prepare the Validation Split

Use the published, audited 60/20 train/validation split. Do not generate a new
random split. See `DATASET_SETUP.md` for audit details and `TRAINING_SETUP.md` for
masked training and checkpoint rules. To rerun the audit from PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -B scripts/validate_msbin_dataset.py --msbin-root '.\MSBin' --output-dir '.\results\dataset_audit'
```

Successful auditing publishes train/val/test key files, preserving identical
existing files and rejecting conflicting lists.

---

### 2. Train

**Future training example (all 12 bands, offline initialization; not run here):**

```powershell
& '.\.venv\Scripts\python.exe' train_unet_msbin_cpu.py --msbin_root '.\MSBin' --train_keys '.\splits\train_keys.txt' --val_keys '.\splits\val_keys.txt' --no-pretrained --epochs 30 --batch 4 --outdir '.\runs\baseline'
```

**With Band Reweighting + Band Dropping:**

```powershell
& '.\.venv\Scripts\python.exe' train_unet_msbin_cpu.py --msbin_root '.\MSBin' --train_keys '.\splits\train_keys.txt' --val_keys '.\splits\val_keys.txt' --no-pretrained --epochs 30 --batch 4 --band_reweight --band_drop_p 0.2 --outdir '.\runs\reweight_drop'
```

Checkpoints are saved as `last.pt` and `best.pt` (strictly improved masked patch
validation F1). New runs refuse nonempty output directories. Resume requires
versioned metadata and unchanged experiment settings/splits; extend `--epochs`
with explicit `--resume`. CLI initialization defaults to offline; future ImageNet
experiments must explicitly select `--pretrained` (may download uncached weights).
`epochs.csv` stores patch-level metrics and timings. `band_weights.csv` contains
the sample-weighted training epoch mean when attention is active. No completed
research experiments are claimed.

---

### 3. Inference

Inference now reconstructs versioned checkpoint settings offline and processes
one tile at a time. It requires an explicit validation key list and supports
original-size masks, reading previews and optional UR-masked full-page scores.
The old inference `--white_only` and `--band_reweight` overrides are removed.

See [INFERENCE_SETUP.md](INFERENCE_SETUP.md) for the unit-test command and the
exact one-page CPU pilot preview command. Review passing tests before running
real inference. This preview accepts only validation pages under `--split train`;
no test-set inference is performed. The current preview code is not runtime
verified in this Codex session, and the one-epoch pilot is not a final experiment.

---

### 4. Evaluate

The legacy external evaluator is currently blocked by missing helper modules and
its evaluator path. It was not repaired or executed in Step 11. The trainer now
computes masked patch-level precision/recall/F1 internally. Historical CLI shape
(not a validated evaluation workflow):

```bash
python eval_msbin_val.py \
    --msbin_root /path/to/msbin \
    --ckpt runs/reweight_drop/best.pt \
    --val_keys splits/val_keys.txt \
    --outdir evaluation/ \
    --band_reweight
```

---

## Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--msbin_root` | *(required)* | Path to the MSBIN dataset root |
| `--train_keys` | *(required)* | Published training page list |
| `--val_keys` | *(required)* | Published disjoint validation page list |
| `--pretrained` / `--no-pretrained` | `False` | Explicit ImageNet / offline fresh initialization; resume always constructs offline |
| `--freeze_encoder` | `False` | Freeze encoder parameters and BatchNorm updates |
| `--epochs` | `30` | Number of training epochs |
| `--batch` | `4` | Batch size |
| `--lr` | `3e-4` | AdamW learning rate |
| `--patch` | `256` | Patch size (pixels) |
| `--stride` | `256` | Stride for patch extraction |
| `--min_fg_frac` | `0.002` | Minimum foreground fraction to include a patch |
| `--max_patches_per_page` | `None` | Cap patches sampled per page |
| `--fg_type` | `1` | Foreground type: `1` = white ink, `2` = gray ink |
| `--white_only` | `False` | Use only band 0 (white-light) instead of all 12 bands |
| `--band_reweight` | `False` | Enable per-band SE-style attention module |
| `--band_drop_p` | `0.0` | Dropout2d probability for band dropping (0 = disabled) |
| `--pos_weight` | `6.0` | Positive class weight for BCE loss |
| `--bce_w` | `0.9` | Weight for BCE term in combined loss |
| `--dice_w` | `0.1` | Weight for Dice term in combined loss |
| `--grad_clip` | `1.0` | Gradient norm clipping value |
| `--val_thr` | `0.5` | Sigmoid threshold for binarization during validation |
| `--seed` | `42` | Random seed for reproducibility |
| `--resume` | `False` | Resume from `last.pt` checkpoint in `--outdir` |
| `--num_workers` | `0` | DataLoader worker processes |
| `--outdir` | `runs_cpu/unet` | Directory for checkpoints and logs |

---

## Loss Function

Training uses a weighted combination of **Binary Cross-Entropy with logits** and **Dice loss**:

```
Loss = bce_w × masked_BCE(logits, target, valid, pos_weight) + dice_w × masked_Dice(logits, target, valid)
```

The positive weight upweights selected foreground. Defaults are configurable and
are not evidence of tuning or completed experiments. BCE averages valid pixels;
Dice averages valid samples with smoothing. UR and padding are excluded. See
`TRAINING_SETUP.md` for definitions and zero-valid-pixel behavior.

---

## Label Format

The dataset uses color-coded PNG labels (not binary masks). The `MSBinDibcoPatchDataset` class converts these to binary foreground masks:

- Foreground type 1 (`--fg_type 1`): pixels where `(R,G,B) == (255,255,255)`.
- Foreground type 2 (`--fg_type 2`): pixels where `(R,G,B) == (122,122,122)`.
- Uncertain Region pixels (`(B,G,R) == (255,0,0)`) have target zero and validity False;
  masked training and patch scoring exclude them. The other ink class remains
  valid background. Unknown label colors fail explicitly.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision
- OpenCV (`opencv-python`)
- NumPy
- tqdm
