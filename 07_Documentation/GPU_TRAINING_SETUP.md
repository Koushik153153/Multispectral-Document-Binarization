# Portable fresh GPU baseline

For the current workflow, start at `PROJECT_HANDOFF.md`, then `EXPERIMENTS_SETUP.md` and
`notebooks/msbin_experiments_colab.ipynb`. The new archive name is
`msbin_gpu_experiments_v3.zip`; preserve the historical `msbin_gpu_training.zip`.
Also preserve the verified/uploaded `msbin_gpu_experiments_v2.zip`; it contains
earlier source and is not the package for this revised notebook.
The commands below now use the new name. Precision defaults to explicit FP32;
historical CPU/GPU checkpoints are inference-only, not resumable training inputs.

This package prepares a **fresh GPU run**, not a continuation of the CPU pilot.
No checkpoints are included. Do not pass `--resume`: the trainer rejects
cross-device resume. Use a new output directory for the real GPU run.

The published split has 60 training pages and 20 validation pages, both loaded
from original `MSBin/train`. All 80 pages' 12 bands and labels are packaged.
The 50 reserved test pages and test key file stay local and untouched. The
duplicate `MSBin/train_val_split` tree is excluded. Packaging does not train,
decode images, install packages, download weights, or evaluate the test set.

## Build and verify locally (PowerShell)

Run from the existing project using your working Python environment:

```powershell
Set-Location 'D:\Run time Redo\SE\Multispectral_Document_Binarization_using_Band_Reweighting_and_Dropping-main'
& '.\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests/test_package_gpu_training.py
& '.\.venv\Scripts\python.exe' -B scripts/package_gpu_training.py create --project-root '.' --output '.\transfer\msbin_gpu_experiments_v3.zip'
& '.\.venv\Scripts\python.exe' -B scripts/package_gpu_training.py verify --archive '.\transfer\msbin_gpu_experiments_v3.zip'
```

Check each command succeeds before continuing. The packaging and verification
script itself requires only Python's standard library (Python 3.9 or newer).
Tests require pytest. Do not replace the historical ZIP. If the new output name
already exists, choose another new name; overwrite is not part of this workflow.

The builder streams each file into a ZIP64-capable, uncompressed ZIP (the PNGs
are already compressed), verifies the temporary ZIP, then publishes it
atomically. Without `--overwrite`, atomic publication uses a hard link and
fails safely if the destination exists or the filesystem does not support
hard links. The project's NTFS drive is a suitable build location; copy the
finished ZIP to a transfer drive afterward. Allow space for the complete ZIP;
replacement also needs space for the old ZIP until publication finishes.

Use a dedicated output folder such as `transfer`. Existing archives and that
folder cannot enter the allowlist. A failed build removes its temporary file
and does not replace an existing archive.

## Contents and verification scope

Paths are relative to the project root, without an extra enclosing folder.
Extract into a new directory and run commands from that directory. Keep
`MSBin/train/images`, `MSBin/train/labels`, and `splits` in their original layout.
The two split files are copied byte for byte, including line endings.

The explicit allowlist includes training, model, dataset, label, metric and
inference modules; their relevant tests; the CPU smoke script; the dataset
validator used by tests; this packager; and setup documentation. AST inspection
checks static local imports, including function-local imports, and rejects
local dependencies outside the allowlist. The validator's dynamic test import
is explicitly included. External runtime dependencies are PyTorch, torchvision,
NumPy and OpenCV; pytest is needed for tests. The environment and pretrained
weights are not bundled. Choose compatible GPU builds after inspecting the
selected GPU environment; this document intentionally supplies no cloud
installation commands.

`gpu_package_manifest.json` records every payload path, size and SHA-256 hash,
split counts, local import dependencies and exclusions. Expected real-data
counts are `train: 60`, `val: 20`, `original_train: 80`, `test_packaged: 0`.
Verification checks the manifest against the exact archive inventory, required
files, 12 bands per page, disjoint split coverage, sizes, CRCs and hashes. It
streams payloads without extracting anything or consulting local test data.
It reads the complete ZIP, so verification time scales with archive size.
Hashes detect corruption; the manifest is not a cryptographic signature.

If present, `results/dataset_audit/dataset_audit.json` and
`results/dataset_audit/dataset_manifest.csv` are preserved unchanged as
**historical full-dataset provenance**. They can mention test pages and duplicate
paths that are absent from this ZIP. The original full audit cannot be rerun
successfully on this training-only subset. The full-dataset loading check is
not packaged because it requires test keys. Use the package verifier for
transfer integrity; the included validator tests use synthetic fixtures.

After transfer, verify again using the copied packager and the destination's
Python. For example, from the extracted project in PowerShell, with the ZIP
stored in the parent directory:

```powershell
python -B scripts/package_gpu_training.py verify --archive '..\msbin_gpu_experiments_v3.zip'
```

## Proposed configuration for the later real run

The baseline is ResNet50 U-Net with FG1, all 12 input channels, patch size 256,
stride 256, seed 42, band reweighting off, band dropping probability 0,
ImageNet-pretrained initialization and an unfrozen encoder. The 12-channel
choice comes from omitting `--white_only`; there is no channel-count CLI flag.
Omit `--band_reweight` and `--freeze_encoder` to leave both flags false.
`--pretrained` requests ImageNet initialization for this fresh run and may need
weights to be available or downloaded in the chosen environment later.

**All training patches:** omit `--max_patches_per_page`. Its default `None`
disables the cap; `0` is rejected, not interpreted as unlimited. Also pass
`--min_fg_frac 0` to disable foreground-fraction filtering. Patches with no valid
label pixels still remain excluded by the dataset implementation. Validation
already uses no cap and zero foreground filtering. Border patch positions
follow the existing dataset implementation.

Choose batch size for the GPU memory available. Epoch count will be chosen for
the real run; it is not established by the CPU pilot. Validation patch F1 selects
`best.pt`; `last.pt` stores the latest completed epoch. These patch validation
metrics are not final test or full-page performance. No accuracy is guaranteed.

The following is a **later-run template**, not part of packaging. Run it only
after selecting and checking the GPU environment. Despite its filename, the
trainer automatically uses CUDA when available; it otherwise falls back to CPU.
The check below stops this template if CUDA is unavailable.

```powershell
$GpuPython = Read-Host 'Path to Python in the prepared GPU environment'
$BatchSize = [int](Read-Host 'Positive batch size chosen for GPU memory')
$Epochs = [int](Read-Host 'Positive epoch count chosen for the real run')
& $GpuPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is required for this GPU run'"
if ($LASTEXITCODE -ne 0) { throw 'GPU environment check failed' }
& $GpuPython train_unet_msbin_cpu.py `
  --msbin_root MSBin `
  --train_keys splits/train_keys.txt --val_keys splits/val_keys.txt `
  --fg_type 1 --patch 256 --stride 256 --seed 42 `
  --min_fg_frac 0 --band_drop_p 0 --pretrained `
  --batch $BatchSize --epochs $Epochs --num_workers 0 `
  --lr 0.0003 --pos_weight 6 --bce_w 0.9 --dice_w 0.1 `
  --grad_clip 1 --val_thr 0.5 --outdir runs_gpu/fg1_baseline
```

Inference reconstructs the architecture offline from checkpoint metadata and
loads the saved model state; it does not fetch ImageNet weights. See
`INFERENCE_SETUP.md` for tiled full-page inference. Keep the final test set local
and untouched while selecting the model using validation.
