# Pilot validation-page preview

For current commands, verified historical scope and source/package revision v3,
start with `PROJECT_HANDOFF.md`. The earlier pilot narrative below is historical.

**Version 2 update:** inference reads both historical version 1 and new version 2
checkpoints, reconstructing offline. Unknown historical precision stays null.
New manifests include split/configuration/source/initialization provenance for
the controlled full-page comparison workflow in `EXPERIMENTS_SETUP.md`.
The earlier pilot instructions and results below remain historical context.

The user reports 64 passing tests, a successful actual U-Net software smoke check,
and a completed one-epoch CPU baseline pilot. The pilot's patch-validation
precision/recall/F1 were 0.540430303548071 / 0.36588199631051965 /
0.43634786454894225. These are limited pilot observations, not final research
results. This implementation has not yet been executed in the Codex environment:
its single interpreter probe again failed with exit 101 (`Unable to create process
using` the base Python executable). No actual inference was launched.

## First run all tests

From the project directory in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests/test_validate_msbin_dataset.py tests/test_msbin_loading_metrics.py tests/test_training_support.py tests/test_inference.py
```

The inference tests use tiny dummy models and synthetic images only; they never
construct ResNet50, download pretrained weights, or inspect the real dataset.
Review the passing test results before running the next command.

## Exactly the first validation page (manual, after test review)

```powershell
& '.\.venv\Scripts\python.exe' -B infer_msbin_cpu.py --msbin_root '.\MSBin' --split train --keys_txt '.\splits\val_keys.txt' --ckpt '.\results\baseline_cpu_pilot\best.pt' --outdir '.\results\baseline_cpu_preview' --tile 256 --stride 256 --thr 0.5 --device cpu --threads 2 --max_pages 1 --compare_gt
```

The first nonblank page key in the existing validation list is selected without
sorting or randomization. All requested keys must be nonempty, unique, complete
12-band groups and members of the checkpoint's validation keys. This step accepts
only `--split train`, because validation pages live in original `MSBin/train`;
test inference is deliberately not exposed. Only selected pages are decoded and
dimension-checked. No split, dataset, audit or checkpoint file is changed.

## Checkpoint compatibility and memory

Only the explicitly supplied trusted local versioned checkpoint is loaded, using
the existing `training_support.read_checkpoint` validator. Architecture settings
come from `model_config`, foreground type from `training_args["fg_type"]`, and
validation membership from `splits["val"]["keys"]`. Unsupported architecture or
format, legacy metadata, and software-smoke checkpoints fail. Reconstruction
uses `build_model(..., restoring=True)`, always disabling pretrained downloads,
followed by strict state loading and `eval()` (band dropping inactive).

The old inference flags `--white_only` and `--band_reweight` were removed. Omit
them: they cannot override checkpoint architecture. The training flags remain
unchanged. No cross-version or model-accuracy claim follows from inference loading.

Checkpoint loading temporarily reads the full training checkpoint (about 536 MB
for this pilot). Optimizer/scaler/RNG/history references are dropped before model
construction; only weights and small inference metadata are retained. A SHA-256
digest is streamed for checkpoint identity. The reader uses trusted pickle
metadata, so supply only a checkpoint you trust and keep it unchanged during a run.

Inference processes one tile at a time, default 256x256. Use a positive stride no
larger than the tile; a smaller stride adds overlap and compute. Edge starts cover
bottom/right borders, small pages use edge-replication padding (safe even for a
one-pixel axis), and predictions are cropped back to original size. The model sees
tile-sized tensors, not a page-sized tensor. All full-page data remain on CPU:
one uint8 CxHxW page, float32 probability sum and uint32 coverage, plus optional
label/comparison images. Memory still scales with page dimensions and checkpoint
size, but accelerator activation memory is bounded by one tile. No dataset cache
or page-sized float32 multispectral stack is created.

Sigmoid probabilities are averaged at overlaps, then thresholded once using
strict `probability > threshold`. Original-pixel zero coverage, shape mismatches,
nonfinite outputs, invalid threshold/tiling, and non-grayscale/non-uint8 bands fail.
`--device` supports cpu/cuda/auto; explicit unavailable CUDA fails. `--threads`
controls PyTorch CPU threads. Tiny tile sizes work in dummy tests but are not
recommended for the actual U-Net; use the documented 256-pixel tile.

## Outputs and score interpretation

For each selected page:

- `<key>_mask.png`: machine mask, selected foreground white=255, background black=0.
- `<key>_reading_preview.png`: inverted for reading, text black on white.
- `<key>_comparison.png` (with `--compare_gt`): left input band 0; middle reading
  ground truth with uncertain pixels magenta; right reading prediction. Panels
  retain original resolution and are concatenated horizontally.
- `manifest.json`: checkpoint path/size/SHA-256 and completed epoch, architecture,
  foreground task, ordered keys, threshold, tiling/device settings, image sizes,
  coverage and output conventions, plus per-page and aggregate scores if enabled.

Ground truth uses the shared BGR decoder: the selected FG class is 1, the other
ink class is valid background, unknown colors fail, and UR is excluded. Metrics
use stitched probabilities and count each original valid pixel once. Aggregate
scores sum TP/FP/FN before calculating precision/recall/F1 (zero denominator => 0).
These are **full-page scores on only the selected validation pages**, not test-set
scores and not equivalent to overlapping patch-validation scores. One page cannot
establish generalization or support a four-configuration research comparison.

Outputs require a new or empty directory. Each PNG is exclusively reserved and
`cv2.imwrite` success is checked. Existing files are never silently overwritten.
If a run fails after some PNGs are written, those partial outputs may remain;
only a completed run writes `manifest.json` with `status=complete`. Investigate
the failure and choose a new output directory rather than overwriting a preview.
