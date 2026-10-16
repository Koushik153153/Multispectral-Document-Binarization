# Masked training and checkpoint setup

Current operations and source/package revision v3: see `PROJECT_HANDOFF.md`.
The checkpoint format stays version 2. CPU profiling now supports explicit device
and thread selection; controlled research runs still require CUDA.

**Version 2 update:** use `EXPERIMENTS_SETUP.md` and the Colab notebook for new
controlled GPU runs. Precision is now explicit (`--precision fp32` default),
checkpoints use version 2 and record precision/source identity, and version 1
training resume is rejected. Historical version 1 checkpoints remain usable for
inference. The earlier pilot narrative below is historical; its test counts do
not validate the new workflow. New cache/profiling/logging settings are described
in the experiments guide.

Step 11 implements software support; it does not establish model performance.
The user previously verified 43 dataset/metric tests and real-data loading
(60 training pages / 2317 unfiltered patches, 20 validation pages / 766 patches).
Training foreground filtering can reduce the training patch count. The 50 test
pages remain reserved for final evaluation and are never read by the trainer.

## Masks, losses, validation

Training and validation request `(image, target, valid_mask)`. FG1 is the default
selected foreground; FG2 is a separate task, not a union. UR and padding are invalid.
Weighted BCE uses `reduction="none"` and averages only valid pixels across the
batch. Dice uses sigmoid probabilities, masks numerator and denominator, adds
1e-6 to both, and averages per sample over samples with at least one valid pixel.
All-invalid samples are removed before model forward; all-invalid batches are
explicitly skipped. An epoch with no usable samples fails. Invalid logits/targets
are replaced before loss arithmetic, giving invalid pixels zero direct objective
gradient. This does not mean invalid input image pixels cannot affect neighboring
predictions through convolutions. Shapes and finite valid loss inputs are checked.

Validation accumulates integer TP/FP/FN across all batches through `pixel_metrics`.
Precision/recall/F1 use foreground=1 and sigmoid probability strictly greater than
`--val_thr`. Zero-denominator scores are zero. These are **patch-level** scores:
overlapping edge patches may count a pixel more than once. They are not full-page
or test-set performance. Validation uses `min_fg_frac=0` and no patch cap.

Both `--train_keys` and `--val_keys` are required, explicitly supplied, nonempty,
unique, disjoint, and checked against complete original `MSBin/train` pages before
model construction. There is no fallback to training on all pages. The existing
published lists must be used; do not substitute test keys or generate a new split.

## Initialization and frozen encoder

The CLI defaults to `--no-pretrained` (offline). `--pretrained` explicitly enables
ImageNet initialization for a future fresh experiment and may download weights if
not cached. No tests or smoke commands use it. Resume constructs with
`pretrained=False` regardless of historical initialization, then loads saved weights.
The checkpoint preserves the original initialization setting, which must match the
resume CLI. `--freeze_encoder` disables encoder gradients and keeps its BatchNorm
in evaluation mode even after `model.train()`. The ResNet50 U-Net layers are unchanged.

## Checkpoints and resume

Version 1 checkpoints record architecture/version, channels, attention/dropout,
encoder freeze and initialization settings; model/AdamW/AMP state; completed and
next epochs; updated best patch F1; Python/NumPy/CPU torch and relevant CUDA RNG;
DataLoader shuffle generator; full arguments; dependency/Python versions; actual
sorted train/val keys and SHA-256 of the key files' exact bytes; and epoch history.
Split hashes cover list content, not images: the separate dataset audit verifies
image content. Keep the audited dataset immutable during an experiment.

Epoch 1 completed means `completed_epoch=1`, `next_epoch=2`. Best score is updated
before committing `last.pt`; `best.pt` is written only on strict improvement.
Individual files are saved to a same-directory temporary file, flushed, then
replaced. `last.pt` is the commit point; on resume its history regenerates CSVs
without duplicate epochs, and an interrupted improved-best write is repaired.
This is file-level atomicity, not a transaction spanning every file or simultaneous
training processes: use only one writer per output directory.

Fresh runs require an empty/new output directory. Explicit `--resume` requires
`last.pt`; missing metadata/legacy checkpoints fail rather than guessing a model.
Channel, attention, dropout, freeze, initialization, loss/data/optimizer settings,
device, dependency versions, or split-content changes are rejected. Total epochs
may be extended, never reduced. Output and key-file paths may change if split
bytes remain identical; dataset root remains fixed. Always supply the same
experiment arguments when resuming, plus `--resume` and the extended `--epochs`.
Only load trusted local checkpoints: restoring Python/NumPy metadata uses pickle.

`epochs.csv` stores epoch, sample-weighted mean of batch train losses, patch
precision/recall/F1, elapsed seconds and best F1. Attention logging sums weights
over all active training samples and divides by sample count, before validation.
`band_weights.csv` therefore is not a last-batch or validation-batch statistic.

## Verification commands (PowerShell, project directory)

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests/test_validate_msbin_dataset.py tests/test_msbin_loading_metrics.py tests/test_training_support.py
```

The new tests use tiny models/fixtures. They do not measure model accuracy or
download weights. The real U-Net smoke check below is **manual, after review of
unit-test results**; it was not launched during Step 11:

```powershell
& '.\.venv\Scripts\python.exe' -B scripts/smoke_train_unet.py --msbin-root '.\MSBin' --train-keys '.\splits\train_keys.txt' --val-keys '.\splits\val_keys.txt' --threads 2
```

It uses CPU, random initialization, one real training page, at most two 256x256
patches, exactly one optimizer update and one validation-mode forward on that
same training page. That forward is a software check, not independent validation.
It prints timings/shapes and checks finite values, then saves/reconstructs a
temporary checkpoint and compares state tensors exactly. No extra model forward
is used for reconstruction. The checkpoint is labeled software-only, removed on
exit, and rejected by research resume. This may still need substantial RAM/disk
because the actual U-Net and AdamW are large. No test page is sampled.

## Current execution status and later experiments

This Codex session's interpreter probe failed with exit 101:
`Unable to create process using` the base Python 3.12 executable. The new and
previous tests were not executed in this session. The smoke check was deliberately
not launched. Run the unit command in the user's working PowerShell first.

CPU checks establish only software behavior. Later GPU research training needs
an explicitly chosen initialization, fixed experiment settings and identical
splits across configurations. Restoring RNG state does not guarantee bitwise
reproducibility across hardware or nondeterministic CUDA kernels. Cross-device or
dependency-version resume is intentionally rejected. No research training or
experiment comparison was performed in Step 11. A subsequent user-run CPU pilot
is documented in `INFERENCE_SETUP.md`, along with new checkpoint-compatible tiled
inference and commands awaiting review of its unit tests. The legacy external
evaluator remains a separate task. These documentation updates do not change
training behavior.
