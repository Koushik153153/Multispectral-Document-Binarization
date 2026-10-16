# Controlled GPU experiments: v3 package and handoff

Start with `PROJECT_HANDOFF.md` for the complete operating guide and recovery
checklist. The user reports **128 tests passed on v2**, and a created/verified v2
archive uploaded to Drive. Those passes precede this final change. The v3 changes
are source-reviewed only: no Python, tests, profiling, notebook or training was
executed here. Preserve both earlier archives; create the new v3 ZIP below.
Archive revision v3 still uses checkpoint format **2**, not a new checkpoint format.

## Historical baseline and comparison protocol

The user reports a T4 FP32 baseline stopped after 7 of 10 intended epochs, best
epoch 4, patch validation F1 0.9100630197871566. Its all-20-page validation counts
were TP 6915234, FP 930748, FN 372767; precision 0.8813726567305405,
recall 0.9488519554264606, F1 0.9138683451672966. These are user-reported historical
validation results, not independently rerun here, not final test results.
The previous runtime manually disabled autocast/scaling; its version 1 metadata
does not establish precision. Existing checkpoints and archive are untouched.

Rerun baseline with this workflow for controlled comparisons. `experiments.json`
is the single place to select the common epoch budget (default **7**), seeds
(default `[42]`), precision (default **fp32**), batch (2), workers (0), and cache
(0 MiB until profiled). The v3 specification also sets PyTorch CPU intra-op
threads to 2 for every GPU run, so preflight and subprocesses do not depend on
different runtime thread defaults. Keep this common setting fixed. It defines:

| Configuration | Band reweighting | Drop probability |
|---|---|---|
| baseline | off | 0 |
| reweight_only | on | 0 |
| drop_only | off | 0.2 |
| reweight_and_drop | on | 0.2 |

Every run uses FG1, 12 bands, ImageNet-pretrained ResNet50, unfrozen encoder,
patch/stride 256, all valid patches (`--min_fg_frac 0`, omit patch cap), AdamW,
lr .0003, pos_weight 6, BCE .9, Dice .1, grad clip 1 and threshold .5.
Zero is invalid for `--max_patches_per_page`; absence means no cap. UR and
padding remain invalid. Select dropout without consulting reserved test data.
Batch memory requirements must be checked on the actual GPU. Change common
settings before all runs; comparisons reject differences. No gains are claimed.

For each seed, the runner builds a seeded baseline on CPU, copies **all shared
backbone/decoder tensors**, and constructs extra attention under CPU seed
`seed + 1000003`. CPU RNG state is restored afterward. Shared tensors overwrite
the variant's common tensors, so extra module initialization cannot shift them.
`run.json` and checkpoints record per-tensor SHA-256 and a combined initialization
identity, exact arguments, splits, software versions and training source hashes.
Initial weights are reconstructed per run, not saved as an extra large artifact.
ImageNet weights must be available in the future GPU environment; fresh training
can download them if absent. Resume and inference reconstruct offline.

Runs are sequential. Additional seeds use the spec's `seeds` list. One seed is
a preliminary comparison, not statistical significance. Reproducibility is
conditional on the recorded environment; CUDA kernels may be nondeterministic.
TF32 is explicitly disabled and backend flags/GPU name are recorded. Do not
claim bitwise cross-GPU or cross-version equivalence.

## Precision, updates and resume

`--precision fp32` disables autocast and GradScaler; `amp` requires CUDA.
Masked BCE/Dice use float32 inputs and reductions outside autocast. Nonfinite
valid loss always fails. Gradients are unscaled before inspection/clipping.
Finite gradients use strict clipping; nonfinite FP32 gradients fail. AMP overflow
skips clipping and delegates the skipped optimizer step and scale reduction to
public GradScaler APIs. It records `overflow_skipped_updates` separately from
`invalid_label_batches` and `optimizer_updates`. An epoch with no updates or
100 consecutive overflows (configurable) fails. It does not weaken strict clipping.
BatchNorm running statistics can still change during an overflowed forward;
optimizer parameters/state are not updated for that batch.
See [PyTorch AMP examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html).

New checkpoints are version 2. Version 1 remains readable for offline inference,
with unknown precision shown as null; **training resume is rejected**, with no
automatic conversion or inference of precision from device. Version 2 resume
requires matching precision, source hashes, dependency versions, GPU name,
device, splits, cache/worker and other training settings. Dataset root must remain
the same. Only total epochs may increase, and output/key paths may change with
identical split bytes. Resume restores optimizer/scaler, global RNG and shuffle
generator state. Epoch history regenerates CSVs without duplicate rows. Epoch
boundary resume repeats an interrupted partial epoch; mid-epoch resume is absent.
In a new Colab runtime, recreate the same `/content/msbin_project` path to resume.
An alternate extraction path changes the dataset root and will be rejected.
Thread count is now explicitly recorded and checked, along with source hashes.
Do not use this revised source to resume a run created by v2 source: re-extract
its exact archived source/environment instead. Compatibility checks are not bypassed.

Keep a backup of completed files. `last.pt` is the epoch commit point; `best.pt`
uses validation **patch F1**, never final test data. Text output is flushed to
`training.log`; configuration goes to `run.json`; timings to `timing.json`.
Checkpoint/CSV/JSON writes use same-directory temporary files, fsync and replace.
Disk/write errors include the destination. A mounted Drive filesystem can have
remote synchronization delays/failures: this is file-level publication, not a
multi-file transaction or a guarantee of remote durability. Verify/read back
saved checkpoints before ending a session; no second writer may share a run.

## Data loading and profiling

Source inspection found one label decode plus 12 full-page band decodes **per
patch** (one band in white-only mode). Indexing also validates every page/band.
The optional LRU stores uint8 bands, float32 target and bool validity per page.
The retained array budget is `cache_mb` MiB per dataset per worker (0 disables).
One 12-band page costs roughly `17 * height * width` bytes; an oversize page is
decoded but not retained. Peak RAM also includes the incoming decoded page,
temporary decoder arrays, index, batches and worker-prefetch buffers; the cache
budget is not a process RAM cap. Training and validation own separate caches.
With workers, each worker owns its own budget. Workers are not persistent and
caches restart each epoch; workers=0 caches persist in the main process.
No entire-dataset preload is performed. Immutable dataset content is required.
Cache returns independent patch tensors; normalization, masks, positions and
sample/shuffle order are unchanged. Keep worker/cache settings fixed across
variants and resume. There are no random worker augmentations today.

The notebook offers paired cache-off/cache-on software profiles using the same
seed/batch count. Profiles use random weights and bounded train/validation batches,
but index the full splits. They separately time indexing, data wait, transfer,
forward/backward/update, validation, model initialization and checkpoint writing.
CUDA synchronizes at stage boundaries **only in profiling mode**; normal timing
is approximate. Validation time includes validation data loading. Cold startup,
OS caches and short shuffled samples can dominate; repeat/reverse order and
increase the batch sample before attributing the reported 2485.86-second epoch
to I/O. No speedup has been measured here. Profile checkpoints are marked
software-only and cannot be resumed or used as research inference inputs.

CPU profiling is now explicit: `--device cpu --threads 2 --precision fp32`.
`--device cuda` fails when CUDA is unavailable; CPU+AMP is rejected before work.
Threads means PyTorch CPU intra-op threads, applied before indexing/model work;
it does not set OpenCV threads or DataLoader worker count. `timing.json` records
device, resolved threads, requested batch limit, actual train/validation batches
and loaded samples, and stage timings. CPU profiling never synchronizes CUDA.
Four batches with batch size 2 normally load eight samples per stage; shorter
datasets are counted honestly. Indexing still scans full train/validation splits.
This is bounded software profiling, **not support for the controlled research
experiments on CPU**: the experiment runner still requires CUDA.
The failed `results/cpu_profile_v2` folder was absent when inspected; use a new
`results/cpu_profile_v3_threads2` folder for the next manual attempt.

## Manual local tests and new package (PowerShell)

```powershell
Set-Location 'D:\Run time Redo\SE\Multispectral_Document_Binarization_using_Band_Reweighting_and_Dropping-main'
& '.\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests
& '.\.venv\Scripts\python.exe' -B scripts/package_gpu_training.py create --project-root '.' --output '.\transfer\msbin_gpu_experiments_v3.zip'
& '.\.venv\Scripts\python.exe' -B scripts/package_gpu_training.py verify --archive '.\transfer\msbin_gpu_experiments_v3.zip'
```

Stop on any failure. Do **not** pass `--overwrite` or replace the historical ZIP.
The allowlist includes the notebook, spec, new tools and tests; all `.pt` files,
test data/keys, duplicate splits, predictions and experiment outputs stay excluded.
The preserved full-dataset audit is historical provenance, not a runnable audit
of the subset. The updated verifier expects the v3 file inventory (including the handoff); use the
historical packaged verifier for the historical archive.

## Colab, profiling and later research commands

Open `notebooks/msbin_experiments_colab.ipynb` in Colab. The notebook inspects
Python/torch/torchvision/OpenCV/NumPy, tests torchvision's CUDA operator, mounts
Drive and preserves the literal trailing dot in `MSBin_GPU_Training.`. It does
not install/replace dependencies. Resolve missing/incompatible dependencies
before proceeding; no particular runtime image or free GPU is promised.

Upload the **new** ZIP yourself to that folder. Only use your trusted ZIP: the
bootstrap runs its packaged verifier. The notebook copies it locally, verifies
hashes, then extracts into a **new** `/content/msbin_project` directory. Existing
extraction directories are rejected; choose a new path rather than merging code.
Images are always read from local `/content`, not Drive. Runs/logs/checkpoints
use a new `runs/experiments_v3` folder on Drive, leaving `runs/fg1_baseline_fp32`
untouched. Disk-space checks are estimates, not a guarantee of enough quota.

These are manual later-run commands from the extracted project, **not executed
during implementation** (use the GPU environment's Python):

```powershell
python scripts/smoke_cuda_precision.py
python scripts/profile_msbin_training.py --device cuda --threads 2 --outdir profile_cache0 --cache_mb 0 --batches 4
python scripts/profile_msbin_training.py --device cuda --threads 2 --outdir profile_cache512 --cache_mb 512 --batches 4
python run_experiments.py run --configuration baseline --runs runs_gpu/experiments_v3
python run_experiments.py run --configuration all --runs runs_gpu/new_all_v3
python run_experiments.py run --configuration baseline --runs runs_gpu/experiments_v3 --resume
python run_experiments.py evaluate --configuration baseline --runs runs_gpu/experiments_v3
python run_experiments.py compare --configuration baseline --runs runs_gpu/experiments_v3 --output results/comparison_v3
```

Choose either one configuration or all in a fresh run tree. Resume is an explicit
single configuration/seed operation; it never falls back to fresh. Use a one-seed
spec when resuming one seed from a larger sweep. Full-page evaluation uses
`best.pt`, all 20 validation keys, tile/stride 256, fixed threshold .5, existing
stitched inference and UR-masked counts. Outputs require new directories.
Evaluation also disables TF32, records its actual dependency/backend/GPU metadata,
and requires validation key bytes to match the checkpoint when comparing runs.
Use `--configuration all` for evaluation/comparison once all runs finish.
Comparison rejects different splits/tasks/settings/budgets/source/environment
and different shared initialization for a given seed. It writes `comparison.csv`,
`comparison.json` and `per_page.csv`, including epoch and checkpoint SHA-256.
Pooled precision/recall/F1 use summed pixels; `page_mean_*` are unweighted page
means. Mask and reading-preview polarity remain unchanged. Download run and
validation directories from Drive, including manifests/logs; avoid duplicating
the dataset or historical checkpoints in result exports. No keep-alive tricks,
test-set selection or completion-time guarantees are part of this workflow.
