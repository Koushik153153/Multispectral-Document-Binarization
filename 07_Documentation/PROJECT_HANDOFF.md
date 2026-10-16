# Multispectral Document Binarization using Band Reweighting and Band Dropping

Operational handoff, 4 October 2026. Project root:
`D:\Run time Redo\SE\Multispectral_Document_Binarization_using_Band_Reweighting_and_Dropping-main`.
This finishes the experiment-infrastructure/handoff implementation, **not the
unperformed research experiments**. Current source is distributed as package
**v3**; checkpoint format remains **2**. This file is the starting point when
the previous chat is unavailable.

## Start here next time

1. Read this file and `EXPERIMENTS_SETUP.md`; inspect actual local source before editing.
2. Run the full tests below. The reported **128 passes were on v2, before this final change**.
3. If useful, run the four-batch CPU software profile below in its fresh folder.
4. Create and verify the **new v3 archive**; preserve v2 and all historical checkpoints.
5. Upload v3 yourself to the correct Drive account/folder and open the revised notebook.
6. If Colab has no GPU quota, stop GPU work. There is no known reset time.
7. When CUDA is available, use one selected fresh configuration, review preflight,
   then explicitly enable the research-launch cell. Keep one writer per run.
8. Preserve logs/checkpoints. Complete matched runs before comparison; leave test data untouched.

## Purpose and output

The model separates selected foreground ink from document background using 12
aligned spectral bands. It produces a foreground probability map and binary
pixel mask, plus an inverted reading preview. It does **not** perform OCR,
transcription, language recognition, semantic document understanding, or
guaranteed document restoration. Binarization metrics do not measure OCR accuracy.

`unet_small.py` actually implements a **ResNet50 encoder with U-Net decoder**,
bilinear upsampling and skip connections. The encoder's RGB first-convolution
weights are repeated/scaled for multispectral input when ImageNet initialization
is requested. Optional Band Reweighting uses global average pooling and two
fully connected layers with sigmoid per-channel weights; Band Dropping uses
input-channel `Dropout2d` during training. It is disabled by `eval()` at inference.
No improvement from these options has yet been established by the planned comparison.

### Dataset and labels

- Fixed published page split: **60 training / 20 validation / 50 reserved test**.
- All training and validation images/labels come from original `MSBin/train`.
- Each page has bands 0–11. Do not regenerate or edit the published split files.
- Selected task is **FG1**; FG2 is a separate foreground task, not unioned with FG1.

| Label (OpenCV BGR) | Meaning | FG1 target / validity |
|---|---|---|
| `(255,255,255)` | FG1 main ink | 1 / valid |
| `(122,122,122)` | FG2 other ink | 0 / valid |
| `(0,0,0)` | Background | 0 / valid |
| `(255,0,0)` | Uncertain region (UR) | excluded |
| Added patch padding | Outside original page | excluded |

Unknown label colors fail rather than being guessed. Losses and metrics exclude
UR and padding. The full-dataset audit is preserved historical provenance; it
mentions test/duplicate paths absent from a portable training-only package and
cannot be rerun in full on that subset.

### Relevant file layout

```text
MSBin/train/images/<page>_<band>.png   original 80 pages, 960 band PNGs
MSBin/train/labels/<page>.png         original 80 label PNGs
MSBin/test/                          50 reserved pages; local only
MSBin/train_val_split/               duplicate trees; excluded from package
splits/train_keys.txt                fixed 60 keys, exact bytes preserved
splits/val_keys.txt                  fixed 20 keys, exact bytes preserved
splits/test_keys.txt                 reserved, excluded from package
results/dataset_audit/               historical audit JSON/CSV
train_unet_msbin_cpu.py              trainer (name is historical; supports CUDA)
training_support.py                 masks/losses/checkpoints/resume/epoch CSV
experiment_support.py               safe updates/shared initialization/timers/logging
msbin_dataset.py                     patch positions, decoding, optional bounded cache
msbin_labels.py, pixel_metrics.py    shared label and metric semantics
unet_small.py                        ResNet50 U-Net
infer_msbin_cpu.py                   offline reconstruction and tiled inference
experiments.json                     common budget/settings and four variants
run_experiments.py                   sequential run / evaluate / compare CLI
scripts/profile_msbin_training.py    bounded CPU/CUDA software profile
scripts/smoke_cuda_precision.py      optional synthetic CUDA check
scripts/package_gpu_training.py      allowlist ZIP create/verify/safe extract
tests/                              synthetic software tests
notebooks/msbin_experiments_colab.ipynb
results/baseline_gpu/                historical best.pt, last.pt, epochs.csv
results/baseline_gpu_val_all/        historical all-validation manifest/outputs
transfer/msbin_gpu_training.zip      historical package; preserve
transfer/msbin_gpu_experiments_v2.zip verified earlier package; preserve
transfer/msbin_gpu_experiments_v3.zip new output to create manually
```

AMDES is a separate existing implementation. It is **not yet connected to the
research checkpoint**; working integration has not been demonstrated. Do not
modify it as part of profiling, packaging or these experiments.

## Status: verified history versus new implementation

The user reports a Windows Python **3.12.6** virtual environment with CPU
torch **2.14.1+cpu** and torchvision **0.29.1+cpu**. The Intel Iris Xe laptop has
no local CUDA GPU. A working PowerShell interpreter does not imply Codex can
launch it: previous Codex probes failed with exit 101. No probe was repeated.

Before this final change, the user verified:

- 128 tests passed on the v2 workflow.
- CPU model smoke, CPU pilot, tiled inference and full-page evaluation worked.
- v2 package creation and hash verification succeeded: **1,073 payload files,
  1,035,997,472 payload bytes** (package manifest is additional).
- v2 was uploaded to Google Drive folder `MSBin_GPU_Training.` (literal trailing dot).

The v2 archive exists locally. In this final change, Codex inspected source,
tests/docs/notebook, directory state and the saved historical validation JSON;
it did **not** execute Python, unit tests, profiling, inference, notebook cells
or research training. No installation, upload, weight download or cloud action
was performed. New test/profile/package commands remain manual. The notebook
is saved with null execution counts and empty outputs.

Final-revision change inventory:

| Files | Concrete change |
|---|---|
| `scripts/profile_msbin_training.py`, `train_unet_msbin_cpu.py` | Explicit CPU/CUDA profile routing, thread selection before work, CPU+AMP rejection; research CUDA requirement retained. |
| `experiment_support.py` | CPU timing explicitly nonsynchronized; bounded loader records actual batch/sample counts. |
| `run_experiments.py` | Evaluation/comparison validates the requested run identity, common settings and budget against saved metadata. |
| `experiments.json` | Explicit common PyTorch thread count 2, avoiding preflight/subprocess differences from implicit runtime defaults. |
| `tests/test_gpu_experiments.py` | Focused routing, thread, CPU timing/count, safeguard, selection and notebook configuration tests; not executed here. |
| `scripts/package_gpu_training.py` | Adds this handoff to the explicit allowlist. |
| `notebooks/msbin_experiments_colab.ipynb` | One editable configuration, v3 archive, per-cell instructions/recovery, shared runtime preflight and changed-plan launch protection; unexecuted. |
| `PROJECT_HANDOFF.md`, `EXPERIMENTS_SETUP.md`, `GPU_TRAINING_SETUP.md`, `TRAINING_SETUP.md`, `INFERENCE_SETUP.md`, `README.md` | Current operating instructions, status boundaries, history and recovery. |

Review retained the existing masked-loss, GradScaler, shared-initialization,
cache, epoch-history and historical-inference implementations. This is a source
review, not a new runtime verification of those paths. Dataset, splits, audit,
historical checkpoints/archives and AMDES were not modified.

### Historical GPU baseline — validation only

The earlier run completed **7 of 10 intended epochs** on a Tesla T4, FP32,
batch 2, ImageNet-pretrained 12-band ResNet50 U-Net, unfrozen encoder, no
reweighting, drop probability 0. Best checkpoint: **epoch 4**. Its reported
patch validation F1 was `0.9100630197871566`.

The local `results/baseline_gpu_val_all/manifest.json` was inspected and confirms
status complete, epoch 4, **all 20 validation pages**, and these pooled scores:

| Quantity | Value |
|---|---:|
| TP | 6915234 |
| FP | 930748 |
| FN | 372767 |
| Precision | 0.8813726567305405 |
| Recall | 0.9488519554264606 |
| F1 | 0.9138683451672966 |

These are historical **validation results, not final test results**. The scores
were read from the saved manifest, not recomputed here. User-reported prior
Colab environment: Python 3.13.15, torch 2.11.0+cu130, torchvision 0.26.0+cu130,
OpenCV 4.14.0, NumPy 2.1.3, Tesla T4 (about 14.56 GiB). Epoch 1 reportedly took
2485.86 seconds; that is not a future-runtime estimate or proof of an I/O bottleneck.

The earlier AMP run failed on nonfinite gradients during strict clipping after
unscale. The user manually disabled autocast and scaling in that Colab runtime,
after which FP32 worked. Historical **version 1 checkpoints do not explicitly
record precision**. Preserve best.pt, last.pt and logs without migration or
training resume. They remain supported as inference inputs; no new full-model
inference was executed during this handoff change.

## Four controlled experiments

| Configuration | Reweighting | Band dropping |
|---|---|---:|
| baseline | off | 0 |
| reweight_only | on | 0 |
| drop_only | off | 0.2 |
| reweight_and_drop | on | 0.2 |

The default shared budget is **7 epochs per run**, seed `[42]`, FP32, batch 2,
workers 0, PyTorch CPU intra-op threads 2, cache 0 MiB until measured. Common settings: FG1, 12 bands, pretrained
ResNet50, unfrozen encoder, patch/stride 256, AdamW lr .0003, positive weight 6,
BCE .9, Dice .1, gradient clip 1, threshold .5. To use all valid training patches,
omit `--max_patches_per_page` (zero is invalid) and use `--min_fg_frac 0`.
Patches with no valid labels still remain excluded.

`experiments.json` supplies the CLI specification. The notebook has **one editable
configuration section**; it applies overrides to a local active specification
without changing the packaged source/spec. Every run records exact effective
arguments, splits and hashes, dependencies, device, resolved PyTorch thread count,
backend settings, source hashes and shared initialization identity.

Fairness controls reviewed in source:

- Every seed creates shared baseline backbone/decoder tensors and copies those
  exact tensors into all variants. Attention initialization uses separate CPU
  seed `seed + 1000003`; per-tensor and combined shared hashes are recorded.
- All variants use the same split bytes, settings and budget; runs are sequential.
- FP32 disables autocast/scaling. AMP uses public GradScaler overflow handling:
  unscale, inspect gradients, clip only finite gradients, skip/reduce scale on
  overflow. Nonfinite valid loss and FP32 gradients fail; persistent lack of
  optimizer updates fails. Overflow and invalid-label counts are separate.
- Masked loss inputs/reductions use float32; TF32 is disabled explicitly.
- The optional page LRU is bounded by retained array bytes per dataset/worker.
  Twelve uint8 bands + float32 target + bool validity cost about 17 bytes/pixel.
  Transient incoming pages/decoder buffers/batches add RAM beyond that budget.
  Cache 0 preserves sequential decoding; cached patches return independent tensors.
- Worker/cache/thread settings are recorded and checked on resume. CUDA kernels
  can still be nondeterministic; no bitwise cross-device/version guarantee is made.
- Best selection uses patch validation F1. Full-page comparison checks all 20
  validation pages exactly once, pooled counts and matching protocol/provenance.
  The selected folder must now match the requested configuration/seed/specification;
  an accidentally copied manifest cannot silently substitute another run.

Rerun baseline under the new workflow alongside variants. Do not silently treat
the historical seven-epoch result as equivalent to a new controlled baseline.
Additional seeds are supported, but one seed is preliminary and no statistical
significance or improvement is claimed. Do not tune dropout or thresholds on test data.

## Local operations (PowerShell)

First select the project and run **all tests**:

```powershell
Set-Location 'D:\Run time Redo\SE\Multispectral_Document_Binarization_using_Band_Reweighting_and_Dropping-main'
& '.\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests
```

Stop on failure; keep the complete output for diagnosis. A passing earlier
128-test report is not a pass for the source in this handoff.

### Four-batch CPU software profile, two threads

```powershell
& '.\.venv\Scripts\python.exe' -B scripts/profile_msbin_training.py --device cpu --threads 2 --outdir '.\results\cpu_profile_v3_threads2' --data '.\MSBin' --splits '.\splits' --cache_mb 512 --num_workers 0 --batches 4 --precision fp32
```

The failed earlier command hardcoded CUDA through the profiler. Inspection found
**no `results/cpu_profile_v2` directory**, so no files from that attempt existed
there. The new `results/cpu_profile_v3_threads2` path was also absent at handoff;
use another fresh suffix if it exists when you run the command.

This is **software profiling, not CPU support for the controlled research sweep**.
It uses random weights, no pretrained download, one bounded pass of at most four
training and four validation batches (batch 2: normally eight samples per stage).
It still indexes the full published splits and serializes sizeable software-only
checkpoints. CPU work can take time and RAM; no completion estimate is established.
Do not use profile checkpoints for research inference/resume.

`--threads 2` sets PyTorch CPU intra-op threads before computation; it does not
set OpenCV threads or DataLoader workers. CPU+AMP is rejected. CPU profiling has
no CUDA synchronization/timing calls. `timing.json` records device, resolved
threads, batch size/limit, actual loaded batches/samples and separate indexing,
data-wait, transfer, forward/backward/update, validation and checkpoint-writing
times. Validation timing includes its data loading. `training.log` and run.json
record context. Compare cache-off/on with identical settings and fresh directories;
short samples and filesystem caches limit conclusions. No cache speedup has been measured here.

### Create and verify v3; preserve v2

```powershell
& '.\.venv\Scripts\python.exe' -B scripts/package_gpu_training.py create --project-root '.' --output '.\transfer\msbin_gpu_experiments_v3.zip'
& '.\.venv\Scripts\python.exe' -B scripts/package_gpu_training.py verify --archive '.\transfer\msbin_gpu_experiments_v3.zip'
```

Do not pass `--overwrite`. A name collision means choose another new name and
update the notebook configuration to match. The packager streams data, verifies
hashes/inventory and atomically publishes on success. The allowlist includes this
handoff, notebook, code/tests/docs, original train images/labels, exact train/val
split bytes and historical audit files. It excludes all checkpoints, test data/keys,
duplicate dataset trees, .venv, caches, secrets, archives/output directories and AMDES.
The verifier is standard-library-only; tests/training need the existing environment.
The updated verifier expects the new inventory; verify an older package with its
own packaged verifier. v3 has not been built or verified in this final Codex turn.

## Colab operations and recovery

1. Upload v3 yourself to the Google Drive account holding the project, at:
   `/content/drive/MyDrive/MSBin_GPU_Training./msbin_gpu_experiments_v3.zip`.
   Preserve the uploaded v2 archive and historical
   `/content/drive/MyDrive/MSBin_GPU_Training./runs/fg1_baseline_fp32`.
2. Open the revised local `notebooks/msbin_experiments_colab.ipynb` in Colab.
   If replacing an older saved notebook, retain a local copy first.
3. Edit only its configuration section: one selected configuration, fresh/resume,
   paths, shared settings and action flags. Research training defaults disabled.
4. Inspect dependencies/CUDA. Colab currently refuses GPU access according to
   the user; there is **no known reset time**. Stop when unavailable, with no bypass.
   No notebook cell silently installs or replaces torch/torchvision.
5. Mount Drive, copy/verify the archive, and extract to new local
   `/content/msbin_project`. Data loading uses local disk, not Drive images.
6. Run synthetic tests and optional tiny CUDA smoke. Optional profiling is
   software-only. Then run fresh/resume preflight and explicitly enable launch.
7. New run folders are under
   `/content/drive/MyDrive/MSBin_GPU_Training./runs/experiments_v3`.
   Text logs, CSVs, configuration, spec snapshots and checkpoints persist there.
8. After the full common budget, enable validation/comparison for selected runs.
   Download/backup run folders, all-validation outputs and comparison files.

Each operational cell explains its job, expected success, failure action and
rerun rules. Existing extraction, fresh-run and evaluation outputs are protected.
If a runtime is replaced, remount Drive, re-extract the **exact matching archived
source at the original dataset root**, and inspect environment compatibility.
Do not use v3 source to resume a checkpoint created by v2 source: the source hash
check intentionally rejects that change. Use its archived source/environment.

Changing ChatGPT accounts does not transfer project files or this conversation.
Google Drive paths require authorized access to the Google account holding those
files. Keep your own source, notebook, handoff and result backups. No secrets or
public sharing are needed. Account changes are not a GPU-quota workaround.

### Supported GPU commands after environment setup

From the extracted project using the prepared GPU Python (not the local CPU environment):

```text
python scripts/smoke_cuda_precision.py
python scripts/profile_msbin_training.py --device cuda --threads 2 --outdir profile_cuda_v3 --cache_mb 512 --num_workers 0 --batches 4 --precision fp32
python run_experiments.py run --configuration baseline --runs runs_gpu/experiments_v3
python run_experiments.py run --configuration baseline --runs runs_gpu/experiments_v3 --resume
python run_experiments.py evaluate --configuration baseline --runs runs_gpu/experiments_v3
python run_experiments.py compare --configuration baseline --runs runs_gpu/experiments_v3 --output results/comparison_v3
```

These are separate choices, not a script to blindly run top-to-bottom. Notebook
commands pass its persistent Drive run path and active spec. CLI defaults use
packaged experiments.json; use `--spec path/to/spec.json` for an alternative.
For all four fresh runs, select `--configuration all` with a new run tree; this
executes sequentially. Resume accepts one configuration and one seed, never a
silent fresh fallback. Use all for evaluation/comparison only when all budgets
are complete. The runner retains mandatory `--require_cuda`; this handoff does
not add CPU research-sweep support.

## Checkpoints, resume, logs and storage

- Historical version 1 checkpoints remain **inference-only**. Their FP32 history
  is operator provenance, not a precision field inferred from device.
- New checkpoints are version 2 with explicit precision/source metadata. Resume
  checks model configuration, split hashes/keys, dataset root, device/GPU, resolved
  thread count, cache/workers and training settings, source hashes and dependency
  versions. Total epochs may increase, never decrease. Operational output/key
  paths may move only with matching split content; dataset root must match.
- Optimizer/scaler, Python/NumPy/torch/CUDA RNG and shuffle-generator state are
  restored. Partial epochs are repeated from the last committed epoch.
- `last.pt` is the epoch commit point; `best.pt` is selected by patch validation
  F1. Checkpoint history regenerates epoch CSVs without duplicate committed rows.
  Repeated attempt messages in the append-only text log are expected after interruption.
- Writes use temporary files, flush/fsync and replacement at the file level.
  These do not promise multi-file transactions or remote Drive durability.
  Read back saved checkpoints and keep backups. Never launch two writers into
  one run folder. On disk/write failure, preserve files and check local free space
  and Drive quota before retrying. Never delete the only usable checkpoint to make room.

## Inference and metrics

Inference reconstructs the architecture **offline from checkpoint metadata**,
then strictly loads saved weights; it does not download ImageNet weights.
It stitches mean sigmoid probabilities for overlapping tiles and thresholds
with strict `probability > 0.5`. Standard tile/stride are 256/256; edge coverage
is preserved. Every original valid pixel is counted once for full-page metrics.

Machine masks: foreground **white 255**, background **black 0**. Reading previews:
foreground **black 0**, background **white 255**. Comparison panels show input
band 0, reading-polarity ground truth (UR magenta), and reading prediction.
These conventions have not been changed. Pooled precision/recall/F1 are computed
after summing TP/FP/FN; `page_mean_*` is an unweighted average of page metrics,
not the pooled score. Patch validation can count edge overlap multiple times and
is not interchangeable with full-page validation.

Optional historical checkpoint inference on CPU, in a **new** output folder:

```powershell
& '.\.venv\Scripts\python.exe' -B infer_msbin_cpu.py --msbin_root '.\MSBin' --keys_txt '.\splits\val_keys.txt' --ckpt '.\results\baseline_gpu\best.pt' --outdir '.\results\baseline_gpu_val_v3_recheck' --device cpu --threads 2 --tile 256 --stride 256 --thr 0.5 --compare_gt
```

This may be slow locally. It is validation inference, not training or final test
evaluation, and was not run in this change. Controlled comparison outputs
`comparison.csv`, `comparison.json` and `per_page.csv` with configuration, seed,
selected epoch, checkpoint identity, TP/FP/FN and metrics for all 20 validation
pages. Mismatched splits, task, settings, budget, source/environment or shared
initialization are rejected. Existing validation_all directories are not overwritten.

## Troubleshooting actual failure modes

| Symptom | Action |
|---|---|
| Old CPU profiler raises “CUDA is required…” | Use the v3 profiler with explicit `--device cpu --threads 2 --precision fp32`; do not remove CUDA safeguards from the GPU runner. |
| CPU+AMP requested | Choose CPU FP32 or an actually available CUDA device. |
| Colab GPU usage limit | Stop GPU work; no known reset or completion date. No keep-alive/quota bypass. |
| AMP nonfinite valid loss/gradients | Keep logs; FP32 is the reliable default. AMP overflow can skip an update, but valid-loss failure or persistent no-update failure stops. Do not disable strict checks globally. |
| Fresh output already exists | Preserve it; use explicit compatible resume or a new run name. |
| Incompatible resume | Stop. Recreate matching source/environment/root/settings or start a separate fresh run; do not bypass/migrate historical files. |
| Runtime disappeared | Remount Drive, re-extract matching source at the original local root and preflight before resume. |
| Extraction stopped partway | Preserve diagnostics; do not merge another archive into that tree. Use a clean extraction/runtime, keeping resume root constraints in mind. |
| Partial validation_all folder | Preserve/rename the partial folder, then deliberately rerun evaluation into the now-new validation_all path. Do not change checkpoint weights. |
| Insufficient storage/write error | Preserve/checkpoint backups, inspect local space and Drive quota; do not assume mount free-space reporting guarantees remote capacity. |
| Notebook save conflict | Download a local .ipynb copy before resolving the conflict or closing the session. |
| Codex Python exit 101 | Use the working manual PowerShell commands; do not repeat probes or bypass that execution limitation. |

## Remaining research and product work

1. Manually test this final source, run useful profiles, then build/verify/upload v3.
2. Obtain legitimate CUDA availability; run the new controlled baseline and all
   three variants with the same seven-epoch budget/settings. Add seeds if feasible.
3. Perform all-validation inference and compare measured outcomes. Do not claim
   a method helps before measurements or infer significance from one seed.
4. Freeze the chosen model and evaluation protocol, then implement/review the
   final locked test-evaluation procedure. Current preview CLI deliberately only
   exposes original train/validation data; do not invent a test command.
5. Connect and validate AMDES against the research checkpoint; it is not connected yet.
6. Prepare the report/PPT using measured results, limitations and correct scope.

No final accuracy, speedup, significance, GPU reset or completion date is promised.

## Ready-to-paste continuation prompt

```text
Work in D:\Run time Redo\SE\Multispectral_Document_Binarization_using_Band_Reweighting_and_Dropping-main.
First read PROJECT_HANDOFF.md, EXPERIMENTS_SETUP.md, actual local source, tests,
experiments.json and notebooks/msbin_experiments_colab.ipynb. Do not assume unseen
chat context or that an earlier test pass validates later edits. The reported
128-test pass predates the final v3 changes; ask for/inspect newer manual output.
The infrastructure/handoff is implemented; new controlled research runs remain.
Preserve dataset/splits/audits, historical v1 best/last checkpoints, existing
archives and AMDES. Historical v1 is inference-only and lacks explicit precision.
Do not weaken precision/source/environment/resume checks or change metric polarity.
Colab GPU quota was unavailable with no known reset; do not bypass it.
No Python probes (known Codex exit 101), installations, downloads, uploads,
cloud actions, model training or subagents unless I explicitly change those bounds.
Source-review and provide exact manual commands; never claim unexecuted tests or
research results. Tell me what evidence is available and what remains unverified.
My next requested task is: [describe the concrete work and paste any new test/profile output].
```
