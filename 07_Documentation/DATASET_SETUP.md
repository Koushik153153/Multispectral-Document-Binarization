# MSBin dataset audit and page lists

## Current verification status

The existing audit reports `valid=true`, no errors, 80 original training pages,
50 test pages, and a supplied split of 60 training / 20 validation pages. It records
1,040 SHA-256 comparisons with no mismatches and confirms split-list publication.
These are measured observations, not hardcoded requirements. The user reports
15 passing validator tests and uncertain regions on 3 original training pages
and 42 test pages. Successful dataset checks do not establish model performance.

During Step 9, the saved audit and split lists were inspected without modification.
The project interpreter still fails to launch within the Codex session (exit 101,
`Unable to create process using` the base Python 3.12 executable). Consequently,
the new loading/metric tests and loading check have not been executed here.

## Run from the project directory in PowerShell

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests/test_validate_msbin_dataset.py
& '.\.venv\Scripts\python.exe' -B scripts/validate_msbin_dataset.py --msbin-root '.\MSBin' --output-dir '.\results\dataset_audit'
```

The validator decodes all PNG bands and labels in the original and supplied split
directories. Inputs must be uint8 grayscale; labels must be uint8 three-channel
BGR with the four documented class colors. Other representations/colors are errors,
not silently converted. It checks exact bands 0–11, labels, dimensions, key separation,
and SHA-256 equality for every supplied file against the corresponding original.
Files are processed sequentially, with chunked hashing and label-color counting.
Pages smaller than 256 pixels on either axis and uncertain-region occurrences are
reported as warnings. Warnings do not block publication; errors do.

On success, sorted unique keys with a final newline are written to:

- `splits/train_keys.txt`: supplied training keys, used with original `MSBin/train`.
- `splits/val_keys.txt`: supplied validation keys, also used with `MSBin/train`.
- `splits/test_keys.txt`: original `MSBin/test` keys.

`--splits-dir` can select another output directory. All output destinations must
remain outside the dataset. Existing lists must be byte-identical; conflicting
lists are preserved and cause failure before any list is published. The top-level
`test_keys.txt` is legacy Python text and is neither executed nor overwritten.

Reports are `results/dataset_audit/dataset_audit.json` (measured counts, per-file
shapes/dtypes, color pixel counts, uncertain-region pages, hashes, errors/warnings)
and `dataset_manifest.csv` (one row per original page, assigned split, dimensions,
and paths relative to MSBin). Validation failures return a nonzero exit status and
may write an error JSON, but do not publish new split lists or a manifest. Reports
from earlier runs may remain: always check the latest JSON's `valid` and
`split_lists_published` fields and the process exit status.

## Split policy

Reuse the supplied split, never generate a replacement random split. The local
`MSBin/README.md` describes `train_val_split` as the training/validation subdivision
used for the dataset's referenced CNN study. Hash checks establish whether this
local copy matches the original training images; they do not prove provenance.

Training and validation must use disjoint pages so patches from one document page
cannot appear in both. Test pages remain reserved for final evaluation and must
not guide model choices, thresholds, or other training parameters. The audit reads
test images only to validate their integrity and representation. Tiny synthetic
pytest fixtures test validation logic only; they are not research training data.

## Shared labels and dataset API (Step 9)

`msbin_labels.decode_label(label_bgr, fg_type)` returns a float32 binary target and
a boolean validity mask, each `(H,W)`. Only `fg_type=1` (white main text) and
`fg_type=2` (gray red ink) are supported. The selected class is 1; the other ink
class is valid background (0), never implicitly added to the selected class.
Black is background. BGR `(255,0,0)` is uncertain: target 0 and validity False.
Unexpected colors and incompatible image representations produce errors.

`MSBinDibcoPatchDataset` retains its default `(image, target)` return. Pass
`return_valid_mask=True` to receive `(image, target, valid_mask)`. Images are
float32 `(12,H,W)` normalized to `[0,1]`, or `(1,H,W)` with `use_white_only=True`.
Targets are float32 `(1,H,W)` and validity masks are bool `(1,H,W)`.

Patches cover the bottom/right edges without duplicate starts: length 600 with
patch/stride 256 uses starts 0, 256, 344. Pages smaller than a patch produce a
padded patch. Image padding replicates edge pixels; target padding is zero and
mask padding is False. Foreground filtering divides selected foreground by valid
pixels only. All-invalid patches are skipped. `min_fg_frac=0` and no patch cap
include all remaining patches, including background-only patches. Filtering and
caps can intentionally reduce coverage. Overlapping patches count some pixels
more than once, so patch-level scores are not equivalent to full-page scores.

Nonpositive/noninteger patch sizes or strides, strides larger than the patch,
invalid foreground fractions/caps, duplicate or unknown explicit keys, incomplete
band groups, missing labels, and dimension mismatches fail explicitly. As before,
single-band mode still requires a complete audited 12-band page group. Indexing
checks bands sequentially and retains positions/shapes only, not a dataset-sized
image cache. This adds startup I/O; each requested patch rereads its page images.
The shared band-reading helpers now reject multichannel or non-uint8 images
instead of silently converting them; the audited dataset has the required format.

`pixel_metrics.PixelCounts.update(probabilities, target, valid_mask, threshold=0.5)`
accumulates integer TP/FP/FN. Call `.scores()` after all batches for micro precision,
recall and F1. `pixel_metrics(...)` is the one-call alternative. Inputs can be
NumPy arrays or detached PyTorch tensors and must have identical shapes; a supplied
mask must be boolean. Probabilities must be finite in `[0,1]`, and predictions use
strict `probability > threshold`. Target values must be 0 or 1. All zero-denominator
scores are 0, including an all-invalid evaluation. No NaNs are used for empty cases.

The default two-item API cannot convey ignored pixels. Existing training losses
and internal metrics therefore still treat UR/padding as background until training
is separately updated to consume validity masks. Neither training nor inference,
model architecture, or AMDES was changed in this step.

## Verify Step 9 without training

From the project directory in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests/test_validate_msbin_dataset.py tests/test_msbin_loading_metrics.py
& '.\.venv\Scripts\python.exe' -B scripts/check_msbin_loading.py --msbin-root '.\MSBin' --audit '.\results\dataset_audit\dataset_audit.json' --splits-dir '.\splits'
```

The read-only loading check verifies the saved audit and split membership, indexes
train/validation from original `MSBin/train` with masks enabled and filtering/caps
disabled, then inspects a few patches including an edge or padded patch when
available. It checks shapes, dtypes, ranges, binary targets and invalid padding,
and prints measured page/patch counts. It reads test keys only; it never samples
test images, constructs a model, trains, writes reports, or regenerates splits.
Its output is dataset checking, not research results.
