"""Held-out MSBin test evaluation."""
import argparse
import csv
import gc
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_msbin_cpu import (
    load_inference_model,
    requested_pages,
    read_page,
    tiled_probabilities,
    mask_images,
    comparison_image,
    write_png,
)
from msbin_dataset import _read_png_bgr
from msbin_labels import decode_label
from pixel_metrics import PixelCounts
from training_support import read_checkpoint, versions

MODELS = {
    "baseline": ("baseline_seed42/best.pt", False, 0.0),
    "reweight_only": ("reweight_only_seed42/best.pt", True, 0.0),
    "drop_only": ("drop_only_seed42/best.pt", False, 0.2),
    "reweight_and_drop": (
        "reweight_and_drop_seed42/best.pt", True, 0.2
    ),
}

THRESHOLD = 0.5
TILE = 256
STRIDE = 256


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path, reweight, drop):
    ckpt = read_checkpoint(path)
    config = ckpt["model_config"]

    if config.get("in_ch") != 12:
        raise ValueError(f"{path}: expected 12 input bands")
    if config.get("band_reweight") is not reweight:
        raise ValueError(f"{path}: incorrect reweight configuration")
    if float(config.get("band_drop_p", -1)) != drop:
        raise ValueError(f"{path}: incorrect band-drop configuration")
    if ckpt.get("training_args", {}).get("fg_type") != 1:
        raise ValueError(f"{path}: expected FG1 labels")
    if len(set(ckpt["splits"]["val"]["keys"])) != 20:
        raise ValueError(f"{path}: unexpected validation metadata")


def validate_setup(root, data_root, output):
    train = (root / "splits/train_keys.txt").read_text(
        encoding="utf-8-sig"
    ).split()
    val = (root / "splits/val_keys.txt").read_text(
        encoding="utf-8-sig"
    ).split()
    test = (root / "splits/test_keys.txt").read_text(
        encoding="utf-8-sig"
    ).split()

    if (len(train), len(val), len(test)) != (60, 20, 50):
        raise ValueError("Unexpected train/validation/test split sizes")
    if any(len(set(x)) != len(x) for x in (train, val, test)):
        raise ValueError("Duplicate page keys")
    if (
        set(train) & set(val)
        or set(train) & set(test)
        or set(val) & set(test)
    ):
        raise ValueError("Dataset splits overlap")
    if output.exists():
        raise FileExistsError(
            f"Output already exists; refusing overwrite: {output}"
        )

    image_dir = data_root / "test/images"
    label_dir = data_root / "test/labels"
    pages, _, requested = requested_pages(
        image_dir, root / "splits/test_keys.txt"
    )
    if len(requested) != 50:
        raise ValueError("Expected 50 test pages")
    if {p.stem for p in label_dir.glob("*.png")} != set(test):
        raise ValueError("Test labels do not match test keys")

    for key in test:
        label = _read_png_bgr(label_dir / f"{key}.png")
        target, valid = decode_label(label, fg_type=1)
        del target, valid
        if set(pages[key]) != set(range(12)):
            raise ValueError(f"{key}: missing spectral bands")
        for path in pages[key].values():
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != label.shape[:2]:
                raise ValueError(f"{key}: invalid band or dimensions")

    return test, pages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("MSBin/MSBin"))
    parser.add_argument("--outdir", type=Path, default=Path("test_evaluation_all"))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    args = parser.parse_args()

    root = args.root.resolve()
    data_root = args.data_root
    if not data_root.is_absolute():
        data_root = (root / data_root).resolve()
    output = args.outdir
    if not output.is_absolute():
        output = (root / output).resolve()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    test, pages = validate_setup(root, data_root, output)
    checkpoints = {}
    for name, (rel, reweight, drop) in MODELS.items():
        path = root / rel
        validate_checkpoint(path, reweight, drop)
        checkpoints[name] = path.resolve()

    output.mkdir(parents=True, exist_ok=False)
    comparison_rows = []
    page_rows = []
    model_manifests = {}
    labels_dir = data_root / "test/labels"

    for name, (_, reweight, drop) in MODELS.items():
        print(f"\nEvaluating {name}...", flush=True)
        model, metadata = load_inference_model(
            checkpoints[name], device=device
        )
        model_dir = output / name
        (model_dir / "masks").mkdir(parents=True)
        (model_dir / "comparisons").mkdir()

        pooled = PixelCounts()
        scores_per_page = []
        for i, key in enumerate(test, start=1):
            page = read_page(pages[key], 12, key)
            probabilities, _ = tiled_probabilities(
                model, page, TILE, STRIDE, device
            )
            label = _read_png_bgr(labels_dir / f"{key}.png")
            target, valid = decode_label(label, fg_type=1)

            pooled.update(probabilities, target, valid, THRESHOLD)
            counts = PixelCounts().update(
                probabilities, target, valid, THRESHOLD
            ).scores()
            row = {
                "model": name,
                "page": key,
                **counts,
                "valid_pixels": int(valid.sum()),
                "uncertain_pixels": int(valid.size - valid.sum()),
            }
            page_rows.append(row)
            scores_per_page.append(row)

            mask, preview = mask_images(probabilities, THRESHOLD)
            panel = comparison_image(page[0], target, valid, preview)
            write_png(model_dir / "masks" / f"{key}.png", mask)
            write_png(
                model_dir / "comparisons" / f"{key}.png", panel
            )
            del page, probabilities, label, target, valid
            del mask, preview, panel

            if i % 5 == 0:
                print(f"  {i}/50 pages", flush=True)

        pooled_scores = pooled.scores()
        page_means = {
            metric: float(np.mean([r[metric] for r in scores_per_page]))
            for metric in ("precision", "recall", "f1")
        }
        comparison_rows.append({
            "model": name,
            "completed_epoch": metadata["completed_epoch"],
            "band_reweight": reweight,
            "band_drop_p": drop,
            "test_pages": len(test),
            "threshold": THRESHOLD,
            **pooled_scores,
            **{f"page_mean_{k}": v for k, v in page_means.items()},
        })
        model_manifests[name] = {
            "checkpoint": metadata["checkpoint"],
            "model_config": metadata["model_config"],
            "completed_epoch": metadata["completed_epoch"],
            "pooled_pixel_scores": pooled_scores,
            "unweighted_page_mean_scores": page_means,
        }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def write_csv(path, rows):
        with path.open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "comparison.csv", comparison_rows)
    write_csv(output / "per_page.csv", page_rows)

    manifest = {
        "purpose": "Held-out test evaluation; no test-driven tuning",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(data_root),
        "test_keys_sha256": sha256_file(root / "splits/test_keys.txt"),
        "test_pages": test,
        "threshold": THRESHOLD,
        "tile": TILE,
        "stride": STRIDE,
        "foreground": "FG1 main text",
        "uncertain_regions": "Excluded from pixel metrics",
        "device": device,
        "python_version": platform.python_version(),
        "software_versions": versions(),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "models": model_manifests,
        "interpretation": (
            "One seed per variant. Descriptive results only; "
            "no statistical significance claim."
        ),
    }
    with (output / "manifest.json").open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write("\n")

    print("\nTEST EVALUATION COMPLETE")
    print("Results folder:", output)
    for row in comparison_rows:
        print(
            f"{row['model']}: P={row['precision']:.5f}, "
            f"R={row['recall']:.5f}, F1={row['f1']:.5f}"
        )


if __name__ == "__main__":
    main()