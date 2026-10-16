"""Fast MSBin live demo: trained checkpoints, small official test sample, CSV metrics."""
import csv, gc, json, sys, time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT = Path(r"D:\MSBin_Final_Project")
SOURCE = PROJECT / "01_Source_Code"
DATA_ROOT = PROJECT / "02_Dataset_and_Splits" / "MSBin" / "MSBin"
SPLIT_ROOT = PROJECT / "02_Dataset_and_Splits" / "splits"
CHECKPOINT_ROOT = PROJECT / "03_Trained_Models"
DEMO_PAGES = 2
THRESHOLD, TILE, STRIDE, THREADS = 0.5, 256, 256, 2
MODELS = {
    "baseline": ("baseline_seed42/best.pt", False, 0.0),
    "reweight_only": ("reweight_only_seed42/best.pt", True, 0.0),
    "drop_only": ("drop_only_seed42/best.pt", False, 0.2),
    "reweight_and_drop": ("reweight_and_drop_seed42/best.pt", True, 0.2),
}

def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def make_band_grid(band_paths, ground_truth, valid_mask, prediction, tile_size=(300, 220)):
    """Create a labeled 4x4 panel with fixed-size, fixed-type tiles."""
    tile_w, tile_h = tile_size
    tiles = []

    def add_label(image, label):
        # Force every tile to the same dimensions and 8-bit, 3-channel format.
        image = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = np.asarray(image, dtype=np.uint8).copy()
        cv2.rectangle(image, (0, 0), (tile_w - 1, 30), (30, 30, 30), -1)
        cv2.putText(image, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return np.ascontiguousarray(image)

    for band_index in range(12):
        image = cv2.imread(str(band_paths[band_index]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read spectral band {band_index}: {band_paths[band_index]}")
        tiles.append(add_label(image, f"Band {band_index}"))

    gt_gray = ((1 - ground_truth) * 255).astype(np.uint8)
    gt = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2BGR)
    gt[~valid_mask] = (255, 0, 255)
    tiles.append(add_label(gt, "Ground truth (UR magenta)"))

    pred = prediction
    if pred.ndim == 3:
        if pred.shape[2] == 3:
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        elif pred.shape[2] == 4:
            pred = cv2.cvtColor(pred, cv2.COLOR_BGRA2GRAY)
    tiles.append(add_label(pred, "Prediction"))

    blank = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
    cv2.putText(blank, "12 spectral bands", (35, 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(blank, "+ ground truth", (35, 130), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(blank, "+ prediction", (35, 165), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (40, 40, 40), 1, cv2.LINE_AA)
    tiles.append(np.ascontiguousarray(blank))

    # The 4x4 layout requires exactly 16 tiles. Add a legend tile as the final cell.
    legend = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
    cv2.putText(legend, "Legend", (35, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(legend, "Black = foreground", (35, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(legend, "White = background", (35, 135), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(legend, "Magenta = uncertain", (35, 170), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (180, 0, 180), 1, cv2.LINE_AA)
    tiles.append(np.ascontiguousarray(legend))

    # Validate tile shape/type before stacking, avoiding OpenCV vconcat assertions.
    expected_shape = (tile_h, tile_w, 3)
    normalized = []
    for i, tile in enumerate(tiles):
        if tile.shape != expected_shape:
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
            if tile.ndim == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        tile = np.ascontiguousarray(tile, dtype=np.uint8)
        if tile.shape != expected_shape:
            raise ValueError(f"Visualization tile {i} has unexpected shape {tile.shape}")
        normalized.append(tile)

    rows = [np.concatenate(normalized[i:i + 4], axis=1) for i in range(0, 16, 4)]
    return np.concatenate(rows, axis=0)

def main():
    if not SOURCE.is_dir(): raise FileNotFoundError(f"Source code folder not found: {SOURCE}")
    if not DATA_ROOT.is_dir(): raise FileNotFoundError(f"MSBin dataset folder not found: {DATA_ROOT}")
    sys.path.insert(0, str(SOURCE))
    from infer_msbin_cpu import load_inference_model, requested_pages, read_page, tiled_probabilities, mask_images, comparison_image, write_png
    from msbin_dataset import _read_png_bgr
    from msbin_labels import decode_label
    from pixel_metrics import PixelCounts

    images_dir, labels_dir = DATA_ROOT / "test" / "images", DATA_ROOT / "test" / "labels"
    keys_file = SPLIT_ROOT / "test_keys.txt"
    if not keys_file.is_file(): raise FileNotFoundError(f"Test split list not found: {keys_file}")
    pages, all_keys, _ = requested_pages(images_dir, keys_file)
    selected = all_keys[:DEMO_PAGES]
    if not selected: raise RuntimeError("No test pages found in test_keys.txt")

    demo_root = Path(__file__).resolve().parent
    results_parent = demo_root / "test_results"
    results_parent.mkdir(parents=True, exist_ok=True)
    outdir = results_parent / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=False, exist_ok=False)
    torch.set_num_threads(THREADS)
    device = "cpu"
    print("=" * 68)
    print("MSBin LIVE DEMO | CPU inference | no training")
    print(f"Pages: {', '.join(selected)}")
    print(f"Models: {len(MODELS)} | Device: {device} | Threshold: {THRESHOLD}")
    print(f"Output: {outdir}")
    print("=" * 68, flush=True)

    summary_rows, page_rows = [], []
    total_start = time.perf_counter()
    for model_name, (relative_ckpt, expected_reweight, expected_drop) in MODELS.items():
        ckpt_path = CHECKPOINT_ROOT / relative_ckpt
        if not ckpt_path.is_file(): raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"\n[{model_name}] Loading checkpoint ({ckpt_path.stat().st_size / (1024**2):.0f} MB)...", flush=True)
        load_start = time.perf_counter()
        model, metadata = load_inference_model(ckpt_path, device=device)
        load_seconds = time.perf_counter() - load_start
        cfg = metadata["model_config"]
        if cfg.get("in_ch") != 12: raise ValueError(f"{model_name}: expected 12 input bands")
        if cfg.get("band_reweight") is not expected_reweight: raise ValueError(f"{model_name}: unexpected band_reweight setting")
        if float(cfg.get("band_drop_p", -1)) != expected_drop: raise ValueError(f"{model_name}: unexpected band_drop_p setting")
        if metadata.get("fg_type") != 1: raise ValueError(f"{model_name}: expected FG1 labels")
        print(f"  Loaded in {load_seconds:.2f} s", flush=True)
        model_dir = outdir / model_name
        (model_dir / "masks").mkdir(parents=True)
        (model_dir / "comparisons").mkdir()
        (model_dir / "band_visualization").mkdir()
        pooled, per_model_rows, total_inference = PixelCounts(), [], 0.0

        for idx, key in enumerate(selected, start=1):
            print(f"  Page {idx}/{len(selected)}: {key} ...", flush=True)
            page = read_page(pages[key], 12, key)
            label_path = labels_dir / f"{key}.png"
            if not label_path.is_file(): raise FileNotFoundError(f"Ground-truth label missing: {label_path}")
            label = _read_png_bgr(label_path)
            target, valid = decode_label(label, fg_type=1)
            if target.shape != page.shape[1:]: raise ValueError(f"{key}: label size does not match spectral input")
            infer_start = time.perf_counter()
            probabilities, coverage = tiled_probabilities(model, page, TILE, STRIDE, device)
            infer_seconds = time.perf_counter() - infer_start
            total_inference += infer_seconds
            page_counts = PixelCounts().update(probabilities, target, valid, THRESHOLD)
            scores = page_counts.scores()
            pooled.update(probabilities, target, valid, THRESHOLD)
            row = {"model": model_name, "page": key, **scores,
                   "valid_pixels": int(valid.sum()), "uncertain_pixels": int(valid.size-valid.sum()),
                   "inference_seconds": round(infer_seconds, 4)}
            page_rows.append(row); per_model_rows.append(row)
            mask, preview = mask_images(probabilities, THRESHOLD)
            panel = comparison_image(page[0], target, valid, preview)
            write_png(model_dir / "masks" / f"{key}.png", mask)
            write_png(model_dir / "comparisons" / f"{key}.png", panel)
            band_grid = make_band_grid(pages[key], target, valid, preview)
            write_png(model_dir / "band_visualization" / f"{key}_12_bands.png", band_grid)
            print(f"    inference {infer_seconds:.2f}s | F1={scores.get('f1', float('nan')):.5f}", flush=True)
            del page, label, target, valid, probabilities, coverage, mask, preview, panel, band_grid, page_counts
            gc.collect()

        pooled_scores = pooled.scores()
        summary_rows.append({"model": model_name, "checkpoint": str(ckpt_path),
            "completed_epoch": metadata.get("completed_epoch"), "test_pages": len(selected),
            "threshold": THRESHOLD, **pooled_scores, "model_load_seconds": round(load_seconds, 4),
            "total_inference_seconds": round(total_inference, 4),
            "mean_inference_seconds_per_page": round(total_inference/len(selected), 4)})
        del model, pooled
        gc.collect()

    write_csv(outdir / "analysis_summary.csv", summary_rows)
    write_csv(outdir / "per_page_metrics.csv", page_rows)
    manifest = {"purpose": "Small live demo using pages from official held-out test split",
        "created_local": datetime.now().isoformat(timespec="seconds"), "pages": selected,
        "models": list(MODELS.keys()), "device": device, "threshold": THRESHOLD,
        "tile": TILE, "stride": STRIDE, "uncertain_regions": "Excluded from pixel-level metrics",
        "metrics_scope": "Descriptive sample only; not a replacement for full 50-page test evaluation",
        "total_wall_seconds": round(time.perf_counter()-total_start, 3)}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print("\\n" + "="*68 + "\\nDEMO COMPLETE")
    print(f"Results saved to: {outdir}\\nSummary:")
    for row in summary_rows:
        print(f"{row['model']}: P={row.get('precision', float('nan')):.5f}, R={row.get('recall', float('nan')):.5f}, F1={row.get('f1', float('nan')):.5f}, inference={row['total_inference_seconds']:.2f}s")

    # Identify the best model for each metric in the final console output.
    metric_specs = [
        ("precision", "Highest Precision", max),
        ("recall", "Highest Recall", max),
        ("f1", "Highest F1-score", max),
        ("total_inference_seconds", "Fastest inference (lowest time)", min),
    ]
    print("\\nBEST MODEL BY METRIC")
    print("-"*68)
    for field, label, chooser in metric_specs:
        valid_rows = [r for r in summary_rows if isinstance(r.get(field), (int, float))]
        if valid_rows:
            best = chooser(valid_rows, key=lambda r: r[field])
            value = (f"{best[field]:.2f} s" if field == "total_inference_seconds"
                     else f"{best[field]:.5f}")
            print(f"{label}: {best['model']} ({value})")
    print("-"*68)
    print(f"\\nTotal elapsed: {manifest['total_wall_seconds']:.2f}s")
    print("Note: small demo sample, not the full test-set result.")
    print("Each model folder now includes band_visualization with 12 bands + ground truth + prediction.")
    print("="*68)

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"\nDEMO FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
