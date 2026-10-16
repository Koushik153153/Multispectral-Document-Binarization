"""Versioned-checkpoint validation preview; tiled inference, never research training."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from msbin_dataset import _read_png_gray, _read_png_bgr, group_books, patch_starts, read_page_keys
from msbin_labels import decode_label, validate_fg_type
from pixel_metrics import PixelCounts
from training_support import read_checkpoint, build_model, versions


def load_inference_model(path, device="cpu", factory=None):
    """Load one explicitly supplied trusted local checkpoint; discard training state."""
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Checkpoint must be a local file")
    checkpoint = read_checkpoint(path)
    if checkpoint.get("software_smoke_only"):
        raise ValueError("Software-smoke checkpoints cannot be used for pilot inference")
    config = dict(checkpoint["model_config"])
    fg_type = checkpoint["training_args"].get("fg_type")
    validate_fg_type(fg_type)
    if config.get("in_ch") not in (1, 12):
        raise ValueError("Unsupported checkpoint input channel count; expected 1 or 12")
    if any(type(config.get(k)) is not bool for k in ("band_reweight", "freeze_encoder", "pretrained")):
        raise ValueError("Invalid boolean model configuration")
    if not isinstance(config.get("band_drop_p"), (int, float)) or not 0 <= config["band_drop_p"] < 1:
        raise ValueError("Invalid checkpoint band_drop_p")
    try:
        val_keys = list(checkpoint["splits"]["val"]["keys"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Checkpoint lacks validation-page metadata") from exc
    if not val_keys or len(set(val_keys)) != len(val_keys):
        raise ValueError("Checkpoint validation keys must be nonempty and unique")
    metadata = dict(model_config=config, fg_type=fg_type,
                    completed_epoch=checkpoint["completed_epoch"],
                    format_version=checkpoint["format_version"], validation_keys=val_keys)
    metadata.update(precision=checkpoint.get("precision"), splits=checkpoint["splits"],
                    training_args=checkpoint["training_args"], initialization=checkpoint.get("initialization"),
                    source_identity=checkpoint.get("source_identity"), versions=checkpoint["versions"])
    metadata["evaluation_source"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                     for name in ("infer_msbin_cpu.py", "msbin_labels.py", "pixel_metrics.py",
                                                  "msbin_dataset.py", "unet_small.py", "training_support.py", "experiment_support.py")}
    state = checkpoint["model"]
    del checkpoint  # Optimizer, RNG, history and scaler must not survive construction.
    model = build_model(config, restoring=True, factory=factory)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    metadata["checkpoint"] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())
    return model, metadata


def requested_pages(image_dir, keys_path, max_pages=None):
    keys = read_page_keys(keys_path)  # Preserves file order, rejects duplicate/empty lists.
    pages, _ = group_books(Path(image_dir))
    unknown = set(keys) - set(pages)
    if unknown:
        raise ValueError(f"Unknown requested page keys: {sorted(unknown)}")
    for key in keys:
        missing = set(range(12)) - set(pages[key])
        extra = set(pages[key]) - set(range(12))
        if missing or extra:
            raise ValueError(f"Page {key}: missing bands {sorted(missing)}, unexpected bands {sorted(extra)}")
    if max_pages is not None and (type(max_pages) is not int or max_pages <= 0):
        raise ValueError("max_pages must be a positive integer")
    return pages, keys, keys if max_pages is None else keys[:max_pages]


def read_page(bands, in_ch, key):
    """Store one uint8 page on CPU; sequential decode avoids a float32 page stack."""
    first = _read_png_gray(bands[0])
    h, w = first.shape
    page = np.empty((in_ch, h, w), dtype=np.uint8)
    page[0] = first
    del first
    for band in range(1, 12):
        image = _read_png_gray(bands[band])
        if image.shape != (h, w):
            raise ValueError(f"Page {key}: mismatched band {band} dimensions {image.shape}, expected {(h, w)}")
        if in_ch == 12:
            page[band] = image
    return page


def validate_tiling(tile, stride, threshold):
    patch_starts(1, tile, stride)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0,1]")


@torch.inference_mode()
def tiled_probabilities(model, page, tile=256, stride=256, device="cpu"):
    """One tile/forward; uniform mean of sigmoid probabilities on CPU.

    Memory: one uint8 CxHxW page, float32 HxW sum, uint32 HxW coverage,
    and tile-sized model activations. No page-sized tensor is sent to the model.
    """
    validate_tiling(tile, stride, 0.5)
    if page.ndim != 3 or page.dtype != np.uint8 or min(page.shape) <= 0:
        raise ValueError("Expected nonempty uint8 (C,H,W) page")
    _, h, w = page.shape
    total = np.zeros((h, w), np.float32)
    coverage = np.zeros((h, w), np.uint32)
    model.eval()
    for y in patch_starts(h, tile, stride):
        for x in patch_starts(w, tile, stride):
            crop = page[:, y:y + tile, x:x + tile]
            ch, cw = crop.shape[1:]
            padded = np.pad(crop, ((0, 0), (0, tile - ch), (0, tile - cw)), mode="edge")
            tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).unsqueeze(0).to(device)
            logits = model(tensor)
            if logits.shape != (1, 1, tile, tile):
                raise ValueError(f"Model output shape {tuple(logits.shape)}; expected {(1, 1, tile, tile)}")
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite inference logits")
            probability = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
            total[y:y + ch, x:x + cw] += probability[:ch, :cw]
            coverage[y:y + ch, x:x + cw] += 1
    if np.any(coverage == 0):
        raise ValueError("Tiling left original pixels with zero coverage")
    total /= coverage
    if not np.isfinite(total).all() or np.any((total < 0) | (total > 1)):
        raise ValueError("Invalid stitched probabilities")
    return total, coverage


def mask_images(probabilities, threshold):
    validate_tiling(1, 1, threshold)
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Expected finite HxW probabilities in [0,1]")
    machine = (probabilities > threshold).astype(np.uint8) * 255
    return machine, 255 - machine


def comparison_image(band0, target, valid, reading_preview):
    """Left: band 0; middle: reading GT, magenta UR; right: reading prediction."""
    if not (band0.shape == target.shape == valid.shape == reading_preview.shape):
        raise ValueError("Comparison panel dimensions differ")
    gt = cv2.cvtColor(((1 - target) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gt[~valid] = (255, 0, 255)
    panels = [cv2.cvtColor(band0, cv2.COLOR_GRAY2BGR), gt,
              cv2.cvtColor(reading_preview, cv2.COLOR_GRAY2BGR)]
    return np.concatenate(panels, axis=1)


def write_png(path, image):
    """Exclusive reservation prevents replacement; failed writes remove our file."""
    path = Path(path)
    with path.open("xb"):
        pass
    try:
        if not cv2.imwrite(str(path), image):
            raise OSError(f"cv2.imwrite failed: {path}")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog="Removed --white_only/--band_reweight: architecture is taken from checkpoint metadata.")
    p.add_argument("--msbin_root", type=Path, required=True)
    p.add_argument("--split", choices=["train"], default="train", help="Validation preview only; no test-set reads")
    p.add_argument("--keys_txt", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True, help="Explicit trusted local versioned checkpoint")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max_pages", type=int)
    p.add_argument("--compare_gt", action="store_true")
    return p.parse_args(argv)


def run(args):
    validate_tiling(args.tile, args.stride, args.thr)
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    root, outdir = args.msbin_root.resolve(), args.outdir.resolve()
    if outdir == root or root in outdir.parents:
        raise ValueError("Output directory must be outside the dataset")
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        raise FileExistsError("Output directory must be new or empty; existing outputs will not be replaced")
    pages, requested, selected = requested_pages(root / args.split / "images", args.keys_txt, args.max_pages)
    names = [f"{key}{suffix}.png" for key in selected
             for suffix in (["_mask", "_reading_preview", "_comparison"] if args.compare_gt else ["_mask", "_reading_preview"])]
    if len(set(names)) != len(names):
        raise ValueError("Output filenames collide")
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, metadata = load_inference_model(args.ckpt, device)
    nonval = set(requested) - set(metadata["validation_keys"])
    if nonval:
        raise ValueError(f"Preview keys are not checkpoint validation pages: {sorted(nonval)}")
    epoch = metadata["completed_epoch"]
    description = ("Inference from a one-epoch pilot — NOT FINAL RESEARCH RESULTS" if epoch == 1
                   else f"Limited checkpoint preview (completed epoch {epoch}) — NOT FINAL RESEARCH RESULTS")
    print(description, flush=True)
    manifest = dict(status="complete", description=description, **metadata, keys=selected,
                    requested_keys=requested, keys_file=str(args.keys_txt.resolve()),
                    source_split=args.split, device=device, threads=args.threads,
                    keys_file_sha256=hashlib.sha256(args.keys_txt.read_bytes()).hexdigest(),
                    evaluation_versions=versions(),
                    evaluation_backend=dict(tf32=False, hardware=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                                            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                            cudnn_benchmark=torch.backends.cudnn.benchmark,
                                            cudnn_deterministic=torch.backends.cudnn.deterministic),
                    threshold=args.thr, threshold_rule="probability > threshold",
                    tiling=dict(tile=args.tile, stride=args.stride, batch_size=1, overlap="mean of sigmoid probabilities", padding="edge replication"),
                    conventions=dict(machine="foreground white=255, background black=0",
                                     reading_preview="foreground black=0, background white=255",
                                     comparison="left=input band 0; middle=reading ground truth with magenta UR; right=reading prediction"),
                    score_scope="Full-page scores on selected validation pages only; each valid pixel counted once. Not test scores or equivalent to overlapping patch validation.",
                    pages=[])
    aggregate = PixelCounts()
    outdir.mkdir(parents=True, exist_ok=True)
    for key in selected:
        page = read_page(pages[key], metadata["model_config"]["in_ch"], key)
        probability, coverage = tiled_probabilities(model, page, args.tile, args.stride, device)
        machine, reading = mask_images(probability, args.thr)
        record = dict(key=key, height=page.shape[1], width=page.shape[2],
                      coverage_min=int(coverage.min()), coverage_max=int(coverage.max()),
                      mask=f"{key}_mask.png", reading_preview=f"{key}_reading_preview.png")
        comparison = None
        if args.compare_gt:
            target, valid = decode_label(_read_png_bgr(root / args.split / "labels" / f"{key}.png"), metadata["fg_type"])
            if target.shape != probability.shape:
                raise ValueError(f"Page {key}: label and input dimensions differ")
            record["scores"] = PixelCounts().update(probability, target, valid, args.thr).scores()
            record["valid_pixels"], record["uncertain_pixels"] = int(valid.sum()), int((~valid).sum())
            aggregate.update(probability, target, valid, args.thr)
            comparison = comparison_image(page[0], target, valid, reading)
            record["comparison"] = f"{key}_comparison.png"
        write_png(outdir / record["mask"], machine)
        write_png(outdir / record["reading_preview"], reading)
        if comparison is not None:
            write_png(outdir / record["comparison"], comparison)
        manifest["pages"].append(record)
        print(f"{key}: {record['width']}x{record['height']} saved", flush=True)
        del page, probability, coverage, machine, reading, comparison
    if args.compare_gt:
        manifest["aggregate_scores"] = aggregate.scores()
        print("Selected validation full-page scores:", manifest["aggregate_scores"])
    with (outdir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main(argv=None):
    args = get_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        raise SystemExit(f"Inference failed: {exc}") from exc


if __name__ == "__main__":
    main()
