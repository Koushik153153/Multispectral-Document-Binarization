"""Masked objectives, epoch checks and versioned checkpoints for MSBin training."""
import csv
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from msbin_dataset import group_books, read_page_keys
from pixel_metrics import PixelCounts
from experiment_support import source_identity, progress

ARCHITECTURE = "resnet50_unet_v1"
CHECKPOINT_VERSION = 2


def _masked_inputs(logits, target, valid):
    if logits.shape != target.shape or valid.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Loss inputs must have identical (N,1,H,W) shapes")
    if valid.dtype != torch.bool or logits.device != target.device or valid.device != target.device:
        raise ValueError("Validity must be boolean; loss inputs must share a device")
    if not torch.isfinite(logits[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("Nonfinite valid loss input")
    if not ((target[valid] == 0) | (target[valid] == 1)).all():
        raise ValueError("Valid targets must be binary")
    if not valid.any():
        return None
    # Replace before nonlinear operations: even NaN/Inf at invalid positions is ignored.
    return torch.where(valid, logits.float(), 0), torch.where(valid, target.float(), 0)


def _finite(loss):
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite masked loss")
    return loss


def masked_weighted_bce(logits, target, valid, pos_weight=6.0):
    """Mean over valid pixels across the batch; None means skip all-invalid batch."""
    if not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("pos_weight must be finite and positive")
    inputs = _masked_inputs(logits, target, valid)
    if inputs is None:
        return None
    z, y = inputs
    loss = F.binary_cross_entropy_with_logits(z, y, reduction="none", pos_weight=z.new_tensor(pos_weight))
    return _finite(loss[valid].mean())


def masked_dice_loss(logits, target, valid, smooth=1e-6):
    """Mean of per-sample Dice losses, excluding samples with zero valid pixels.

    Add smooth to numerator AND denominator for stable empty-foreground cases.
    """
    if not math.isfinite(smooth) or smooth <= 0:
        raise ValueError("smooth must be finite and positive")
    inputs = _masked_inputs(logits, target, valid)
    if inputs is None:
        return None
    z, y = inputs
    p = torch.sigmoid(z) * valid
    y = y * valid
    axes = (1, 2, 3)
    active = valid.sum(dim=axes) > 0
    dice = (2 * (p * y).sum(dim=axes) + smooth) / (p.sum(dim=axes) + y.sum(dim=axes) + smooth)
    return _finite((1 - dice[active]).mean())


def masked_loss(logits, target, valid, pos_weight=6.0, bce_w=0.9, dice_w=0.1):
    if not all(math.isfinite(w) and w >= 0 for w in (bce_w, dice_w)) or bce_w + dice_w <= 0:
        raise ValueError("Loss weights must be finite, nonnegative, and not both zero")
    bce = masked_weighted_bce(logits, target, valid, pos_weight)
    if bce is None:
        return None
    return _finite(bce_w * bce + dice_w * masked_dice_loss(logits, target, valid))


@torch.no_grad()
def validate_patches(model, loader, device, threshold=0.5, epoch=0, log_every=25):
    model.eval()
    counts, valid_pixels = PixelCounts(), 0
    started = time.perf_counter()
    for batch, (x, y, valid) in enumerate(loader, 1):
        progress("validation", epoch, batch, len(loader), started, log_every)
        if not valid.any():
            continue
        # Remove all-invalid samples before forward (same policy as training).
        active = valid.flatten(1).any(1)
        x, y, valid = x[active].to(device), y[active].to(device), valid[active].to(device)
        logits = model(x)
        if logits.shape != y.shape:
            raise ValueError("Validation output and target shapes differ")
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite validation output")
        counts.update(torch.sigmoid(logits), y, valid, threshold)
        valid_pixels += int(valid.sum())
    if not valid_pixels:
        raise ValueError("Validation has no valid pixels")
    return counts.scores()


class BandWeightMean:
    def __init__(self):
        self.total, self.samples = None, 0

    def update(self, weights):
        if weights is None:
            return
        weights = weights.detach().double().cpu()
        if weights.ndim != 2 or not torch.isfinite(weights).all():
            raise ValueError("Band weights must be finite (N,C)")
        total = weights.sum(0)
        self.total = total if self.total is None else self.total + total
        self.samples += len(weights)

    def mean(self):
        return None if not self.samples else (self.total / self.samples).tolist()


def split_metadata(root, train_path, val_path):
    if not train_path or not val_path:
        raise ValueError("Both --train_keys and --val_keys are required")
    pages, complete = group_books(Path(root) / "train/images")
    result = {}
    for name, path in (("train", train_path), ("val", val_path)):
        keys = sorted(read_page_keys(path))
        unknown = set(keys) - set(pages)
        incomplete = set(keys) - set(complete)
        if unknown or incomplete:
            raise ValueError(f"{name}: unknown/incomplete original training pages: {sorted(unknown | incomplete)}")
        for key in keys:
            if not (Path(root) / "train/labels" / f"{key}.png").is_file():
                raise ValueError(f"{name}: missing label for page {key}")
        result[name] = {"keys": keys, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    overlap = set(result["train"]["keys"]) & set(result["val"]["keys"])
    if overlap:
        raise ValueError(f"Overlapping train/validation keys: {sorted(overlap)}")
    return result


def model_config(args):
    return {"architecture": ARCHITECTURE, "in_ch": 1 if args.white_only else 12,
            "band_reweight": args.band_reweight, "band_drop_p": args.band_drop_p,
            "freeze_encoder": args.freeze_encoder, "pretrained": args.pretrained}


def build_model(config, restoring=False, factory=None):
    required = {"architecture", "in_ch", "band_reweight", "band_drop_p", "freeze_encoder", "pretrained"}
    if set(config) != required or config["architecture"] != ARCHITECTURE:
        raise ValueError("Unsupported or incomplete model configuration")
    if factory is None:
        from unet_small import UNetSmall
        factory = UNetSmall
    kwargs = {k: v for k, v in config.items() if k != "architecture"}
    if restoring:
        kwargs["pretrained"] = False  # Initialization metadata retained; never download on restore.
    return factory(**kwargs)


def versions():
    return {"python": sys.version, **{name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "opencv-python")}}


def experiment_args(args):
    # Only operational paths, resume flag and final epoch limit may change.
    ignored = {"epochs", "resume", "outdir", "train_keys", "val_keys"}
    return {"precision": getattr(args, "precision", "fp32"),
            **{k: v for k, v in vars(args).items() if k not in ignored}}


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise OSError(f"Atomic write failed for {path}; check free space and mounted filesystem: {exc}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, checkpoint):
    try:
        atomic_write(path, lambda stream: torch.save(checkpoint, stream))
    except (OSError, RuntimeError) as exc:
        raise OSError(f"Checkpoint write failed for {path}; check disk space and mounted filesystem: {exc}") from exc


def make_checkpoint(model, optimizer, scaler, generator, epoch, best, config, args, splits, history):
    return {"format_version": CHECKPOINT_VERSION, "completed_epoch": epoch, "next_epoch": epoch + 1,
            "best_val_f1": best, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if str(args.device).startswith("cuda") else None},
            "shuffle_rng": generator.get_state(), "model_config": config,
            "training_args": vars(args).copy(), "experiment_args": experiment_args(args),
            "versions": versions(), "splits": splits, "history": history,
            "precision": getattr(args, "precision", "fp32"), "source_identity": source_identity()}


def read_checkpoint(path):
    # Only load trusted locally produced checkpoints (pickle metadata is intentional).
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = {"completed_epoch", "next_epoch", "best_val_f1", "model", "optimizer", "scaler", "rng",
                "shuffle_rng", "model_config", "training_args", "experiment_args", "versions", "splits", "history"}
    if not isinstance(ckpt, dict) or ckpt.get("format_version") not in (1, CHECKPOINT_VERSION) or not required <= ckpt.keys():
        raise ValueError("Legacy/incomplete checkpoint unsupported: explicit versioned metadata required; configuration will not be guessed")
    if not isinstance(ckpt["rng"], dict) or not {"python", "numpy", "torch", "cuda"} <= ckpt["rng"].keys():
        raise ValueError("Incomplete checkpoint RNG metadata")
    if not isinstance(ckpt["training_args"], dict) or "epochs" not in ckpt["training_args"]:
        raise ValueError("Incomplete checkpoint training metadata")
    if not isinstance(ckpt["completed_epoch"], int) or ckpt["completed_epoch"] < 1:
        raise ValueError("Invalid completed epoch")
    if ckpt["next_epoch"] != ckpt["completed_epoch"] + 1:
        raise ValueError("Invalid checkpoint epoch semantics")
    if not isinstance(ckpt["history"], list) or any(not isinstance(r, dict) or "epoch" not in r for r in ckpt["history"]):
        raise ValueError("Invalid checkpoint history metadata")
    if [r["epoch"] for r in ckpt["history"]] != list(range(1, ckpt["next_epoch"])):
        raise ValueError("Checkpoint epoch history is incomplete or duplicated")
    if ckpt["format_version"] == 2 and (ckpt.get("precision") not in ("fp32", "amp") or not ckpt.get("source_identity")):
        raise ValueError("Version 2 checkpoint missing precision/source metadata")
    return ckpt


def validate_resume(ckpt, config, args, splits):
    if ckpt.get("format_version") != 2 or ckpt.get("precision") not in ("fp32", "amp"):
        raise ValueError("Historical training resume rejected: explicit precision metadata required; version 1 remains inference-only")
    if ckpt["precision"] != getattr(args, "precision", "fp32"):
        raise ValueError("Incompatible precision for resume")
    if ckpt.get("source_identity") != source_identity():
        raise ValueError("Training source changed; exact resume rejected")
    if ckpt["model_config"] != config:
        raise ValueError("Incompatible model configuration (channels/attention/dropout/freeze/initialization)")
    if ckpt["splits"] != splits:
        raise ValueError("Changed split content or keys; resume rejected")
    if ckpt["experiment_args"] != experiment_args(args):
        raise ValueError("Changed training configuration; only total epochs and operational paths may change")
    if ckpt["versions"] != versions():
        raise ValueError("Dependency versions changed; exact resume rejected")
    if args.epochs < ckpt["training_args"]["epochs"] or args.epochs < ckpt["completed_epoch"]:
        raise ValueError("Resume may extend, but not reduce, total epochs")


def restore_checkpoint(ckpt, model, optimizer, scaler, generator):
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None:
        if ckpt["scaler"] is None:
            raise ValueError("Missing scaler state")
        scaler.load_state_dict(ckpt["scaler"])
    random.setstate(ckpt["rng"]["python"])
    np.random.set_state(ckpt["rng"]["numpy"])
    torch.set_rng_state(ckpt["rng"]["torch"])
    if ckpt["rng"]["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint RNG cannot be restored on this device")
        torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    generator.set_state(ckpt["shuffle_rng"])
    return ckpt["next_epoch"], ckpt["best_val_f1"]


def check_output_dir(outdir, resume):
    outdir = Path(outdir)
    if resume:
        if not (outdir / "last.pt").is_file():
            raise ValueError("Resume requires an existing last.pt; refusing fresh fallback")
    elif outdir.exists() and any(outdir.iterdir()):
        raise ValueError("Fresh run requires a new or empty output directory; use explicit --resume")


def write_logs(outdir, history):
    import io
    fields = ["epoch", "train_loss", "patch_val_precision", "patch_val_recall", "patch_val_f1", "elapsed_seconds", "best_val_f1",
              "invalid_label_batches", "overflow_skipped_updates", "optimizer_updates"]
    def write_csv(path, names, rows):
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(path, lambda stream: stream.write(text.getvalue().encode("utf-8")))
    write_csv(Path(outdir) / "epochs.csv", fields, [{k: row.get(k, "") for k in fields} for row in history])
    attention = [r for r in history if r["band_weights"] is not None]
    if attention:
        bands = [f"band_{i}" for i in range(len(attention[0]["band_weights"]))]
        write_csv(Path(outdir) / "band_weights.csv", ["epoch"] + bands,
                  [{"epoch": r["epoch"], **dict(zip(bands, r["band_weights"]))} for r in attention])


def commit_epoch(outdir, checkpoint, improved):
    # last.pt is the commit point. History regenerates logs after interrupted writes.
    save_checkpoint(Path(outdir) / "last.pt", checkpoint)
    if improved:
        save_checkpoint(Path(outdir) / "best.pt", checkpoint)
    write_logs(outdir, checkpoint["history"])


def write_json(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    atomic_write(path, lambda stream: stream.write(raw))
