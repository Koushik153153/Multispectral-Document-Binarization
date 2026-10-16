"""Mask-aware patch training. No test-set reads or evaluation."""
import argparse
from pathlib import Path
import math
import random
import time
import json
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from msbin_dataset import MSBinDibcoPatchDataset
from experiment_support import (optimizer_update, Timer, LimitedLoader, progress, text_log,
                                shared_model, source_identity, tensor_identity)
from training_support import (
    BandWeightMean, build_model, check_output_dir, commit_epoch, make_checkpoint,
    masked_loss, model_config, read_checkpoint, restore_checkpoint,
    save_checkpoint, split_metadata, validate_patches, validate_resume, write_logs, write_json, versions,
)


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--msbin_root", required=True)
    p.add_argument("--train_keys", required=True)
    p.add_argument("--val_keys", required=True)
    p.add_argument("--fg_type", type=int, choices=(1, 2), default=1)
    for name, default in (("patch", 256), ("stride", 256), ("epochs", 30), ("batch", 4),
                          ("num_workers", 0), ("seed", 42)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--max_patches_per_page", type=int)
    for name, default in (("min_fg_frac", 0.002), ("lr", 3e-4), ("band_drop_p", 0.0),
                          ("pos_weight", 6.0), ("bce_w", 0.9), ("dice_w", 0.1),
                          ("grad_clip", 1.0), ("val_thr", 0.5)):
        p.add_argument("--" + name, type=float, default=default)
    for name in ("white_only", "band_reweight", "freeze_encoder", "resume"):
        p.add_argument("--" + name, action="store_true")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False,
                   help="Fresh-run ImageNet initialization; default offline. Resume always constructs offline.")
    p.add_argument("--outdir", default="runs_cpu/unet")
    p.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    p.add_argument("--cache_mb", type=float, default=0, help="Decoded-page LRU byte budget per dataset per worker")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--max_overflow_streak", type=int, default=100)
    p.add_argument("--profile_batches", type=int, default=0, help="Software-only bounded run; synchronized timing, no research checkpoint")
    p.add_argument("--experiment_name", default="")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, help="PyTorch CPU intra-op threads; default records the runtime setting")
    return p.parse_args(argv)


def validate_args(args):
    if args.threads is not None and args.threads < 1:
        raise ValueError("threads must be positive")
    if args.log_every < 1 or args.max_overflow_streak < 1 or args.profile_batches < 0:
        raise ValueError("Invalid logging/overflow/profile limits")
    if not math.isfinite(args.cache_mb) or args.cache_mb < 0:
        raise ValueError("cache_mb must be finite and nonnegative")
    if args.profile_batches and (args.resume or args.epochs != 1):
        raise ValueError("Software profiling requires a fresh one-epoch bounded run")
    if args.epochs < 1 or args.batch < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch must be positive; num_workers nonnegative")
    for key in ("lr", "pos_weight"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("bce_w", "dice_w", "grad_clip"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if args.bce_w + args.dice_w <= 0:
        raise ValueError("At least one loss weight must be positive")
    if not 0 <= args.band_drop_p < 1 or not 0 <= args.val_thr <= 1:
        raise ValueError("band_drop_p must be in [0,1); val_thr in [0,1]")


def train_epoch(model, loader, optimizer, scaler, device, args, stats=None, timer=None, epoch=0):
    model.train()
    running, samples, skipped = 0.0, 0, 0
    stats = stats if stats is not None else {}
    stats.update(overflow_skipped_updates=0, optimizer_updates=0, invalid_label_batches=0)
    timer = timer or Timer(device)
    streak = 0
    started = time.perf_counter()
    weights = BandWeightMean()
    with timer.stage("data_wait"):
        iterator = iter(loader)
    batch = 0
    while True:
        with timer.stage("data_wait"):
            try:
                x, y, valid = next(iterator)
            except StopIteration:
                break
        batch += 1
        progress("train", epoch, batch, len(loader), started, getattr(args, "log_every", 25))
        active = valid.flatten(1).any(1)
        if not active.any():
            skipped += 1
            stats["invalid_label_batches"] += 1
            print("Skipping all-invalid training batch", flush=True)
            continue
        precision = getattr(args, "precision", "fp32")
        with timer.stage("transfer"):
            x, y, valid = x[active].to(device), y[active].to(device), valid[active].to(device)
        with timer.stage("forward_backward_update"):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(torch.device(device).type, enabled=(precision == "amp")):
                logits = model(x)
            # Explicitly outside autocast: sigmoid/BCE/Dice reductions are float32.
            with torch.amp.autocast(torch.device(device).type, enabled=False):
                loss = masked_loss(logits.float(), y, valid, args.pos_weight, args.bce_w, args.dice_w)
            if loss is None:
                raise RuntimeError("Unexpected all-invalid batch after filtering")
            updated = optimizer_update(model, optimizer, scaler, loss, precision, args.grad_clip)
        stats["optimizer_updates" if updated else "overflow_skipped_updates"] += 1
        streak = 0 if updated else streak + 1
        if not updated:
            print(f"AMP overflow: skipped update batch={batch} scale={scaler.get_scale()}", flush=True)
        if streak >= getattr(args, "max_overflow_streak", 100):
            raise FloatingPointError("Persistent AMP overflow: too many consecutive skipped updates")
        running += loss.item() * len(x)
        samples += len(x)
        weights.update(getattr(model, "last_band_weights", None))
    if not samples:
        raise ValueError("Training epoch has no valid samples")
    if not stats["optimizer_updates"]:
        raise FloatingPointError("Training epoch made no optimizer updates")
    # Descriptive sample-weighted mean of batch objectives, not a full-page loss.
    return running / samples, weights.mean(), skipped


def configure_runtime(args):
    """Resolve device/threads before computation; also used by notebook preflight."""
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu" and args.precision == "amp":
        raise ValueError("CPU requires precision fp32; AMP requires CUDA")
    if args.device != "cuda" and (args.require_cuda or args.precision == "amp"):
        raise ValueError("CUDA is required for this run/AMP; refusing CPU fallback")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable; refusing CPU fallback")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    args.threads = torch.get_num_threads()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.backend = dict(tf32=False, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                        cudnn_benchmark=torch.backends.cudnn.benchmark,
                        cudnn_deterministic=torch.backends.cudnn.deterministic)
    args.hardware = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"


def main(argv=None):
    args = get_args(argv)
    validate_args(args)
    args.msbin_root = str(Path(args.msbin_root).resolve())
    configure_runtime(args)
    splits = split_metadata(args.msbin_root, args.train_keys, args.val_keys)
    config = model_config(args)
    outdir = Path(args.outdir)
    check_output_dir(outdir, args.resume)
    checkpoint = read_checkpoint(outdir / "last.pt") if args.resume else None
    if checkpoint is not None:
        if checkpoint.get("software_smoke_only"):
            raise ValueError("Software smoke checkpoints cannot be resumed as research training")
        validate_resume(checkpoint, config, args, splits)
    with text_log(outdir / "training.log"):
        return run_training(args, splits, config, outdir, checkpoint)


def run_training(args, splits, config, outdir, checkpoint):
    print("RESUME" if args.resume else "FRESH RUN", json.dumps(vars(args), sort_keys=True), flush=True)
    print(f"device={args.device} precision={args.precision}; free_output_bytes={shutil.disk_usage(outdir).free}", flush=True)
    timer = Timer(args.device, synchronize=bool(args.profile_batches))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    common = dict(msbin_root=args.msbin_root, split="train", fg_type=args.fg_type,
                  patch_size=args.patch, stride=args.stride, use_white_only=args.white_only,
                  return_valid_mask=True, cache_mb=args.cache_mb)
    with timer.stage("dataset_indexing"):
        train_ds = MSBinDibcoPatchDataset(**common, keys_txt=args.train_keys,
                                        min_fg_frac=args.min_fg_frac, max_patches_per_page=args.max_patches_per_page)
        val_ds = MSBinDibcoPatchDataset(**common, keys_txt=args.val_keys, min_fg_frac=0, max_patches_per_page=None)
    print(f"train_pages={len(getattr(train_ds, 'keys', []))} train_patches={len(train_ds)} "
          f"val_pages={len(getattr(val_ds, 'keys', []))} val_patches={len(val_ds)}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=generator,
                              num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    if args.profile_batches:
        train_loader = LimitedLoader(train_loader, args.profile_batches)
        val_loader = LimitedLoader(val_loader, args.profile_batches)
    with timer.stage("model_initialization"):
        if checkpoint is None and args.experiment_name:
            model, initialization = shared_model(config, args.seed, build_model)
        else:
            model = build_model(config, restoring=checkpoint is not None)
            initialization = checkpoint.get("initialization") if checkpoint is not None else tensor_identity(model.state_dict())
        model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(args.device, enabled=(args.precision == "amp"))
    run_manifest = dict(format_version=2, arguments=vars(args).copy(), model_config=config,
                        splits=splits, initialization=initialization, source_identity=source_identity(),
                        versions=versions(), software_smoke_only=bool(args.profile_batches),
                        counts=dict(train_patches=len(train_ds), val_patches=len(val_ds),
                                    train_pages=len(getattr(train_ds, "keys", [])), val_pages=len(getattr(val_ds, "keys", []))))
    write_json(outdir / "run.json", run_manifest)
    start, best, history = 1, -1.0, []
    if checkpoint is not None:
        start, best = restore_checkpoint(checkpoint, model, optimizer, scaler, generator)
        history = checkpoint["history"]
        # Repair a best/log write interrupted after the last.pt commit.
        if history and history[-1]["improved"]:
            save_checkpoint(outdir / "best.pt", checkpoint)
        write_logs(outdir, history)
        del checkpoint
    for epoch in range(start, args.epochs + 1):
        started = time.perf_counter()
        stats = {}
        train_loss, band_weights, skipped = train_epoch(model, train_loader, optimizer, scaler, args.device, args, stats, timer, epoch)
        with timer.stage("validation"):
            metrics = validate_patches(model, val_loader, args.device, args.val_thr, epoch, args.log_every)
        improved = metrics["f1"] > best
        best = max(best, metrics["f1"])
        history.append(dict(epoch=epoch, train_loss=train_loss, patch_val_precision=metrics["precision"],
                            patch_val_recall=metrics["recall"], patch_val_f1=metrics["f1"],
                            elapsed_seconds=time.perf_counter() - started, best_val_f1=best,
                            band_weights=band_weights, improved=improved, skipped_batches=skipped, **stats))
        checkpoint = make_checkpoint(model, optimizer, scaler, generator, epoch, best, config, args, splits, history)
        checkpoint["initialization"] = initialization
        checkpoint["software_smoke_only"] = bool(args.profile_batches)
        with timer.stage("checkpoint_writing"):
            commit_epoch(outdir, checkpoint, improved)
        del checkpoint
        print(f"Epoch {epoch}: loss={train_loss:.6f}; PATCH validation={metrics}; best_patch_f1={best:.6f}", flush=True)
        timing = dict(seconds=timer.seconds, synchronized=timer.synchronize,
                      device=args.device, threads=args.threads, batch_size=args.batch,
                      profile_batch_limit=args.profile_batches,
                      profile_observed=({"train": train_loader.observed, "validation": val_loader.observed}
                                        if args.profile_batches else None),
                      note="cumulative; validation includes its data wait; ordinary CUDA timings are approximate",
                      cache_stats=({"hits": getattr(train_ds, "cache_hits", 0), "misses": getattr(train_ds, "cache_misses", 0),
                                    "resident_bytes": getattr(train_ds, "cache_bytes", 0)} if args.num_workers == 0 else "worker-local; not aggregated"))
        write_json(outdir / "timing.json", timing)
        print("TIMING", json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
