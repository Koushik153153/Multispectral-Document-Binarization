"""Numerical updates, reproducible initialization, profiling and durable run logs."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch


def source_identity():
    root = Path(__file__).resolve().parent
    names = ("train_unet_msbin_cpu.py", "training_support.py", "experiment_support.py",
             "msbin_dataset.py", "msbin_labels.py", "pixel_metrics.py", "unet_small.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def tensor_identity(state):
    hashes = {}
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
        hashes[name] = digest.hexdigest()
    return {"sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "tensors": hashes}


def shared_model(config, seed, builder):
    """CPU construction with independent initialization streams; no training RNG consumed."""
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        common_config = dict(config, band_reweight=False, band_drop_p=0.0)
        base = builder(common_config)
        shared = {k: v.detach().clone() for k, v in base.state_dict().items()}
        del base
        # The extra module has its own recorded seed; all common tensors are replaced.
        attention_seed = seed + 1000003
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(attention_seed).get_state())
        model = builder(config, restoring=True)
        state = model.state_dict()
        if set(shared) != {k for k in state if not k.startswith("band_reweight.")}:
            raise ValueError("Shared initialization keys differ across model variants")
        for name, value in shared.items():
            if state[name].shape != value.shape:
                raise ValueError(f"Shared tensor shape mismatch: {name}")
            state[name] = value
        model.load_state_dict(state, strict=True)
        identity = tensor_identity(shared)
        identity.update(seed=seed, additional_module_seed=attention_seed,
                        scheme="common tensors copied from seeded baseline; additional module separate CPU RNG")
    return model, identity


def optimizer_update(model, optimizer, scaler, loss, precision, grad_clip):
    """Return True on update, False on GradScaler-detected overflow; public APIs only."""
    if precision not in ("fp32", "amp") or scaler.is_enabled() != (precision == "amp"):
        raise ValueError("Precision and GradScaler enablement disagree")
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite valid loss; optimizer update refused")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise ValueError("No gradients: optimizer cannot update")
    finite = bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all())
    if not finite:
        if precision == "fp32":
            raise FloatingPointError("FP32 nonfinite gradients; optimizer update refused")
        before = scaler.get_scale()
        # unscale_ already recorded overflow: step skips parameter/optimizer mutation.
        scaler.step(optimizer)
        scaler.update()
        if not math.isfinite(scaler.get_scale()) or not 0 < scaler.get_scale() < before:
            raise FloatingPointError("AMP overflow did not reduce scale to a finite positive value")
        return False
    torch.nn.utils.clip_grad_norm_(parameters, grad_clip or float("inf"), error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return True


class Timer:
    def __init__(self, device, synchronize=False):
        self.device = device
        self.synchronize = bool(synchronize and str(device).startswith("cuda"))
        self.seconds = {}

    @contextmanager
    def stage(self, name):
        if self.synchronize and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.synchronize and str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started


class LimitedLoader:
    def __init__(self, loader, limit):
        self.loader, self.limit = loader, limit
        self.observed = {"batches": 0, "samples_loaded": 0}

    def __len__(self):
        return min(len(self.loader), self.limit)

    def __iter__(self):
        self.observed = {"batches": 0, "samples_loaded": 0}
        iterator = iter(self.loader)
        for _ in range(len(self)):
            batch = next(iterator)
            self.observed["batches"] += 1
            self.observed["samples_loaded"] += len(batch[0])
            yield batch


def progress(stage, epoch, batch, total, started, interval=25):
    if batch == 1 or batch == total or batch % interval == 0:
        print(f"stage={stage} epoch={epoch} batch={batch}/{total} elapsed={time.perf_counter()-started:.1f}s", flush=True)


@contextmanager
def text_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        class Tee:
            def __init__(self, console):
                self.console = console
            def write(self, text):
                try:
                    stream.write(text)
                    stream.flush()
                except OSError as exc:
                    raise OSError(f"Text log write failed for {path}: {exc}") from exc
                return self.console.write(text)
            def flush(self):
                stream.flush()
                self.console.flush()
        with redirect_stdout(Tee(sys.stdout)), redirect_stderr(Tee(sys.stderr)):
            try:
                yield
            except Exception as exc:
                print(f"RUN FAILED: {type(exc).__name__}: {exc}", flush=True)
                raise
