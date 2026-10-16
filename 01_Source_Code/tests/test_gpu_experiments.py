"""CPU-only synthetic tests; no weights, real dataset, GPU or research results."""
import ast
import copy
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment_support as workflow
import training_support as support
from train_unet_msbin_cpu import get_args, train_epoch
from msbin_dataset import MSBinDibcoPatchDataset
from msbin_labels import FG1, UR
from run_experiments import load_spec, training_command, comparison_rows, VARIANTS
from pixel_metrics import PixelCounts
from infer_msbin_cpu import load_inference_model


def test_fp32_default_and_finite_update():
    args = get_args(["--msbin_root", "data", "--train_keys", "train", "--val_keys", "val"])
    assert args.precision == "fp32"
    model = nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    before = model.weight.detach().clone()
    assert workflow.optimizer_update(model, opt, scaler, model(torch.ones(2, 1)).sum(), "fp32", 1)
    assert not torch.equal(before, model.weight)


def test_genuine_scaler_overflow_skips_clip_and_optimizer(monkeypatch):
    model = nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    before = copy.deepcopy(model.state_dict())
    scale = scaler.get_scale()
    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **kw: pytest.fail("Overflow must not be clipped"))
        assert not workflow.optimizer_update(model, opt, scaler, model(torch.ones(2, 1)).sum(), "amp", 1)
    assert not opt.state
    assert scaler.get_scale() < scale
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    hook.remove()
    opt.zero_grad(set_to_none=True)
    assert workflow.optimizer_update(model, opt, scaler, model(torch.ones(2, 1)).sum(), "amp", 1)


def test_fp32_nonfinite_gradient_and_nonfinite_loss_fail():
    model = nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    before = model.weight.detach().clone()
    with pytest.raises(FloatingPointError, match="FP32"):
        workflow.optimizer_update(model, opt, scaler, model(torch.ones(1, 1)).sum(), "fp32", 1)
    torch.testing.assert_close(before, model.weight)
    with pytest.raises(ValueError, match="Nonfinite valid loss"):
        workflow.optimizer_update(model, opt, scaler, torch.tensor(float("nan")), "fp32", 1)


def test_half_precision_loss_reduces_in_float32():
    logits = torch.zeros(1, 1, 256, 256, dtype=torch.float16, requires_grad=True)
    target = torch.ones_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss = support.masked_loss(logits, target, valid)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_skips_counted_separately_and_persistent_failure(monkeypatch):
    import train_unet_msbin_cpu as trainer
    args = get_args(["--msbin_root", "data", "--train_keys", "a", "--val_keys", "b"])
    model = nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    x = torch.ones(1, 1, 2, 2)
    batch = (x, x, torch.ones_like(x, dtype=torch.bool))
    invalid = (x, x, torch.zeros_like(x, dtype=torch.bool))
    decisions = iter([False, True])
    monkeypatch.setattr(trainer, "optimizer_update", lambda *a: next(decisions))
    stats = {}
    train_epoch(model, [invalid, batch, batch], optimizer, scaler, "cpu", args, stats)
    assert stats == dict(invalid_label_batches=1, overflow_skipped_updates=1, optimizer_updates=1)
    monkeypatch.setattr(trainer, "optimizer_update", lambda *a: False)
    args.max_overflow_streak = 2
    with pytest.raises(FloatingPointError, match="Persistent"):
        train_epoch(model, [batch, batch], optimizer, scaler, "cpu", args)
    with pytest.raises(FloatingPointError, match="no optimizer updates"):
        train_epoch(model, [batch], optimizer, scaler, "cpu", args)


def tiny_checkpoint():
    args = get_args(["--msbin_root", "data", "--train_keys", "a", "--val_keys", "b"])
    args.device = "cpu"
    model = nn.Conv2d(12, 1, 1)
    splits = {"train": {"keys": ["A"], "sha256": "a"}, "val": {"keys": ["B"], "sha256": "b"}}
    config = support.model_config(args)
    ckpt = support.make_checkpoint(model, torch.optim.AdamW(model.parameters()),
                                   torch.amp.GradScaler("cpu", enabled=False), torch.Generator(),
                                   1, .5, config, args, splits, [{"epoch": 1}])
    return ckpt, config, args, splits


def test_precision_source_worker_resume_checks_and_historical_inference(tmp_path):
    ckpt, config, args, splits = tiny_checkpoint()
    assert ckpt["format_version"] == 2 and ckpt["precision"] == "fp32"
    support.validate_resume(ckpt, config, args, splits)
    args.precision = "amp"
    with pytest.raises(ValueError, match="precision"):
        support.validate_resume(ckpt, config, args, splits)
    args.precision = "fp32"
    args.num_workers = 1
    with pytest.raises(ValueError, match="configuration"):
        support.validate_resume(ckpt, config, args, splits)
    args.num_workers = 0
    changed = copy.deepcopy(ckpt)
    changed["source_identity"]["unet_small.py"] = "modified"
    with pytest.raises(ValueError, match="source changed"):
        support.validate_resume(changed, config, args, splits)
    old = copy.deepcopy(ckpt)
    old["format_version"] = 1
    old.pop("precision")
    old.pop("source_identity")
    old["training_args"].pop("precision")
    old["experiment_args"].pop("precision")
    with pytest.raises(ValueError, match="Historical"):
        support.validate_resume(old, config, args, splits)
    path = tmp_path / "historical.pt"
    torch.save(old, path)
    _, metadata = load_inference_model(path, factory=lambda **kw: nn.Conv2d(12, 1, 1))
    assert metadata["precision"] is None and metadata["format_version"] == 1


def write_pages(root):
    (root / "train/images").mkdir(parents=True)
    (root / "train/labels").mkdir()
    for key, value in (("A", 32), ("B", 224)):
        for band in range(12):
            assert cv2.imwrite(str(root / f"train/images/{key}_{band}.png"), np.full((2, 3), value + band, np.uint8))
        label = np.zeros((2, 3, 3), np.uint8)
        label[0, 0], label[0, 1] = FG1, UR
        assert cv2.imwrite(str(root / f"train/labels/{key}.png"), label)


def test_bounded_cache_equivalence_eviction_and_independent_patches(tmp_path):
    write_pages(tmp_path)
    kwargs = dict(msbin_root=tmp_path, patch_size=2, stride=2, min_fg_frac=0, return_valid_mask=True)
    uncached = MSBinDibcoPatchDataset(**kwargs)
    cached = MSBinDibcoPatchDataset(**kwargs, cache_mb=150 / 2**20)  # One 102-byte page.
    assert cached.index == uncached.index
    for index in (0, 1, 0, 2, 3, 0):
        for left, right in zip(uncached[index], cached[index]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert cached.cache_bytes <= 150
    assert cached.cache_hits > 0 and cached.cache_misses > 1 and len(cached._cache) == 1
    cached[0][1].fill_(42)
    torch.testing.assert_close(cached[0][1], uncached[0][1])
    tiny = MSBinDibcoPatchDataset(**kwargs, cache_mb=1 / 2**20)
    tiny[0]
    assert tiny.cache_bytes == 0
    loaders = [torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True,
               generator=torch.Generator().manual_seed(42)) for ds in (cached, uncached)]
    for left, right in zip(*loaders):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_shared_initialization_across_variants_and_rng_isolation():
    class Tiny(nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.band_reweight = nn.Linear(2, 2) if attention else nn.Identity()
            self.enc = nn.Linear(2, 2)
            self.decoder = nn.Linear(2, 1)
    def builder(config, restoring=False):
        return Tiny(config["band_reweight"])
    before = torch.get_rng_state().clone()
    identities, states = [], []
    for attention, drop in VARIANTS.values():
        model, identity = workflow.shared_model(dict(band_reweight=attention, band_drop_p=drop), 42, builder)
        identities.append(identity["sha256"])
        states.append(model.state_dict())
    assert len(set(identities)) == 1
    torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
    for state in states[1:]:
        for key, value in states[0].items():
            torch.testing.assert_close(state[key], value, atol=0, rtol=0)
    _, other = workflow.shared_model(dict(band_reweight=False, band_drop_p=0), 43, builder)
    assert other["sha256"] != identities[0]


def test_progress_is_flushed_and_persisted(tmp_path, capsys):
    path = tmp_path / "training.log"
    with workflow.text_log(path):
        workflow.progress("train", 2, 1, 10, time.perf_counter(), 5)
        assert "stage=train epoch=2 batch=1/10" in path.read_text()
    assert "stage=train" in capsys.readouterr().out
    with pytest.raises(RuntimeError):
        with workflow.text_log(path):
            raise RuntimeError("write problem example")
    assert "RUN FAILED" in path.read_text()


def test_timer_synchronizes_only_when_requested(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    for profiling in (False, True):
        timer = workflow.Timer("cuda", synchronize=profiling)
        with timer.stage("forward_backward_update"):
            pass
        assert timer.seconds["forward_backward_update"] >= 0
        assert len(calls) == (2 if profiling else 0)


def test_checkpoint_write_failure_names_destination(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    path.write_bytes(b"old checkpoint")
    def fail(*a, **kw):
        raise RuntimeError("disk write error")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="last.pt.*disk"):
        support.save_checkpoint(path, {})
    assert path.read_bytes() == b"old checkpoint"


def test_four_commands_share_common_settings():
    spec = load_spec(ROOT / "experiments.json")
    parsed = []
    for name in VARIANTS:
        command = training_command(spec, name, 42, "MSBin", "splits", "run")
        args = vars(get_args(command[3:]))
        assert args["pretrained"] and not args["freeze_encoder"] and not args["white_only"]
        assert args["require_cuda"]  # Profiler CPU support must not relax research routing.
        assert args["threads"] == 2
        assert args["max_patches_per_page"] is None and args["precision"] == "fp32"
        assert (args["band_reweight"], args["band_drop_p"]) == VARIANTS[name]
        parsed.append({k: v for k, v in args.items() if k not in ("experiment_name", "band_reweight", "band_drop_p")})
    assert all(p == parsed[0] for p in parsed)


def evaluation(name="baseline"):
    keys = [f"V{i}" for i in range(20)]
    scores = [PixelCounts(i + 1, 1, 2).scores() for i in range(20)]
    pooled = PixelCounts(sum(s["tp"] for s in scores), 20, 40).scores()
    return dict(status="complete", format_version=2, training_args=dict(experiment_name=name, seed=42,
                epochs=7, precision="fp32"), model_config=dict(in_ch=12, pretrained=True, freeze_encoder=False,
                band_reweight=VARIANTS[name][0], band_drop_p=VARIANTS[name][1]),
                splits={"train": {"keys": ["T"], "sha256": "t"}, "val": {"keys": keys, "sha256": "v"}},
                keys_file_sha256="v", evaluation_versions={"torch": "version"}, evaluation_backend={"tf32": False},
                keys=keys, pages=[dict(key=k, scores=s) for k, s in zip(keys, scores)],
                fg_type=1, threshold=.5, tiling=dict(tile=256, stride=256), precision="fp32",
                threshold_rule="probability > threshold", conventions={"machine": "white"}, source_split="train",
                evaluation_source={"infer": "hash"}, device="cuda", source_identity={"trainer": "hash"}, versions={"torch": "version"},
                initialization={"sha256": "shared"}, aggregate_scores=pooled, completed_epoch=4,
                checkpoint={"sha256": "checkpoint", "path": "best.pt"}, run_completed_epoch=7)


def test_comparison_pooled_counts_and_page_means():
    report = comparison_rows([evaluation(), evaluation("reweight_only")])
    assert len(report["runs"]) == 2 and len(report["pages"]) == 40
    row = report["runs"][0]
    assert row["tp"] == 210 and row["fp"] == 20 and row["fn"] == 40
    assert row["f1"] != row["page_mean_f1"]


@pytest.mark.parametrize("change", ["split", "task", "threshold", "tile", "missing_page", "duplicate_page", "init", "budget", "precision", "counts", "unfinished"])
def test_comparison_rejects_mismatch(change):
    other = evaluation("drop_only")
    if change == "split":
        other["splits"]["val"]["sha256"] = "changed"
    elif change == "task":
        other["fg_type"] = 2
    elif change == "threshold":
        other["threshold"] = .6
    elif change == "tile":
        other["tiling"]["tile"] = 128
    elif change == "missing_page":
        other["pages"].pop()
    elif change == "duplicate_page":
        other["pages"].append(other["pages"][0])
    elif change == "init":
        other["initialization"]["sha256"] = "different"
    elif change == "budget":
        other["training_args"]["epochs"] = 10
    elif change == "precision":
        other["precision"] = None
    elif change == "counts":
        other["aggregate_scores"]["tp"] += 1
    else:
        other["run_completed_epoch"] = 6
    with pytest.raises(ValueError):
        comparison_rows([evaluation(), other])


def test_notebook_is_unexecuted_and_python_cells_parse():
    notebook = json.loads((ROOT / "notebooks/msbin_experiments_colab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_profiler_explicit_routing_and_bounded_random_initialization(monkeypatch, device):
    import scripts.profile_msbin_training as profiler
    captured = []
    monkeypatch.setattr(profiler, "train", lambda argv: captured.append(get_args(argv)))
    profiler.main(["--outdir", "fresh", "--device", device, "--threads", "2", "--batches", "4"])
    args = captured[0]
    assert args.device == device and args.threads == 2 and args.precision == "fp32"
    assert args.require_cuda == (device == "cuda")
    assert args.profile_batches == 4 and args.batch == 2 and args.epochs == 1
    assert not args.pretrained and not args.resume and not args.experiment_name


def test_profiler_cpu_amp_rejected_before_trainer(monkeypatch, capsys):
    import scripts.profile_msbin_training as profiler
    monkeypatch.setattr(profiler, "train", lambda *a: pytest.fail("must reject before trainer"))
    with pytest.raises(SystemExit):
        profiler.main(["--outdir", "fresh", "--device", "cpu", "--precision", "amp"])
    assert "CPU profiling requires" in capsys.readouterr().err


def test_device_resolution_threads_and_gpu_safeguards(monkeypatch):
    import train_unet_msbin_cpu as trainer
    base = ["--msbin_root", "data", "--train_keys", "a", "--val_keys", "b"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    args = get_args(base + ["--device", "cpu", "--threads", "2"])
    trainer.configure_runtime(args)
    assert args.device == "cpu" and args.hardware == "cpu" and args.threads == 2 and calls == [2]
    for extra in (["--device", "cpu", "--precision", "amp"], ["--device", "cpu", "--require_cuda"]):
        with pytest.raises(ValueError, match="CPU|CUDA"):
            trainer.configure_runtime(get_args(base + extra))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for extra in (["--device", "cuda"], ["--require_cuda"]):
        with pytest.raises(ValueError, match="CUDA"):
            trainer.configure_runtime(get_args(base + extra))


def test_cpu_profile_records_actual_counts_threads_and_no_cuda_sync(tmp_path, monkeypatch):
    import train_unet_msbin_cpu as trainer
    import scripts.profile_msbin_training as profiler
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: pytest.fail("CPU profile must not synchronize CUDA"))
    class TinyDataset:
        def __init__(self, **kwargs):
            assert calls == [2]  # Applied before indexing or model computation.
            self.keys = ["A"]
        def __len__(self):
            return 9
        def __getitem__(self, index):
            return torch.ones(12, 2, 2), torch.ones(1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool)
    monkeypatch.setattr(trainer, "MSBinDibcoPatchDataset", TinyDataset)
    monkeypatch.setattr(trainer, "split_metadata", lambda *a: {"train": "A", "val": "B"})
    monkeypatch.setattr(trainer, "build_model", lambda *a, **kw: nn.Conv2d(12, 1, 1))
    output = tmp_path / "cpu_profile"
    profiler.main(["--outdir", str(output), "--device", "cpu", "--threads", "2", "--batches", "4"])
    timing = json.loads((output / "timing.json").read_text())
    assert timing["device"] == "cpu" and timing["threads"] == 2 and not timing["synchronized"]
    assert timing["profile_observed"] == {stage: {"batches": 4, "samples_loaded": 8} for stage in ("train", "validation")}
    assert {"dataset_indexing", "data_wait", "forward_backward_update", "validation", "checkpoint_writing"} <= timing["seconds"].keys()
    assert support.read_checkpoint(output / "last.pt")["software_smoke_only"]


@pytest.mark.parametrize("field,value", [("experiment_name", "drop_only"), ("seed", 43), ("epochs", 10), ("batch", 4)])
def test_comparison_selection_cannot_silently_use_another_run(field, value):
    from run_experiments import validate_run_selection
    spec = load_spec(ROOT / "experiments.json")
    saved = dict(spec["common"], experiment_name="baseline", seed=42, epochs=spec["epochs"])
    validate_run_selection(saved, spec, "baseline", 42)
    saved[field] = value
    with pytest.raises(ValueError, match="selected experiment"):
        validate_run_selection(saved, spec, "baseline", 42)


def test_thread_setting_is_resume_configuration():
    ckpt, config, args, splits = tiny_checkpoint()
    args.threads = 2
    with pytest.raises(ValueError, match="training configuration"):
        support.validate_resume(ckpt, config, args, splits)


def test_notebook_has_one_editable_configuration_and_v3_archive():
    notebook = json.loads((ROOT / "notebooks/msbin_experiments_colab.ipynb").read_text(encoding="utf-8"))
    assignments = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert index > 0 and notebook["cells"][index - 1]["cell_type"] == "markdown"
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append((index, node.value))
    editable = ("ARCHIVE_NAME", "CONFIGURATION", "MODE", "ENABLE_RESEARCH_TRAINING",
                "RUN_PROFILE", "RUN_VALIDATION", "RUN_COMPARISON", "DOWNLOAD_COMPARISON")
    assert all(len(assignments[name]) == 1 for name in editable)
    assert len({assignments[name][0][0] for name in editable}) == 1
    assert ast.literal_eval(assignments["ARCHIVE_NAME"][0][1]) == "msbin_gpu_experiments_v3.zip"
    assert ast.literal_eval(assignments["CONFIGURATION"][0][1]) == "baseline"
    for name in editable[3:]:
        assert ast.literal_eval(assignments[name][0][1]) is False
