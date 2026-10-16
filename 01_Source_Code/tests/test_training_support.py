"""Software behavior only. No research model fitting or pretrained downloads."""
import copy
import csv
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import training_support as support
from train_unet_msbin_cpu import get_args, train_epoch


def config():
    return dict(architecture=support.ARCHITECTURE, in_ch=1, band_reweight=False,
                band_drop_p=0.0, freeze_encoder=False, pretrained=False)


def args():
    return SimpleNamespace(epochs=2, lr=0.001, device="cpu", outdir="unused", resume=False,
                           train_keys="train.txt", val_keys="val.txt", pos_weight=6.0,
                           bce_w=0.9, dice_w=0.1, grad_clip=1.0)


def row(epoch=1, score=0.6):
    return dict(epoch=epoch, train_loss=0.5, patch_val_precision=score, patch_val_recall=score,
                patch_val_f1=score, elapsed_seconds=1.0, best_val_f1=score,
                band_weights=[0.2], improved=True)


def fixture_checkpoint():
    model = nn.Conv2d(1, 1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    # Populate optimizer state with a tiny software update.
    model(torch.ones(1, 1, 2, 2)).sum().backward()
    opt.step()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(19)
    splits = {"train": {"keys": ["A"], "sha256": "a"}, "val": {"keys": ["B"], "sha256": "b"}}
    ckpt = support.make_checkpoint(model, opt, scaler, generator, 1, 0.6, config(), args(), splits, [row()])
    return ckpt, model, opt, scaler, generator


@pytest.mark.parametrize("loss_fn", [support.masked_weighted_bce, support.masked_dice_loss, support.masked_loss])
def test_invalid_pixels_have_no_loss_or_gradient_influence(loss_fn):
    valid = torch.tensor([[[[True, False], [True, False]]]])
    y = torch.tensor([[[[1., 0.], [0., 1.]]]])
    z = torch.tensor([[[[0.2, 0.3], [-0.4, 0.7]]]], requires_grad=True)
    original = loss_fn(z, y, valid)
    original.backward()
    changed = z.detach().clone()
    changed[~valid] = float("nan")
    changed.requires_grad_()
    changed_y = y.clone()
    changed_y[~valid] = float("inf")
    actual = loss_fn(changed, changed_y, valid)
    actual.backward()
    torch.testing.assert_close(actual, original)
    torch.testing.assert_close(changed.grad, z.grad)
    assert (z.grad[~valid] == 0).all()


def test_empty_foreground_and_invalid_samples():
    z = torch.zeros(1, 1, 2, 2, requires_grad=True)
    y, mask = torch.zeros_like(z), torch.ones_like(z, dtype=torch.bool)
    base = support.masked_loss(z, y, mask)
    assert torch.isfinite(base)
    base.backward()
    assert torch.isfinite(z.grad).all()
    assert support.masked_loss(z, y, ~mask) is None
    combined = support.masked_loss(torch.cat([z, z]), torch.cat([y, y]), torch.cat([mask, ~mask]))
    torch.testing.assert_close(combined, base)
    nearly_empty = support.masked_dice_loss(torch.full_like(z, -100), y, mask)
    assert nearly_empty.item() == pytest.approx(0, abs=1e-6)


def test_loss_shapes_and_finite_checks():
    z = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match="shapes"):
        support.masked_loss(z, z[0], torch.ones_like(z, dtype=torch.bool))
    with pytest.raises(ValueError, match="Nonfinite"):
        support.masked_loss(z + float("nan"), z, torch.ones_like(z, dtype=torch.bool))


def test_validation_aggregates_counts():
    batch1 = (torch.tensor([[[[8., -8.]]]]), torch.tensor([[[[1., 1.]]]]), torch.ones(1, 1, 1, 2, dtype=torch.bool))
    batch2 = (torch.tensor([[[[8., 8., 8.]]]]), torch.tensor([[[[0., 1., 0.]]]]), torch.tensor([[[[True, True, False]]]]))
    metrics = support.validate_patches(nn.Identity(), [batch1, batch2], "cpu")
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 1, 1)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_all_invalid_training_batch_skipped():
    model = nn.Conv2d(1, 1, 1)
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    x = torch.ones(1, 1, 2, 2)
    invalid = (x, x, torch.zeros_like(x, dtype=torch.bool))
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="no valid samples"):
        train_epoch(model, [invalid], opt, scaler, "cpu", args())
    for key in before:
        torch.testing.assert_close(before[key], model.state_dict()[key])


def test_split_checks(tmp_path):
    images, labels = tmp_path / "train/images", tmp_path / "train/labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for key in ("A", "B"):
        for band in range(12):
            (images / f"{key}_{band}.png").touch()
        (labels / f"{key}.png").touch()
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text("A\n")
    val.write_text("B\n")
    assert support.split_metadata(tmp_path, train, val)["train"]["keys"] == ["A"]
    with pytest.raises(ValueError, match="Both"):
        support.split_metadata(tmp_path, None, val)
    for content, match in (("A\n", "Overlapping"), ("B\nB\n", "Duplicate"), ("", "Empty"), ("C\n", "unknown")):
        val.write_text(content)
        with pytest.raises(ValueError, match=match):
            support.split_metadata(tmp_path, train, val)
    with pytest.raises(SystemExit):
        get_args(["--msbin_root", str(tmp_path)])


def test_checkpoint_best_epoch_restore_rng_and_logs(tmp_path):
    ckpt, model, opt, scaler, generator = fixture_checkpoint()
    support.commit_epoch(tmp_path, ckpt, True)
    loaded = support.read_checkpoint(tmp_path / "last.pt")
    assert loaded["best_val_f1"] == 0.6 and loaded["completed_epoch"] == 1 and loaded["next_epoch"] == 2
    assert (tmp_path / "best.pt").is_file()
    expected = (random.random(), np.random.random(), torch.rand(2), torch.rand(2, generator=generator))
    new = nn.Conv2d(1, 1, 1)
    new_opt = torch.optim.AdamW(new.parameters())
    start, best = support.restore_checkpoint(loaded, new, new_opt, scaler, generator)
    assert (start, best) == (2, 0.6)
    assert random.random() == expected[0] and np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(2), expected[2])
    torch.testing.assert_close(torch.rand(2, generator=generator), expected[3])
    for key in model.state_dict():
        torch.testing.assert_close(new.state_dict()[key], model.state_dict()[key])
    assert len(new_opt.state) == len(opt.state)
    support.write_logs(tmp_path, loaded["history"])
    support.write_logs(tmp_path, loaded["history"])
    with (tmp_path / "epochs.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
    with pytest.raises(ValueError, match="Fresh"):
        support.check_output_dir(tmp_path, False)


@pytest.mark.parametrize("field,value", [("in_ch", 12), ("band_reweight", True), ("band_drop_p", 0.2)])
def test_model_configuration_rejection(field, value):
    ckpt, *_ = fixture_checkpoint()
    changed = config()
    changed[field] = value
    with pytest.raises(ValueError, match="model configuration"):
        support.validate_resume(ckpt, changed, args(), ckpt["splits"])


def test_changed_splits_and_config_rejected_epoch_extension_allowed():
    ckpt, *_ = fixture_checkpoint()
    changed = copy.deepcopy(ckpt["splits"])
    changed["train"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="split"):
        support.validate_resume(ckpt, config(), args(), changed)
    extended = args()
    extended.epochs = 5
    extended.resume = True
    support.validate_resume(ckpt, config(), extended, ckpt["splits"])
    extended.lr = 1
    with pytest.raises(ValueError, match="training configuration"):
        support.validate_resume(ckpt, config(), extended, ckpt["splits"])


def test_legacy_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model": {}}, path)
    with pytest.raises(ValueError, match="Legacy"):
        support.read_checkpoint(path)


def test_training_main_updates_best_before_last_and_resumes(tmp_path, monkeypatch):
    import train_unet_msbin_cpu as trainer
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    class TinyDataset:
        def __init__(self, **kwargs):
            assert kwargs["return_valid_mask"] and kwargs["split"] == "train"
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return torch.ones(12, 2, 2), torch.ones(1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool)
    monkeypatch.setattr(trainer, "MSBinDibcoPatchDataset", TinyDataset)
    monkeypatch.setattr(trainer, "split_metadata", lambda *a: {"train": "A", "val": "B"})
    constructions = []
    def factory(cfg, restoring=False):
        assert not cfg["pretrained"]
        constructions.append(restoring)
        return nn.Conv2d(12, 1, 1)
    monkeypatch.setattr(trainer, "build_model", factory)
    scores = iter([0.7, 0.4, 0.5])
    monkeypatch.setattr(trainer, "validate_patches", lambda *a: dict(precision=0.5, recall=0.5, f1=next(scores)))
    argv = ["--msbin_root", str(tmp_path), "--train_keys", "train.txt", "--val_keys", "val.txt",
            "--outdir", str(tmp_path / "run"), "--no-pretrained", "--epochs", "2"]
    trainer.main(argv)
    last = support.read_checkpoint(tmp_path / "run/last.pt")
    best = support.read_checkpoint(tmp_path / "run/best.pt")
    assert last["best_val_f1"] == best["best_val_f1"] == 0.7
    assert last["completed_epoch"] == 2 and best["completed_epoch"] == 1
    trainer.main(argv[:-1] + ["3", "--resume"])
    assert constructions == [False, True]
    last = support.read_checkpoint(tmp_path / "run/last.pt")
    assert [r["epoch"] for r in last["history"]] == [1, 2, 3]
    assert last["next_epoch"] == 4
    with (tmp_path / "run/epochs.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_restore_construction_forces_offline():
    seen = []
    def factory(**kwargs):
        seen.append(kwargs)
        assert kwargs["pretrained"] is False
        return nn.Conv2d(kwargs["in_ch"], 1, 1)
    cfg = config()
    cfg["pretrained"] = True  # Historic initialization must NOT trigger a download.
    support.build_model(cfg, restoring=True, factory=factory)
    assert cfg["pretrained"] is True and len(seen) == 1


def test_band_weight_sample_average():
    mean = support.BandWeightMean()
    mean.update(torch.tensor([[0., 1.], [0., 1.]]))
    mean.update(torch.tensor([[1., 0.]]))
    assert mean.mean() == pytest.approx([1 / 3, 2 / 3])


def test_train_loop_logs_training_weights_only():
    class TinyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(()))
        def forward(self, x):
            self.last_band_weights = x.mean((2, 3)).detach()
            return x + self.bias
    model = TinyAttention()
    opt = torch.optim.AdamW(model.parameters())
    batches = []
    for value, size in ((0., 2), (1., 1)):
        x = torch.full((size, 1, 2, 2), value)
        batches.append((x, torch.zeros_like(x), torch.ones_like(x, dtype=torch.bool)))
    _, mean, skipped = train_epoch(model, batches, opt, torch.amp.GradScaler("cuda", enabled=False), "cpu", args())
    assert mean == pytest.approx([1 / 3]) and skipped == 0
    support.validate_patches(model, batches, "cpu")
    assert mean == pytest.approx([1 / 3])


def test_frozen_encoder_train_method_without_large_construction():
    from unet_small import UNetSmall
    # Exercise the actual train override on a tiny shell, no ResNet constructed.
    model = UNetSmall.__new__(UNetSmall)
    nn.Module.__init__(model)
    model.freeze_encoder = True
    model.enc = nn.Sequential(nn.BatchNorm2d(1))
    model.decoder = nn.BatchNorm2d(1)
    before = model.enc[0].running_mean.clone()
    model.train()
    model.enc(torch.ones(2, 1, 2, 2))
    assert not model.enc.training and model.decoder.training
    torch.testing.assert_close(before, model.enc[0].running_mean)
    model.freeze_encoder = False
    model.train()
    assert model.enc.training


def test_resumed_tiny_loop_matches_uninterrupted_training(tmp_path, monkeypatch):
    import train_unet_msbin_cpu as trainer
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    class TinyDataset:
        def __init__(self, **kwargs):
            pass
        def __len__(self):
            return 5
        def __getitem__(self, index):
            x = torch.full((12, 2, 2), (index + 1) / 5)
            y = torch.full((1, 2, 2), float(index % 2))
            valid = torch.ones(1, 2, 2, dtype=torch.bool)
            valid[:, 0, 0] = False
            return x, y, valid
    monkeypatch.setattr(trainer, "MSBinDibcoPatchDataset", TinyDataset)
    monkeypatch.setattr(trainer, "split_metadata", lambda *a: {"train": "A", "val": "B"})
    def factory(cfg, restoring=False):
        assert cfg["pretrained"] is False
        return nn.Sequential(nn.Conv2d(12, 1, 1), nn.Dropout(0.3))
    monkeypatch.setattr(trainer, "build_model", factory)
    common = ["--msbin_root", str(tmp_path), "--train_keys", "train.txt", "--val_keys", "val.txt",
              "--no-pretrained", "--batch", "2", "--seed", "123"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    trainer.main(common + ["--outdir", str(full), "--epochs", "3"])
    trainer.main(common + ["--outdir", str(resumed), "--epochs", "1"])
    trainer.main(common + ["--outdir", str(resumed), "--epochs", "3", "--resume"])
    expected = support.read_checkpoint(full / "last.pt")
    actual = support.read_checkpoint(resumed / "last.pt")
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)
    torch.testing.assert_close(actual["shuffle_rng"], expected["shuffle_rng"], rtol=0, atol=0)
    for left, right in zip(actual["history"], expected["history"]):
        assert left["train_loss"] == right["train_loss"]
        assert left["patch_val_f1"] == right["patch_val_f1"]


def test_atomic_write_failure_preserves_previous_file(tmp_path):
    path = tmp_path / "last.pt"
    path.write_bytes(b"previous complete checkpoint")
    def interrupted(stream):
        stream.write(b"partial replacement")
        raise OSError("simulated interrupted save")
    with pytest.raises(OSError, match="interrupted"):
        support.atomic_write(path, interrupted)
    assert path.read_bytes() == b"previous complete checkpoint"
    assert list(tmp_path.iterdir()) == [path]
