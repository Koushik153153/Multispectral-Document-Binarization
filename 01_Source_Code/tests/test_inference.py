"""Synthetic inference behavior tests; no real ResNet or dataset access."""
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import infer_msbin_cpu as inference
from msbin_labels import decode_label, FG1, BG, UR
from pixel_metrics import PixelCounts
import training_support as support


class FirstBand(nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []
        self.drop = nn.Dropout2d(0.9)
    def forward(self, x):
        self.shapes.append(tuple(x.shape))
        return self.drop(x[:, :1])


@pytest.mark.parametrize("height,width", [(6, 7), (1, 1), (1, 7), (3, 2), (8, 8)])
def test_edge_small_page_dimensions_and_eval(height, width):
    model = FirstBand().train()
    page = np.full((1, height, width), 128, np.uint8)
    probabilities, coverage = inference.tiled_probabilities(model, page, tile=4, stride=4)
    assert probabilities.shape == coverage.shape == (height, width)
    assert coverage.min() >= 1
    assert set(model.shapes) == {(1, 1, 4, 4)}
    assert not model.training and not model.drop.training
    np.testing.assert_allclose(probabilities, torch.sigmoid(torch.tensor(128 / 255)).item(), rtol=1e-6)


def test_overlap_averages_probabilities_before_threshold():
    class TileConstant(nn.Module):
        def __init__(self):
            super().__init__()
            self.values = iter([0.9, 0.4])
        def forward(self, x):
            p = next(self.values)
            return torch.full((1, 1, 2, 2), float(np.log(p / (1 - p))))
    probabilities, coverage = inference.tiled_probabilities(TileConstant(), np.zeros((1, 2, 3), np.uint8), 2, 1)
    np.testing.assert_allclose(probabilities, [[0.9, 0.65, 0.4]] * 2, atol=1e-6)
    np.testing.assert_array_equal(coverage, [[1, 2, 1]] * 2)
    mask, _ = inference.mask_images(probabilities, 0.6)
    assert np.all(mask[:, 1] == 255)  # Thresholding tiles first would incorrectly vote 0.5.


@pytest.mark.parametrize("tile,stride,thr", [(0, 1, .5), (4, 0, .5), (4, 5, .5), (4, 4, float("nan")), (4, 4, 2)])
def test_invalid_tiling(tile, stride, thr):
    with pytest.raises(ValueError):
        inference.validate_tiling(tile, stride, thr)


def test_shape_finite_and_coverage_failures(monkeypatch):
    class Bad(nn.Module):
        def forward(self, x):
            return x[:, :, :1]
    page = np.zeros((1, 5, 5), np.uint8)
    with pytest.raises(ValueError, match="shape"):
        inference.tiled_probabilities(Bad(), page, 2, 2)
    class Nan(nn.Module):
        def forward(self, x):
            return x * float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        inference.tiled_probabilities(Nan(), page, 2, 2)
    monkeypatch.setattr(inference, "patch_starts", lambda *args: [0])
    with pytest.raises(ValueError, match="zero coverage"):
        inference.tiled_probabilities(FirstBand(), page, 2, 2)


def make_page(root, key="A", shape=(2, 3)):
    images = root / "train/images"
    labels = root / "train/labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for band in range(12):
        assert cv2.imwrite(str(images / f"{key}_{band}.png"), np.full(shape, 128, np.uint8))
    label = np.zeros((*shape, 3), np.uint8)
    label[0, 0] = FG1
    label[0, 1] = UR
    assert cv2.imwrite(str(labels / f"{key}.png"), label)


def test_requested_order_and_invalid_keys(tmp_path):
    make_page(tmp_path, "A")
    make_page(tmp_path, "B")
    keys = tmp_path / "keys.txt"
    keys.write_text("B\nA\n")
    pages, requested, selected = inference.requested_pages(tmp_path / "train/images", keys, 1)
    assert requested == ["B", "A"] and selected == ["B"]
    for contents, message in (("", "Empty"), ("A\nA\n", "Duplicate"), ("Z\n", "Unknown")):
        keys.write_text(contents)
        with pytest.raises(ValueError, match=message):
            inference.requested_pages(tmp_path / "train/images", keys)
    keys.write_text("A\n")
    (tmp_path / "train/images/A_5.png").unlink()
    with pytest.raises(ValueError, match="missing bands"):
        inference.requested_pages(tmp_path / "train/images", keys)
    assert cv2.imwrite(str(tmp_path / "train/images/B_5.png"), np.zeros((1, 2), np.uint8))
    with pytest.raises(ValueError, match="mismatched"):
        inference.read_page(pages["B"], 12, "B")


def checkpoint_file(path, **changes):
    config = dict(architecture=support.ARCHITECTURE, in_ch=1, band_reweight=False,
                  band_drop_p=0.2, pretrained=True, freeze_encoder=False)
    tiny = nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.AdamW(tiny.parameters())
    args = SimpleNamespace(epochs=1, device="cpu", fg_type=1)
    checkpoint = support.make_checkpoint(tiny, optimizer, None, torch.Generator(), 1, 0.2, config,
                                         args, {"val": {"keys": ["A"]}}, [{"epoch": 1}])
    checkpoint.update(changes)
    torch.save(checkpoint, path)
    return tiny


def test_metadata_reconstruction_offline_and_strict(tmp_path):
    path = tmp_path / "tiny.pt"
    original = checkpoint_file(path)
    def factory(**kwargs):
        assert kwargs == dict(in_ch=1, band_reweight=False, band_drop_p=0.2, pretrained=False, freeze_encoder=False)
        return nn.Conv2d(1, 1, 1)
    model, metadata = inference.load_inference_model(path, factory=factory)
    assert metadata["fg_type"] == 1 and metadata["completed_epoch"] == 1
    assert metadata["model_config"]["pretrained"] is True
    assert "optimizer" not in metadata and not model.training
    assert len(metadata["checkpoint"]["sha256"]) == 64
    torch.testing.assert_close(model.weight, original.weight)
    with pytest.raises(RuntimeError):
        inference.load_inference_model(path, factory=lambda **kw: nn.Conv2d(12, 1, 1))


def test_smoke_legacy_unsupported_rejected(tmp_path):
    path = tmp_path / "tiny.pt"
    checkpoint_file(path, software_smoke_only=True)
    def forbidden(**kwargs):
        pytest.fail("Should reject before model construction")
    with pytest.raises(ValueError, match="Software-smoke"):
        inference.load_inference_model(path, factory=forbidden)
    torch.save({"model": {}}, path)
    with pytest.raises(ValueError, match="Legacy"):
        inference.load_inference_model(path, factory=forbidden)
    checkpoint_file(path)
    checkpoint = support.read_checkpoint(path)
    checkpoint["model_config"]["architecture"] = "unsupported"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Unsupported"):
        inference.load_inference_model(path, factory=forbidden)
    checkpoint_file(path, format_version=999)
    with pytest.raises(ValueError, match="Legacy"):
        inference.load_inference_model(path, factory=forbidden)


def test_removed_flags_and_test_split_are_rejected():
    common = ["--msbin_root", "data", "--keys_txt", "keys.txt", "--ckpt", "trusted.pt", "--outdir", "preview"]
    for extra in (["--white_only"], ["--band_reweight"], ["--split", "test"]):
        with pytest.raises(SystemExit):
            inference.get_args(common + extra)
def test_full_page_ur_metrics_and_panel():
    target, valid = decode_label(np.array([[FG1, BG, UR]], np.uint8))
    probability = np.array([[0.9, 0.1, 0.99]], np.float32)
    scores = PixelCounts().update(probability, target, valid).scores()
    assert (scores["tp"], scores["fp"], scores["fn"], scores["f1"]) == (1, 0, 0, 1)
    machine, reading = inference.mask_images(probability, 0.5)
    np.testing.assert_array_equal(machine, [[255, 0, 255]])
    np.testing.assert_array_equal(reading, [[0, 255, 0]])
    panel = inference.comparison_image(np.zeros((1, 3), np.uint8), target, valid, reading)
    assert panel.shape == (1, 9, 3)
    np.testing.assert_array_equal(panel[0, 5], [255, 0, 255])


def test_write_failure_and_overwrite_protection(tmp_path, monkeypatch):
    path = tmp_path / "result.png"
    monkeypatch.setattr(inference.cv2, "imwrite", lambda *a: False)
    with pytest.raises(OSError, match="imwrite failed"):
        inference.write_png(path, np.zeros((2, 2), np.uint8))
    assert not path.exists()
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        inference.write_png(path, np.zeros((2, 2), np.uint8))
    assert path.read_bytes() == b"keep"


def test_end_to_end_dummy_preview(tmp_path, monkeypatch):
    root = tmp_path / "data"
    make_page(root)
    keys = tmp_path / "keys.txt"
    keys.write_text("A\n")
    checkpoint = tmp_path / "tiny.pt"
    checkpoint_file(checkpoint)
    real_loader = inference.load_inference_model
    monkeypatch.setattr(inference, "load_inference_model", lambda path, device: real_loader(path, device, factory=lambda **k: nn.Conv2d(1, 1, 1)))
    argv = ["--msbin_root", str(root), "--keys_txt", str(keys), "--ckpt", str(checkpoint),
            "--outdir", str(tmp_path / "preview"), "--tile", "2", "--stride", "1", "--max_pages", "1", "--compare_gt"]
    manifest = inference.run(inference.get_args(argv))
    assert manifest["keys"] == ["A"] and manifest["pages"][0]["uncertain_pixels"] == 1
    assert manifest["aggregate_scores"] == manifest["pages"][0]["scores"]
    assert cv2.imread(str(tmp_path / "preview/A_mask.png"), 0).shape == (2, 3)
    assert (tmp_path / "preview/manifest.json").is_file()
    with pytest.raises(FileExistsError):
        inference.run(inference.get_args(argv))
