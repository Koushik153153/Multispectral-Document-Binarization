"""Tiny synthetic images verify software behavior, not model performance."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msbin_labels import decode_label, FG1, FG2, BG, UR
from msbin_dataset import MSBinDibcoPatchDataset, patch_starts
from pixel_metrics import PixelCounts, pixel_metrics


def write_page(root, label, key="PAGE"):
    for folder in ("images", "labels"):
        (root / "train" / folder).mkdir(parents=True, exist_ok=True)
    h, w = label.shape[:2]
    for band in range(12):
        assert cv2.imwrite(str(root / "train/images" / f"{key}_{band}.png"), np.full((h, w), 128, np.uint8))
    assert cv2.imwrite(str(root / "train/labels" / f"{key}.png"), label)


def dataset(root, **kwargs):
    return MSBinDibcoPatchDataset(str(root), patch_size=4, stride=4, min_fg_frac=0, **kwargs)


def test_label_semantics_and_ur():
    label = np.array([[FG1, FG2, BG, UR]], dtype=np.uint8)
    for fg_type, expected in ((1, [[1, 0, 0, 0]]), (2, [[0, 1, 0, 0]])):
        target, valid = decode_label(label, fg_type)
        np.testing.assert_array_equal(target, expected)
        np.testing.assert_array_equal(valid, [[True, True, True, False]])
        assert target.dtype == np.float32 and valid.dtype == np.bool_


@pytest.mark.parametrize("fg_type", [0, 3, -1, True, "1", 1.5])
def test_invalid_fg_type(fg_type):
    with pytest.raises(ValueError, match="fg_type"):
        decode_label(np.zeros((1, 1, 3), np.uint8), fg_type)


def test_unknown_label_color():
    with pytest.raises(ValueError, match="Unexpected BGR.*17"):
        decode_label(np.full((1, 1, 3), 17, np.uint8))


def test_return_api_padding_and_single_band(tmp_path):
    label = np.array([[FG1, UR, BG], [FG2, BG, BG]], dtype=np.uint8)
    write_page(tmp_path, label)
    assert len(dataset(tmp_path)[0]) == 2
    ds = dataset(tmp_path, return_valid_mask=True)
    image, target, valid = ds[0]
    assert image.shape == (12, 4, 4)
    assert target.shape == valid.shape == (1, 4, 4)
    assert image.dtype == target.dtype == torch.float32
    assert valid.dtype == torch.bool
    assert torch.allclose(image, torch.full_like(image, 128 / 255))
    assert int(valid.sum()) == 5
    assert not valid[:, 2:, :].any() and not valid[:, :, 3:].any()
    assert (target[~valid] == 0).all()
    assert dataset(tmp_path, use_white_only=True)[0][0].shape == (1, 4, 4)


@pytest.mark.parametrize("length,expected", [(512, [0, 256]), (600, [0, 256, 344]), (20, [0]), (256, [0])])
def test_patch_starts(length, expected):
    starts = patch_starts(length, 256, 256)
    assert starts == expected and len(starts) == len(set(starts))


def test_full_edge_and_background_coverage(tmp_path):
    write_page(tmp_path, np.zeros((6, 7, 3), np.uint8))
    ds = dataset(tmp_path, return_valid_mask=True)
    assert ds.index == [("PAGE", 0, 0), ("PAGE", 0, 3), ("PAGE", 2, 0), ("PAGE", 2, 3)]
    covered = np.zeros((6, 7), bool)
    for i, (_, y, x) in enumerate(ds.index):
        covered[y:y + 4, x:x + 4] = True
        _, target, valid = ds[i]
        assert not target.any() and valid.all()
    assert covered.all()


def test_filter_uses_only_valid_pixels(tmp_path):
    label = np.full((2, 2, 3), UR, np.uint8)
    label[0, 0] = FG1
    write_page(tmp_path, label)
    ds = MSBinDibcoPatchDataset(str(tmp_path), patch_size=4, stride=4, min_fg_frac=1, return_valid_mask=True)
    assert len(ds) == 1 and int(ds[0][2].sum()) == 1
    write_page(tmp_path, np.full((2, 2, 3), UR, np.uint8))
    with pytest.raises(RuntimeError, match="no valid patches"):
        dataset(tmp_path)


@pytest.mark.parametrize("patch,stride", [(0, 1), (-1, 1), (4, 0), (4, -1), (4, 5)])
def test_invalid_patch_parameters(tmp_path, patch, stride):
    with pytest.raises(ValueError):
        MSBinDibcoPatchDataset(str(tmp_path), patch_size=patch, stride=stride)


@pytest.mark.parametrize("contents,message", [("PAGE\nPAGE\n", "Duplicate"), ("MISSING\n", "Unknown")])
def test_invalid_explicit_keys(tmp_path, contents, message):
    write_page(tmp_path, np.zeros((4, 4, 3), np.uint8))
    keys = tmp_path / "keys.txt"
    keys.write_text(contents)
    with pytest.raises(ValueError, match=message):
        dataset(tmp_path, keys_txt=str(keys))


@pytest.mark.parametrize("damage,message", [("band", "PAGE.*missing bands"), ("label", "PAGE.png"), ("shape", "PAGE.*mismatched dimensions")])
def test_page_integrity_errors(tmp_path, damage, message):
    write_page(tmp_path, np.zeros((4, 4, 3), np.uint8))
    path = tmp_path / "train/images/PAGE_7.png"
    if damage == "band":
        path.unlink()
    elif damage == "label":
        (tmp_path / "train/labels/PAGE.png").unlink()
    else:
        assert cv2.imwrite(str(path), np.zeros((3, 4), np.uint8))
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        dataset(tmp_path)


def test_masked_metrics_ignore_ur_and_accumulate():
    label = np.array([[FG1, BG, UR], [FG1, FG1, BG]], np.uint8)
    target, valid = decode_label(label)
    probabilities = np.array([[0.9, 0.8, 1.0], [0.1, 0.9, 0.1]])
    direct = pixel_metrics(probabilities, target, valid)
    assert (direct["tp"], direct["fp"], direct["fn"]) == (2, 1, 1)
    assert direct["precision"] == pytest.approx(2 / 3)
    assert direct["recall"] == pytest.approx(2 / 3)
    assert direct["f1"] == pytest.approx(2 / 3)
    accumulated = PixelCounts()
    for i in range(2):
        accumulated.update(torch.tensor(probabilities[i:i + 1]), torch.tensor(target[i:i + 1]), torch.tensor(valid[i:i + 1]))
    assert accumulated.scores() == direct
    probabilities[0, 2] = 0
    assert pixel_metrics(probabilities, target, valid) == direct


def test_metric_empty_and_threshold_conventions():
    target = np.zeros((2, 2))
    assert pixel_metrics(target, target)["f1"] == 0
    assert pixel_metrics(np.ones((2, 2)), target, np.zeros((2, 2), bool)) == PixelCounts().scores()
    assert pixel_metrics(np.full((2, 2), 0.5), np.ones((2, 2)))["fn"] == 4


def test_metric_shape_and_value_validation():
    y = np.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="identical"):
        pixel_metrics(np.zeros((2, 2)), y)
    with pytest.raises(ValueError, match="valid_mask"):
        pixel_metrics(y, y, np.ones((2, 2), bool))
    with pytest.raises(ValueError, match="valid_mask"):
        pixel_metrics(y, y, np.ones_like(y))
    with pytest.raises(ValueError, match="Probabilities"):
        pixel_metrics(np.full_like(y, np.nan), y)
    with pytest.raises(ValueError, match="binary"):
        pixel_metrics(y, np.full_like(y, 2))
