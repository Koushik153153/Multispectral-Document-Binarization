"""Synthetic tiny fixtures test audit behavior, never research performance."""
import importlib.util
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_msbin_dataset.py"
SPEC = importlib.util.spec_from_file_location("validate_msbin_dataset", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def page(root, split, key):
    for band in range(12):
        write_image(root / split / "images" / f"{key}_{band}.png", np.full((8, 9), band, np.uint8))
    label = np.zeros((8, 9, 3), np.uint8)
    label[0] = (255, 255, 255)
    label[1] = (122, 122, 122)
    label[2] = (255, 0, 0)
    write_image(root / split / "labels" / f"{key}.png", label)


def copy_page(root, key, destination):
    for folder in ("images", "labels"):
        target = root / "train_val_split" / destination / folder
        target.mkdir(parents=True, exist_ok=True)
        pattern = f"{key}_*.png" if folder == "images" else f"{key}.png"
        for source in (root / "train" / folder).glob(pattern):
            shutil.copyfile(source, target / source.name)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset with spaces"
    page(root, "train", "BOOK_A")
    page(root, "train", "BOOK_B")
    page(root, "test", "BOOK_TEST")
    copy_page(root, "BOOK_A", "train")
    copy_page(root, "BOOK_B", "val")
    return root


def test_valid_complete_pages_and_publication(dataset, tmp_path):
    report = audit.audit_dataset(dataset)
    assert report["valid"], report["errors"]
    assert report["assigned_counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["content_checks"]["compared"] == 26
    assert report["datasets"]["train"]["decoded_files"] == 26
    assert report["datasets"]["train"]["pages"]["BOOK_A"]["height"] == 8
    assert report["datasets"]["train"]["label_colors_bgr"]["(255, 0, 0)"] == 18
    assert audit.publish(report, tmp_path / "report", tmp_path / "splits")
    assert (tmp_path / "splits" / "train_keys.txt").read_bytes() == b"BOOK_A\n"
    assert (tmp_path / "splits" / "val_keys.txt").read_bytes() == b"BOOK_B\n"
    assert (tmp_path / "splits" / "test_keys.txt").read_bytes() == b"BOOK_TEST\n"
    assert len((tmp_path / "report" / "dataset_manifest.csv").read_text().splitlines()) == 4
    assert audit.publish(report, tmp_path / "report", tmp_path / "splits")


@pytest.mark.parametrize("damage,expected", [
    ("missing", "missing bands [4]"),
    ("dimensions", "mismatched dimensions"),
    ("unreadable", "Unreadable image"),
    ("rgb", "expected grayscale"),
    ("color", "unexpected BGR colors"),
    ("label", "missing label"),
    ("extra", "unexpected band indices"),
    ("malformed", "Malformed band filename"),
    ("orphan", "label without input page"),
])
def test_rejects_invalid_data_without_publishing(dataset, tmp_path, damage, expected):
    path = dataset / "train" / "images" / "BOOK_A_4.png"
    if damage == "missing":
        path.unlink()
    elif damage == "dimensions":
        write_image(path, np.zeros((7, 9), np.uint8))
    elif damage == "unreadable":
        path.write_bytes(b"not a PNG")
    elif damage == "rgb":
        write_image(path, np.zeros((8, 9, 3), np.uint8))
    elif damage == "color":
        write_image(dataset / "train/labels/BOOK_A.png", np.full((8, 9, 3), 17, np.uint8))
    elif damage == "label":
        (dataset / "train/labels/BOOK_A.png").unlink()
    elif damage == "extra":
        shutil.copyfile(path, path.with_name("BOOK_A_12.png"))
    elif damage == "malformed":
        shutil.copyfile(path, path.with_name("bad.png"))
    elif damage == "orphan":
        shutil.copyfile(dataset / "train/labels/BOOK_A.png", dataset / "train/labels/ORPHAN.png")
    report = audit.audit_dataset(dataset)
    assert not report["valid"]
    assert any(expected in error for error in report["errors"])
    assert not audit.publish(report, tmp_path / "report", tmp_path / "splits")
    assert not (tmp_path / "splits").exists()
    assert not (tmp_path / "report/dataset_manifest.csv").exists()
    assert (tmp_path / "report/dataset_audit.json").exists()


def test_overlapping_supplied_keys(dataset):
    copy_page(dataset, "BOOK_A", "val")
    report = audit.audit_dataset(dataset)
    assert not report["valid"]
    assert report["split_checks"]["supplied_train_val_overlap"] == ["BOOK_A"]


def test_supplied_content_must_match(dataset):
    write_image(dataset / "train_val_split/val/images/BOOK_B_3.png", np.full((8, 9), 200, np.uint8))
    report = audit.audit_dataset(dataset)
    assert not report["valid"]
    assert report["content_checks"]["compared"] == 26
    assert len(report["content_checks"]["mismatches"]) == 1
    assert any("SHA-256 content differs" in error for error in report["errors"])


def test_conflicting_existing_lists_are_preserved(dataset, tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    existing = splits / "val_keys.txt"
    existing.write_bytes(b"different\n")
    report = audit.audit_dataset(dataset)
    assert not audit.publish(report, tmp_path / "report", splits)
    assert existing.read_bytes() == b"different\n"
    assert not (splits / "train_keys.txt").exists()
    assert not (splits / "test_keys.txt").exists()


def test_cli_failure_exit_code(dataset, tmp_path):
    (dataset / "test/images/BOOK_TEST_0.png").unlink()
    assert audit.main(["--msbin-root", str(dataset), "--output-dir", str(tmp_path / "report"),
                       "--splits-dir", str(tmp_path / "splits")]) == 1


def test_cli_rejects_output_inside_dataset(dataset):
    with pytest.raises(SystemExit) as exc:
        audit.main(["--msbin-root", str(dataset), "--output-dir", str(dataset / "audit")])
    assert exc.value.code != 0
