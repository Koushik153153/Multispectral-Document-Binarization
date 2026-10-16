from pathlib import Path
from collections import Counter, OrderedDict
from numbers import Integral
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from msbin_labels import decode_label, validate_fg_type


def _read_png(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except (ValueError, cv2.error) as exc:
        raise ValueError(f"Unreadable image: {path}") from exc
    if img is None:
        raise ValueError(f"Unreadable image: {path}")
    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image: {path}; got {img.dtype}")
    return img


def _read_png_gray(path: Path) -> np.ndarray:
    img = _read_png(path)
    if img.ndim != 2:
        raise ValueError(f"Expected grayscale band: {path}; got shape {img.shape}")
    return img


def _read_png_bgr(path: Path) -> np.ndarray:
    img = _read_png(path)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected three-channel BGR label: {path}; got shape {img.shape}")
    return img


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def patch_starts(length, patch_size, stride):
    """Cover an axis including its final edge; small axes start at zero."""
    for value, name in ((length, "length"), (patch_size, "patch_size"), (stride, "stride")):
        _positive_integer(value, name)
    if stride > patch_size:
        raise ValueError("stride must not exceed patch_size (would leave gaps)")
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def pad_to_size(img: np.ndarray, size: int, mode="constant") -> np.ndarray:
    """Bottom/right padding; constant zero by default for API compatibility."""
    _positive_integer(size, "size")
    if img.ndim not in (2, 3) or min(img.shape[:2]) == 0:
        raise ValueError("pad_to_size expects a nonempty 2D or 3D array")
    pads = [(0, max(0, size - img.shape[0])), (0, max(0, size - img.shape[1]))]
    if img.ndim == 3:
        pads.append((0, 0))
    return np.pad(img, pads, mode=mode)


def group_books(image_dir: Path):
    books = {}
    for p in sorted(Path(image_dir).glob("*.png")):
        m = re.fullmatch(r"(.+)_([0-9]+)", p.stem)
        if not m:
            raise ValueError(f"Malformed band filename: {p}")
        key, band = m[1], int(m[2])
        group = books.setdefault(key, {})
        if band in group:
            raise ValueError(f"Page {key}: duplicate band {band}: {p}")
        group[band] = p
    keys = sorted(k for k, bands in books.items() if set(bands) == set(range(12)))
    return books, keys


def read_page_keys(path):
    keys = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicates = sorted(k for k, n in Counter(keys).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate page keys in {path}: {duplicates}")
    if not keys:
        raise ValueError(f"Empty page-key list: {path}")
    return keys


class MSBinDibcoPatchDataset(Dataset):
    """Page-filtered patches; optional bool masks exclude UR and padding.

    No decoded-page cache is retained. Edge patches may overlap, so accumulated
    patch metrics weight some pixels multiple times and are not full-page scores.
    """
    def __init__(self, msbin_root: str, split="train", fg_type=1,
                 patch_size=256, stride=256, use_white_only=False,
                 min_fg_frac=0.002, max_patches_per_page=None,
                 keys_txt: str | None = None, return_valid_mask=False, cache_mb=0):
        validate_fg_type(fg_type)
        patch_starts(1, patch_size, stride)
        if not np.isfinite(min_fg_frac) or not 0 <= min_fg_frac <= 1:
            raise ValueError("min_fg_frac must be finite and between 0 and 1")
        if max_patches_per_page is not None:
            _positive_integer(max_patches_per_page, "max_patches_per_page")
        self.root = Path(msbin_root)
        self.split, self.fg_type = split, fg_type
        self.patch_size, self.stride = patch_size, stride
        self.use_white_only = bool(use_white_only)
        self.min_fg_frac = float(min_fg_frac)
        self.max_patches_per_page = max_patches_per_page
        self.return_valid_mask = bool(return_valid_mask)
        if not np.isfinite(cache_mb) or cache_mb < 0:
            raise ValueError("cache_mb must be finite and nonnegative")
        self.cache_limit = int(cache_mb * 1024 * 1024)
        self._cache, self.cache_bytes = OrderedDict(), 0
        self.cache_hits, self.cache_misses = 0, 0
        self.image_dir, self.label_dir = self.root / split / "images", self.root / split / "labels"
        self.books, _ = group_books(self.image_dir)
        self.keys = sorted(read_page_keys(keys_txt)) if keys_txt is not None else sorted(self.books)
        unknown = sorted(set(self.keys) - set(self.books))
        if unknown:
            raise ValueError(f"Unknown requested page keys in {self.image_dir}: {unknown}")
        self.index, self.page_shapes = [], {}
        for key in self.keys:
            bands = self.books[key]
            missing, extra = set(range(12)) - set(bands), set(bands) - set(range(12))
            if missing or extra:
                raise ValueError(f"Page {key}: missing bands {sorted(missing)}, unexpected bands {sorted(extra)}")
            target, valid = self._read_target(key)
            h, w = target.shape
            self.page_shapes[key] = (h, w)
            # Decode sequentially during indexing; retain only shapes and positions.
            for path in bands.values():
                self._check_shape(key, _read_png_gray(path).shape, (h, w), path)
            positions = []
            for y in patch_starts(h, patch_size, stride):
                for x in patch_starts(w, patch_size, stride):
                    mask = valid[y:y + patch_size, x:x + patch_size]
                    count = np.count_nonzero(mask)
                    if count == 0:
                        continue
                    crop = target[y:y + patch_size, x:x + patch_size]
                    if crop[mask].sum() / count >= self.min_fg_frac:
                        positions.append((key, y, x))
            if max_patches_per_page is not None and len(positions) > max_patches_per_page:
                step = max(1, len(positions) // max_patches_per_page)
                positions = positions[::step][:max_patches_per_page]
            self.index.extend(positions)
        if not self.index:
            raise RuntimeError(f"Dataset empty: no valid patches for {self.image_dir}; check keys, validity and foreground filtering")

    @staticmethod
    def _check_shape(key, actual, expected, path):
        if actual != expected:
            raise ValueError(f"Page {key}: mismatched dimensions at {path}: {actual}, expected {expected}")

    def _read_target(self, key):
        path = self.label_dir / f"{key}.png"
        try:
            return decode_label(_read_png_bgr(path), self.fg_type)
        except ValueError as exc:
            raise ValueError(f"Page {key}, label {path}: {exc}") from exc

    def _gt_to_fgmask(self, gt_bgr):
        # Legacy helper preserves target-only callers, with strict shared decoding.
        target, _ = decode_label(gt_bgr, self.fg_type)
        return pad_to_size(target, self.patch_size)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key, y, x = self.index[idx]
        images = None
        if self.cache_limit:
            images, target, valid = self._page(key)
        else:
            # Preserve the original sequential band decode and peak memory when disabled.
            self.cache_misses += 1
            target, valid = self._read_target(key)
            self._check_shape(key, target.shape, self.page_shapes[key], self.label_dir / f"{key}.png")
        crop = (slice(y, y + self.patch_size), slice(x, x + self.patch_size))
        channels = []
        for channel, band in enumerate([0] if self.use_white_only else range(12)):
            path = self.books[key][band]
            image = images[channel] if images is not None else _read_png_gray(path)
            self._check_shape(key, image.shape, self.page_shapes[key], path)
            channels.append(pad_to_size(image[crop], self.patch_size, mode="edge").astype(np.float32) / 255.0)
        image = torch.from_numpy(np.stack(channels))
        target = torch.from_numpy(pad_to_size(target[crop], self.patch_size).copy()).unsqueeze(0)
        valid = torch.from_numpy(pad_to_size(valid[crop], self.patch_size).copy()).unsqueeze(0)
        return (image, target, valid) if self.return_valid_mask else (image, target)

    def _page(self, key):
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.cache_misses += 1
        target, valid = self._read_target(key)
        expected = self.page_shapes[key]
        self._check_shape(key, target.shape, expected, self.label_dir / f"{key}.png")
        images = []
        for band in ([0] if self.use_white_only else range(12)):
            path = self.books[key][band]
            image = _read_png_gray(path)
            self._check_shape(key, image.shape, expected, path)
            images.append(image)
        size = sum(image.nbytes for image in images) + target.nbytes + valid.nbytes
        if size <= self.cache_limit:
            while self._cache and self.cache_bytes + size > self.cache_limit:
                _, old = self._cache.popitem(last=False)
                self.cache_bytes -= sum(a.nbytes for a in old[0]) + old[1].nbytes + old[2].nbytes
            self._cache[key] = (images, target, valid)
            self.cache_bytes += size
        return images, target, valid


group_pages = group_books
