"""Masked pixel metrics; zero denominator => 0, including all-invalid input.

Accumulate counts before computing micro scores; never average batch F1.
Accept matching NumPy arrays or torch tensors; foreground=1, p > threshold.
"""
from dataclasses import dataclass
import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class PixelCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, probabilities, target, valid_mask=None, threshold=0.5):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and in [0,1]")
        p, y = _array(probabilities), _array(target)
        if p.shape != y.shape or p.ndim < 2:
            raise ValueError(f"Probabilities and target must have identical spatial shapes: {p.shape} vs {y.shape}")
        valid = np.ones(y.shape, dtype=bool) if valid_mask is None else _array(valid_mask)
        if valid.shape != y.shape or valid.dtype != np.bool_:
            raise ValueError("valid_mask must be boolean and exactly match target shape")
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Probabilities must be finite and in [0,1]")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Target must be binary (foreground=1)")
        pred, truth = p[valid] > threshold, y[valid] == 1
        self.tp += int(np.count_nonzero(pred & truth))
        self.fp += int(np.count_nonzero(pred & ~truth))
        self.fn += int(np.count_nonzero(~pred & truth))
        return self

    def scores(self):
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        denom = 2 * self.tp + self.fp + self.fn
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": precision, "recall": recall,
                "f1": 2 * self.tp / denom if denom else 0.0}


def pixel_metrics(probabilities, target, valid_mask=None, threshold=0.5):
    return PixelCounts().update(probabilities, target, valid_mask, threshold).scores()
