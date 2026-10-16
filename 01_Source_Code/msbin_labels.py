"""MSBin BGR labels: selected foreground=1; UR is always invalid."""
import numpy as np

FG1 = (255, 255, 255)
FG2 = (122, 122, 122)
BG = (0, 0, 0)
UR = (255, 0, 0)


def validate_fg_type(fg_type):
    if isinstance(fg_type, (bool, np.bool_)) or fg_type not in (1, 2):
        raise ValueError(f"fg_type must be 1 (main text) or 2 (red ink), got {fg_type!r}")


def decode_label(label_bgr, fg_type=1):
    """Return float32 target and bool validity (H,W); other ink is valid BG."""
    validate_fg_type(fg_type)
    label = np.asarray(label_bgr)
    if label.ndim != 3 or label.shape[2] != 3 or min(label.shape[:2]) == 0:
        raise ValueError(f"Expected nonempty (H,W,3) BGR label, got {label.shape}")
    if label.dtype != np.uint8:
        raise ValueError(f"Expected uint8 BGR label, got {label.dtype}")
    known = np.zeros(label.shape[:2], dtype=bool)
    for color in (FG1, FG2, BG, UR):
        known |= np.all(label == color, axis=-1)
    if not known.all():
        colors = np.unique(label[~known], axis=0)
        raise ValueError(f"Unexpected BGR label colors: {colors[:16].tolist()} ({len(colors)} distinct)")
    target = np.all(label == (FG1 if fg_type == 1 else FG2), axis=-1).astype(np.float32)
    valid = ~np.all(label == UR, axis=-1)
    return target, valid
