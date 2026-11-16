"""Canonical input normalization for symbol images (docs/BACKUP_PLAN.md A4, PLAN_PHASE_C 1b).

The one-shot match compares a DigitizePID crop (raster, scanned-looking, a 10% margin of
context with pipes and text, any aspect ratio) with an ISO template (clean vector artwork cut
tight to its bbox). Both are then squashed to a square by `Resize((imgsz, imgsz))`, which
distorts the aspect ratio (template width / height spans 0.38-3.58). `ink_letterbox` brings
both sides to the same form before any transform:

1. grayscale -> Otsu binarization (ink black, background white): drops scan tone and grey
   levels, so only the shape is left;
2. tight cut around the ink of the connected components that reach the central part of the
   image (CENTER_FRAC), so pipes and text at the rim of a crop do not set the box, unless they
   are attached to the symbol (then the box keeps them: a known limit);
3. letterbox: pad with white to a square around that cut, keeping the aspect ratio, with
   MARGIN_FRAC on each side (the same margin as utils.crop_box.CROP_MARGIN_FRAC).

Applied IDENTICALLY to queries and templates; an encoder trained without it sees a different
input distribution under it (PLAN_PHASE_C: only the ImageNet-style encoders are a fair test
without retraining).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

MARGIN_FRAC = 0.10  # each side, as a fraction of the longer side of the ink box
CENTER_FRAC = 0.5  # the central box a component must touch to count as the symbol
MIN_CONTRAST = 32  # gray levels; below this max - min the image is treated as blank
MIN_COMPONENT_PX = 4  # specks smaller than this never set the box


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean ink mask of a uint8 grayscale image by Otsu; all False when the image is
    (nearly) uniform, where Otsu would split noise."""
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale array, got shape {gray.shape}")
    if int(gray.max()) - int(gray.min()) < MIN_CONTRAST:
        return np.zeros(gray.shape, dtype=bool)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray <= thr


def symbol_box(mask: np.ndarray, center_frac: float = CENTER_FRAC) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1), exclusive ends: union box of the ink components that overlap the
    central `center_frac` box; every component if none does. `mask` must hold some ink."""
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = [stats[i] for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX]
    if not comps:
        comps = [stats[i] for i in range(1, n)]
    if not comps:
        raise ValueError("symbol_box needs a mask with ink")
    cx0, cx1 = w * (1 - center_frac) / 2, w * (1 + center_frac) / 2
    cy0, cy1 = h * (1 - center_frac) / 2, h * (1 + center_frac) / 2

    def central(s) -> bool:
        x, y, bw, bh = (int(s[k]) for k in (0, 1, 2, 3))
        return x < cx1 and x + bw > cx0 and y < cy1 and y + bh > cy0

    keep = [s for s in comps if central(s)] or comps
    x0 = min(int(s[0]) for s in keep)
    y0 = min(int(s[1]) for s in keep)
    x1 = max(int(s[0] + s[2]) for s in keep)
    y1 = max(int(s[1] + s[3]) for s in keep)
    return x0, y0, x1, y1


def ink_letterbox(
    img: Image.Image, margin: float = MARGIN_FRAC, center_frac: float = CENTER_FRAC
) -> Image.Image:
    """Binarized symbol, cut to its ink box, letterboxed to a white square (RGB). A blank
    image comes back as a white square of its longer side."""
    gray = np.asarray(img.convert("L"))
    mask = ink_mask(gray)
    if not mask.any():
        side = max(gray.shape)
        return Image.new("RGB", (side, side), (255, 255, 255))
    x0, y0, x1, y1 = symbol_box(mask, center_frac)
    cut = np.where(mask[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    h, w = cut.shape
    side = max(1, int(round(max(h, w) * (1 + 2 * margin))))
    canvas = np.full((side, side), 255, dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top : top + h, left : left + w] = cut
    return Image.fromarray(canvas).convert("RGB")


# `oneshot.input_norm` values (null = off). Applied to templates (pid-iso-prototypes) and to
# every query crop (pipeline.oneshot_infer.OneShotClassifier); recorded in the prototype and
# CL2N-center files so a mismatch is refused, as with the encoder itself.
INPUT_NORMS = {"ink_letterbox": ink_letterbox}


def input_norm_fn(name):
    """The callable for an `oneshot.input_norm` value; None when off (null / empty)."""
    if not name:
        return None
    if str(name) not in INPUT_NORMS:
        raise ValueError(
            f"oneshot.input_norm must be null or one of {sorted(INPUT_NORMS)}, got {name!r}"
        )
    return INPUT_NORMS[str(name)]
