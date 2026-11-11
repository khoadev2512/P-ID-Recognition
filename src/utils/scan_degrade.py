"""Scan-like degradation as a training augment, for both stages (off unless configured).

The V7 page degradation (`utils.iso_degrade`) re-used online: a training image is degraded
when it is loaded, with fresh parameters each time, instead of training on stored degraded
images. Only the steps that leave coordinates where they were are kept: stroke dilation /
erosion, pixelation, blur, uneven lighting, noise and JPEG. Rotation and shear are left out,
because they would move the boxes; YOLO has its own geometric augment.

Strength follows the stroke width `w`, as in V7: |morph| <= w / 2, blur sigma <= w / 2,
pixelation block <= w / 2, and a line of width `w` keeps `iso_degrade.MIN_PEAK` of its
darkness (`iso_degrade.clamp_params`). V7 knows `w` (it drew the page); here it is measured
(`stroke_width`): the local width of each ink pixel is the shorter of its horizontal and
vertical ink runs, and `w` is a low percentile of those, so the thinnest common strokes of a
page (DigitizePID mixes 2 px and 6 px lines) are the ones the limits protect.

Config, read from `cfg.detector.aug.degrade` / `cfg.fgc.aug.degrade` (absent: off). Ranges
are those of `render_iso.yaml`'s `degrade:` and mean the same (morph / blur / pixelate as
fractions of the limits for `w`), so a training image is degraded like the V7 test pages:

    pid-detect-train +detector.aug.degrade.p=0.5
    pid-fgc-train    +fgc.aug.degrade.p=0.5 '+fgc.aug.degrade.noise_sigma=[0,6]'
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from PIL import Image

from utils.augment import SeededRandom
from utils.iso_degrade import DegradeParams, degrade_image, params_from

# The V7 ranges (configs/render_iso.yaml `degrade:`), rotation and shear left out.
DEFAULTS: dict[str, tuple[float, float]] = {
    "morph": (-1.0, 0.4),
    "blur": (0.0, 0.35),
    "pixelate": (0.0, 0.5),
    "lighting": (0.0, 0.2),
    "noise_sigma": (0.0, 10.0),
    "jpeg_quality": (55.0, 90.0),
}
_BOUNDS = {
    "morph": (-1.0, 1.0),
    "blur": (0.0, 1.0),
    "pixelate": (0.0, 1.0),
    "lighting": (0.0, 0.9),
    "noise_sigma": (0.0, 64.0),
    "jpeg_quality": (1.0, 100.0),
}

STROKE_PERCENTILE = 10  # of the local widths: the thin end of the strokes on the image
_MIN_CONTRAST = 60  # paper minus ink, gray levels; less: nothing to measure, left as it is
_MIN_INK_PX = 50


def _runs(mask: np.ndarray) -> np.ndarray:
    """For each True pixel, the length of the horizontal run of True it belongs to (0 elsewhere)."""
    h, w = mask.shape
    pad = np.zeros((h, w + 2), np.int8)  # a False column on each side: no run crosses a row
    pad[:, 1:-1] = mask
    flat = pad.ravel()
    start = np.diff(flat, prepend=0) == 1
    starts = np.flatnonzero(start)
    ends = np.flatnonzero(np.diff(flat, append=0) == -1)
    run = np.cumsum(start) - 1
    out = np.where(flat > 0, (ends - starts + 1)[np.maximum(run, 0)], 0)
    return out.reshape(h, w + 2)[:, 1:-1]


def _quantile(counts: np.ndarray, q: float) -> int:
    """The smallest value whose cumulative count reaches the share `q` (counts by value)."""
    return int(np.searchsorted(np.cumsum(counts), q * counts.sum()))


def stroke_width(gray: np.ndarray, percentile: float = STROKE_PERCENTILE) -> float | None:
    """The stroke width of a gray uint8 image (dark ink on lighter paper of any shade), in px:
    the `percentile` of the per-pixel local width (shorter of the horizontal and vertical ink
    run). None when the image has too little ink or contrast to tell."""
    levels = np.bincount(gray.ravel(), minlength=256)
    paper, ink_level = _quantile(levels, 0.5), _quantile(levels, 0.005)
    if paper - ink_level < _MIN_CONTRAST:
        return None
    ink = gray < (paper + ink_level) / 2
    if int(ink.sum()) < _MIN_INK_PX:
        return None
    local = np.minimum(_runs(ink), _runs(ink.T).T)[ink]
    return float(_quantile(np.bincount(local), percentile / 100))


def gray_of(img: np.ndarray) -> np.ndarray:
    """uint8 gray (H, W) of a gray (H, W), one-channel or BGR image."""
    if img.ndim == 2:
        return img
    return img[:, :, 0] if img.shape[2] == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _range(node: Mapping[str, Any], key: str) -> tuple[float, float]:
    lo, hi = (float(v) for v in node.get(key, DEFAULTS[key]))
    b_lo, b_hi = _BOUNDS[key]
    if not b_lo <= lo <= hi <= b_hi:
        raise ValueError(f"degrade.{key} must be a range inside [{b_lo}, {b_hi}], got {[lo, hi]}")
    return lo, hi


class ScanDegrade(SeededRandom):
    """Training augment: with probability `p`, the image degraded like a V7 page (no geometry).

    `__call__` takes and returns a PIL image (Stage 2 crops, on the native crop before
    Resize); `array` takes and returns a uint8 gray or BGR array. Stage 1 calls `fires` and
    `apply` itself, so it measures the stroke width of a training image only once.
    Colour is dropped: the image is degraded in gray and given back with its channels.

    Args:
        p: probability of degrading a given image; 0 disables.
        morph, blur, pixelate, lighting, noise_sigma, jpeg_quality: (lo, hi) ranges, drawn
            uniformly; see the module doc and `DEFAULTS`.
        seed: fixed seed for reproducible tests. None uses the global `random` module, which
            torch's DataLoader reseeds per worker (see `augment.SubpartWarp`).
    """

    def __init__(
        self,
        p: float = 0.5,
        ranges: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        unknown = set(ranges or {}) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown degrade keys {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        self.p = p
        self.ranges = {k: _range(ranges or {}, k) for k in DEFAULTS}
        self._rng = random.Random(seed) if seed is not None else random

    @classmethod
    def from_cfg(cls, node: Mapping[str, Any] | None) -> ScanDegrade | None:
        """From a `cfg.<stage>.aug.degrade` node: None when the node is absent or p is 0."""
        if node is None:
            return None
        values = dict(node)
        p = float(values.pop("p", 0.0))
        return cls(p, values) if p > 0 else None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p}, ranges={self.ranges})"

    def params(self, w: float) -> DegradeParams:
        """One draw for stroke width `w`, held to the V7 limits."""
        u = {k: self._rng.uniform(*r) for k, r in self.ranges.items()}
        return params_from(
            {**u, "rotate_deg": 0.0, "shear_deg": 0.0}, w, self._rng.getrandbits(62),
            u["jpeg_quality"],
        )  # fmt: skip

    def fires(self) -> bool:
        """Whether to degrade the next image (probability p)."""
        return self.p > 0.0 and self._rng.random() < self.p

    def apply(self, img: np.ndarray, w: float | None) -> np.ndarray:
        """`img` (uint8 gray (H, W) or BGR (H, W, C)) degraded for stroke width `w`, same shape;
        unchanged when `w` is None (nothing measured)."""
        if w is None:
            return img
        out, _, _ = degrade_image(gray_of(img), self.params(w))
        return out if img.ndim == 2 else np.repeat(out[:, :, None], img.shape[2], axis=2)

    def array(self, img: np.ndarray) -> np.ndarray:
        """`img` degraded with probability p, its stroke width measured on it (see `apply`)."""
        return self.apply(img, stroke_width(gray_of(img))) if self.fires() else img

    def __call__(self, img: Image.Image) -> Image.Image:
        mode = img.mode
        gray = np.asarray(img.convert("L"))
        out = self.array(gray)
        if out is gray:
            return img
        result = Image.fromarray(out, mode="L")
        return result if mode == "L" else result.convert(mode)
