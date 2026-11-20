"""Crop geometry shared by crop extraction, the FGC dataset and FGC inference.

One rule everywhere: a symbol is classified on its box grown by CROP_MARGIN_FRAC of its own
size per side, cut from the image and clipped to the image edges (`cut`). Before this module
the training crops had that 10% margin and the inference crops (`fgc_infer`) had none.

Stored context. `pid-prep-crops` does not store that view itself: it stores a wider region
(CROP_CONTEXT_FRAC per side, `context_region`) and, per class folder, a `_boxes.csv` with the
symbol box and the region's origin, both in source-image pixels (`StoredCrop`). The dataset
then cuts the classification view out of it (`view`): exactly the `cut` view at val / test
time (same float box, same bounds, so the same pixels as at inference), and a jittered one at
train time.

Box jitter (`fgc.aug.box_jitter`, off unless p > 0; `BoxJitter`). A detector box is not the
ground-truth box: it is shifted, looser or tighter. Training on views cut around a jittered
box teaches the classifier that, from the real surroundings of the symbol (pipe stubs,
neighbours) instead of padding. The centre moves by up to `shift` of the box size, the size
scales log-uniformly in `scale`, the aspect ratio log-uniformly in `aspect`. The ranges are
checked to keep the view inside the stored context.

    pid-fgc-train ++fgc.aug.box_jitter.p=0.5
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.augment import SeededRandom

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

# Margin around a box, per side, as a fraction of the box's own size. A little context
# (stems, adjoining lines) helps tell fine classes apart; 10% does not drag in neighbouring
# symbols at typical P&ID spacing.
CROP_MARGIN_FRAC = 0.10
# Region stored per symbol by pid-prep-crops: wide enough for every jittered view.
CROP_CONTEXT_FRAC = 0.40
BOXES_CSV = "_boxes.csv"
_BOXES_FIELDS = ("file", "x1", "y1", "x2", "y2", "ox", "oy")


def grow(box: Box, frac: float) -> Box:
    """`box` grown by `frac` of its own width / height on each side."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * frac, (y2 - y1) * frac
    return (x1 - mx, y1 - my, x2 + mx, y2 + my)


def _floor(v: float) -> int:
    return math.floor(round(v, 6))  # 1e-6: float noise must not move a pixel edge


def _ceil(v: float) -> int:
    return math.ceil(round(v, 6))


def region(box: Box, frac: float, bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Integer pixel region covering `box` grown by `frac`, clipped to `bounds` (x1, y1, x2, y2;
    end-exclusive, as for slicing)."""
    bx1, by1, bx2, by2 = bounds
    x1, y1, x2, y2 = grow(box, frac)
    return (
        min(max(_floor(x1), bx1), bx2),
        min(max(_floor(y1), by1), by2),
        min(max(_ceil(x2), bx1), bx2),
        min(max(_ceil(y2), by1), by2),
    )


def cut(image: np.ndarray, box: Box, frac: float = CROP_MARGIN_FRAC) -> np.ndarray:
    """`image` (H, W[, C]) cut to `box` grown by `frac`, clipped to the image: the view a
    symbol is classified on."""
    x1, y1, x2, y2 = region(box, frac, (0, 0, image.shape[1], image.shape[0]))
    return image[y1:y2, x1:x2]


def context_region(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """The region pid-prep-crops stores for `box` in a `width` x `height` image."""
    return region(box, CROP_CONTEXT_FRAC, (0, 0, width, height))


@dataclass(frozen=True)
class StoredCrop:
    """Where a stored context crop sits: the symbol `box` and the crop's top-left `origin`,
    both in source-image pixels."""

    box: Box
    origin: tuple[int, int]


def view(
    stored: Image.Image | np.ndarray,
    where: StoredCrop,
    box: Box | None = None,
    frac: float = CROP_MARGIN_FRAC,
) -> Image.Image | np.ndarray:
    """The classification view cut out of a stored context crop: around `box` (default: the
    stored symbol box; a jittered box at train time) grown by `frac`, clipped to the stored
    region. With the stored box it equals `cut` on the source image."""
    ox, oy = where.origin
    w, h = stored.size if isinstance(stored, Image.Image) else (stored.shape[1], stored.shape[0])
    x1, y1, x2, y2 = region(where.box if box is None else box, frac, (ox, oy, ox + w, oy + h))
    if isinstance(stored, Image.Image):
        return stored.crop((x1 - ox, y1 - oy, x2 - ox, y2 - oy))
    return stored[y1 - oy : y2 - oy, x1 - ox : x2 - ox]


def write_boxes(class_dir: Path, rows: list[tuple[str, StoredCrop]]) -> None:
    """Write `class_dir/_boxes.csv` (one row per stored crop file name)."""
    with (class_dir / BOXES_CSV).open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(_BOXES_FIELDS)
        for name, s in rows:
            out.writerow([name, *(repr(float(v)) for v in s.box), *s.origin])


def read_boxes(class_dir: Path) -> dict[str, StoredCrop]:
    """`class_dir/_boxes.csv` as {file name: StoredCrop}; empty when the file is absent
    (crops cut before the stored-context format: used as they are, never jittered)."""
    path = class_dir / BOXES_CSV
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {
            r["file"]: StoredCrop(
                box=(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
                origin=(int(r["ox"]), int(r["oy"])),
            )
            for r in csv.DictReader(f)
        }


def _pair(node: Mapping[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    lo, hi = (float(v) for v in node.get(key, default))
    if not 0.0 < lo <= hi:
        raise ValueError(f"box_jitter.{key} must be a range 0 < lo <= hi, got {[lo, hi]}")
    return lo, hi


class BoxJitter(SeededRandom):
    """Training augment: with probability `p`, a box shifted / rescaled like a detector box.

    Args:
        p: probability of jittering a given box; 0 disables.
        shift: largest centre shift, as a fraction of the box width / height.
        scale: (lo, hi) size factor, drawn log-uniformly.
        aspect: (lo, hi) width / height factor, drawn log-uniformly (width * sqrt(a),
            height / sqrt(a)).
        seed: fixed seed for tests; None uses the global `random` module, which torch's
            DataLoader reseeds per worker.
    """

    DEFAULTS: dict[str, Any] = {"shift": 0.08, "scale": (0.9, 1.15), "aspect": (0.9, 1.1)}

    def __init__(
        self,
        p: float = 0.5,
        ranges: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        node = dict(ranges or {})
        unknown = set(node) - set(self.DEFAULTS)
        if unknown:
            known = sorted(self.DEFAULTS)
            raise ValueError(f"unknown box_jitter keys {sorted(unknown)}; known: {known}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"box_jitter.p must be in [0, 1], got {p}")
        self.p = p
        self.shift = float(node.get("shift", self.DEFAULTS["shift"]))
        if self.shift < 0:
            raise ValueError(f"box_jitter.shift must be >= 0, got {self.shift}")
        self.scale = _pair(node, "scale", self.DEFAULTS["scale"])
        self.aspect = _pair(node, "aspect", self.DEFAULTS["aspect"])
        # Farthest a view edge can reach from the box centre, in box sizes; the stored
        # context reaches 0.5 + CROP_CONTEXT_FRAC.
        stretch = self.scale[1] * max(math.sqrt(self.aspect[1]), 1 / math.sqrt(self.aspect[0]))
        reach = 0.5 * stretch * (1 + 2 * CROP_MARGIN_FRAC) + self.shift
        if reach > 0.5 + CROP_CONTEXT_FRAC:
            raise ValueError(
                f"box_jitter ranges reach {reach:.3f} box sizes from the centre, beyond the "
                f"stored context ({0.5 + CROP_CONTEXT_FRAC}); narrow shift / scale / aspect"
            )
        self._rng = random.Random(seed) if seed is not None else random

    @classmethod
    def from_cfg(cls, node: Mapping[str, Any] | None) -> BoxJitter | None:
        """From a `cfg.fgc.aug.box_jitter` node: None when the node is absent or p is 0."""
        if node is None:
            return None
        values = dict(node)
        p = float(values.pop("p", 0.0))
        return cls(p, values) if p > 0 else None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(p={self.p}, shift={self.shift}, scale={self.scale}, "
            f"aspect={self.aspect})"
        )

    def _log_uniform(self, lo: float, hi: float) -> float:
        return math.exp(self._rng.uniform(math.log(lo), math.log(hi)))

    def __call__(self, box: Box) -> Box:
        if not (self.p > 0.0 and self._rng.random() < self.p):
            return box
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2 + self._rng.uniform(-self.shift, self.shift) * w
        cy = (y1 + y2) / 2 + self._rng.uniform(-self.shift, self.shift) * h
        s = self._log_uniform(*self.scale)
        a = math.sqrt(self._log_uniform(*self.aspect))
        hw, hh = w * s * a / 2, h * s / a / 2
        return (cx - hw, cy - hh, cx + hw, cy + hh)
