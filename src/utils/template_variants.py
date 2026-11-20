"""Drawing-like variants of an ISO template (PLAN_PHASE_C §2D.1 "H2", PLAN_STAGE2_ONESHOT "B9").

A DigitizePID crop is not the clean, tight template: pipes run from the symbol's ports to the
crop edge, strokes are thicker or thinner, the scan blurs and pixelates. H2 makes the template
side look more like that instead of training anything: each template gives `1 + k` images, the
clean one and `k` "dirty" ones, and its prototype is the mean of their embeddings (domain
randomization, Tobin et al. 2017; the legend-augment views of Hain et al. 2025).

A dirty view, drawn with a seeded `random.Random`:

1. the template padded by CROP_MARGIN_FRAC of its ink box per side (the crop's own margin);
2. a pipe at each REAL port (`utils.iso_templates`, side + position), each port with
   probability STUB_PROB, at least one: the pipe axis is the port's position across its side,
   mapped through the PNG's ink box (the SVG and the PNG frame the same drawing, but not to the
   pixel); the pipe starts at the first ink met walking in from that side along the axis (the
   port's own position when the axis meets no ink) and runs out past the ink box by
   STUB_REACH x the margin; its width is the measured stroke width;
3. strokes thickened or thinned by one pixel (MORPH_PX; thinning skipped on strokes <= 3 px);
4. `utils.scan_degrade.ScanDegrade(p=1)` on the narrow DEGRADE_RANGES (no morph: step 3 does it).

No off-centre shift (B9 point 3): A4 (`utils.input_norm.ink_letterbox`), applied after, cuts to
the ink again and would undo it. A template without ports (`pipeline`) gets steps 3-4 only.
Constants are fixed before any score (§2D.1: "K = 8, không dò").
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from utils.crop_box import CROP_MARGIN_FRAC
from utils.input_norm import ink_mask
from utils.scan_degrade import ScanDegrade, stroke_width

K_VIEWS = 8
STUB_PROB = 0.8
STUB_REACH = (0.3, 1.0)  # how far past the ink box a pipe runs, as a share of the margin
MORPH_PX = (-1, 0, 1)
MIN_THIN_STROKE = 3  # px; thinner strokes are never thinned (they would break)
DEGRADE_RANGES = {
    "morph": (0.0, 0.0),
    "blur": (0.0, 0.3),
    "pixelate": (0.0, 0.3),
    "lighting": (0.0, 0.1),
    "noise_sigma": (0.0, 6.0),
    "jpeg_quality": (70.0, 90.0),
}
SIDES = ("L", "R", "T", "B")


@dataclass(frozen=True)
class PipeAxis:
    """A port: the side a pipe leaves from and its position across that side, as a share of
    the ink box (v for L / R, u for T / B), plus the port's own position along the pipe."""

    side: str
    at: float
    depth: float  # u for L / R, v for T / B: where the port sits along the pipe direction

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")


def axes_of(template) -> list[PipeAxis]:
    """The pipe axes of a `utils.iso_templates.Template` (None -> no ports)."""
    if template is None:
        return []
    return [
        PipeAxis(p.side, p.v if p.side in "LR" else p.u, p.u if p.side in "LR" else p.v)
        for p in template.ports
    ]


def pick_template(candidates: Sequence, aspect: float):
    """Of the SVG drawings of one REG#, the one whose w / h is closest to `aspect` (log scale);
    None when there is none."""
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(np.log((t.w_mm / t.h_mm) / aspect)))


def ink_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1), exclusive ends, of all ink."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("ink_box needs a mask with ink")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def contact(mask: np.ndarray, box, axis: PipeAxis) -> tuple[int, int]:
    """(x, y) where a pipe coming in from `axis.side` meets the symbol: the first ink pixel on
    the axis line walking inward from the box edge; the port's own position when the line meets
    no ink."""
    x0, y0, x1, y1 = box
    if axis.side in "LR":
        y = min(y1 - 1, max(y0, int(round(y0 + axis.at * (y1 - y0)))))
        line = np.flatnonzero(mask[y, x0:x1])
        fallback = int(round(x0 + axis.depth * (x1 - x0)))
        if not len(line):
            return min(x1 - 1, max(x0, fallback)), y
        return (x0 + int(line[0]) if axis.side == "L" else x0 + int(line[-1])), y
    x = min(x1 - 1, max(x0, int(round(x0 + axis.at * (x1 - x0)))))
    line = np.flatnonzero(mask[y0:y1, x])
    fallback = int(round(y0 + axis.depth * (y1 - y0)))
    if not len(line):
        return x, min(y1 - 1, max(y0, fallback))
    return x, (y0 + int(line[0]) if axis.side == "T" else y0 + int(line[-1]))


def draw_pipe(gray: np.ndarray, start: tuple[int, int], side: str, end: int, width: int) -> None:
    """A black bar of `width` from `start` outward on `side` to coordinate `end` (x for L / R,
    y for T / B), in place."""
    x, y = start
    lo, hi = width // 2, width - width // 2
    if side == "L":
        gray[max(0, y - lo) : y + hi, max(0, end) : x + 1] = 0
    elif side == "R":
        gray[max(0, y - lo) : y + hi, x : end + 1] = 0
    elif side == "T":
        gray[max(0, end) : y + 1, max(0, x - lo) : x + hi] = 0
    else:
        gray[y : end + 1, max(0, x - lo) : x + hi] = 0


def _pad(gray: np.ndarray, box) -> tuple[np.ndarray, tuple[int, int, int, int], int, int]:
    """`gray` on a white canvas grown by the crop margin of its ink box per side."""
    x0, y0, x1, y1 = box
    mx = int(round(CROP_MARGIN_FRAC * (x1 - x0))) + 1
    my = int(round(CROP_MARGIN_FRAC * (y1 - y0))) + 1
    canvas = np.full((gray.shape[0] + 2 * my, gray.shape[1] + 2 * mx), 255, dtype=np.uint8)
    canvas[my : my + gray.shape[0], mx : mx + gray.shape[1]] = gray
    return canvas, (x0 + mx, y0 + my, x1 + mx, y1 + my), mx, my


def dirty_view(
    img: Image.Image,
    axes: Sequence[PipeAxis],
    rng: random.Random,
    *,
    pipes: bool = True,
    morph: bool = True,
    degrade: bool = True,
) -> Image.Image:
    """One drawing-like variant of a template (steps 1-4 of the module doc), RGB.

    `pipes` / `morph` / `degrade` switch steps 2 / 3 / 4 off (PLAN_PHASE_C §2K diagnosis); with
    all three on (the default) the output and the rng draws are exactly H2's."""
    gray = np.asarray(img.convert("L"))
    mask = ink_mask(gray)
    if not mask.any():
        return img.convert("RGB")
    w = stroke_width(gray) or max(1.0, max(gray.shape) / 60)
    canvas, box, mx, my = _pad(gray, ink_box(mask))
    cmask = ink_mask(canvas)
    if axes and pipes:
        on = [rng.random() < STUB_PROB for _ in axes]
        if not any(on):
            on[rng.randrange(len(axes))] = True
        x0, y0, x1, y1 = box
        for axis, keep in zip(axes, on, strict=True):
            if not keep:
                continue
            reach = rng.uniform(*STUB_REACH)
            px, py = contact(cmask, box, axis)
            end = {
                "L": x0 - int(round(reach * mx)),
                "R": x1 - 1 + int(round(reach * mx)),
                "T": y0 - int(round(reach * my)),
                "B": y1 - 1 + int(round(reach * my)),
            }[axis.side]
            draw_pipe(canvas, (px, py), axis.side, end, max(1, int(round(w))))
    if morph:
        step = rng.choice(MORPH_PX)
        if step < 0 and w <= MIN_THIN_STROKE:
            step = 0
        if step:
            kernel = np.ones((2, 2), np.uint8)
            # ink is dark: eroding the image thickens it, dilating thins it
            canvas = cv2.erode(canvas, kernel) if step > 0 else cv2.dilate(canvas, kernel)
    if degrade:
        scan = ScanDegrade(p=1.0, ranges=DEGRADE_RANGES, seed=rng.getrandbits(32))
        canvas = scan.apply(canvas, w)
    return Image.fromarray(canvas).convert("RGB")


def template_seed(seed: int, name: str) -> int:
    """A per-template seed that does not depend on the template order."""
    return (seed * 1_000_003) ^ zlib.crc32(name.encode())


def template_views(
    img: Image.Image, axes: Sequence[PipeAxis], k: int = K_VIEWS, seed: int = 0, name: str = ""
) -> list[Image.Image]:
    """[clean] + `k` dirty views of one template, reproducible from (seed, name)."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    rng = random.Random(template_seed(seed, name))
    return [img.convert("RGB")] + [dirty_view(img, axes, rng) for _ in range(k)]
