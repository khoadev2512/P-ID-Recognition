"""Per-subpart augmentation for FGC training crops (OSSR-PID technique #2, raster form).

OSSR-PID (arXiv 2109.03849) affine-transforms EACH subpart of a symbol independently so
the encoder learns that a part may drift slightly (a centre dot, an actuator, the two
cones of a bow-tie) while the identity stays the same. They do it on a vector graph,
where subparts are explicit. On raster crops they are not:

1. Connected components do not find the parts of a valve. Body, pipe stubs and
   decoration are one stroke — measured 2026-10-18 on ~1,600 DigitizePID crops, only
   0-13% of valve crops (Symbol_1..16; Symbol_5 37%) have >=2 significant components.
   A component-only augment therefore degenerates to whole-crop affine for exactly the
   family it was meant for.
2. Splitting the stroke artificially (k-means on ink pixels) tears lines at the cut —
   an artefact real drawings never contain.

So `SubpartWarp` keeps the component affine where components exist and adds two
connectivity-preserving perturbations that act on parts without needing to find them:

- a smooth low-frequency displacement field (cubic-upsampled coarse grid), which moves
  regions of the symbol relative to each other — cone proportions, dot size, actuator
  offset — without breaking a stroke;
- a stroke-width jitter, biased toward THINNER strokes because ISO 10628-2 templates
  are drawn ~1.1-2x thinner (stroke / symbol size) than DigitizePID crops, and the
  one-shot prototypes are those templates.

Mini leave-out experiment behind the choice (12 valve classes trained, 4 held out,
recognized from their ISO template, 5 seeds; docs/PLAN_STAGE2_ONESHOT.md B1): one-shot
accuracy 0.770 +/- 0.201 with this augment vs 0.670 (component-only) and 0.691 (none).
Seed variance is as large as the gap, so this is a direction, not a result — the
subpart on/off ablation at full scale (R1/R4) is what decides it.

Ink is handled as "darkness above background" so the varying grey page backgrounds of
DigitizePID survive: background = median of non-ink pixels, ink layers are warped in
darkness space, and the result is composited back with a darkest-wins rule.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image

# Share of total ink a connected component needs to count as a subpart. Smaller
# components (speckle, fragments of a neighbouring tag) stay where they are.
_MIN_PART_FRAC = 0.03

# Below this many pixels per side there is no symbol to perturb meaningfully.
_MIN_SIDE = 8

# Thinning must not erase strokes: an erosion that removes more than this share of the
# ink mass is discarded (hairline symbols, tiny crops).
_MAX_THIN_LOSS = 0.5


class SeededRandom:
    """Picklable `_rng` for augments: a seeded `random.Random`, or the global `random` module
    (seed=None), which torch's DataLoader reseeds per worker. A module does not pickle, so it
    is dropped on the way to a worker started with spawn / forkserver (macOS, Python 3.14+)
    and the worker's own global `random` is used instead."""

    _rng: random.Random | object

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if state.get("_rng") is random:
            state["_rng"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._rng is None:
            self._rng = random


class SubpartWarp(SeededRandom):
    """PIL -> PIL training augment: per-component affine + smooth warp + stroke jitter.

    Applied to the NATIVE crop, before Resize: after downscaling, thin strokes blur
    into the background and the ink mask stops being reliable. Output has the input's
    size and mode.

    Args:
        p: probability of augmenting a given crop; 0 disables (the ablation axis).
        rotate: per-component rotation range, degrees (uniform in [-rotate, rotate]).
            Small on purpose: whole-symbol orientation is fgc.aug.rotate's job, and
            OSSR-PID's +/-20 deg is for parts that are cleanly separated.
        scale: per-component isotropic scale range (lo, hi).
        shear: per-component shear range (uniform in [-shear, shear]).
        warp: max displacement of the smooth field, as a fraction of the crop side.
        warp_grid: control-grid resolution of the field (grid x grid points).
        p_thin: probability of thinning strokes by one pixel.
        p_thick: probability of thickening strokes by one pixel (exclusive with thin).
        seed: fixed seed for reproducible tests. None uses the global `random` module,
            which torch's DataLoader reseeds per worker — a private RNG would be cloned
            into every worker and hand them all the same augmentation stream.
    """

    def __init__(
        self,
        p: float = 0.5,
        rotate: float = 8.0,
        scale: tuple[float, float] = (0.92, 1.08),
        shear: float = 0.05,
        warp: float = 0.04,
        warp_grid: int = 3,
        p_thin: float = 0.35,
        p_thick: float = 0.15,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if p_thin + p_thick > 1.0:
            raise ValueError(f"p_thin + p_thick must be <= 1, got {p_thin} + {p_thick}")
        if warp_grid < 2:
            raise ValueError(f"warp_grid must be >= 2, got {warp_grid}")
        self.p = p
        self.rotate = rotate
        self.scale = (float(scale[0]), float(scale[1]))
        self.shear = shear
        self.warp = warp
        self.warp_grid = warp_grid
        self.p_thin = p_thin
        self.p_thick = p_thick
        self._rng = random.Random(seed) if seed is not None else random

    @classmethod
    def from_cfg(cls, node) -> SubpartWarp:
        """Build from a cfg.fgc.aug.subpart node."""
        return cls(
            p=float(node.p),
            rotate=float(node.rotate),
            scale=(float(node.scale[0]), float(node.scale[1])),
            shear=float(node.shear),
            warp=float(node.warp),
            warp_grid=int(node.warp_grid),
            p_thin=float(node.p_thin),
            p_thick=float(node.p_thick),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(p={self.p}, rotate={self.rotate}, scale={self.scale}, "
            f"shear={self.shear}, warp={self.warp}, warp_grid={self.warp_grid}, "
            f"p_thin={self.p_thin}, p_thick={self.p_thick})"
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0.0 or self._rng.random() >= self.p:
            return img
        if min(img.size) < _MIN_SIDE:
            return img
        mode = img.mode
        rgb = np.asarray(img.convert("RGB"))
        out = self._augment(rgb)
        result = Image.fromarray(out, mode="RGB")
        return result if mode == "RGB" else result.convert(mode)

    # ------------------------------------------------------------------ #

    def _augment(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if not ink.any() or ink.all():
            return rgb.copy()  # blank or solid crop: no ink/background split to work with

        # Per-channel background level, and ink as darkness above it. The mask is
        # dilated by one pixel so anti-aliased stroke edges travel with their stroke
        # instead of staying behind as a ghost outline.
        bg = np.median(rgb[ink == 0], axis=0).astype(np.float32)
        kernel = np.ones((3, 3), np.uint8)
        halo = cv2.dilate(ink, kernel) > 0
        dark = np.clip(bg - rgb.astype(np.float32), 0.0, None) * halo[..., None]

        dark = self._move_components(ink, dark, kernel)
        dark = self._smooth_warp(dark)
        dark = self._stroke_jitter(dark)

        base = rgb.astype(np.float32)
        base[halo] = bg
        return np.clip(np.minimum(base, bg - dark), 0, 255).astype(np.uint8)

    def _move_components(self, ink: np.ndarray, dark: np.ndarray, kernel) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        parts = [i + 1 for i, a in enumerate(areas) if a >= _MIN_PART_FRAC * areas.sum()]
        if len(parts) < 2:
            return dark  # one stroke: nothing to move relative to anything else

        # Ownership map grown by the same 1-px halo as `dark`, so each layer carries its
        # component's anti-aliased edge.
        owner = cv2.dilate(labels.astype(np.float32), kernel).astype(np.int32)
        h, w = ink.shape
        out = np.zeros_like(dark)
        moved = np.zeros(ink.shape, dtype=bool)
        for label in parts:
            mask = owner == label
            moved |= mask
            layer = dark * mask[..., None]
            out = np.maximum(out, self._warp_affine(layer, mask, w, h))
        # Unmoved remainder (small components) is composited unchanged.
        return np.maximum(out, dark * ~moved[..., None])

    def _warp_affine(self, layer: np.ndarray, mask: np.ndarray, w: int, h: int) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        angle = math.radians(self._rng.uniform(-self.rotate, self.rotate))
        s = self._rng.uniform(*self.scale)
        sh = self._rng.uniform(-self.shear, self.shear)
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        a = rot @ np.array([[1.0, sh], [0.0, 1.0]]) * s
        t = np.array([cx, cy]) - a @ np.array([cx, cy])  # transform about the part's centroid
        m = np.hstack([a, t[:, None]]).astype(np.float32)
        return cv2.warpAffine(layer, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    def _smooth_warp(self, dark: np.ndarray) -> np.ndarray:
        h, w = dark.shape[:2]
        g = self.warp_grid
        dx = np.array([self._rng.uniform(-1, 1) for _ in range(g * g)], dtype=np.float32)
        dy = np.array([self._rng.uniform(-1, 1) for _ in range(g * g)], dtype=np.float32)
        dx = cv2.resize(dx.reshape(g, g) * self.warp * w, (w, h), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(dy.reshape(g, g) * self.warp * h, (w, h), interpolation=cv2.INTER_CUBIC)
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return cv2.remap(
            dark, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

    def _stroke_jitter(self, dark: np.ndarray) -> np.ndarray:
        r = self._rng.random()
        kernel = np.ones((2, 2), np.uint8)
        if r < self.p_thin:
            thinned = cv2.erode(dark, kernel)
            if thinned.sum() >= (1.0 - _MAX_THIN_LOSS) * dark.sum():
                return thinned
            return dark
        if r < self.p_thin + self.p_thick:
            return cv2.dilate(dark, kernel)
        return dark
