"""Whole-image degradation of a V6 variant, geometry included (V7 of PLAN_ISO_RENDER_PID2GRAPH).

`degrade_image` makes a clean page look scanned: stroke dilation / erosion, a small rotation and
shear, pixelation, blur, uneven lighting, noise and JPEG, in that order (the order of a scan:
the paper is skewed, the optics blur, the sensor adds noise, the file is compressed). The idea
of the noise / blur / pixelate steps is `scripts/render_pid.py:degrade`, rewritten here.

Strength follows the variant's stroke width `w`, so a thin line is never degraded out of
existence: stroke-width change |morph| <= w / 2 px, blur sigma <= w / 2 px, pixelation block
<= w / 2 px. `cfg["degrade"]` holds ranges as fractions of those limits; `sample_params` draws
inside them from the variant's own seed (`derive_seed(.., "degrade")`, a stage no other step
reads) and `clamp_params` enforces the limits whatever the caller passes.

Rotation and shear move pixels, so everything that has coordinates moves with them. The affine
`M` (2x3, forward: clean -> degraded) is in page coordinates, the SVG / COCO convention (origin
at the top-left corner of the top-left pixel, a pixel's centre at +0.5), and `M` is exact: the
same matrix goes to the image (`cv2.warpAffine`, after the +-0.5 shift) and to every point, box
and polyline (`apply_affine`). Boxes are moved by their ink (`ink_box_after`: the tight box of
the symbol's own ink pixels, each moved by `M`), not by their corners, which would grow a tall
box by the rotation (a 3000 px column, 1 degree: ~50 px).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import cv2
import numpy as np

Affine = tuple[tuple[float, float, float], tuple[float, float, float]]
Box = tuple[float, float, float, float]

_STRIP = 512  # rows per noise strip (bounds the float32 temporary)


@dataclass(frozen=True)
class DegradeParams:
    """What was done to one image (all lengths in page px)."""

    rotate_deg: float  # clockwise in the image (y down), about the page centre
    shear_deg: float  # horizontal shear angle, about the page centre
    morph_px: float  # stroke-width change: < 0 erodes the ink, > 0 dilates it
    blur_sigma: float
    pixelate_block: float  # >= 1; 1 = off
    lighting: float  # depth of the brightness falloff, 0 = flat
    noise_sigma: float  # gray levels
    jpeg_quality: int
    noise_seed: int  # seeds the noise and the lighting field


def limits(w: float) -> dict[str, float]:
    """The largest morph (width change), blur sigma and pixelation block for stroke width `w`."""
    return {"morph_px": w / 2, "blur_sigma": w / 2, "pixelate_block": max(1.0, w / 2)}


MIN_PEAK = 0.55  # a line must keep this much of its darkness after erosion, blur and pixelation


def peak_contrast(w: float, p: DegradeParams) -> float:
    """The darkness (0 = paper, 1 = full ink) left at the middle of a straight line of stroke
    width `w` after `p`'s erosion / dilation, pixelation and blur, measured on a drawn line (at
    four sub-pixel positions, the faintest counts; the geometry and the noise are left out)."""
    worst = 1.0
    for phase in (0.0, 0.25, 0.5, 0.75):
        big = np.full((24 * 8, 64 * 8), 255, np.uint8)
        top = int((10 + phase) * 8)
        big[top : top + int(round(w * 8)), :] = 0
        line = cv2.resize(big, (64, 24), interpolation=cv2.INTER_AREA)
        line = _optics(line, p)
        worst = min(worst, float((255 - line[:, 16:48].min(axis=0)).mean()) / 255)
    return worst


def clamp_params(p: DegradeParams, w: float) -> DegradeParams:
    """`p` with morph, blur and pixelation held to the limits for stroke width `w`, and lowered
    (blur and pixelation first, then erosion) where together they would leave a line of width `w`
    fainter than `MIN_PEAK`."""
    out = _clamp_limits(p, w)
    while peak_contrast(w, out) < MIN_PEAK:
        if out.blur_sigma > 0 or out.pixelate_block > 1:
            out = replace(
                out,
                blur_sigma=max(out.blur_sigma - 0.05, 0.0),
                pixelate_block=max(out.pixelate_block - 0.05, 1.0),
            )
        elif out.morph_px < 0:
            out = replace(out, morph_px=min(out.morph_px + 0.05, 0.0))
        else:
            break
    return out


def _clamp_limits(p: DegradeParams, w: float) -> DegradeParams:
    lim = limits(w)
    return DegradeParams(
        rotate_deg=p.rotate_deg,
        shear_deg=p.shear_deg,
        morph_px=float(np.clip(p.morph_px, -lim["morph_px"], lim["morph_px"])),
        blur_sigma=float(np.clip(p.blur_sigma, 0.0, lim["blur_sigma"])),
        pixelate_block=float(np.clip(p.pixelate_block, 1.0, lim["pixelate_block"])),
        lighting=p.lighting,
        noise_sigma=p.noise_sigma,
        jpeg_quality=p.jpeg_quality,
        noise_seed=p.noise_seed,
    )


def params_from(r: Mapping[str, float], w: float, noise_seed: int, jpeg_q: float) -> DegradeParams:
    """Parameters from a draw `r` of the config's fractions (see `degrade:` in render_iso.yaml)."""
    lim = limits(w)
    return clamp_params(
        DegradeParams(
            rotate_deg=r["rotate_deg"],
            shear_deg=r["shear_deg"],
            morph_px=r["morph"] * lim["morph_px"],
            blur_sigma=r["blur"] * lim["blur_sigma"],
            pixelate_block=1.0 + r["pixelate"] * (lim["pixelate_block"] - 1.0),
            lighting=r["lighting"],
            noise_sigma=r["noise_sigma"],
            jpeg_quality=int(round(jpeg_q)),
            noise_seed=noise_seed,
        ),
        w,
    )


_RANGE_KEYS = ("rotate_deg", "shear_deg", "morph", "blur", "pixelate", "lighting", "noise_sigma")


def sample_params(cfg: Mapping[str, Any], w: float, rng: random.Random) -> DegradeParams:
    """Draw every parameter uniformly in its `cfg["degrade"]` range (always the same draws in the
    same order, so a variant's degradation does not depend on which ranges are set to a point)."""
    d = cfg["degrade"]
    r = {k: rng.uniform(float(d[k][0]), float(d[k][1])) for k in _RANGE_KEYS}
    jpeg = rng.uniform(float(d["jpeg_quality"][0]), float(d["jpeg_quality"][1]))
    return params_from(r, w, rng.getrandbits(62), jpeg)


def extreme_params(
    cfg: Mapping[str, Any], w: float, morph: int = -1, only: str | None = None
) -> DegradeParams:
    """Every parameter at its strongest end of the `cfg["degrade"]` range; `morph` -1 erodes, +1
    dilates. With `only` (a key of `_RANGE_KEYS` or "jpeg_quality") the others stay at the mild
    end, to attribute a failure to one parameter."""
    d = cfg["degrade"]

    def strong(key: str) -> float:
        lo, hi = float(d[key][0]), float(d[key][1])
        if key == "morph":
            return hi if morph > 0 else lo
        if key in ("rotate_deg", "shear_deg"):
            return max(abs(lo), abs(hi))
        return hi

    def mild(key: str) -> float:
        return {"morph": 0.0, "rotate_deg": 0.0, "shear_deg": 0.0}.get(key, float(d[key][0]))

    r = {k: strong(k) if only in (None, k) else mild(k) for k in _RANGE_KEYS}
    jpeg = (
        float(d["jpeg_quality"][0])
        if only in (None, "jpeg_quality")
        else float(d["jpeg_quality"][1])
    )
    return params_from(r, w, 0, jpeg)


# --- the affine ---------------------------------------------------------------------------------


def affine_for(width: int, height: int, rotate_deg: float, shear_deg: float) -> Affine:
    """Clean -> degraded page coordinates: a horizontal shear then a rotation, both about the
    page centre. Rotation is clockwise on screen (y points down) for positive degrees."""
    th, ph = math.radians(rotate_deg), math.radians(shear_deg)
    c, s, t = math.cos(th), math.sin(th), math.tan(ph)
    a, b, d, e = c, c * t - s, s, s * t + c  # R @ [[1, t], [0, 1]]
    cx, cy = width / 2, height / 2
    return ((a, b, cx - a * cx - b * cy), (d, e, cy - d * cx - e * cy))


def apply_affine(m: Affine, pts: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """`pts` (N x 2, page coordinates) through `m`."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    a = np.array([m[0][:2], m[1][:2]])
    return p @ a.T + np.array([m[0][2], m[1][2]])


def _cv_matrix(m: Affine) -> np.ndarray:
    """`m` for cv2.warpAffine: pixel (i, j) of an array sits at page (j + .5, i + .5)."""
    a = np.array([m[0][:2], m[1][:2]])
    t = np.array([m[0][2], m[1][2]]) + a @ [0.5, 0.5] - [0.5, 0.5]
    return np.hstack([a, t[:, None]])


def corner_box(m: Affine, box: Box) -> Box:
    """The axis-aligned box of the four corners of `box` moved by `m` (loose: it grows with the
    rotation times the box's length)."""
    x0, y0, x1, y1 = box
    q = apply_affine(m, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))


def ink_box_after(
    m: Affine, ink: np.ndarray, box: Box, pad: float = 0.0, thresh: int = 245
) -> Box | None:
    """The tight box of the ink inside `box` after `m`. `ink` is a gray image (dark = ink, the
    symbol layer of the clean page); every pixel darker than `thresh` inside `box` is moved by
    `m` (its centre, then +-0.5 for its extent) and `pad` is added on each side. None when `box`
    holds no ink."""
    h, w = ink.shape
    x0, y0 = max(int(math.floor(box[0])), 0), max(int(math.floor(box[1])), 0)
    x1, y1 = min(int(math.ceil(box[2])), w), min(int(math.ceil(box[3])), h)
    if x1 <= x0 or y1 <= y0:
        return None
    ys, xs = np.nonzero(ink[y0:y1, x0:x1] < thresh)
    if xs.size == 0:
        return None
    centres = np.stack([xs + x0 + 0.5, ys + y0 + 0.5], axis=1)
    q = apply_affine(m, centres)
    return (
        float(q[:, 0].min() - 0.5 - pad),
        float(q[:, 1].min() - 0.5 - pad),
        float(q[:, 0].max() + 0.5 + pad),
        float(q[:, 1].max() + 0.5 + pad),
    )


# --- the image ----------------------------------------------------------------------------------


def _morph(img: np.ndarray, px: float) -> np.ndarray:
    """Ink (dark) wider by `px` (< 0: narrower): a gray max / min filter of radius px / 2, a
    fractional radius blending the two neighbouring odd kernels."""
    r = abs(px) / 2
    if r < 1e-3:
        return img
    n = int(math.floor(r))
    f = r - n
    op = cv2.erode if px > 0 else cv2.dilate  # erode the gray values = grow the dark ink

    def run(radius: int) -> np.ndarray:
        if radius == 0:
            return img
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return op(img, k)

    lo = run(n)
    return lo if f < 1e-3 else cv2.addWeighted(lo, 1 - f, run(n + 1), f, 0)


def _soften(img: np.ndarray, p: DegradeParams) -> np.ndarray:
    """Pixelation (down, then up) and Gaussian blur."""
    h, w = img.shape
    out = img
    if p.pixelate_block > 1.01:
        small = cv2.resize(
            out, (max(1, round(w / p.pixelate_block)), max(1, round(h / p.pixelate_block))),
            interpolation=cv2.INTER_AREA,
        )  # fmt: skip
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if p.blur_sigma > 0.1:
        out = cv2.GaussianBlur(out, (0, 0), p.blur_sigma)
    return out


def _optics(img: np.ndarray, p: DegradeParams) -> np.ndarray:
    """What the lens and the stroke change do to a line, without the geometry or the noise."""
    return _soften(_morph(img, p.morph_px), p)


def _lighting(shape: tuple[int, int], depth: float, seed: int) -> np.ndarray:
    """A smooth brightness factor in [1 - depth, 1]: a coarse random field, enlarged."""
    rng = np.random.default_rng(seed)
    coarse = rng.random((4, 6)).astype(np.float32)
    coarse = (coarse - coarse.min()) / max(float(np.ptp(coarse)), 1e-6)
    field = cv2.resize(coarse, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return 1.0 - depth * np.clip(field, 0, 1)


def degrade_image(img: np.ndarray, p: DegradeParams) -> tuple[np.ndarray, Affine, bytes]:
    """`img` (gray uint8, white paper) degraded by `p`. Returns (the image as decoded from the
    JPEG, the clean -> degraded affine, the JPEG bytes)."""
    h, w = img.shape
    out = _morph(img, p.morph_px)
    m = affine_for(w, h, p.rotate_deg, p.shear_deg)
    if p.rotate_deg or p.shear_deg:
        out = cv2.warpAffine(
            out, _cv_matrix(m), (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )  # fmt: skip
    out = _soften(out, p)
    if p.lighting > 0:
        out = np.clip(out * _lighting((h, w), p.lighting, p.noise_seed ^ 0x5F3759DF), 0, 255)
        out = out.astype(np.uint8)
    if p.noise_sigma > 0:
        rng = np.random.default_rng(p.noise_seed)
        out = out.copy()
        for y in range(0, h, _STRIP):
            strip = out[y : y + _STRIP]
            noise = rng.standard_normal(strip.shape, dtype=np.float32) * p.noise_sigma
            out[y : y + _STRIP] = np.clip(strip + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(p.jpeg_quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), m, buf.tobytes()


def params_dict(p: DegradeParams) -> dict[str, float]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(p).items()}


# --- a whole variant ----------------------------------------------------------------------------


def _round(v: Sequence[float]) -> list[float]:
    return [round(float(x), 2) for x in v]


def moved_geometry(
    m: Affine, rendered: Any, symbols: np.ndarray, params: DegradeParams
) -> dict[str, Any]:
    """Everything of a drawn page that has coordinates, after `m`: the COCO annotations (the ink
    box moved by the symbol's own pixels; `bbox_clean` and `bbox_corners` kept for comparison),
    the V4 node boxes and centres (a bubble's are those of its circle, the annotation box, not
    the box of the PID2Graph node), the points where pipes meet symbols and the signal lines.
    `rendered` is `iso_variants.Rendered`; `symbols` the clean symbol layer (gray)."""
    pad = max(params.morph_px, 0.0) / 2  # dilation widens the ink by half its amount a side
    anns = []
    for a in rendered.page.annotations:
        box = tuple(a["bbox"])
        moved = ink_box_after(m, symbols, box, pad) or corner_box(m, box)
        anns.append(
            {
                **a,
                "bbox_clean": a["bbox"],
                "bbox": _round(moved),
                "bbox_corners": _round(corner_box(m, box)),
            }
        )
    by_node: dict[str, list[dict]] = {}
    for a in anns:
        by_node.setdefault(a.get("node", a["id"]), []).append(a)
    nodes = {}
    for nid, n in rendered.sheet.nodes.items():
        clean = (n.xmin, n.ymin, n.xmax, n.ymax)
        own = by_node.get(nid, [])
        if n.label == "instrumentation" and own and own[0]["kind"] == "bubble":
            clean = tuple(own[0]["bbox_clean"])  # the circle that was drawn, not the node box
        cx, cy = (clean[0] + clean[2]) / 2, (clean[1] + clean[3]) / 2
        nodes[nid] = {
            "box": _round(corner_box(m, clean)),
            "center": _round(apply_affine(m, [(cx, cy)])[0]),
            "ink_boxes": [a["bbox"] for a in own],
        }
    ports = [
        {**p, "x": round(float(q[0]), 2), "y": round(float(q[1]), 2)}
        for p, q in zip(
            rendered.page.ports,
            apply_affine(m, [(p["x"], p["y"]) for p in rendered.page.ports]),
            strict=True,
        )
    ]
    stems = [
        {**s, "points": [_round(q) for q in apply_affine(m, s["points"])]}
        for s in rendered.page.stems
    ]
    return {"annotations": anns, "nodes": nodes, "ports": ports, "stems": stems}
