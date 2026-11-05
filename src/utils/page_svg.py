"""Draw a whole sheet as one SVG (V5 of PLAN_ISO_RENDER_PID2GRAPH).

`render_page(sheet, runs, assignments, cfg, source)` takes the sheet after V4 (`sheet`, `runs`), the
V3 assignments and the sheet *before* V4 (`source`: its bboxes give the symbol sizes and which
pairs overlapped originally). Pipes and symbols are written into one SVG with one stroke width
`page.stroke_px`: a template's own strokes are rewritten to `stroke_px / k` (k = px per template
mm) inside a `<g transform>`, so a symbol's line is as thick as a pipe's.

Order of work: place every ISO symbol by its ports -> shrink or replace symbols that newly
overlap -> connect the pipes to the final ports -> draw. A symbol's size comes from its original
bbox (x S), its position from where the pipes reach it after V4 (see plan §V5).

Vessels (`multi` templates: 2062 with dished ends, X8100 column) are drawn from the template's own
cap curves on the node's box (`_Vessel`): the box gives the diameter and the length, so the
strokes stay square at any aspect ratio. Pipes stop on the drawn outline (`_Vessel.target`), and
a standing 2062 may carry an agitator (V5b).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from utils.iso_layout import NOT_SYMBOLS
from utils.iso_mapping import OTHER_ISO, Assignment
from utils.iso_templates import Matrix, Template, TemplatePort, _apply, _compose
from utils.pid2graph import PipeRun, Port, Sheet, _simplify, symbol_ports

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x0, y0, x1, y1

EPS = 1e-6
_INK_PX_PER_MM = 40
_INK_STROKE_MM = 0.06  # stroke used to measure a template's ink box (geometry + a hair)
_SIGN = {"L": 1, "T": 1, "R": -1, "B": -1}  # travel direction of a pipe entering that side
_OPPOSITE_SIDE = {"L": "R", "R": "L", "T": "B", "B": "T"}
# flanges around a disc (template 511, flanged_connection): its two marks are 2.5 mm apart
# (centre to centre); a mark is never closer to the disc's stem than _FLANGE_MIN_GAP strokes,
# so it does not merge with it
_FLANGE_PITCH_MM = 2.5
# discs and flange pairs are drawn at the mm scale of these symbols on the same sheet (see
# `_sheet_scale`)
_SCALE_REF_GROUP = "G1"
# an actuator's stem goes into the middle of these (valves); any other symbol takes the line on
# its outline, nearest the bubble
_STEM_CENTRE_GROUPS = ("G1", "G2")
# a flange pair (template 511): the signal line of its tag enters the gap between the marks
_FLANGE_PAIR = "flanged_connection"
# a leader arrow (an arrowhead on a thin line from a tag, touching the pipe): the line's stroke,
# in pipe strokes
_LEADER_STROKE = 0.5
_FLANGE_MIN_GAP = 2.0
GENERIC_LABEL = "other_generic"
# a vessel (2062, X8100) in BK-2020: 20 mm across, elliptic ends 2.5 mm deep (a cubic of these
# control points, in mm: x across, d outwards from the end of the straight body)
_VESSEL_MM = 20.0
_CAP_MM = 2.5
_CAP_CURVE = (
    (0.0, 0.0),
    (0.0, 0.0),
    (2.5, 2.5),
    (10.0, 2.5),
    (17.5, 2.5),
    (20.0, 0.0),
    (20.0, 0.0),
)
_CURVE_STEPS = 24
# the column's tray lines: three dashes across, this far inside each end of the body
_TRAY_MM = 2.5
_TRAY_DASHES_MM = ((0.0, 4.0), (8.0, 12.0), (16.0, 20.0))
COLUMN = "column"
# an agitator's shaft sticks out of the vessel's top by this much (ISO 10628-2 X8006), and the
# template is at most this fraction of the vessel's length (else at the vessel's own scale: the
# blade is half its diameter)
_SHAFT_OUT_MM = 3.0
_AGITATOR_MAX_LENGTH = 0.7
# the shaft stays within this fraction of the vessel's width from its axis: farther, it reads
# as a side-mounted agitator (no agitator rather than that)
_AGITATOR_MAX_OFFSET = 0.1
# two bubbles joined by a run are moved to touch when their circles are this many strokes apart
# at most (a run of no length: always)
_BUBBLE_JOIN_STROKES = 2
# passes over all pairs of bubbles before `_bubble_circles` gives up (a row settles in a few)
_BUBBLE_ROUNDS = 100


# --- geometry helpers -------------------------------------------------------------------------


def _axis(side: str) -> str:
    return "H" if side in "LR" else "V"


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] - EPS and b[0] < a[2] - EPS and a[1] < b[3] - EPS and b[1] < a[3] - EPS


def _area(b: Box) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _grow(b: Box, m: float) -> Box:
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)


def _node_box(n) -> Box:
    return (n.xmin, n.ymin, n.xmax, n.ymax)


def _box_of(points: list[Point]) -> Box:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _pts(points: list[Point]) -> str:
    return " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in points)


# --- template ink box -------------------------------------------------------------------------

_INK_CACHE: dict[Any, Box] = {}


def _ink_body_box(base: Template) -> Box:
    """Ink box of a template's canonical drawing in *body* (sheet) coordinates, from a raster."""
    import pymupdf

    key = (base.file, str(base.svg_path))
    if key in _INK_CACHE:
        return _INK_CACHE[key]
    vx, vy = -base.matrix[4], -base.matrix[5]
    s = _INK_PX_PER_MM
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{base.w_mm * s:.3f}" '
        f'height="{base.h_mm * s:.3f}" viewBox="{vx} {vy} {base.w_mm} {base.h_mm}">'
        f'<rect x="{vx}" y="{vy}" width="{base.w_mm}" height="{base.h_mm}" fill="white"/>'
        f"{base.body(stroke=_INK_STROKE_MM)}</svg>"
    )
    pix = pymupdf.open(stream=svg.encode(), filetype="svg")[0].get_pixmap(
        dpi=72, colorspace=pymupdf.csGRAY, alpha=False
    )
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    ys, xs = np.nonzero(img < 250)
    if xs.size == 0:
        raise ValueError(f"{base.file}: template renders blank")
    sx, sy = pix.width / base.w_mm, pix.height / base.h_mm
    box = (
        vx + xs.min() / sx,
        vy + ys.min() / sy,
        vx + (xs.max() + 1) / sx,
        vy + (ys.max() + 1) / sy,
    )
    _INK_CACHE[key] = box
    return box


def oriented_ink_box(t: Template) -> Box:
    """Ink box of an oriented template in its own local frame (mm, origin top-left)."""
    x0, y0, x1, y1 = _ink_body_box(t.base or t)
    corners = [_apply(t.matrix, x, y) for x in (x0, x1) for y in (y0, y1)]
    return _box_of(corners)


_RASTER_CACHE: dict[Any, np.ndarray] = {}


def _ink_raster(t: Template) -> np.ndarray:
    """Boolean ink image of an oriented template, `_INK_PX_PER_MM` px per mm, origin top-left."""
    import pymupdf

    key = (t.file, t.rot, t.flip)
    if key not in _RASTER_CACHE:
        s = _INK_PX_PER_MM
        base = t.base or t
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{t.w_mm * s:.3f}" '
            f'height="{t.h_mm * s:.3f}" viewBox="0 0 {t.w_mm} {t.h_mm}">'
            f'<rect width="{t.w_mm}" height="{t.h_mm}" fill="white"/>'
            f'<g transform="{t.transform}">{base.body(stroke=_INK_STROKE_MM)}</g></svg>'
        )
        pix = pymupdf.open(stream=svg.encode(), filetype="svg")[0].get_pixmap(
            dpi=72, colorspace=pymupdf.csGRAY, alpha=False
        )
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        _RASTER_CACHE[key] = img < 250
    return _RASTER_CACHE[key]


def ink_hit(t: Template, side: str, cross_mm: float) -> Point:
    """Where a line entering `t` through `side` first meets its ink, in the template's local mm.

    The line runs along `cross_mm` (y for L / R, x for T / B); when nothing is drawn there, the
    nearest line that meets ink is used. A line from the left meets the leftmost ink, and so on.
    """
    img = _ink_raster(t)
    s = _INK_PX_PER_MM
    if side in "TB":
        img = img.T
    has = img.any(axis=1)
    if not has.any():
        raise ValueError(f"{t.file}: template renders blank")
    rows = np.nonzero(has)[0]
    r = int(rows[np.argmin(np.abs(rows + 0.5 - cross_mm * s))])
    cols = np.nonzero(img[r])[0]
    depth = (cols.min() if side in "LT" else cols.max() + 1) / s
    along = (r + 0.5) / s
    return (depth, along) if side in "LR" else (along, depth)


# --- placed symbols ---------------------------------------------------------------------------


@dataclass
class _Placed:
    nid: str
    tpl: Template  # oriented
    k0: float  # px per mm before any shrink
    pos0: Point  # page position of the template's local origin before any shrink
    pairs: list[tuple[Port, TemplatePort]]
    factor: float = 1.0  # shrink factor applied around the anchor (see `pos`)
    min_factor: float | None = None  # may shrink past the shrink steps down to this (a disc)

    @property
    def k(self) -> float:
        return self.k0 * self.factor

    def ink_local(self) -> Box:
        return oriented_ink_box(self.tpl)

    @property
    def pos(self) -> Point:
        """Page position of the local origin. A shrink keeps the template's ports' centre in
        place (on the pipe axis for a fitting, the stem's tip for a disc), so the pipes stay
        straight; a symbol without ports keeps its ink centre."""
        if self.tpl.ports:
            cx = sum(q.x_mm for q in self.tpl.ports) / len(self.tpl.ports)
            cy = sum(q.y_mm for q in self.tpl.ports) / len(self.tpl.ports)
        else:
            x0, y0, x1, y1 = self.ink_local()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return (
            self.pos0[0] + self.k0 * cx - self.k * cx,
            self.pos0[1] + self.k0 * cy - self.k * cy,
        )

    def to_page(self, x_mm: float, y_mm: float) -> Point:
        px, py = self.pos
        return (px + self.k * x_mm, py + self.k * y_mm)

    def ink_box(self, stroke_px: float) -> Box:
        x0, y0, x1, y1 = self.ink_local()
        a, b = self.to_page(x0, y0), self.to_page(x1, y1)
        # the measured box includes half of _INK_STROKE_MM; the drawn stroke is stroke_px wide
        return _grow((a[0], a[1], b[0], b[1]), stroke_px / 2 - self.k * _INK_STROKE_MM / 2)

    def ink_target(self, p: Port) -> Point:
        """Page point where the pipe of `p` meets this symbol's ink, going in through `p.side`."""
        px, py = self.pos
        cross = (p.y - py) / self.k if p.side in "LR" else (p.x - px) / self.k
        x, y = ink_hit(self.tpl, p.side, cross)
        px_, py_ = self.to_page(x, y)
        # the raster is coarser than the pipe: do not make a sub-pixel jog out of that
        if p.side in "LR" and abs(py_ - p.y) < 1.0:
            py_ = p.y
        elif p.side in "TB" and abs(px_ - p.x) < 1.0:
            px_ = p.x
        return (px_, py_)

    def ink_along(self, start: Point, direction: Point) -> Point | None:
        """First point of this symbol's ink on the ray from `start` along `direction` (a unit
        vector, page px): `start` itself when it is on ink. None when the ray meets none."""
        img = _ink_raster(self.tpl)
        px, py = self.pos
        step = self.k / _INK_PX_PER_MM  # one raster pixel
        h, w = img.shape
        for i in range(int(1.5 * max(h, w))):
            x = start[0] + direction[0] * step * i
            y = start[1] + direction[1] * step * i
            c, r = int((x - px) / step), int((y - py) / step)
            if 0 <= r < h and 0 <= c < w and img[r, c]:
                return (x, y)
        return None

    def ink_nearest(self, target: Point) -> Point | None:
        """The point of this symbol's ink nearest to `target` (page px); None for blank ink. The
        segment from it to `target` meets no other ink of the symbol."""
        rows, cols = np.nonzero(_ink_raster(self.tpl))
        if rows.size == 0:
            return None
        px, py = self.pos
        step = self.k / _INK_PX_PER_MM
        xs, ys = px + (cols + 0.5) * step, py + (rows + 0.5) * step
        i = int(np.argmin((xs - target[0]) ** 2 + (ys - target[1]) ** 2))
        return (float(xs[i]), float(ys[i]))

    def transform(self) -> str:
        px, py = self.pos
        m: Matrix = _compose((self.k, 0.0, 0.0, self.k, px, py), self.tpl.matrix)
        return "matrix(" + " ".join(f"{v:.6g}" for v in m) + ")"


def _match_ports(
    node_ports: list[Port], tpl: Template, k: float, pos: Point
) -> list[tuple[Port, TemplatePort]]:
    """Node ports <-> template ports: same side in edge order, then the nearest of the rest."""
    pairs: list[tuple[Port, TemplatePort]] = []
    free_n, free_t = list(node_ports), list(tpl.ports)

    def along(side: str, x: float, y: float) -> float:
        return y if side in "LR" else x

    for side in "LRTB":
        ns = sorted((p for p in free_n if p.side == side), key=lambda p: along(side, p.x, p.y))
        ts = sorted(
            (q for q in free_t if q.side == side),
            key=lambda q: along(side, q.x_mm, q.y_mm),
        )
        if ns and len(ns) == len(ts):
            for p, q in zip(ns, ts, strict=True):
                pairs.append((p, q))
                free_n.remove(p)
                free_t.remove(q)
    costs = sorted(
        (
            math.hypot(pos[0] + k * q.x_mm - p.x, pos[1] + k * q.y_mm - p.y),
            i,
            j,
        )
        for i, p in enumerate(free_n)
        for j, q in enumerate(free_t)
    )
    used_i: set[int] = set()
    used_j: set[int] = set()
    for _, i, j in costs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((free_n[i], free_t[j]))
    return pairs


def _place(
    nid: str,
    tpl: Template,
    node_ports: list[Port],
    bbox4: Box,
    bbox_src: Box,
    scale: float,
    k: float | None = None,
) -> _Placed:
    """Size (`k`, px per mm: the ink fitted into `bbox_src` x `scale` unless given) and position
    (by the ports that agree on the side) of one ISO symbol."""
    ix0, iy0, ix1, iy1 = oriented_ink_box(tpl)
    if k is None:
        k = _fit_scale(tpl, bbox_src, scale)
    # first guess: the ink centred on the V4 bbox; refined by the ports below
    cx, cy = (bbox4[0] + bbox4[2]) / 2, (bbox4[1] + bbox4[3]) / 2
    guess = (cx - k * (ix0 + ix1) / 2, cy - k * (iy0 + iy1) / 2)
    pairs = _match_ports(node_ports, tpl, k, guess)
    # a port that met a template port on another side (a pump piped from one side) cannot say
    # where the symbol goes; only the pairs that agree on the side do
    sure = [(p, q) for p, q in pairs if p.side == q.side]
    if sure:
        pos = (
            sum(p.x - k * q.x_mm for p, q in sure) / len(sure),
            sum(p.y - k * q.y_mm for p, q in sure) / len(sure),
        )
    elif node_ports and len({p.side for p in node_ports}) == 1:
        # no port says where it goes and all pipes come from one side (a pump piped from the top
        # whose box is wide): put the symbol right behind the pipes, centred on them
        side = node_ports[0].side
        mx = sum(p.x for p in node_ports) / len(node_ports)
        my = sum(p.y for p in node_ports) / len(node_ports)
        if side in "TB":
            pos = (mx - k * (ix0 + ix1) / 2, my - k * (iy0 if side == "T" else iy1))
        else:
            pos = (mx - k * (ix0 if side == "L" else ix1), my - k * (iy0 + iy1) / 2)
    else:
        pos = guess
    return _Placed(nid, tpl, k, pos, pairs)


# --- pipe ends --------------------------------------------------------------------------------


def _attach(
    pts: list[Point],
    target: Point,
    side: str,
    stub: float,
) -> list[Point]:
    """`pts` end where a pipe reached a symbol; carry it to `target`, a port on `side`.

    The pipe has to arrive along the port's axis. An offset along the last segment is absorbed
    by moving its end (longer or shorter); an offset across it adds a Z jog, placed at the old
    end when the target is ahead of it and no closer than `stub` to the target. Never diagonal.
    A target behind the pipe's previous vertex is reached by backing up to that vertex.
    """
    axis_t = _axis(side)
    sg = _SIGN[side]
    end = pts[-1]
    prev = pts[-2] if len(pts) > 1 else None
    last_axis = axis_t
    if prev is not None:
        dx, dy = abs(end[0] - prev[0]), abs(end[1] - prev[1])
        if dx > EPS or dy > EPS:
            last_axis = "H" if dx >= dy else "V"
    if last_axis != axis_t:
        # arrives along the other axis than the port: an L into the port
        corner = (target[0], end[1]) if last_axis == "H" else (end[0], target[1])
        return [*pts[:-1], corner, target]
    u = 0 if axis_t == "H" else 1

    def mk(uu: float, vv: float) -> Point:
        return (uu, vv) if u == 0 else (vv, uu)

    eu, ev = end[u], end[1 - u]
    tu, tv = target[u], target[1 - u]
    pu = prev[u] if prev is not None else eu
    if abs(tv - ev) < EPS and sg * (tu - pu) >= -EPS:
        return [*pts[:-1], target]
    a, b, e = sg * pu, sg * tu, sg * eu
    m = min(max(e, a), max(b - stub, a))
    return [*pts[:-1], mk(sg * m, ev), mk(sg * m, tv), target]


def _extend(pts: list[Point], reach: float) -> list[Point]:
    """`pts` with its last (axis-aligned) segment lengthened by `reach`."""
    if len(pts) < 2 or reach <= 0:
        return pts
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    if abs(x1 - x0) >= abs(y1 - y0):
        return [*pts[:-1], (x1 + math.copysign(reach, x1 - x0), y1)]
    return [*pts[:-1], (x1, y1 + math.copysign(reach, y1 - y0))]


def _attach_end(
    pts: list[Point],
    end: str,
    target: Point,
    side: str,
    stub: float,
    reach: float = 0.0,
) -> list[Point]:
    """`_attach` at the `end` ("a" = first point, "b" = last) of a run, then `reach` px further
    into the symbol: the butt cap of the pipe is then covered by the symbol's own stroke."""
    out = _extend(_attach(pts[::-1] if end == "a" else pts, target, side, stub), reach)
    return out[::-1] if end == "a" else out


def _ray_polygon(poly: list[Point], start: Point, side: str) -> Point | None:
    """First point of `poly`'s outline met by a ray from `start` entering through `side`."""
    sg = _SIGN[side]
    u = 0 if _axis(side) == "H" else 1
    v = 1 - u
    hits: list[float] = []
    for p, q in zip(poly, [*poly[1:], poly[0]], strict=True):
        v0, v1 = p[v], q[v]
        if abs(v1 - v0) < EPS or not (min(v0, v1) - EPS <= start[v] <= max(v0, v1) + EPS):
            continue
        hits.append(p[u] + (start[v] - v0) * (q[u] - p[u]) / (v1 - v0))
    ahead = [h for h in hits if sg * (h - start[u]) >= -EPS]
    if not ahead:
        return None
    h = min(ahead, key=lambda t: sg * (t - start[u]))
    return (h, start[v]) if u == 0 else (start[v], h)


Circle = tuple[Point, float]  # centre, radius


def _bubble_circles(
    sheet: Sheet,
    runs: list[PipeRun],
    kind: dict[str, str],
    w_px: float,
    stats: dict[str, int] | None = None,
    rounds: int = _BUBBLE_ROUNDS,
) -> dict[str, Circle]:
    """The circle drawn for each bubble, computed once: node -> (centre, radius).

    A circle is inscribed in its PID2Graph box. Tags that stand side by side (TE ..A / ..B,
    FO / FT) just touch in the original, but their boxes overlap by a few px, or are not as tall
    as wide, so the circles overlap or leave a gap. Two bubbles joined by a run of no length
    (or a very short one: a gap of at most two strokes), and any two whose circles overlap by
    more than a stroke, are moved to touch from outside: each takes half the overlap (or the gap)
    along the line between the centres, and the radii stay. Moving one pair can upset another
    (a row of tags), so it passes over all pairs until none moves, at most `rounds` times.

    `stats`, when given, gets `moved`: bubbles whose circle left its box centre, and `unsettled`:
    pairs still to be moved when the passes run out (0 unless `rounds` was too few)."""
    circ: dict[str, list[float]] = {}
    for nid, kd in kind.items():
        if kd == "bubble":
            b = _node_box(sheet.nodes[nid])
            circ[nid] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, min(b[2] - b[0], b[3] - b[1]) / 2]
    linked: dict[frozenset[str], bool] = {}  # pair -> some run between them has no length
    for run in runs:
        if run.internal or run.a == run.b or run.a not in circ or run.b not in circ:
            continue
        key = frozenset((run.a, run.b))
        linked[key] = linked.get(key, False) or len({tuple(q) for q in run.points}) <= 1

    def gap(a: str, b: str) -> float:
        return (
            math.hypot(circ[a][0] - circ[b][0], circ[a][1] - circ[b][1]) - circ[a][2] - circ[b][2]
        )

    close = {
        key
        for key, zero in linked.items()
        if zero or gap(*sorted(key)) <= _BUBBLE_JOIN_STROKES * w_px
    }

    def to_move(a: str, b: str) -> float | None:
        """The gap to close (< 0: an overlap to undo), or None when the pair is settled."""
        g = gap(a, b)
        if frozenset((a, b)) in close:
            return None if abs(g) <= 0.05 else g
        return None if g >= -w_px else g

    start = {n: (c[0], c[1]) for n, c in circ.items()}
    pairs = [(a, b) for a, b in combinations(sorted(circ), 2)]
    for _ in range(rounds):
        moved = False
        for a, b in pairs:
            g = to_move(a, b)
            if g is None:
                continue
            dx, dy = circ[b][0] - circ[a][0], circ[b][1] - circ[a][1]
            d = math.hypot(dx, dy)
            ux, uy = (dx / d, dy / d) if d > EPS else (1.0, 0.0)
            circ[a][0] += ux * g / 2
            circ[a][1] += uy * g / 2
            circ[b][0] -= ux * g / 2
            circ[b][1] -= uy * g / 2
            moved = True
        if not moved:
            break
    if stats is not None:
        stats["moved"] = sum(math.dist(start[n], (c[0], c[1])) > 0.01 for n, c in circ.items())
        stats["unsettled"] = sum(to_move(a, b) is not None for a, b in pairs)
    return {n: ((c[0], c[1]), c[2]) for n, c in circ.items()}


def _circle_hit(centre: Point, r: float, start: Point, side: str) -> Point:
    """Where a pipe entering a circle through `side` meets it (clamped to the circle's extent)."""
    sg = _SIGN[side]
    u = 0 if _axis(side) == "H" else 1
    v = 1 - u
    dv = max(-0.999 * r, min(0.999 * r, start[v] - centre[v]))
    hu = centre[u] - sg * math.sqrt(max(r * r - dv * dv, 0.0))
    return (hu, centre[v] + dv) if u == 0 else (centre[v] + dv, hu)


def _fit_scale(tpl: Template, bbox_src: Box, scale: float) -> float:
    """px per mm that fits the ink of `tpl` (as oriented) into `bbox_src` x `scale`."""
    ix0, iy0, ix1, iy1 = oriented_ink_box(tpl)
    sw, sh = bbox_src[2] - bbox_src[0], bbox_src[3] - bbox_src[1]
    return scale * min(sw / (ix1 - ix0), sh / (iy1 - iy0))


def _sheet_scale(
    assignments: Mapping[str, Assignment], source: Sheet, scale: float
) -> float | None:
    """The sheet's own symbol scale: median px per mm of its `_SCALE_REF_GROUP` symbols (straight
    valves), each fitted to its box. None when the sheet has none.

    BK-2020 draws every symbol at one mm scale (a valve body, a flange mark and a disc's head
    are all 5 mm), but a PID2Graph box says little about a disc or a flange pair: a flange box
    is a thin sliver and a disc has no box of its own. Drawn at the valves' scale, a flange pair
    is about as tall as a valve body, as in the OPEN100 drawings (ratio ~1 on the sheets whose
    valve boxes hold the body only)."""
    ks = [
        _fit_scale(a.template, _node_box(source.nodes[n]), scale)
        for n, a in assignments.items()
        if a.kind == "iso" and a.template is not None and a.template.group == _SCALE_REF_GROUP
    ]
    return float(np.median(ks)) if ks else None


def _disc_scale(tpl: Template, bbox_src: Box, scale: float) -> float:
    """px per mm of a disc or plate on a stem when the sheet has no valves to take the scale
    from: its head (the ink across the stem, i.e. along the pipe) as wide as the node box is
    across the pipe, x `scale` (the node is the flange joint the disc sits in)."""
    ix0, iy0, ix1, iy1 = oriented_ink_box(tpl)
    sw, sh = bbox_src[2] - bbox_src[0], bbox_src[3] - bbox_src[1]
    if tpl.ports and tpl.ports[0].side in "LR":  # the pipe runs along x
        return scale * sh / (ix1 - ix0)
    return scale * sw / (iy1 - iy0)


def _place_flanges(pl: _Placed, flanges: Template, w_px: float, k_ref: float) -> _Placed:
    """The flange pair a disc is held between (`flanges`, already turned to the disc's pipe
    axis), centred on the tip of the disc's stem, where its two ports are. Drawn at `k_ref`, the
    scale a lone flange pair on this node gets, so a flange pair has one size with or without a
    disc; it shrinks with the disc, but a mark never runs into the stem."""
    ports = pl.tpl.ports
    tip = pl.to_page(
        sum(q.x_mm for q in ports) / len(ports), sum(q.y_mm for q in ports) / len(ports)
    )
    k = max(k_ref * pl.factor, 2 * _FLANGE_MIN_GAP * w_px / _FLANGE_PITCH_MM)
    fx = sum(q.x_mm for q in flanges.ports) / len(flanges.ports)
    fy = sum(q.y_mm for q in flanges.ports) / len(flanges.ports)
    return _Placed(f"{pl.nid}/flanges", flanges, k, (tip[0] - k * fx, tip[1] - k * fy), [])


def _flag_polygon(box: Box, axis: str, forward: bool) -> list[Point]:
    x0, y0, x1, y1 = box
    length = (x1 - x0) if axis == "H" else (y1 - y0)
    across = (y1 - y0) if axis == "H" else (x1 - x0)
    tip = min(0.4 * length, across / 2)
    local = [(0, 0), (length - tip, 0), (length, across / 2), (length - tip, across), (0, across)]
    out: list[Point] = []
    for uu, vv in local:
        if axis == "H":
            out.append(((x0 + uu) if forward else (x1 - uu), y0 + vv))
        else:
            out.append((x0 + vv, (y0 + uu) if forward else (y1 - uu)))
    return out


def _arrow_box(box: Box, axis: str, ports: list[Port]) -> Box:
    """`box` (an arrow's, after V4) moved across `axis` onto the pipe it lies on: the line of the
    port on that axis nearest the box's middle. A PID2Graph arrow box is not centred on its pipe
    (off by 2 px typically, up to ~30 px after V4), so an arrow drawn in its middle sat beside the
    line. Unchanged when no port lies on the axis."""
    u = 1 if axis == "H" else 0  # the coordinate across the pipe
    lines = [(p.y if u == 1 else p.x) for p in ports if (p.side in "LR") == (axis == "H")]
    if not lines:
        return box
    mid = (box[u] + box[u + 2]) / 2
    d = min(lines, key=lambda v: abs(v - mid)) - mid
    return (
        (box[0], box[1] + d, box[2], box[3] + d)
        if u == 1
        else (box[0] + d, box[1], box[2] + d, box[3])
    )


def _place_overlay(
    nid: str, tpl: Template, box: Box, bbox_src: Box, scale: float, ports: list[Port]
) -> _Placed:
    """An overlay template (a flow arrow, 241: just the arrowhead) fitted to the node's
    original box x `scale`, on `box` (the arrow's box, already moved onto its pipe). Its tip
    (the base drawing's R port) goes on the node's port on that side, where the pipe meets a
    bend, a junction or a symbol in OPEN100; with no port there it is centred on `box`. The pipe
    runs on underneath it."""
    k = _fit_scale(tpl, bbox_src, scale)
    ix0, iy0, ix1, iy1 = oriented_ink_box(tpl)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    pos = (cx - k * (ix0 + ix1) / 2, cy - k * (iy0 + iy1) / 2)
    base_tip = next((q.port_id for q in (tpl.base or tpl).ports if q.side == "R"), None)
    tip = next((q for q in tpl.ports if q.port_id == base_tip), None)
    at = next((p for p in ports if tip is not None and p.side == tip.side), None)
    if tip is not None and at is not None:
        # the ink's edge on that side, not the template port: a port found on the ink sits a
        # little inside a sharp tip (~0.2 mm, several px at an arrow's size)
        edge = {"L": ix0, "R": ix1, "T": iy0, "B": iy1}[tip.side]
        if tip.side in "LR":  # along the pipe from the port, across it from the box's line
            pos = (at.x - k * edge, pos[1])
        else:
            pos = (pos[0], at.y - k * edge)
    return _Placed(nid, tpl, k, pos, [])


def _leader_shape(
    far: Point, centre: Point, side: str, tpl: Template | None, length: float, w_px: float
) -> tuple[str, list[Point], Box]:
    """A leader arrow, as OPEN100 draws one: a thin straight line from `far` (the other end of
    the leader, where its graph edge ends) to an arrowhead `length` px long centred on `centre`
    (its box, against the pipe it points at), pointing away from `far`, so usually at a slant.
    `side` is the side of its box the leader leaves from, used when `far` is on the box itself.
    Returns (SVG of the head, the line, the head's box). The head is the flow arrow template
    (`tpl`, 241 pointing right) turned to any angle; a plain 2:1 triangle without it."""
    dx, dy = centre[0] - far[0], centre[1] - far[1]
    n = math.hypot(dx, dy)
    if n < EPS:  # point away from the side the leader leaves from
        dx, dy = {"L": (1.0, 0.0), "R": (-1.0, 0.0), "T": (0.0, 1.0), "B": (0.0, -1.0)}[side]
    else:
        dx, dy = dx / n, dy / n
    tip = (centre[0] + dx * length / 2, centre[1] + dy * length / 2)
    base = (centre[0] - dx * length / 2, centre[1] - dy * length / 2)
    line = [far, base] if n > length / 2 else [base, base]
    if tpl is None:
        h = length / 4  # half the base of a 2:1 head
        pts = [tip, (base[0] - dy * h, base[1] + dx * h), (base[0] + dy * h, base[1] - dx * h)]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        svg = f'<polygon points="{_pts(pts)}" fill="black" stroke="none"/>'
        return svg, line, (min(xs), min(ys), max(xs), max(ys))
    ix0, iy0, ix1, iy1 = oriented_ink_box(tpl)
    k = length / (ix1 - ix0)
    # local -> page: tip of the ink (right edge, mid height) on `tip`, turned to (dx, dy)
    place: Matrix = (k * dx, k * dy, -k * dy, k * dx, 0.0, 0.0)
    tx, ty = _apply(place, ix1, (iy0 + iy1) / 2)
    place = (place[0], place[1], place[2], place[3], tip[0] - tx, tip[1] - ty)
    m = _compose(place, tpl.matrix)
    svg = (
        '<g transform="matrix('
        + " ".join(f"{v:.6g}" for v in m)
        + ')">'
        + tpl.body(stroke=w_px / k)
        + "</g>"
    )
    corners = [_apply(place, x, y) for x in (ix0, ix1) for y in (iy0, iy1)]
    xs, ys = [q[0] for q in corners], [q[1] for q in corners]
    return svg, line, (min(xs), min(ys), max(xs), max(ys))


def _arrow_polygon(box: Box, axis: str, forward: bool) -> list[Point]:
    """A filled triangle filling `box`: the arrow drawn when no arrow template is in scope."""
    x0, y0, x1, y1 = box
    length = (x1 - x0) if axis == "H" else (y1 - y0)
    across = (y1 - y0) if axis == "H" else (x1 - x0)
    out: list[Point] = []
    for uu, vv in [(0, 0), (length, across / 2), (0, across)]:
        if axis == "H":
            out.append(((x0 + uu) if forward else (x1 - uu), y0 + vv))
        else:
            out.append((x0 + vv, (y0 + uu) if forward else (y1 - uu)))
    return out


def _at_end(run: PipeRun, end: str, nid: str, here: Point) -> bool:
    """Does `run` start (end "a") or finish ("b") at `here`, a point of node `nid`?"""
    node, pt = (run.a, run.points[0]) if end == "a" else (run.b, run.points[-1])
    return node == nid and tuple(pt) == here


def _stems(
    sheet: Sheet,
    runs: list[PipeRun],
    placed: dict[str, _Placed],
    kind: dict[str, str],
    w_px: float,
    circles: Mapping[str, Circle],
) -> tuple[dict[tuple[str, str], tuple[list[int], list[Point]]], set[int]]:
    """Signal lines from a bubble to an ISO symbol (an actuator's stem), one per pair of nodes:
    ({(symbol, bubble): (runs of the pair, points: start, end)}, runs of pairs that
    touch).

    The symbol has no port for the line (`symbol_ports` ignores `instrumentation`), so the line is
    drawn to the bubble's circle. A valve (group 1 or 2) takes it from its middle (its ink box
    centre, or the first ink on the way to the bubble when the middle is empty), along the line
    between their centres: that is where an actuator's stem enters. A flange pair (511) takes it
    from the middle of the gap between its marks, on the pipe axis, and runs it square to the pipe
    to the circle (OPEN100: the line of an FO / RO tag enters the gap); the gap has no ink, so the
    start is not on the symbol. When the circle is off to the side (that ray misses it), the line
    runs diagonally from the port of the mark on the circle's side, where the pipe meets it, as
    OPEN100 draws it: never across a mark. Any other symbol (a safety device, a disc) takes it
    from the point of its ink nearest the bubble, so the line never runs through its body nor
    starts in the gap of a hollow drawing.
    That needs no jog and no side, and it does not depend on the run: PID2Graph measured the line
    on a box wider than the drawn symbol, and a run of no length (the boxes overlap) still needs
    its stem, because the symbol drawn is narrower. It is the one place a pipe is allowed to run
    diagonally. A bubble whose circle reaches the symbol's ink box gets none: the two touch, and
    their runs are returned to be left undrawn.
    """
    pairs: dict[tuple[str, str], list[int]] = {}
    for idx, run in enumerate(runs):
        if run.internal or run.a == run.b:
            continue
        for nd, bubble in ((run.a, run.b), (run.b, run.a)):
            if nd in placed and kind.get(nd) == "iso" and kind.get(bubble) == "bubble":
                pairs.setdefault((nd, bubble), []).append(idx)
    out: dict[tuple[str, str], tuple[list[int], list[Point]]] = {}
    touching: set[int] = set()
    for (nd, bubble), idxs in pairs.items():
        x0, y0, x1, y1 = placed[nd].ink_box(w_px)
        start = ((x0 + x1) / 2, (y0 + y1) / 2)
        centre, r = circles[bubble]
        dist = math.hypot(centre[0] - start[0], centre[1] - start[1])
        if dist <= r + w_px:
            touching.update(idxs)
            continue
        if (gap := _gap_of(placed[nd])) is not None:
            out[(nd, bubble)] = (idxs, _gap_stem(gap, centre, r))
            continue
        if placed[nd].tpl.group in _STEM_CENTRE_GROUPS:
            # a hollow valve (a check valve) has no ink at its middle: begin on its outline
            start = (
                placed[nd].ink_along(
                    start, ((centre[0] - start[0]) / dist, (centre[1] - start[1]) / dist)
                )
                or start
            )
        else:
            hit = placed[nd].ink_nearest(centre)
            if hit is not None:
                # the pipe's butt cap ends at the middle of the outline's stroke: run half a
                # stroke into it so the two join (the line is straight, so it stays inside it)
                d = math.hypot(centre[0] - hit[0], centre[1] - hit[1]) or 1.0
                start = (
                    hit[0] - (centre[0] - hit[0]) / d * w_px / 2,
                    hit[1] - (centre[1] - hit[1]) / d * w_px / 2,
                )
        dist = math.hypot(centre[0] - start[0], centre[1] - start[1])
        # to the middle of the circle's stroke, which covers the butt cap
        end = (
            centre[0] - (centre[0] - start[0]) / dist * r,
            centre[1] - (centre[1] - start[1]) / dist * r,
        )
        out[(nd, bubble)] = (idxs, [start, end])
    return out, touching


def _gap_of(pl: _Placed) -> tuple[Point, str, float] | None:
    """For a flange pair, (centre of the gap on the pipe axis, the axis "H" | "V", half the gap in
    px); None for any other symbol. The gap is between the ports: the marks' outer faces."""
    ports = pl.tpl.ports
    if pl.tpl.fine_name != _FLANGE_PAIR or len(ports) != 2:
        return None
    (ax, ay), (bx, by) = (pl.to_page(q.x_mm, q.y_mm) for q in ports)
    axis = "H" if abs(bx - ax) >= abs(by - ay) else "V"
    return ((ax + bx) / 2, (ay + by) / 2), axis, math.hypot(bx - ax, by - ay) / 2


def _gap_stem(gap: tuple[Point, str, float], centre: Point, r: float) -> list[Point]:
    """(start, end) of a flange pair's signal line: from the middle of the gap on the pipe axis,
    square to the pipe towards the bubble, to the circle's stroke. When that ray misses the
    circle, from the port of the mark on the bubble's side (where the pipe meets that mark)
    straight to the circle, as OPEN100 draws it (RO 1603): it leaves the mark on its outer face,
    so it crosses neither mark, and needs no elbow."""
    (sx, sy), axis, half = gap
    across = 1 if axis == "H" else 0  # the coordinate the square line runs along
    along = 1 - across
    off = (sx, sy)[along] - centre[along]  # how far the ray is from the circle's centre line
    if abs(off) < r - EPS:
        reach = math.sqrt(r * r - off * off)
        sign = 1.0 if centre[across] >= (sx, sy)[across] else -1.0
        end = [0.0, 0.0]
        end[along] = (sx, sy)[along]
        end[across] = centre[across] - sign * reach
        return [(sx, sy), (end[0], end[1])]
    start = [sx, sy]
    start[along] -= math.copysign(half, off)  # the port on the bubble's side of the gap
    d = math.hypot(centre[0] - start[0], centre[1] - start[1])
    end = (centre[0] - (centre[0] - start[0]) / d * r, centre[1] - (centre[1] - start[1]) / d * r)
    return [(start[0], start[1]), end]


def _degenerate_runs(sheet: Sheet, runs: list[PipeRun], kind: dict[str, str]) -> set[int]:
    """Runs that join two symbols whose boxes overlap: a link of no length, not a pipe. Not a
    symbol and its bubble: drawn smaller than its box, the symbol is not touching the bubble."""
    out: set[int] = set()
    for idx, run in enumerate(runs):
        a, b = sheet.nodes[run.a], sheet.nodes[run.b]
        if a.label in NOT_SYMBOLS or b.label in NOT_SYMBOLS or run.a == run.b:
            continue
        if {kind.get(run.a), kind.get(run.b)} == {"iso", "bubble"}:
            continue
        if _overlap(_node_box(a), _node_box(b)):
            out.add(idx)
    return out


def _nozzle_links(
    sheet: Sheet,
    runs: list[PipeRun],
    ports: dict[str, list[Port]],
    placed: dict[str, _Placed],
    kind: dict[str, str],
) -> dict[int, tuple[str, str, Port]]:
    """Links of no length between a placed ISO symbol and a vessel or a plain box: run index ->
    (symbol, vessel / box node, the symbol's port on it).

    In OPEN100 these are vessel nozzles: a flange or a closed nozzle against the vessel. The
    ISO symbol is drawn smaller than its box, so without a pipe it would float off the vessel
    (2-9 px) or, at the end of a line, hang on its outline; `_nozzle_stub` draws that pipe.
    """
    degenerate = _degenerate_runs(sheet, runs, kind)
    out: dict[int, tuple[str, str, Port]] = {}
    for nid in placed:
        for p in ports[nid]:
            run = runs[p.run]
            other = run.b if run.a == nid else run.a
            if run.internal or other == nid or kind.get(other) not in ("generic", "vessel"):
                continue
            if len({tuple(q) for q in run.points}) <= 1 or p.run in degenerate:
                out[p.run] = (nid, other, p)
    return out


def _nozzle_stub(box: Box, target: Point, side: str) -> list[Point] | None:
    """The pipe from the edge of `box` (the vessel) to `target`, a port on `side` of the symbol
    beside it; None when the port is not in front of the box or already reaches it."""
    sg = _SIGN[side]
    u = 0 if side in "LR" else 1
    lo, hi = (box[1], box[3]) if u == 0 else (box[0], box[2])
    if not lo <= target[1 - u] <= hi:
        return None
    edge = (box[2] if u == 0 else box[3]) if sg > 0 else (box[0] if u == 0 else box[1])
    if sg * (target[u] - edge) <= EPS:  # touching or inside the box already
        return None
    start = (edge, target[1]) if u == 0 else (target[0], edge)
    return [start, target]


def _drawn_ends(
    runs: list[PipeRun], skip: set[int]
) -> dict[tuple[int, str], tuple[str, float, float, str, str]]:
    """(run index, "a" | "b") -> (node, x, y, bbox side, node at the other end) for every pipe
    end that is drawn: not an `internal` run, not a zero-length one (nor one in `skip`), and
    with a known side."""
    out: dict[tuple[int, str], tuple[str, float, float, str, str]] = {}
    for idx, run in enumerate(runs):
        if idx in skip or run.internal or len(set(map(tuple, run.points))) <= 1:
            continue
        for end, node, other, side, pt in (
            ("a", run.a, run.b, run.a_side, run.points[0]),
            ("b", run.b, run.a, run.b_side, run.points[-1]),
        ):
            if side is not None:
                out[(idx, end)] = (node, pt[0], pt[1], side, other)
    return out


# --- vessels ----------------------------------------------------------------------------------


@dataclass
class _Vessel:
    """A vessel drawn on a node's box: elliptic ends on a straight body, as BK-2020 draws 2062.

    The box gives the diameter (its short side: `k` = diameter / 20 mm px per mm) and the length
    (the long side). The cap curves keep the template's shape at that `k`, the body is as long
    as the box needs, so a stroke is never stretched. A box wider than tall is drawn lying down,
    with the ends left and right. `poly` is the outline (the curves in short chords), for finding
    where a pipe meets it."""

    nid: str
    box: Box
    lying: bool
    column: bool
    k: float
    poly: list[Point]

    @property
    def cap(self) -> float:
        return _CAP_MM * self.k

    @property
    def length(self) -> float:
        x0, y0, x1, y1 = self.box
        return (x1 - x0) if self.lying else (y1 - y0)

    def _page(self, v: float, u: float) -> Point:
        """Page point of `v` across the vessel and `u` along it (from its first end)."""
        x0, y0, _, _ = self.box
        return (x0 + u, y0 + v) if self.lying else (x0 + v, y0 + u)

    def outline_d(self) -> str:
        return "M" + " L".join(f"{_fmt(x)},{_fmt(y)}" for x, y in self.poly) + " Z"

    def trays_d(self) -> str:
        """The column's tray lines as a path (empty for a plain vessel)."""
        if not self.column:
            return ""
        parts = []
        for u in (self.cap + _TRAY_MM * self.k, self.length - self.cap - _TRAY_MM * self.k):
            for a, b in _TRAY_DASHES_MM:
                (xa, ya), (xb, yb) = self._page(a * self.k, u), self._page(b * self.k, u)
                parts.append(f"M{_fmt(xa)},{_fmt(ya)} L{_fmt(xb)},{_fmt(yb)}")
        return " ".join(parts)

    def ink_box(self, w_px: float) -> Box:
        return _grow(self.box, w_px / 2)

    def target(self, x: float, y: float, side: str) -> Point:
        """Where a pipe end at (x, y) entering through `side` meets the outline. A pipe on the
        flank of a dished end (a side port in the cap's zone) is first moved to the straight
        body, so that it joins the wall and not the curve far from its box edge."""
        bx0, by0, bx1, by1 = self.box
        if not self.lying and side in "LR":
            y = min(max(y, by0 + self.cap), by1 - self.cap)
        elif self.lying and side in "TB":
            x = min(max(x, bx0 + self.cap), bx1 - self.cap)
        return _ray_polygon(self.poly, (x, y), side) or (x, y)

    def y_at(self, x: float) -> float:
        """Page y of the top of a standing vessel's outline at `x`."""
        hit = _ray_polygon(self.poly, (x, self.box[1]), "T")
        return hit[1] if hit else self.box[1]

    def bottom_at(self, x: float) -> float:
        """Page y of the bottom of a standing vessel's outline at `x`."""
        hit = _ray_polygon(self.poly, (x, self.box[3]), "B")
        return hit[1] if hit else self.box[3]

    @property
    def diameter(self) -> float:
        return _VESSEL_MM * self.k


@dataclass
class _Agitator:
    """An agitator placed in a vessel: the template (`pl`, at the vessel's scale, its blade low in
    the vessel) and the shaft drawn above it at `x`, from `top` (above the vessel's top) down to
    `joint`, inside the template's own shaft. `joint` <= `top`: the template's shaft is enough."""

    pl: _Placed
    x: float
    top: float
    joint: float

    @property
    def nid(self) -> str:
        return self.pl.nid

    @property
    def tpl(self) -> Template:
        return self.pl.tpl

    def ink_box(self, w_px: float) -> Box:
        x0, y0, x1, y1 = self.pl.ink_box(w_px)
        if self.joint <= self.top:
            return (x0, y0, x1, y1)
        return (min(x0, self.x - w_px / 2), min(y0, self.top), max(x1, self.x + w_px / 2), y1)

    def svg(self, w_px: float) -> str:
        body = f'<g transform="{self.pl.transform()}">{self.tpl.body(stroke=w_px / self.pl.k)}</g>'
        if self.joint <= self.top:
            return body
        shaft = (
            f'<path d="M{_fmt(self.x)},{_fmt(self.top)} L{_fmt(self.x)},{_fmt(self.joint)}" '
            f'fill="none" stroke="black" stroke-width="{_fmt(w_px)}"/>'
        )
        return shaft + body


def _cubic(p0: Point, c1: Point, c2: Point, p1: Point) -> list[Point]:
    out = []
    for i in range(1, _CURVE_STEPS + 1):
        t = i / _CURVE_STEPS
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3
        out.append(
            (
                a * p0[0] + b * c1[0] + c * c2[0] + d * p1[0],
                a * p0[1] + b * c1[1] + c * c2[1] + d * p1[1],
            )
        )
    return out


def _make_vessel(nid: str, tpl: Template, box: Box) -> _Vessel:
    """The vessel of template `tpl` (2062, X8100) drawn on `box`."""
    x0, y0, x1, y1 = box
    lying = tpl.rot in (90, 270)
    diameter, length = ((y1 - y0), (x1 - x0)) if lying else ((x1 - x0), (y1 - y0))
    k = diameter / _VESSEL_MM
    cap = _CAP_MM * k
    # v across, u along: the first end's cap runs out from u = cap to u = 0, the last end's from
    # u = length - cap to u = length
    p0, c1, c2, p1, c3, c4, p2 = ((x * k, d * k) for x, d in _CAP_CURVE)

    def at_first(q: Point) -> Point:
        return (q[0], cap - q[1])

    def at_last(q: Point) -> Point:
        return (q[0], length - cap + q[1])

    vu: list[Point] = [at_first(p0)]
    vu += _cubic(at_first(p0), at_first(c1), at_first(c2), at_first(p1))
    vu += _cubic(at_first(p1), at_first(c3), at_first(c4), at_first(p2))
    vu += [at_last(p2)]  # down the straight body
    vu += _cubic(at_last(p2), at_last(c4), at_last(c3), at_last(p1))
    vu += _cubic(at_last(p1), at_last(c2), at_last(c1), at_last(p0))
    vessel = _Vessel(nid, box, lying, tpl.fine_name == COLUMN, k, [])
    vessel.poly = [vessel._page(v, u) for v, u in vu]
    return vessel


def _inside(poly: list[Point], pt: Point) -> bool:
    """Even-odd test: is `pt` inside the closed polygon `poly`?"""
    x, y = pt
    inside = False
    for (xa, ya), (xb, yb) in zip(poly, [*poly[1:], poly[0]], strict=True):
        if (ya > y) != (yb > y) and x < xa + (y - ya) * (xb - xa) / (yb - ya):
            inside = not inside
    return inside


def _vessel_stub(vessel: _Vessel, target: Point, side: str) -> list[Point] | None:
    """The pipe from the vessel's outline to `target`, a port on `side` of the symbol beside it
    (as `_nozzle_stub` for a box); None when the port is inside the outline, or the vessel is not
    in front of it."""
    if _inside(vessel.poly, target):
        return None
    start = _ray_polygon(vessel.poly, target, _OPPOSITE_SIDE[side])
    if start is None or math.dist(start, target) <= EPS:
        return None
    return [start, target]


def _place_agitator(
    vessel: _Vessel,
    tpl: Template,
    w_px: float,
    top_ends: list[float],
    obstacles: list[Box],
    clearance: float = 0.5,
) -> _Agitator | None:
    """An agitator in a standing vessel: shaft vertical, through the top, the blade low inside.

    As ISO 10628-2 draws it (X8006): the template at the vessel's scale (the blade is half the
    vessel's diameter; no longer than `_AGITATOR_MAX_LENGTH` of its length), its blade
    `clearance` diameters above the vessel's bottom, the shaft drawn on from the template's up
    through the top. When the vessel is too short for that, the template hangs from the top as
    it is. The shaft is on the axis unless a pipe enters the top there (`top_ends`: x of the
    pipe ends on the top); then it moves to the nearest place that clears them, at most
    `_AGITATOR_MAX_OFFSET` of the vessel's width away. A shaft that would reach a symbol
    outside the vessel (`obstacles`: the other symbols' boxes) stops at the outline instead; an
    agitator that overlaps one anyway (a symbol inside the vessel) is left out. None when there
    is no place."""
    ix0, iy0, ix1, iy1 = oriented_ink_box(tpl)
    k = min(vessel.k, _AGITATOR_MAX_LENGTH * vessel.length / (iy1 - iy0))
    blade = (ix1 - ix0) * k
    x0, _, x1, _ = vessel.box
    mid = (x0 + x1) / 2
    off = _AGITATOR_MAX_OFFSET * (x1 - x0)
    lo = max(x0 + blade / 2 + w_px, mid - off)
    hi = min(x1 - blade / 2 - w_px, mid + off)
    clear = 8 * w_px  # only the shaft meets a pipe on the top: the blades are inside
    step = 2 * w_px
    shaft = tpl.ports[0]
    for i in range(int((hi - lo) / step) + 2):
        for x in (mid,) if i == 0 else (mid - i * step, mid + i * step):
            if not (lo <= x <= hi and all(abs(x - t) >= clear for t in top_ends)):
                continue
            low = vessel.bottom_at(x) - clearance * vessel.diameter - (iy1 - iy0) * k
            for out_mm in (_SHAFT_OUT_MM, 0.0):
                top = vessel.y_at(x) - out_mm * k
                ink_top = max(low, top)  # the template's ink top
                pl = _Placed(
                    f"{vessel.nid}/agitator", tpl, k, (x - k * shaft.x_mm, ink_top - k * iy0), []
                )
                ag = _Agitator(pl, x, top, ink_top + w_px if ink_top > top else top)
                if not any(_overlap(ag.ink_box(w_px), o) for o in obstacles):
                    return ag
    return None


# --- the page ---------------------------------------------------------------------------------


@dataclass
class Page:
    """A drawn sheet: `pipes` and `symbols` are SVG fragments, `annotations` one dict per drawn
    symbol (id, node, label, kind, fine_name, template, bbox = ink box; `node` is the PID2Graph
    node it stands for, which is `id` except for the flanges around a disc: id `<node>/flanges`,
    a second box on the disc's node), `log` counters of the cases
    worth knowing about (shrunk / replaced symbols, routes that could not arrive cleanly...),
    `ports` the point where each pipe was joined to a symbol, `stems` each signal line to a bubble
    (for the G1 check)."""

    width: int
    height: int
    pipes: str
    symbols: str
    annotations: list[dict[str, Any]] = field(default_factory=list)
    log: dict[str, int] = field(default_factory=dict)
    ports: list[dict[str, Any]] = field(default_factory=list)  # id, kind, x, y: where a pipe stops
    # signal lines to a bubble: id (symbol), bubble, points, from_centre (a valve's stem)
    stems: list[dict[str, Any]] = field(default_factory=list)

    def _doc(self, body: str) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
            f'<rect width="{self.width}" height="{self.height}" fill="white"/>{body}</svg>'
        )

    @property
    def svg(self) -> str:
        return self._doc(self.pipes + self.symbols)

    def layer(self, which: str) -> str:
        """A document with only the pipes, or only the symbols (for measuring)."""
        return self._doc({"pipes": self.pipes, "symbols": self.symbols}[which])


def rasterize(svg: str, dpi: int = 72) -> np.ndarray:
    """Gray image of an SVG as a uint8 array; at 72 dpi one SVG unit is one pixel."""
    import pymupdf

    pix = pymupdf.open(stream=svg.encode(), filetype="svg")[0].get_pixmap(
        dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False
    )
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def render_page(
    sheet: Sheet,
    runs: list[PipeRun],
    assignments: Mapping[str, Assignment],
    cfg: Mapping[str, Any],
    source: Sheet,
) -> Page:
    """Draw `sheet` (after V4) with the V3 `assignments`. `source` is the sheet before V4."""
    scale = float(cfg["layout"]["scale"])
    page_cfg = cfg["page"]
    w_px = float(page_cfg["stroke_px"])
    steps = [float(s) for s in page_cfg["shrink_steps"]]
    dots = bool(page_cfg["junction_dot"])
    nozzle_frac = float(page_cfg.get("nozzle_stub", 0.0))
    replaced_px = page_cfg.get("replaced_size")
    disc_classes = set((cfg.get("flanges") or {}).get("classes", []))
    flange_symbol = (cfg.get("flanges") or {}).get("symbol")
    k_sheet = _sheet_scale(assignments, source, scale)

    def joint_scale(tpl: Template, bbox_src: Box) -> float:
        """px per mm of a flange pair (alone or around a disc) on a node with this box."""
        return k_sheet if k_sheet is not None else _fit_scale(tpl, bbox_src, scale)

    stub = 2 * w_px
    log = dict.fromkeys(
        (
            "iso", "shrunk", "replaced", "vessels", "agitators", "agitators_skipped",
            "zero_length_runs", "skipped_internal_runs", "unmatched_ports", "loose_ends", "stems",
            "flanges", "nozzles", "resized", "turned", "leaders", "bubbles_moved",
            "bubbles_unsettled",
        ),
        0,
    )  # fmt: skip

    ports = {nid: symbol_ports(sheet, nid, runs) for nid in assignments}
    # arrows lie on their pipe, not in the middle of their PID2Graph box
    arrow_boxes: dict[str, Box] = {}
    arrow_placed: dict[str, _Placed] = {}  # drawn from the arrow template (241)
    leaders: dict[str, tuple[str, list[Point]]] = {}  # head SVG, tail
    for nid, a in assignments.items():
        if a.kind == "leader" and ports[nid]:
            p = ports[nid][0]
            run = runs[p.run]
            far = tuple(run.points[-1] if run.a == nid else run.points[0])  # the line's end
            src_box = _node_box(source.nodes[nid])
            length = scale * max(src_box[2] - src_box[0], src_box[3] - src_box[1])
            head, tail, box = _leader_shape(
                far, sheet.nodes[nid].center, p.side, a.template, length, w_px
            )
            leaders[nid] = (head, tail)
            arrow_boxes[nid] = box
            continue
        if a.kind != "arrow":
            continue
        box = _arrow_box(_node_box(sheet.nodes[nid]), a.axis or "H", ports[nid])
        if a.template is not None:
            arrow_placed[nid] = _place_overlay(
                nid, a.template, box, _node_box(source.nodes[nid]), scale, ports[nid]
            )
            box = arrow_placed[nid].ink_box(w_px)
        arrow_boxes[nid] = box
    placed: dict[str, _Placed] = {}
    size_boxes: dict[str, Box] = {}
    kind: dict[str, str] = {nid: a.kind for nid, a in assignments.items()}
    bubble_stats: dict[str, int] = {}
    circles = _bubble_circles(sheet, runs, kind, w_px, bubble_stats)
    log["bubbles_moved"] = bubble_stats["moved"]  # circles moved to touch their neighbour
    log["bubbles_unsettled"] = bubble_stats["unsettled"]  # pairs left overlapping / apart
    vessels: dict[str, _Vessel] = {}
    for nid, a in assignments.items():
        if a.kind != "iso":
            continue
        if a.template.port_config == "multi":  # drawn on its box, not placed by ports
            kind[nid] = "vessel"
            vessels[nid] = _make_vessel(nid, a.template, _node_box(sheet.nodes[nid]))
            log["vessels"] += 1
            continue
        size_box = _node_box(source.nodes[nid])
        if replaced_px is not None and sheet.nodes[nid].label == "instrumentation":
            # a group-5 device standing in for a bubble: sized like one on a `general` node, not
            # like the bubble (5x larger)
            cx, cy = source.nodes[nid].center
            h = float(replaced_px) / 2
            size_box = (cx - h, cy - h, cx + h, cy + h)
            log["resized"] += 1
        k = None
        disc = a.template.fine_name in disc_classes  # with or without flanges
        if disc:
            k = k_sheet if k_sheet is not None else _disc_scale(a.template, size_box, scale)
        elif a.template.fine_name == flange_symbol:
            k = joint_scale(a.template, size_box)
        size_boxes[nid] = size_box
        placed[nid] = _place(
            nid, a.template, ports[nid], _node_box(sheet.nodes[nid]), size_box, scale, k
        )
        if disc and k is not None:  # in a tight spot it may shrink back to fit its whole box
            placed[nid].min_factor = min(1.0, _fit_scale(a.template, size_box, scale) / k)
        log["iso"] += 1
        log["unmatched_ports"] += max(len(ports[nid]) - len(placed[nid].pairs), 0)

    # an agitator in a standing vessel; the shaft keeps clear of the pipes entering its top
    agitators: dict[str, _Agitator] = {}
    all_ends = _drawn_ends(runs, set()) if vessels else {}
    for nid, vessel in vessels.items():
        if assignments[nid].agitator is None:
            continue
        top_ends = [x for nd, x, _, side, _ in all_ends.values() if nd == nid and side == "T"]
        others = [
            _node_box(sheet.nodes[n]) for n, kd in kind.items() if n != nid and kd != "nested"
        ]
        a = assignments[nid]
        ag = _place_agitator(vessel, a.agitator, w_px, top_ends, others, a.clearance)
        if ag is None:
            log["agitators_skipped"] += 1
        else:
            agitators[nid] = ag
            log["agitators"] += 1

    # a device at the end of a line that sits against a vessel goes to the end of a short nozzle
    nozzles = _nozzle_links(sheet, runs, ports, placed, kind)
    for nid, _, p in nozzles.values():
        if len(ports[nid]) == 1 and nozzle_frac > 0:
            pl = placed[nid]
            x0, y0, x1, y1 = pl.ink_local()
            d = _SIGN[p.side] * nozzle_frac * pl.k0 * ((x1 - x0) if p.side in "LR" else (y1 - y0))
            pl.pos0 = (
                (pl.pos0[0] + d, pl.pos0[1]) if p.side in "LR" else (pl.pos0[0], pl.pos0[1] + d)
            )

    def turns(nid: str) -> list[_Placed]:
        """`nid` placed in its other orientations with the same port sides (a disc's stem on the
        other side of the pipe), at the same size."""
        pl = placed[nid]
        base = pl.tpl.base or pl.tpl
        sides = Counter(pl.tpl.sides)
        out = [
            _place(nid, o, ports[nid], _node_box(sheet.nodes[nid]), size_boxes[nid], scale, pl.k0)
            for o in base.all_orientations()
            if Counter(o.sides) == sides and (o.rot, o.flip) != (pl.tpl.rot, pl.tpl.flip)
        ]
        for alt in out:
            alt.min_factor = pl.min_factor
        return out

    _resolve_overlaps(
        sheet,
        source,
        assignments,
        placed,
        kind,
        steps,
        w_px,
        log,
        float(cfg["layout"]["column_tol"]),
        turns,
        lambda pl: (
            None
            if assignments[pl.nid].flanges is None
            else _place_flanges(
                pl,
                assignments[pl.nid].flanges,
                w_px,
                joint_scale(assignments[pl.nid].flanges, _node_box(source.nodes[pl.nid])),
            ).ink_box(w_px)
        ),
        arrow_boxes,
        circles,
    )

    drawn = [n for n, kd in kind.items() if kd != "nested"]
    boxes = {n: _shape_box(sheet, n, kind[n], placed, w_px, arrow_boxes, circles) for n in drawn}
    # the flanges a disc is held between: placed before the pipes, which stop on them
    flanges = {
        n: _place_flanges(
            placed[n],
            assignments[n].flanges,
            w_px,
            joint_scale(assignments[n].flanges, _node_box(source.nodes[n])),
        )
        for n in placed
        if kind[n] == "iso" and assignments[n].flanges is not None
    }

    # --- pipes -----------------------------------------------------------------------------
    targets: dict[tuple[int, str], tuple[Point, str]] = {}
    port_log: list[dict[str, Any]] = []
    stems, touching = _stems(sheet, runs, placed, kind, w_px, circles)
    stem_runs = {idx for idxs, _ in stems.values() for idx in idxs}
    degenerate = _degenerate_runs(sheet, runs, kind) | touching
    ends = _drawn_ends(runs, degenerate | stem_runs)
    node_ends = {
        (p.run, e)
        for nid in drawn
        for p in ports[nid]
        for e in "ab"
        if _at_end(runs[p.run], e, nid, (p.x, p.y))
    }
    stem_log: list[dict[str, Any]] = []
    for (valve, bubble), (_, pts) in stems.items():
        start, end = pts[0], pts[-1]
        centre_start = placed[valve].tpl.group in _STEM_CENTRE_GROUPS
        stem_log.append({"id": valve, "bubble": bubble, "points": pts, "from_centre": centre_start})
        entry: dict[str, Any] = {
            "id": valve,
            "kind": "iso",
            "x": start[0],
            "y": start[1],
            "loose": True,
        }
        gap = _gap_of(placed[valve])
        if gap is not None and math.dist(start, gap[0]) < EPS:  # not when it leaves from a port
            entry["gap"] = {"axis": gap[1], "half": gap[2]}  # the start is in the flange gap
        port_log.append(entry)
        port_log.append({"id": bubble, "kind": "bubble", "x": end[0], "y": end[1], "loose": True})
        log["stems"] += 1
    for nid in drawn:
        if kind[nid] in ("arrow", "leader"):  # drawn to the centre of its box / no pipe: below
            continue
        loose = [
            Port(x, y, side, other, idx)
            for (idx, e), (nd, x, y, side, other) in ends.items()
            if nd == nid and (idx, e) not in node_ends
        ]
        log["loose_ends"] += len(loose)
        for p, target, side in _port_targets(
            sheet,
            nid,
            kind[nid],
            placed,
            ports[nid],
            assignments[nid],
            loose,
            flanges.get(nid),
            circles.get(nid),
            vessels.get(nid),
        ):
            for e in "ab":
                if (p.run, e) in ends and ends[(p.run, e)][:3] == (nid, p.x, p.y):
                    targets[(p.run, e)] = (target, side)
                    port_log.append(
                        {
                            "id": nid,
                            "kind": kind[nid],
                            "x": target[0],
                            "y": target[1],
                            "loose": p in loose,
                        }
                    )
                    break
    polylines: list[list[Point]] = []
    for idx, run in enumerate(runs):
        if run.internal:
            log["skipped_internal_runs"] += 1
            continue
        if idx in stem_runs:  # drawn as a straight line below
            continue
        if "leader" in (kind.get(run.a), kind.get(run.b)):  # a tag's line, not a pipe
            continue
        pts = [tuple(q) for q in run.points]
        if len(set(pts)) <= 1 or idx in degenerate:
            log["zero_length_runs"] += 1
            continue
        for end in ("a", "b"):
            if (idx, end) in targets:
                target, side = targets[(idx, end)]
                pts = _attach_end(pts, end, target, side, stub, w_px / 2)
        polylines.append(_simplify(pts))
    for idx, (nid, box_node, _) in nozzles.items():
        hits = [
            (t, sd)
            for q, t, sd in _port_targets(
                sheet, nid, "iso", placed, ports[nid], assignments[nid], [], flanges.get(nid)
            )
            if q.run == idx
        ]
        if not hits:
            continue
        target, side = hits[0]
        if box_node in vessels:
            line = _vessel_stub(vessels[box_node], target, side)
        else:
            line = _nozzle_stub(_node_box(sheet.nodes[box_node]), target, side)
        if line is None:
            continue
        polylines.append(_extend(line, w_px / 2))
        port_log.append({"id": nid, "kind": "iso", "x": target[0], "y": target[1], "loose": False})
        log["nozzles"] += 1

    pipe_parts = [f'<polyline points="{_pts(pl)}"/>' for pl in polylines if len(pl) >= 2]
    pipe_parts += [f'<polyline points="{_pts(pts)}"/>' for _, pts in stems.values()]
    arrows: list[str] = []
    for nid in drawn:
        if kind[nid] != "arrow":
            continue
        a = assignments[nid]
        box = arrow_boxes[nid]
        centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        for p in ports[nid]:
            corner = (centre[0], p.y) if p.side in "LR" else (p.x, centre[1])
            pipe_parts.append(
                f'<polyline points="{_pts(_simplify([(p.x, p.y), corner, centre]))}"/>'
            )
        if nid in arrow_placed:
            pl = arrow_placed[nid]
            arrows.append(f'<g transform="{pl.transform()}">{pl.tpl.body(stroke=w_px / pl.k)}</g>')
        else:
            poly = _arrow_polygon(box, a.axis or "H", bool(a.forward))
            arrows.append(f'<polygon points="{_pts(poly)}" fill="black" stroke="none"/>')
    for nid, (head, tail) in leaders.items():
        if kind.get(nid) != "leader":
            continue
        log["leaders"] += 1
        arrows.append(
            f'<polyline points="{_pts(tail)}" fill="none" stroke="black" '
            f'stroke-width="{_fmt(w_px * _LEADER_STROKE)}"/>'
        )
        arrows.append(head)
    if dots:
        for nid, node in sheet.nodes.items():
            if node.label == "crossing" and len(sheet.adjacency[nid]) == 3:
                cx, cy = node.center
                pipe_parts.append(
                    f'<circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(w_px)}" fill="black" '
                    f'stroke="none"/>'
                )
    pipes = (
        f'<g fill="none" stroke="black" stroke-width="{_fmt(w_px)}" stroke-linecap="butt" '
        f'stroke-linejoin="miter">{"".join(pipe_parts)}</g>'
    )

    # --- symbols and annotations -----------------------------------------------------------
    sym_parts: list[str] = list(arrows)
    outline: list[str] = []
    annotations: list[dict[str, Any]] = []
    for nid in drawn:
        a = assignments[nid]
        kd = kind[nid]
        label = a.label
        fine = None
        tpl_file = None
        if kd == "arrow" and nid in arrow_placed:
            fine, tpl_file = arrow_placed[nid].tpl.fine_name, arrow_placed[nid].tpl.file
        if kd == "leader" and a.template is not None:
            tpl_file = a.template.file  # the head; the class is leader_arrow, not 241's
        if kd == "iso":
            pl = placed[nid]
            sym_parts.append(
                f'<g transform="{pl.transform()}">{pl.tpl.body(stroke=w_px / pl.k)}</g>'
            )
            fine, tpl_file = pl.tpl.fine_name, pl.tpl.file
            if nid in flanges:  # a symbol of its own, labelled apart from the disc
                fl = flanges[nid]
                sym_parts.append(
                    f'<g transform="{fl.transform()}">{fl.tpl.body(stroke=w_px / fl.k)}</g>'
                )
                annotations.append(
                    {
                        "id": fl.nid,
                        "node": nid,
                        "label": fl.tpl.fine_name if fl.tpl.in_vocab else OTHER_ISO,
                        "kind": "iso",
                        "fine_name": fl.tpl.fine_name,
                        "template": fl.tpl.file,
                        "bbox": [round(v, 2) for v in fl.ink_box(w_px)],
                    }
                )
                log["flanges"] += 1
        elif kd == "vessel":
            vessel = vessels[nid]
            fine, tpl_file = a.template.fine_name, a.template.file
            outline.append(f'<path d="{vessel.outline_d()}"/>')
            if vessel.column:
                outline.append(f'<path d="{vessel.trays_d()}"/>')
            if nid in agitators:
                ag = agitators[nid]
                sym_parts.append(ag.svg(w_px))
                annotations.append(
                    {
                        "id": ag.nid,
                        "node": nid,
                        "label": ag.tpl.fine_name if ag.tpl.in_vocab else OTHER_ISO,
                        "kind": "iso",
                        "fine_name": ag.tpl.fine_name,
                        "template": ag.tpl.file,
                        "bbox": [round(v, 2) for v in ag.ink_box(w_px)],
                        "added": True,  # PID2Graph has no such node: picture and COCO only
                    }
                )
        elif kd == "generic":
            label = GENERIC_LABEL
            b = _node_box(sheet.nodes[nid])
            outline.append(
                f'<rect x="{_fmt(b[0])}" y="{_fmt(b[1])}" width="{_fmt(b[2] - b[0])}" '
                f'height="{_fmt(b[3] - b[1])}"/>'
            )
        elif kd == "bubble":
            (cx, cy), r = circles[nid]
            outline.append(f'<circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}"/>')
        elif kd == "flag":
            poly = _flag_polygon(_node_box(sheet.nodes[nid]), a.axis or "H", bool(a.forward))
            outline.append(f'<polygon points="{_pts(poly)}"/>')
        annotations.append(
            {
                "id": nid,
                "node": nid,
                "label": label,
                "kind": "iso" if kd == "vessel" else kd,
                "fine_name": fine,
                "template": tpl_file,
                "bbox": [round(v, 2) for v in boxes[nid]],
            }
        )
    symbols = "".join(sym_parts)
    if outline:
        symbols += (
            f'<g fill="none" stroke="black" stroke-width="{_fmt(w_px)}" '
            f'stroke-linejoin="miter">{"".join(outline)}</g>'
        )
    return Page(sheet.width, sheet.height, pipes, symbols, annotations, log, port_log, stem_log)


def _shape_box(
    sheet: Sheet,
    nid: str,
    kd: str,
    placed: Mapping[str, _Placed],
    w_px: float,
    arrows: Mapping[str, Box] | None = None,
    circles: Mapping[str, Circle] | None = None,
) -> Box:
    """Ink box of whatever is drawn for `nid`, in page px. `arrows`: arrow boxes moved onto
    their pipe (`_arrow_box`); `circles`: the bubbles' circles (`_bubble_circles`)."""
    if kd == "iso":
        return placed[nid].ink_box(w_px)
    if kd in ("arrow", "leader") and arrows is not None and nid in arrows:
        return arrows[nid]
    box = _node_box(sheet.nodes[nid])
    if kd == "bubble":
        if circles is not None and nid in circles:
            (cx, cy), r = circles[nid]
        else:
            r = min(box[2] - box[0], box[3] - box[1]) / 2
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        box = (cx - r, cy - r, cx + r, cy + r)
    return box if kd == "arrow" else _grow(box, w_px / 2)


def _port_targets(
    sheet: Sheet,
    nid: str,
    kd: str,
    placed: dict[str, _Placed],
    node_ports: list[Port],
    a: Assignment,
    loose: list[Port],
    flanges: _Placed | None = None,
    circle: Circle | None = None,
    vessel: _Vessel | None = None,
) -> list[tuple[Port, Point, str]]:
    """(pipe end, where it should stop, side it enters) for each pipe end of a symbol.

    `node_ports` are the node's ports, `loose` pipe ends that are not (a signal line to an
    actuator): they stop on the outline of whatever is drawn. So does a port of an ISO symbol
    that has no template port on its side (a pump piped from one side only). A disc held between
    `flanges` has its pipes stop on the flanges' outer faces, as on a lone flanged connection:
    no pipe runs between a flange mark and the disc's stem.
    """
    out: list[tuple[Port, Point, str]] = []
    if kd == "iso":
        pl = placed[nid]
        x0, y0, x1, y1 = pl.ink_local()
        free = list(loose)
        for p, q in pl.pairs:
            if p.side != q.side:
                free.append(p)
                continue
            fq = next((f for f in flanges.tpl.ports if f.side == q.side), None) if flanges else None
            if fq is not None:
                out.append((p, flanges.to_page(fq.x_mm, fq.y_mm), q.side))
                continue
            # a port a fraction of a mm off the drawing: stop on the ink's edge instead
            x = min(max(q.x_mm, x0), x1) if q.side in "LR" else q.x_mm
            y = min(max(q.y_mm, y0), y1) if q.side in "TB" else q.y_mm
            out.append((p, pl.to_page(x, y), q.side))
        out += [(p, pl.ink_target(p), p.side) for p in free]
        return out
    ends = [*node_ports, *loose]
    if kd == "vessel" and vessel is not None:
        out += [(p, vessel.target(p.x, p.y, p.side), p.side) for p in ends]
    elif kd == "bubble":
        if circle is None:
            b = _node_box(sheet.nodes[nid])
            circle = (((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), min(b[2] - b[0], b[3] - b[1]) / 2)
        centre, r = circle
        for p in ends:
            out.append((p, _circle_hit(centre, r, (p.x, p.y), p.side), p.side))
    elif kd == "flag":
        poly = _flag_polygon(_node_box(sheet.nodes[nid]), a.axis or "H", bool(a.forward))
        ys = [q[1] for q in poly]
        xs = [q[0] for q in poly]
        for p in ends:
            hit = _ray_polygon(poly, (p.x, p.y), p.side)
            if hit is None:  # the line misses the outline: aim at its nearest extent
                lo, hi = (min(ys), max(ys)) if p.side in "LR" else (min(xs), max(xs))
                aim = (
                    (p.x, min(max(p.y, lo + EPS), hi - EPS))
                    if p.side in "LR"
                    else (min(max(p.x, lo + EPS), hi - EPS), p.y)
                )
                hit = _ray_polygon(poly, aim, p.side) or aim
            out.append((p, hit, p.side))
    elif kd == "generic":
        out += [(p, (p.x, p.y), p.side) for p in loose]  # the box edge is where V4 put the end
    return out


def touching_pairs(source: Sheet, ids: list[str], tol: float) -> set[frozenset[str]]:
    """Pairs of `ids` whose boxes in `source` overlap or lie within `tol` px of each other.

    V4 closes every gap of at most `column_tol` to zero, and the ink box is a stroke wider than
    the bbox, so such pairs (a valve and its actuator, a flange against a tank) end up
    overlapping on the page although they never did in the data. They are not "new".
    """
    grown = {n: _grow(_node_box(source.nodes[n]), tol / 2 + EPS) for n in ids}
    return {frozenset((a, b)) for a, b in combinations(ids, 2) if _overlap(grown[a], grown[b])}


def _resolve_overlaps(
    sheet: Sheet,
    source: Sheet,
    assignments: Mapping[str, Assignment],
    placed: dict[str, _Placed],
    kind: dict[str, str],
    steps: list[float],
    w_px: float,
    log: dict[str, int],
    tol: float,
    turns: Callable[[str], list[_Placed]] | None = None,
    extra: Callable[[_Placed], Box | None] | None = None,
    arrows: Mapping[str, Box] | None = None,
    circles: Mapping[str, Circle] | None = None,
) -> None:
    """Newly overlapping pairs (not touching in `source`, see `touching_pairs`): the larger ISO
    symbol is first turned (`turns`: its other orientations with the same port sides) when one
    of them overlaps nothing new, else shrunk by `steps`, then replaced with a plain box. Pairs of
    two non-ISO shapes are left alone. `extra` gives more of a symbol's drawing (the flanges
    around a disc), counted as part of its box."""
    ids = [n for n, kd in kind.items() if kd != "nested"]

    def shape(n: str, pls: Mapping[str, _Placed]) -> Box:
        b = _shape_box(sheet, n, kind[n], pls, w_px, arrows, circles)
        more = extra(pls[n]) if extra is not None and kind[n] == "iso" else None
        if more is None:
            return b
        return (min(b[0], more[0]), min(b[1], more[1]), max(b[2], more[2]), max(b[3], more[3]))

    stage = dict.fromkeys(placed, -1)  # last shrink step applied; len(steps) = replaced
    originally = touching_pairs(source, ids, tol)
    tried: set[str] = set()
    while True:
        boxes = {n: shape(n, placed) for n in ids}
        victim: str | None = None
        for a, b in combinations(ids, 2):
            if frozenset((a, b)) in originally or not _overlap(boxes[a], boxes[b]):
                continue
            cands = [n for n in (a, b) if kind[n] == "iso"]
            if not cands:
                continue
            victim = max(cands, key=lambda n: _area(boxes[n]))
            break
        if victim is None:
            return
        if turns is not None and victim not in tried:
            tried.add(victim)
            others = [n for n in ids if n != victim]
            for alt in turns(victim):
                box = shape(victim, {victim: alt})
                if not any(
                    frozenset((victim, n)) not in originally and _overlap(box, boxes[n])
                    for n in others
                ):
                    placed[victim] = alt
                    log["turned"] += 1
                    break
            else:
                alt = None
            if alt is not None:
                continue
        stage[victim] += 1
        pl = placed[victim]
        if stage[victim] < len(steps):
            pl.factor = steps[stage[victim]]
            if stage[victim] == 0:
                log["shrunk"] += 1
        elif pl.min_factor is not None and pl.factor > pl.min_factor + EPS:
            # a disc in a tight spot: smaller, down to its old size, rather than a box
            pl.factor = max(pl.factor - 0.1, pl.min_factor)
            if stage[victim] == 0:
                log["shrunk"] += 1
        else:
            kind[victim] = "generic"
            log["replaced"] += 1
            log["iso"] -= 1
