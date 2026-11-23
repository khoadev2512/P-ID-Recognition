"""Layout jitter that keeps the topology (V4 of PLAN_ISO_RENDER_PID2GRAPH).

`transform_layout` scales a sheet by S and varies the spacing of its rows and columns, without
touching the graph: same node ids, labels and edges, pipes still axis-aligned, pipe ends still on
their symbol's edge. It maps *every* coordinate (bbox edges, connector / crossing centres, pipe
points) through one monotone map per axis, so nothing can drift apart:

  - the two ports of a symbol piped straight through (`2_straight`) are tied: their coordinate
    across the pipe lands on one position, so the pipes either side of it stay on one line;
    with `layout.straighten_pipes` (PLAN_PHASE_C §2M) so are the two pipe ends meeting from
    opposite sides at any node, and the two ends of a small step inside one run (`_pipe_ties`);
  - values are grouped into columns (a new column opens once a value is more than `column_tol`
    past the column's first value; no chaining) and every value of a column lands on one position;
  - the gap between two neighbouring columns is the original gap x S x a factor drawn from
    `gap_scale`, except where a symbol's bbox spans the gap: that gap is exactly x S, so symbols
    keep their size (up to +-column_tol x S from the edges snapping to column starts).

Because the map is strictly increasing across columns and constant inside one, equal values stay
equal (axis-aligned stays axis-aligned), the order of all edges is kept (an overlap can vanish
when its depth is within `column_tol`, never appear), and V3's port sides stay valid.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from utils.pid2graph import Node, PipeRun, Sheet, _simplify, port_config, symbol_ports

NOT_SYMBOLS = ("connector", "crossing", "background")
CENTRE_ONLY = ("connector", "crossing")  # drawn as a point: centre mapped, size x S
BUBBLE_LABEL = "instrumentation"  # an ISA bubble: V5 draws its circle from the V4 box


class AxisMap:
    """Monotone, piecewise-constant map of one axis; built by `build_axis_map`."""

    def __init__(self, anchors: list[float], positions: list[float]) -> None:
        self.anchors = anchors
        self.positions = positions

    def __call__(self, value: float) -> float:
        """Position of `value`'s column; anything below the first column takes the first one."""
        return self.positions[max(bisect_right(self.anchors, value) - 1, 0)]


def column_anchors(values: list[float], tol: float) -> list[float]:
    """First value of every column: a value joins the open column while it is within `tol` of it."""
    anchors: list[float] = []
    for v in sorted(values):
        if not anchors or v - anchors[-1] > tol:
            anchors.append(v)
    return anchors


def build_axis_map(
    values: list[float],
    spans: list[tuple[float, float]],
    scale: float,
    tol: float,
    gap_scale: tuple[float, float],
    rng: random.Random,
    ties: list[tuple[float, float]] | None = None,
    free_ties: list[tuple[float, float]] | None = None,
    hard_spans: list[tuple[float, float]] | None = None,
) -> AxisMap:
    """Map of one axis. `values` are every coordinate that will go through it, `spans` the
    [lo, hi] extent of each symbol bbox (gaps they cover are not jittered), `ties` pairs of
    values that must land on one position (the two ports of a straight fitting): every column
    from the first value's to the second's becomes one, so the map stays monotone. A tie that
    would move the edge of a symbol bbox (and so resize the symbol) is not made. `free_ties`
    (PLAN_PHASE_C §2M) are made whatever edge lies there: V5 fits every symbol to its source
    bbox, so a V4 box edge moving a few px resizes nothing that is drawn) — except the edges of
    `hard_spans`: symbols drawn from their V4 box (ISA bubbles: a smaller circle would leave two
    bubbles that touch in the data apart)."""
    anchors = column_anchors(values, tol)
    edges = sorted(e for span in spans for e in span)
    hard = sorted(e for span in hard_spans or [] for e in span)
    for i, (lo, hi) in enumerate([*(ties or []), *(free_ties or [])]):
        first, last = bisect_right(anchors, min(lo, hi)) - 1, bisect_right(anchors, max(lo, hi)) - 1
        if last <= first:
            continue
        moved = anchors[first + 1], (anchors[last + 1] if last + 1 < len(anchors) else math.inf)
        guarded = i < len(ties or [])
        if bisect_left(edges if guarded else hard, moved[1]) > bisect_left(
            edges if guarded else hard, moved[0]
        ):
            continue  # a (guarding) symbol edge lies in the columns that would move
        anchors = anchors[: first + 1] + anchors[last + 1 :]
    rigid = [False] * max(len(anchors) - 1, 0)
    for lo, hi in spans:
        first, last = bisect_right(anchors, lo) - 1, bisect_right(anchors, hi) - 1
        for k in range(first, last):
            rigid[k] = True
    positions = [anchors[0] * scale]
    for k in range(len(anchors) - 1):
        factor = 1.0 if rigid[k] else rng.uniform(*gap_scale)
        positions.append(positions[-1] + (anchors[k + 1] - anchors[k]) * scale * factor)
    return AxisMap(anchors, positions)


def _straight_ties(
    sheet: Sheet, runs: list[PipeRun], limit: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """(x ties, y ties): for every symbol piped straight through (`2_straight`), the coordinate of
    its two ports across the pipe, so that after the map the two pipes lie on one line and a
    template with aligned ports needs no jog. Offsets above `limit` source px are real jogs of the
    pipe, not measurement noise, and are kept."""
    x_ties: list[tuple[float, float]] = []
    y_ties: list[tuple[float, float]] = []
    for nid, node in sheet.nodes.items():
        if node.label in NOT_SYMBOLS:
            continue
        ports = symbol_ports(sheet, nid, runs)
        cfg = port_config(ports)
        if cfg.kind != "2_straight":
            continue
        a, b = ports
        if cfg.axis == "H" and 0 < abs(a.y - b.y) <= limit:
            y_ties.append((a.y, b.y))
        elif cfg.axis == "V" and 0 < abs(a.x - b.x) <= limit:
            x_ties.append((a.x, b.x))
    return x_ties, y_ties


def _axis_of(a: tuple[float, float], b: tuple[float, float]) -> str | None:
    if a[1] == b[1] and a[0] != b[0]:
        return "H"
    if a[0] == b[0] and a[1] != b[1]:
        return "V"
    return None


def _pipe_ties(
    runs: list[PipeRun], limit: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """(x ties, y ties) that take the small steps out of near-straight pipes (PLAN_PHASE_C §2M):

      - at any node (an arrow, a crossing, a tee, a symbol), two run ends arriving along one axis
        from opposite sides: their coordinate across the axis;
      - inside one run, a short perpendicular step between two parallel stretches running the
        same way: the coordinates of its two ends.

    Only offsets of at most `limit` source px are tied (measurement noise of the source drawing,
    as `_straight_ties`); larger ones are real jogs of the pipe and are kept."""
    x_ties: list[tuple[float, float]] = []
    y_ties: list[tuple[float, float]] = []

    def tie(axis: str, u: float, v: float) -> None:
        if 0 < abs(u - v) <= limit:
            (y_ties if axis == "H" else x_ties).append((u, v))

    ends: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = {}
    for run in runs:
        pts = run.points
        if run.internal or run.nozzle or len(pts) < 2:
            continue
        ends.setdefault(run.a, []).append((pts[0], pts[1]))
        ends.setdefault(run.b, []).append((pts[-1], pts[-2]))
        for i in range(len(pts) - 3):
            a, b, c, d = pts[i : i + 4]
            s1, s2, s3 = _axis_of(a, b), _axis_of(b, c), _axis_of(c, d)
            if s1 is None or s1 != s3 or s2 is None or s2 == s1:
                continue
            k = 0 if s1 == "H" else 1
            if (b[k] - a[k]) * (d[k] - c[k]) > 0:
                tie(s1, b[1 - k], c[1 - k])
    for arrivals in ends.values():
        for axis in ("H", "V"):
            k = 0 if axis == "H" else 1  # along the pipe
            side: dict[int, list[float]] = {1: [], -1: []}
            for p, p_in in arrivals:
                if _axis_of(p_in, p) == axis:
                    side[1 if p[k] > p_in[k] else -1].append(p[1 - k])
            # exactly one pipe in from each side: one pipe through the node (a vessel's nozzles
            # on opposite walls, or a valve's signal line, are other pipes and are left alone)
            if len(side[1]) == 1 and len(side[-1]) == 1:
                tie(axis, side[1][0], side[-1][0])
    return x_ties, y_ties


def transform_layout(
    sheet: Sheet,
    runs: list[PipeRun],
    rng: random.Random,
    cfg: Mapping[str, Any],
) -> tuple[Sheet, list[PipeRun]]:
    """The sheet and its pipe runs after scaling and axis remapping (see the module doc).

    `cfg` is the `render_iso` mapping; its `layout` holds `scale`, `column_tol` (source px) and
    `gap_scale` ([low, high]). The x map is drawn before the y map, so a seed reproduces both.
    """
    layout = cfg["layout"]
    scale, tol = float(layout["scale"]), float(layout["column_tol"])
    gap_scale = (float(layout["gap_scale"][0]), float(layout["gap_scale"][1]))

    xs = [0.0, float(sheet.width)]
    ys = [0.0, float(sheet.height)]
    x_spans: list[tuple[float, float]] = []
    y_spans: list[tuple[float, float]] = []
    x_hard: list[tuple[float, float]] = []  # boxes drawn as they come out of V4 (bubbles)
    y_hard: list[tuple[float, float]] = []
    for node in sheet.nodes.values():
        if node.label in CENTRE_ONLY:
            cx, cy = node.center
            xs.append(cx)
            ys.append(cy)
            continue
        xs += [node.xmin, node.xmax]
        ys += [node.ymin, node.ymax]
        if node.label not in NOT_SYMBOLS:
            x_spans.append((node.xmin, node.xmax))
            y_spans.append((node.ymin, node.ymax))
        if node.label == BUBBLE_LABEL:
            x_hard.append((node.xmin, node.xmax))
            y_hard.append((node.ymin, node.ymax))
    for run in runs:
        xs += [p[0] for p in run.points]
        ys += [p[1] for p in run.points]

    limit = float(layout.get("straight_max", 15))
    x_ties, y_ties = _straight_ties(sheet, runs, limit)
    x_free: list[tuple[float, float]] = []
    y_free: list[tuple[float, float]] = []
    if layout.get("straighten_pipes", False):  # PLAN_PHASE_C §2M (OPEN100-ISO v2)
        x_free, y_free = _pipe_ties(runs, limit)
    x_map = build_axis_map(xs, x_spans, scale, tol, gap_scale, rng, x_ties, x_free, x_hard)
    y_map = build_axis_map(ys, y_spans, scale, tol, gap_scale, rng, y_ties, y_free, y_hard)

    nodes: dict[str, Node] = {}
    for nid, node in sheet.nodes.items():
        if node.label in CENTRE_ONLY:
            cx, cy = node.center
            mx, my = x_map(cx), y_map(cy)
            hw, hh = node.width * scale / 2, node.height * scale / 2
            nodes[nid] = replace(node, xmin=mx - hw, xmax=mx + hw, ymin=my - hh, ymax=my + hh)
        else:
            nodes[nid] = replace(
                node,
                xmin=x_map(node.xmin),
                xmax=x_map(node.xmax),
                ymin=y_map(node.ymin),
                ymax=y_map(node.ymax),
            )

    new_runs: list[PipeRun] = []
    for run in runs:
        points = [(x_map(x), y_map(y)) for x, y in run.points]
        # a nozzle is a zero-length link whose two points must both survive
        new_runs.append(replace(run, points=tuple(points if run.nozzle else _simplify(points))))

    new_sheet = Sheet(
        name=sheet.name,
        width=int(round(x_map(sheet.width))),
        height=int(round(y_map(sheet.height))),
        nodes=nodes,
        edges=list(sheet.edges),
    )
    return new_sheet, new_runs
