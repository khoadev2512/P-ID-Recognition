"""PID2Graph OPEN100 GraphML -> pipe skeleton + symbol ports (V2 of PLAN_ISO_RENDER_PID2GRAPH).

What the GraphML actually holds (measured on the 12 OPEN100 sheets, see the plan §0):
  - nodes carry a label and a bbox; there is no text and no pipe geometry on edges (edges are
    centre-to-centre links);
  - the pipe's real path lives in the `connector` nodes: 8x8 px, always degree 2, placed at
    every bend of a pipe and flush with the end of each symbol (the connector centre sits on
    the symbol's bbox edge, within 0.5 px);
  - `crossing` nodes are junctions (degree 3), true crossings (4), and also plain bends or
    dead ends (degrees 2 / 1).

So a *pipe run* is a chain of connectors between two "terminals" (any non-connector,
non-background node), and a symbol's *ports* are where those chains leave its bbox. Nothing
here re-routes pipes: the polyline is the annotated path, with near-axis segments snapped
onto the axis and genuinely diagonal ones replaced by an L (never a diagonal in the output).

The connector/crossing structure is read here for geometry only; it is not collapsed the way
SynthPID's collapse_connectors does, because that loses the bends and the port positions.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_NS = "{http://graphml.graphdrawing.org/xmlns}"

CONNECTOR = "connector"
BACKGROUND = "background"
# Neighbours that are not pipes into the symbol: a valve's actuator link goes through an
# instrumentation node (40 direct edges in OPEN100), and `background` is the page frame.
DEFAULT_IGNORE_LABELS = ("instrumentation", BACKGROUND)

# Segments whose off-axis offset is at most this many px are snapped onto the axis.
AXIS_TOL_PX = 2.0
# A first connector within this many px of its symbol's bbox edge *is* the port. A deeper one
# (tank / large `general` nozzles are drawn up to ~50 px inside) is not: the port is then
# wherever the pipe's polyline actually crosses the bbox boundary.
PORT_TOL_PX = 2.0
# A polyline point counts as outside a bbox only beyond this margin.
EXIT_TOL_PX = 1.0
# PID2Graph marks some tank nozzles as a zero-length loop `tank - connector - connector - tank`
# (the two connectors coincide) and leaves the nozzle's flange, a small `general` ~10-18 px
# away, attached only to the outgoing pipe. A loop shorter than NOZZLE_LOOP_PX is such a mark;
# the flange is the nearest unclaimed `general` of degree <= 1 within NOZZLE_RADIUS_PX whose
# bbox is at most NOZZLE_MAX_SIZE_PX across.
NOZZLE_LOOP_PX = 1.0
NOZZLE_RADIUS_PX = 25.0
NOZZLE_MAX_SIZE_PX = 40.0

Point = tuple[float, float]


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center(self) -> Point:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass
class Sheet:
    name: str
    width: int
    height: int
    nodes: dict[str, Node]
    edges: list[tuple[str, str]]
    adjacency: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for u, v in self.edges:
            adj[u].append(v)
            adj[v].append(u)
        self.adjacency = adj

    def is_terminal(self, node_id: str) -> bool:
        """Ends a pipe run: anything but a degree-2 connector (background is never a pipe)."""
        node = self.nodes[node_id]
        if node.label == BACKGROUND:
            return False
        if node.label != CONNECTOR:
            return True
        return len(self.adjacency[node_id]) != 2


@dataclass(frozen=True)
class PipeRun:
    """One pipe between two terminals; `points` run from `a`'s end to `b`'s end."""

    a: str
    b: str
    points: tuple[Point, ...]
    connectors: tuple[str, ...]
    elbows: int = 0
    warnings: tuple[str, ...] = ()
    # Side of the bbox each end leaves from; None at a crossing, and on internal runs.
    a_side: str | None = None
    b_side: str | None = None
    # Zero-length link that seats a nozzle flange (`b`, a small `general`) on the tank `a`: both
    # ends are the same point, `b_side` faces the tank. Place the flange at that point.
    nozzle: bool = False
    # The pipe never leaves a symbol's bbox: its other end is nested inside (an arrow or a
    # `general` within a tank). ISO templates have no interior, so V3 / V5 skip these runs.
    internal: bool = False


@dataclass(frozen=True)
class Port:
    """Where a pipe leaves a symbol. (x, y) lies on the bbox edge named by `side`, except the
    flange end of a `nozzle` run, which is the tank's port (the flange is to be moved there)."""

    x: float
    y: float
    side: str  # "L" | "R" | "T" | "B"
    neighbor: str  # the terminal at the far end of the pipe
    run: int  # index into pipe_runs(sheet)


@dataclass(frozen=True)
class PortConfig:
    kind: str  # "0" | "1" | "2_straight" | "2_angle" | "2_same" | "3" | "4" | "multi"
    axis: str | None = None  # "H" / "V" for 2_straight; None otherwise
    sides: tuple[str, ...] = ()


def load_graphml(path: str | Path, image_size: tuple[int, int] | None = None) -> Sheet:
    """Read one OPEN100 GraphML. `image_size=(w, h)` overrides the size inferred from the
    nodes (they disagree on sheet 11: 2792 px of boxes on a 2786 px image)."""
    path = Path(path)
    root = ET.parse(path).getroot()
    key_names = {
        k.attrib["id"]: k.attrib.get("attr.name", k.attrib["id"]) for k in root.iter(f"{_NS}key")
    }

    nodes: dict[str, Node] = {}
    for xml_node in root.iter(f"{_NS}node"):
        values = {
            key_names[d.attrib["key"]]: (d.text or "") for d in xml_node.findall(f"{_NS}data")
        }
        nid = xml_node.attrib["id"]
        nodes[nid] = Node(
            id=nid,
            label=values["label"].strip(),
            xmin=float(values["xmin"]),
            ymin=float(values["ymin"]),
            xmax=float(values["xmax"]),
            ymax=float(values["ymax"]),
        )

    edges = [(e.attrib["source"], e.attrib["target"]) for e in root.iter(f"{_NS}edge")]
    if image_size is None:
        width = int(round(max(n.xmax for n in nodes.values())))
        height = int(round(max(n.ymax for n in nodes.values())))
    else:
        width, height = image_size
    return Sheet(name=path.stem, width=width, height=height, nodes=nodes, edges=edges)


def _nearest_side(node: Node, p: Point) -> str:
    dists = {
        "L": abs(p[0] - node.xmin),
        "R": abs(p[0] - node.xmax),
        "T": abs(p[1] - node.ymin),
        "B": abs(p[1] - node.ymax),
    }
    return min(dists, key=dists.__getitem__)


def _edge_point(node: Node, side: str, p: Point) -> Point:
    """`p` projected onto the `side` edge of `node`'s bbox, clamped to the edge's extent."""
    x = min(max(p[0], node.xmin), node.xmax)
    y = min(max(p[1], node.ymin), node.ymax)
    if side == "L":
        return (node.xmin, y)
    if side == "R":
        return (node.xmax, y)
    if side == "T":
        return (x, node.ymin)
    return (x, node.ymax)


@dataclass(frozen=True)
class _End:
    point: Point  # where the polyline starts at this terminal
    side: str | None  # bbox side it leaves from; None at a crossing / when internal
    skip: int  # connectors (from this end) already accounted for by `point`
    internal: bool = False


def _outside(node: Node, p: Point, tol: float) -> bool:
    return (
        p[0] < node.xmin - tol
        or p[0] > node.xmax + tol
        or p[1] < node.ymin - tol
        or p[1] > node.ymax + tol
    )


def _exit_point(node: Node, path: list[Point]) -> tuple[int, Point, str] | None:
    """Where `path` (path[0] inside `node`) first leaves the bbox.

    Returns (index of the first point outside, the point on the bbox edge, the edge side),
    or None when the whole path stays inside.
    """
    for k in range(1, len(path)):
        if not _outside(node, path[k], EXIT_TOL_PX):
            continue
        p0, p1 = path[k - 1], path[k]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        hits: list[tuple[float, str]] = []
        if dx > 0:
            hits.append(((node.xmax - p0[0]) / dx, "R"))
        elif dx < 0:
            hits.append(((node.xmin - p0[0]) / dx, "L"))
        if dy > 0:
            hits.append(((node.ymax - p0[1]) / dy, "B"))
        elif dy < 0:
            hits.append(((node.ymin - p0[1]) / dy, "T"))
        t, side = min(hits)
        t = max(t, 0.0)
        q = (p0[0] + t * dx, p0[1] + t * dy)
        return k, _edge_point(node, side, q), side
    return None


def _terminal_end(node: Node, chain: list[Point], other: Point) -> _End:
    """The polyline's end at `node`. `chain` are the connector centres leaving it, nearest
    first; `other` is the far terminal's centre (the direction when there is no chain).

    A crossing is a point: its centre. A symbol's port is on its bbox edge: the first
    connector when that sits on the edge (the usual case, and it fixes the side), otherwise
    the place the polyline crosses the boundary, so a nozzle drawn inside a tank leaves
    through the side the pipe really exits. No crossing at all means the other end is nested
    in this symbol: the run is internal.
    """
    if node.label == "crossing":
        return _End(node.center, None, 0)
    if chain:
        side = _nearest_side(node, chain[0])
        on_edge = _edge_point(node, side, chain[0])
        if abs(on_edge[0] - chain[0][0]) + abs(on_edge[1] - chain[0][1]) <= PORT_TOL_PX:
            return _End(on_edge, side, 1)
    found = _exit_point(node, [node.center, *chain, other])
    if found is None:
        return _End(node.center, None, 0, internal=True)
    k, point, side = found
    return _End(point, side, k - 1)


@dataclass
class _Walk:
    a: str
    b: str
    chain: list[str]
    a_first: Node | None
    b_first: Node | None


def _walk_chains(sheet: Sheet) -> list[_Walk]:
    """Every maximal connector chain between two terminals, each reported once."""
    seen: set[frozenset[str]] = set()
    walks: list[_Walk] = []
    for start in sheet.nodes:
        if not sheet.is_terminal(start):
            continue
        for nxt in sheet.adjacency[start]:
            if frozenset((start, nxt)) in seen or sheet.nodes[nxt].label == BACKGROUND:
                continue
            prev, cur = start, nxt
            seen.add(frozenset((prev, cur)))
            chain: list[str] = []
            while not sheet.is_terminal(cur):
                chain.append(cur)
                # a non-terminal connector has exactly two neighbours
                a, b = sheet.adjacency[cur]
                prev, cur = cur, (b if a == prev else a)
                seen.add(frozenset((prev, cur)))
            walks.append(
                _Walk(
                    a=start,
                    b=cur,
                    chain=chain,
                    a_first=sheet.nodes[chain[0]] if chain else None,
                    b_first=sheet.nodes[chain[-1]] if chain else None,
                )
            )
    return walks


class _Axes:
    """Union-find over the x / y coordinates of one polyline.

    A horizontal segment ties its two points' y together, a vertical one their x. A class
    that contains a fixed point (a port or a crossing centre) takes that value; otherwise the
    mean of its members. Two different fixed values never end up in one class: that segment
    is reported as unsatisfiable and becomes an elbow instead.
    """

    def __init__(self, n: int):
        self.parent = list(range(2 * n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i


def _straighten(
    pts: list[Point],
    fixed: list[tuple[bool, bool]],
    tol: float,
    start_axis: str | None = None,
    end_axis: str | None = None,
) -> tuple[list[Point], int, list[str]]:
    """Snap near-axis segments onto the axis; turn the rest into L-shaped elbows.

    Returns (points, elbows, warnings). `fixed[i]` says which of point i's (x, y) are pinned
    and never change.
    `start_axis` / `end_axis` ("H" / "V") are the directions the pipe must leave the first
    / last point along (perpendicular to a symbol's edge); an elbow at either end honours it.
    """
    n = len(pts)
    axes = _Axes(n)
    fixed_val: dict[int, float] = {}
    for i, (fix_x, fix_y) in enumerate(fixed):
        if fix_x:
            fixed_val[2 * i] = pts[i][0]
        if fix_y:
            fixed_val[2 * i + 1] = pts[i][1]

    kinds: list[str] = []  # "H" | "V" | "D" per segment
    for i in range(n - 1):
        dx = abs(pts[i + 1][0] - pts[i][0])
        dy = abs(pts[i + 1][1] - pts[i][1])
        if dx <= 1e-9 and dy <= 1e-9:
            kinds.append("Z")
            continue
        if min(dx, dy) > tol:
            kinds.append("D")
            continue
        # H ties y (index 2i+1), V ties x (index 2i)
        kind, off = ("H", 1) if dx >= dy else ("V", 0)
        ra, rb = axes.find(2 * i + off), axes.find(2 * (i + 1) + off)
        va, vb = fixed_val.get(ra), fixed_val.get(rb)
        if ra != rb:
            if va is not None and vb is not None and abs(va - vb) > 1e-9:
                kinds.append("D")
                continue
            axes.parent[rb] = ra
            if va is None and vb is not None:
                fixed_val[ra] = vb
        kinds.append(kind)

    members: dict[int, list[float]] = defaultdict(list)
    for i, (x, y) in enumerate(pts):
        members[axes.find(2 * i)].append(x)
        members[axes.find(2 * i + 1)].append(y)
    value = {r: fixed_val.get(r, sum(v) / len(v)) for r, v in members.items()}
    snapped = [(value[axes.find(2 * i)], value[axes.find(2 * i + 1)]) for i in range(n)]

    out: list[Point] = [snapped[0]]
    elbows = 0
    warnings: list[str] = []
    for i, kind in enumerate(kinds):
        p, q = snapped[i], snapped[i + 1]
        if kind == "D":
            elbows += 1
            off = min(abs(q[0] - p[0]), abs(q[1] - p[1]))
            warnings.append(f"segment {i} off-axis by {off:.1f}px -> L elbow")
            prev_kind = next((k for k in reversed(kinds[:i]) if k in "HV"), None)
            next_kind = next((k for k in kinds[i + 1 :] if k in "HV"), None)
            if i == 0 and start_axis is not None:
                horizontal_first = start_axis == "H"
            elif i == len(kinds) - 1 and end_axis is not None:
                horizontal_first = end_axis != "H"
            elif prev_kind is not None:
                horizontal_first = prev_kind == "H"
            elif next_kind is not None:
                horizontal_first = next_kind == "V"
            else:
                horizontal_first = abs(q[0] - p[0]) >= abs(q[1] - p[1])
            out.append((q[0], p[1]) if horizontal_first else (p[0], q[1]))
        out.append(q)
    return out, elbows, warnings


def _simplify(pts: list[Point], eps: float = 1e-6) -> list[Point]:
    """Drop repeated points and interior points on a straight run."""
    dedup: list[Point] = []
    for p in pts:
        if not dedup or abs(p[0] - dedup[-1][0]) > eps or abs(p[1] - dedup[-1][1]) > eps:
            dedup.append(p)
    out: list[Point] = []
    for p in dedup:
        while len(out) >= 2:
            a, b = out[-2], out[-1]
            same_x = abs(a[0] - b[0]) <= eps and abs(b[0] - p[0]) <= eps
            same_y = abs(a[1] - b[1]) <= eps and abs(b[1] - p[1]) <= eps
            if not (same_x or same_y):
                break
            out.pop()
        out.append(p)
    return out


def _pinned(side: str | None) -> tuple[bool, bool]:
    """Which coordinates of a run's end may not move. A crossing (side None) is a point; a
    symbol port is pinned across its edge but may slide along it (annotations are a pixel
    or two off)."""
    if side is None:
        return (True, True)
    return (True, False) if side in "LR" else (False, True)


def _leave_axis(side: str | None) -> str | None:
    """Axis a pipe leaves a symbol along (perpendicular to its edge); free at a crossing."""
    if side is None:
        return None
    return "H" if side in "LR" else "V"


def _chain_length(centres: list[Point]) -> float:
    return sum(math.dist(p, q) for p, q in zip(centres, centres[1:], strict=False))


_OPPOSITE = {"L": "R", "R": "L", "T": "B", "B": "T"}


def _nozzle_flange(sheet: Sheet, tank_id: str, anchor: Point, claimed: set[str]) -> str | None:
    best: tuple[float, str] | None = None
    for nid, node in sheet.nodes.items():
        if node.label != "general" or nid == tank_id or nid in claimed:
            continue
        if len(sheet.adjacency[nid]) > 1 or max(node.width, node.height) > NOZZLE_MAX_SIZE_PX:
            continue
        d = math.dist(node.center, anchor)
        if d <= NOZZLE_RADIUS_PX and (best is None or d < best[0]):
            best = (d, nid)
    return None if best is None else best[1]


def _nozzle_run(sheet: Sheet, walk: _Walk, claimed: set[str]) -> PipeRun | None:
    """Replace a zero-length self-loop by a link that seats the nozzle's flange on the tank.

    The tank's port is where a ray from the loop's position through the flange leaves the
    bbox (the flange sits just outside the nozzle's inner end). Without a flange nearby the
    loop is dropped: it carries no pipe.
    """
    tank = sheet.nodes[walk.a]
    anchor = sheet.nodes[walk.chain[0]].center
    flange_id = _nozzle_flange(sheet, walk.a, anchor, claimed)
    if flange_id is None:
        return None
    claimed.add(flange_id)
    fc = sheet.nodes[flange_id].center
    dx, dy = fc[0] - anchor[0], fc[1] - anchor[1]
    norm = math.hypot(dx, dy)
    found = None
    if norm > 1e-6:
        far = (anchor[0] + dx / norm * 1e5, anchor[1] + dy / norm * 1e5)
        found = _exit_point(tank, [anchor, far])
    if found is None:
        side = _nearest_side(tank, anchor)
        point = _edge_point(tank, side, anchor)
    else:
        _, point, side = found
    return PipeRun(
        a=walk.a,
        b=flange_id,
        points=(point, point),
        connectors=tuple(walk.chain),
        a_side=side,
        b_side=_OPPOSITE[side],
        nozzle=True,
    )


def pipe_runs(sheet: Sheet, axis_tol: float = AXIS_TOL_PX) -> list[PipeRun]:
    """Pipe skeleton: one axis-aligned polyline per connector chain / direct link.

    Deterministic: runs follow the node order of the GraphML. Links to `background` are
    skipped; links to `instrumentation` stay (the caller decides what a signal line is).
    Zero-length tank loops are not pipes: each becomes a `nozzle` link to its flange, or is
    dropped when no flange is near (see NOZZLE_LOOP_PX).
    """
    runs: list[PipeRun] = []
    claimed_flanges: set[str] = set()
    for walk in _walk_chains(sheet):
        na, nb = sheet.nodes[walk.a], sheet.nodes[walk.b]
        if walk.a == walk.b and na.label != "crossing":
            if _chain_length([sheet.nodes[c].center for c in walk.chain]) < NOZZLE_LOOP_PX:
                nozzle = _nozzle_run(sheet, walk, claimed_flanges)
                if nozzle is not None:
                    runs.append(nozzle)
                continue
        centres = [sheet.nodes[c].center for c in walk.chain]
        ea = _terminal_end(na, centres, nb.center)
        eb = _terminal_end(nb, centres[::-1], na.center)
        internal = ea.internal or eb.internal
        if internal:
            # inside a symbol: keep the geometry, pin only crossings, expose no port
            lead = [na.center] if na.label == "crossing" else []
            tail = [nb.center] if nb.label == "crossing" else []
            pts = [*lead, *centres, *tail]
            if len(pts) < 2:
                lead, tail = [], []
                pts = [na.center, nb.center]
            fixed = [(False, False)] * len(pts)
            if lead:
                fixed[0] = (True, True)
            if tail:
                fixed[-1] = (True, True)
            a_side = b_side = None
        else:
            lo, hi = ea.skip, len(centres) - eb.skip
            pts = [ea.point, *(centres[lo:hi] if lo < hi else []), eb.point]
            a_side, b_side = ea.side, eb.side
            fixed = [(False, False)] * len(pts)
            fixed[0], fixed[-1] = _pinned(a_side), _pinned(b_side)
        straightened, elbows, warnings = _straighten(
            pts, fixed, axis_tol, _leave_axis(a_side), _leave_axis(b_side)
        )
        runs.append(
            PipeRun(
                a=walk.a,
                b=walk.b,
                points=tuple(_simplify(straightened)),
                connectors=tuple(walk.chain),
                elbows=elbows,
                warnings=tuple(warnings),
                a_side=a_side,
                b_side=b_side,
                internal=internal,
            )
        )
    return runs


def symbol_ports(
    sheet: Sheet,
    node_id: str,
    runs: list[PipeRun] | None = None,
    ignore_labels: tuple[str, ...] = DEFAULT_IGNORE_LABELS,
) -> list[Port]:
    """Ports of one node: where its pipes leave the bbox, one per pipe run.

    A run whose far end is a node in `ignore_labels` is not a process pipe (actuator or
    signal link) and gives no port; neither does an `internal` run.
    The port point is the run's first (or last) polyline point, on the bbox edge `side`.
    """
    runs = pipe_runs(sheet) if runs is None else runs
    ports: list[Port] = []
    for idx, run in enumerate(runs):
        if run.internal:
            continue
        ends = []  # a loop run (a == b) contributes both ends
        if run.a == node_id:
            ends.append((run.points[0], run.a_side, run.b))
        if run.b == node_id:
            ends.append((run.points[-1], run.b_side, run.a))
        for point, side, other in ends:
            if side is None or sheet.nodes[other].label in ignore_labels:
                continue
            ports.append(Port(point[0], point[1], side, other, idx))
    return ports


def port_config(ports: list[Port]) -> PortConfig:
    """Classify a port set: 0, 1, 2_straight (axis H/V), 2_angle (adjacent sides), 2_same (both
    on one side, e.g. a pump piped from the top), 3, 4 or multi (5+)."""
    sides = tuple(sorted(p.side for p in ports))
    n = len(sides)
    if n <= 1:
        return PortConfig(str(n), None, sides)
    if n == 2:
        if set(sides) == {"L", "R"}:
            return PortConfig("2_straight", "H", sides)
        if set(sides) == {"T", "B"}:
            return PortConfig("2_straight", "V", sides)
        if sides[0] == sides[1]:
            return PortConfig("2_same", None, sides)
        return PortConfig("2_angle", None, sides)
    if n == 3:
        return PortConfig("3", None, sides)
    if n == 4:
        return PortConfig("4", None, sides)
    return PortConfig("multi", None, sides)
