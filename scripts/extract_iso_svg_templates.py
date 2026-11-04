"""Export the ISO 10628-2 taxonomy as standalone SVG templates + measured connection ports.

For the PID2Graph re-rendering branch (docs/PLAN_ISO_RENDER_PID2GRAPH.md, step V1). Reads the
symbol list from docs/symbol_mapping/iso_taxonomy.csv (NOT vocab.csv — Stage 2 is untouched)
and the BK-2020-04 vector sheets (CC BY-SA 4.0, https://gitlab.com/baltakatei/BK-2020-04).

Per taxonomy row it writes <out>/<reg>.svg (or <reg>_<variant>.svg when one REG# has several
drawings, e.g. the two butterfly-valve forms of X8075): the bare symbol, stubs removed, with
the viewBox cropped to it in the sheet's millimetre units, so the renderer can scale it and
set the stroke width itself.

Ports come from the connection stubs the sheets draw (5-unit straight segments). BK-2020 leaves
a gap (2.5 mm on most symbols) between a stub's inner end and the body, so each port then slides
inward along the stub's axis to the body's first ink — where a pipe must stop (override ports
too). The stub test differs from extract_iso_symbols_svg.find_stubs in one way: a candidate is
a stub when at least 25% of its ink lies outside the body's box (there: more than 50%).
X8087's left stub fails the old test because the control valve's diagonal arrow stretches the
body box over it. Symbols the sheets
draw without stubs (discs and flanges the pipe passes through, the funnel and vent stems) get
their ports from the taxonomy's `port_override` column: sides of the body box (L, R, T, B), or
`L,R@tail` for a disc on a stem, whose L/R ports go on the tip of the stem. An `overlay`
symbol (241, flow direction: a line with an arrowhead marker) keeps only its marker art: the
line is the pipe it is drawn on. Pieces of pipe such
a drawing still carries along those ports' axis, outside the symbol (511's flange marks sit
between two 5 mm pieces), are dropped first, like stubs.
SVG markers are baked into plain shapes (MuPDF ignores markers; X8087's arrowhead uses one).

Outputs, all under --out:
  <file>.svg      standalone template (renders on its own)
  ports.csv       file, reg, variant, fine_name, port_id, side, x_mm, y_mm, u, v, source
                  (x/y relative to the viewBox origin; u/v = x/width, y/height)
  templates.csv   one row per template: expected vs detected port configuration, the largest
                  port-to-ink gap (`ok` needs <= 0.15 mm), and the taxonomy's `scope`
                  (html = the 5-group proposal's symbols; extra = the rest)
  --preview       optional contact sheet with the ports drawn in red

Run:
    git clone --depth 1 https://gitlab.com/baltakatei/BK-2020-04.git third_party/bk2020
    uv run python scripts/extract_iso_svg_templates.py \\
        --preview docs/experiments/iso_render/template_iso_svg_preview.png
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_iso_symbols_svg as ext  # noqa: E402

for _prefix, _uri in {
    "": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
    "xlink": "http://www.w3.org/1999/xlink",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "cc": "http://creativecommons.org/ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}.items():
    ET.register_namespace(_prefix, _uri)

# Stage 2 reads its prototypes from here; this script must never write into it.
_PROTECTED = Path("data/raw/iso_symbols_clean")

_MIN_OUTSIDE = 0.25  # fraction of a candidate's ink box that must lie outside the body
_DEDUP_MM = 1.0  # stubs whose inner ends are this close are one port (duplicated paths)
_GAP_OK_MM = 0.15  # a port farther than this from the body's ink fails the `ok` check


@dataclass
class Port:
    side: str  # L, R, T, B — which side of the body the pipe leaves from
    x: float  # sheet user units (mm)
    y: float
    source: str  # "stub" or "override"; "+nearest" when snapped off its pipe axis


def load_taxonomy(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def template_name(reg: str, variant: int, n_variants: int) -> str:
    return reg if n_variants == 1 else f"{reg}_{variant}"


def select_variant(root: ET.Element, reg: str, variant: int) -> bool:
    """Relabel every other group carrying this REG#, so ext.isolate() keeps the chosen one."""
    label = ext.symbol_label(reg)
    groups = [e for e in root.iter() if e.get(ext._INK_LABEL) == label]
    if not 1 <= variant <= len(groups):
        return False
    for i, g in enumerate(groups, start=1):
        if i != variant:
            g.set(ext._INK_LABEL, f"{label}__other{i}")
    return True


def user_per_px(root: ET.Element, dpi: int) -> tuple[float, float, float]:
    """(scale, x0, y0): sheet user units per rendered pixel, and the viewBox origin."""
    width_mm = float(re.match(r"[\d.]+", root.get("width", "")).group())
    vb = [float(v) for v in re.split(r"[\s,]+", root.get("viewBox", "").strip())]
    return 25.4 / dpi * vb[2] / width_mm, vb[0], vb[1]


def _side(box: tuple[int, int, int, int], body: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0, bx1, by1 = body
    gaps = {"L": bx0 - cx, "R": cx - bx1, "T": by0 - cy, "B": cy - by1}
    return max(gaps, key=gaps.get)


def _inner_end(box: tuple[int, int, int, int], side: str) -> tuple[float, float]:
    """Point of a stub's ink box that touches the body."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {"L": (x1, cy), "R": (x0, cy), "T": (cx, y1), "B": (cx, y0)}[side]


def find_ports(
    root: ET.Element, target: ET.Element, dpi: int = 150
) -> tuple[list[Port], list[ET.Element]]:
    """Ports from the connection stubs of an isolated symbol, and the stub elements to drop."""
    leaves = [e for e in target.iter() if e is not target and len(e) == 0]
    candidates = [e for e in leaves if ext.is_stub(e)]
    if not candidates or len(candidates) == len(leaves):
        return [], []
    body = ext._ink_box(ext._without(root, target, candidates), dpi)
    if body is None:
        return [], []
    bx0, by0, bx1, by1 = body
    scale, ox, oy = user_per_px(root, dpi)

    found: list[tuple[Port, ET.Element]] = []
    for cand in candidates:
        box = ext._ink_box(ext._without(root, target, [e for e in leaves if e is not cand]), dpi)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        ix = max(0, min(x1, bx1) - max(x0, bx0) + 1)
        iy = max(0, min(y1, by1) - max(y0, by0) + 1)
        if ix * iy > (1 - _MIN_OUTSIDE) * area:
            continue  # part of the symbol (a gate valve's stem, a silencer's baffles)
        side = _side(box, body)
        px, py = _inner_end(box, side)
        found.append((Port(side, ox + px * scale, oy + py * scale, "stub"), cand))

    ports: list[Port] = []
    for port, _ in found:
        if all(abs(port.x - p.x) + abs(port.y - p.y) > _DEDUP_MM for p in ports):
            ports.append(port)
    return ports, [elem for _, elem in found]


_SNAP_DPI = 1200  # ~0.02 mm per pixel
_SNAP_MAX_MM = 5.0  # how far a port may move onto the body before it is left where it was
_SNAP_BAND_MM = 0.1  # half-width of the pipe line searched for body ink
_INWARD = {"L": (1, 0), "R": (-1, 0), "T": (0, 1), "B": (0, -1)}


def _ink_points(
    root: ET.Element, box: tuple[float, float, float, float], dpi: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sheet coordinates (mm) of the ink pixels inside `box` (x0, y0, x1, y1, sheet mm)."""
    page = pymupdf.open(stream=ET.tostring(root), filetype="svg")[0]
    vb = [float(v) for v in re.split(r"[\s,]+", root.get("viewBox", "").strip())]
    pt = page.rect.width / vb[2]  # PDF points per sheet unit
    x0, y0, x1, y1 = box
    clip = pymupdf.Rect((x0 - vb[0]) * pt, (y0 - vb[1]) * pt, (x1 - vb[0]) * pt, (y1 - vb[1]) * pt)
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    rows, cols = np.nonzero(img.min(axis=2) < 128)
    px = 72.0 / dpi / pt
    return vb[0] + (pix.x + cols + 0.5) * px, vb[1] + (pix.y + rows + 0.5) * px


def snap_to_body(root: ET.Element, ports: list[Port]) -> list[Port]:
    """Move each port onto the body's ink, where a pipe drawn to it must stop.

    BK-2020 draws its stubs with a gap: 2101's bow-tie spans x 62.5–72.5 but its stubs end at
    60 and 75, so a pipe stopped at the stub's inner end would leave 2.5 mm of white. Override
    ports sit on the body box, which for the discs on a stem (C0096–C0098, 2044/2045) is as wide
    as the disc, 2.5 mm off the stem the pipe crosses. Each port slides inward along its pipe
    axis to the first ink; if the axis misses the body (X411's hose ends 0.25 mm off the stub
    line) it takes the nearest ink instead and its source gets a ``+nearest`` suffix. A port
    with no ink within _SNAP_MAX_MM stays put; the caller's gap check reports it. `root` must
    already have the stubs removed.
    """
    reach, band = _SNAP_MAX_MM, _SNAP_BAND_MM
    out = []
    for p in ports:
        xs, ys = _ink_points(root, (p.x - reach, p.y - reach, p.x + reach, p.y + reach), _SNAP_DPI)
        dx, dy = _INWARD[p.side]
        along = (xs - p.x) * dx + (ys - p.y) * dy
        across = np.abs((xs - p.x) * dy - (ys - p.y) * dx)
        on_axis = (along >= 0) & (across <= band)
        if on_axis.any():
            t = float(along[on_axis].min())
            out.append(Port(p.side, p.x + dx * t, p.y + dy * t, p.source))
        elif xs.size:
            i = int(np.argmin((xs - p.x) ** 2 + (ys - p.y) ** 2))
            out.append(Port(p.side, float(xs[i]), float(ys[i]), f"{p.source}+nearest"))
        else:
            out.append(p)
    return out


_ALIGN_MM = 0.3  # opposite ports closer than this to one axis are put on it
_BACKOFF_MM = 1.0  # a re-aligned port restarts its slide this far out


def align_opposite(ports: list[Port]) -> list[Port]:
    """Put each L-R (T-B) pair of ports that is off by at most _ALIGN_MM on their mean y (x).

    BK-2020 draws one stub of many inline symbols 0.125 mm off the other (X8071: y -95.5 and
    -95.625), and X411's left port lands 0.22 mm off its hose's axis; left alone, V5 would have to
    jog the pipe by under a pixel. Larger offsets are deliberate (2511's nozzles) and stay.
    """
    out = list(ports)
    for i, j in itertools.combinations(range(len(out)), 2):
        a, b = out[i], out[j]
        if {a.side, b.side} == {"L", "R"}:
            attr = "y"
        elif {a.side, b.side} == {"T", "B"}:
            attr = "x"
        else:
            continue
        va, vb = getattr(a, attr), getattr(b, attr)
        if 0 < abs(va - vb) <= _ALIGN_MM:
            mid = (va + vb) / 2
            out[i], out[j] = replace(a, **{attr: mid}), replace(b, **{attr: mid})
    return out


def straighten_pairs(root: ET.Element, ports: list[Port]) -> list[Port]:
    """`align_opposite`, then slide each moved port back onto the body along its new axis
    (starting _BACKOFF_MM outside, so a port nudged off a corner still finds the edge)."""
    aligned = align_opposite(ports)
    out = []
    for old, new in zip(ports, aligned, strict=True):
        if new == old:
            out.append(old)
            continue
        dx, dy = _INWARD[new.side]
        start = replace(new, x=new.x - dx * _BACKOFF_MM, y=new.y - dy * _BACKOFF_MM)
        out.append(snap_to_body(root, [start])[0])
    return out


def port_gap(root: ET.Element, port: Port, dpi: int = 600) -> float:
    """Distance (mm) from a port to the nearest ink of the stub-less symbol."""
    r = _SNAP_MAX_MM
    xs, ys = _ink_points(root, (port.x - r, port.y - r, port.x + r, port.y + r), dpi)
    if xs.size == 0:
        return float("inf")
    return float(np.sqrt((xs - port.x) ** 2 + (ys - port.y) ** 2).min())


_TAIL_BAND_MM = 0.5  # depth of the end strips compared when looking for a symbol's tail


def tail_y(root: ET.Element, body_mm: tuple[float, float, float, float]) -> float:
    """Sheet y (mm) of the tip of a symbol's tail: the end of the stem a disc sits on.

    The tail is whichever of the body's top and bottom ends is narrower in ink (a stem 0.25 mm
    wide against a disc); the tip is that end's outermost ink, where a pipe drawn to it stops.
    """
    x0, y0, x1, y1 = body_mm
    xs, ys = _ink_points(root, (x0 - 0.5, y0 - 0.5, x1 + 0.5, y1 + 0.5), _SNAP_DPI)
    top, bottom = ys.min(), ys.max()
    top_w = np.ptp(xs[ys <= top + _TAIL_BAND_MM])
    bottom_w = np.ptp(xs[ys >= bottom - _TAIL_BAND_MM])
    if abs(top_w - bottom_w) < 2 * _SNAP_BAND_MM:
        raise RuntimeError(f"no tail: both ends are {top_w:.2f} / {bottom_w:.2f} mm wide")
    return float(top if top_w < bottom_w else bottom)


def pipe_pieces(
    root: ET.Element, target: ET.Element, spec: str, dpi: int = 150
) -> list[ET.Element]:
    """Stub-length straight segments that a symbol drawn without stubs still carries along the
    axis of its `port_override` ports, wholly outside the rest of its drawing: pieces of the
    pipe it sits on (511's two 5 mm pieces, 2.5 mm short of the flange marks). They are dropped
    like stubs, so the ink box and the ports are the symbol's own. `find_ports` cannot tell
    them apart there: 511's flange marks are stub-length too."""
    sides = {s.strip() for s in spec.partition("@")[0].split(",") if s.strip()}
    axes = {"H" if s in "LR" else "V" for s in sides}
    leaves = [e for e in target.iter() if e is not target and len(e) == 0]
    found: list[tuple[ET.Element, tuple[int, int, int, int]]] = []
    for cand in (e for e in leaves if ext.is_stub(e)):
        box = ext._ink_box(ext._without(root, target, [e for e in leaves if e is not cand]), dpi)
        if box is not None and ("H" if box[2] - box[0] > box[3] - box[1] else "V") in axes:
            found.append((cand, box))
    if not found or len(found) == len(leaves):
        return []
    rest = ext._ink_box(ext._without(root, target, [e for e, _ in found]), dpi)
    if rest is None:
        return []

    def outside(b: tuple[int, int, int, int]) -> bool:
        if b[2] - b[0] > b[3] - b[1]:  # along x
            return b[2] <= rest[0] or b[0] >= rest[2]
        return b[3] <= rest[1] or b[1] >= rest[3]

    return [e for e, b in found if outside(b)]


def override_ports(
    root: ET.Element, body_mm: tuple[float, float, float, float], spec: str
) -> list[Port]:
    """Ports from a taxonomy `port_override`: sides of the body box, e.g. ``L,R``.

    ``L,R@tail`` puts the L and R ports on the tip of the symbol's tail instead of halfway up
    the body box (discs on a stem, whose pipe ends at the stem). Only L/R take ``@tail``.
    """
    sides, _, where = spec.partition("@")
    names = [s for s in (t.strip() for t in sides.split(",")) if s]
    if where and where != "tail":
        raise ValueError(f"unknown port_override position {where!r} in {spec!r}")
    if where and not set(names) <= {"L", "R"}:
        raise ValueError(f"@{where} only places L/R ports: {spec!r}")
    x0, y0, x1, y1 = body_mm
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if where:
        cy = tail_y(root, body_mm)
    at = {"L": (x0, cy), "R": (x1, cy), "T": (cx, y0), "B": (cx, y1)}
    return [Port(s, *at[s], "override") for s in names]


_SVG = "{http://www.w3.org/2000/svg}"
_MARKER_PROPS = ("marker-start", "marker-end")
_PATH_TOKEN = re.compile(rf"[MmLlHhVvZz]|{ext._NUM}")


def _polyline(d: str) -> list[tuple[float, float]] | None:
    """Vertices of a path made only of M/L/H/V segments (absolute or relative), else None."""
    pts: list[tuple[float, float]] = []
    x = y = 0.0
    start = (x, y)
    cmd = ""
    tokens = _PATH_TOKEN.findall(d or "")
    i = 0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                x, y = start
                pts.append(start)
                continue
        if cmd in "MmLl":
            if i + 1 >= len(tokens):
                return None
            dx, dy = float(tokens[i]), float(tokens[i + 1])
            i += 2
            x, y = (x + dx, y + dy) if cmd.islower() and pts else (dx, dy)
            if cmd in "Mm":
                start = (x, y)
                cmd = "L" if cmd == "M" else "l"  # extra pairs after M are line-tos
        elif cmd in "HhVv":
            v = float(tokens[i])
            i += 1
            if cmd in "Hh":
                x = x + v if cmd == "h" else v
            else:
                y = y + v if cmd == "v" else v
        else:
            return None
        pts.append((x, y))
    return pts


def _style(elem: ET.Element) -> dict[str, str]:
    return dict(
        (k.strip(), v.strip())
        for k, v in (kv.split(":", 1) for kv in elem.get("style", "").split(";") if ":" in kv)
    )


def bake_markers(root: ET.Element, target: ET.Element, made: list[ET.Element] | None = None) -> int:
    """Replace marker-start/-end on the symbol's paths with the marker art itself.

    MuPDF ignores SVG markers, so X8087's control-valve arrowhead — the detail that tells the
    class apart — never rendered. The marker's children are copied into a <g> placed at the
    path's end, turned by `orient` and scaled by `markerUnits`, as a browser would draw them.
    Only straight-segment paths and markers without a viewBox are supported (all the taxonomy
    needs); anything else raises rather than dropping the arrowhead silently. The <g>s made are
    appended to `made` when given.
    """
    markers = {m.get("id"): m for m in root.iter(f"{_SVG}marker")}
    parent = {c: p for p in target.iter() for c in p}
    baked = 0
    for elem in list(target.iter()):
        style = _style(elem)
        refs = {}
        for k in _MARKER_PROPS:
            in_style, in_attr = style.pop(k, None), elem.attrib.pop(k, None)
            if (in_style or in_attr or "none") != "none":
                refs[k] = in_style or in_attr
        if not refs:
            continue
        pts = _polyline(elem.get("d", ""))
        pts = [p for i, p in enumerate(pts or []) if i == 0 or p != pts[i - 1]]
        if len(pts) < 2:
            raise RuntimeError(f"marker on a non-polyline path: {elem.get('id')}")
        stroke = float(re.match(ext._NUM, style.get("stroke-width", "1")).group())
        at = list(parent[elem]).index(elem) + 1
        for prop, ref in refs.items():
            marker = markers.get(re.fullmatch(r"url\(#(.+)\)", ref).group(1))
            if marker is None or marker.get("viewBox"):
                raise RuntimeError(f"unsupported marker {ref} on {elem.get('id')}")
            (x0, y0), (x1, y1) = (pts[0], pts[1]) if prop == "marker-start" else (pts[-2], pts[-1])
            x, y = (x0, y0) if prop == "marker-start" else (x1, y1)
            orient = marker.get("orient", "0")
            if orient.startswith("auto"):
                angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
                if orient == "auto-start-reverse" and prop == "marker-start":
                    angle += 180.0
            else:
                angle = float(orient)
            scale = 1.0 if marker.get("markerUnits") == "userSpaceOnUse" else stroke
            ref_x, ref_y = float(marker.get("refX", 0)), float(marker.get("refY", 0))
            transform = (
                f"{elem.get('transform', '')} translate({x:.5f},{y:.5f}) "
                f"rotate({angle:.5f}) scale({scale:.5f}) "
                f"translate({-ref_x:.5f},{-ref_y:.5f})"
            ).strip()
            g = ET.Element(f"{_SVG}g", {"transform": transform})
            for child in marker:
                art = copy.deepcopy(child)
                for e in art.iter():
                    e.attrib.pop("id", None)
                g.append(art)
            parent[elem].insert(at, g)
            if made is not None:
                made.append(g)
            baked += 1
        elem.set("style", ";".join(f"{k}:{v}" for k, v in style.items()))
    return baked


def port_config(ports: list[Port]) -> str:
    sides = "".join(sorted(p.side for p in ports))
    if len(ports) == 2:
        return "2_straight" if sides in ("LR", "BT") else "2_angle"
    return str(len(ports))


def config_matches(expected: str, detected: str) -> bool:
    if expected in ("multi", "attach"):
        return True  # vessels take ports anywhere; motors/agitators are not on a pipe
    if expected == "2":
        return detected.startswith("2")  # pumps: L-jog routing, either layout works
    return expected == detected


def keep_markers_only(target: ET.Element, made: list[ET.Element]) -> int:
    """Drop every shape of the symbol but its baked markers; returns how many were dropped.

    For an `overlay` symbol drawn on a pipe (241, flow direction: a 6.25 mm line with an
    arrowhead marker) the line is the pipe itself; the symbol is the arrowhead."""
    keep = {e for g in made for e in g.iter()}
    parent = {c: p for p in target.iter() for c in p}
    dropped = 0
    for e in list(target.iter()):
        if e is not target and len(e) == 0 and e not in keep:
            parent[e].remove(e)
            dropped += 1
    return dropped


def export_one(
    sheet: Path,
    reg: str,
    variant: int,
    override: str,
    dpi: int,
    pad_mm: float,
    placement: str = "",
) -> tuple[bytes, list[Port], tuple[float, float, float, float], float]:
    """Standalone SVG bytes, ports relative to its viewBox origin, its viewBox, and the
    largest port-to-ink gap in mm. An `overlay` symbol keeps only its markers."""
    tree = ET.parse(sheet)
    root = tree.getroot()
    if not select_variant(root, reg, variant) or not ext.isolate(root, reg, keep_stubs=True):
        raise RuntimeError(f"REG# {reg} variant {variant} not found in {sheet.name}")
    target = next(e for e in root.iter() if e.get(ext._INK_LABEL) == ext.symbol_label(reg))

    ports, stubs = find_ports(root, target, dpi)
    if not ports and override:
        stubs = pipe_pieces(root, target, override, dpi)
    parent = {c: p for p in target.iter() for c in p}
    for s in stubs:
        parent[s].remove(s)
    made: list[ET.Element] = []
    bake_markers(root, target, made)  # after the ports, so the arrowhead cannot shift a stub's side
    if placement == "overlay":
        if not made:
            raise RuntimeError(f"REG# {reg}: an overlay symbol needs a marker to keep")
        keep_markers_only(target, made)

    scale, ox, oy = user_per_px(root, dpi)
    box = ext._ink_box(root, dpi)
    if box is None:
        raise RuntimeError(f"REG# {reg} variant {variant} renders blank")
    bx0, by0, bx1, by1 = box
    body = (ox + bx0 * scale, oy + by0 * scale, ox + bx1 * scale, oy + by1 * scale)  # mm
    if not ports and override:
        ports = override_ports(root, body, override)
    ports = straighten_pairs(root, snap_to_body(root, ports))
    gap = max((port_gap(root, p) for p in ports), default=0.0)

    xs = [body[0], body[2], *(p.x for p in ports)]
    ys = [body[1], body[3], *(p.y for p in ports)]
    vx0, vy0 = min(xs) - pad_mm, min(ys) - pad_mm
    vw, vh = max(xs) + pad_mm - vx0, max(ys) + pad_mm - vy0
    root.set("viewBox", f"{vx0:.4f} {vy0:.4f} {vw:.4f} {vh:.4f}")
    root.set("width", f"{vw:.4f}mm")
    root.set("height", f"{vh:.4f}mm")
    rel = [Port(p.side, p.x - vx0, p.y - vy0, p.source) for p in ports]
    svg = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return svg, rel, (vx0, vy0, vw, vh), gap


def _preview(out_dir: Path, rows: list[dict], ports: dict[str, list[Port]], path: Path) -> None:
    cell, cols = 200, 8
    tiles = []
    for row in rows:
        doc = pymupdf.open(stream=(out_dir / f"{row['file']}.svg").read_bytes(), filetype="svg")
        pix = doc[0].get_pixmap(dpi=150, alpha=False)
        img = Image.fromarray(
            np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
        )
        draw = ImageDraw.Draw(img)
        w_mm, h_mm = float(row["width_mm"]), float(row["height_mm"])
        for p in ports[row["file"]]:
            x, y = p.x / w_mm * img.width, p.y / h_mm * img.height
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="red")
        img.thumbnail((cell - 20, cell - 40))
        tile = Image.new("RGB", (cell, cell), "white")
        tile.paste(img, ((cell - img.width) // 2, 8))
        mark = "" if row["ok"] == "yes" else "  !!"
        ImageDraw.Draw(tile).text(
            (6, cell - 28), f"{row['group']} {row['file']}{mark}", fill="black"
        )
        ImageDraw.Draw(tile).text((6, cell - 15), row["fine_name"][:30], fill="gray")
        tiles.append(tile)
    sheet = Image.new("RGB", (cols * cell, -(-len(tiles) // cols) * cell), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * cell, (i // cols) * cell))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def extract(
    svg_dir: Path, taxonomy: Path, out_dir: Path, dpi: int = 150, pad_mm: float = 0.5
) -> tuple[list[dict], dict[str, list[Port]]]:
    if out_dir.resolve() == _PROTECTED.resolve():
        raise ValueError(f"refusing to write into {_PROTECTED} (Stage 2 prototypes)")
    sheets = sorted(svg_dir.glob(ext._SHEET_GLOB))
    if not sheets:
        raise FileNotFoundError(f"no {ext._SHEET_GLOB} under {svg_dir}")
    rows_in = load_taxonomy(taxonomy)
    n_variants: dict[str, int] = {}
    for r in rows_in:
        n_variants[r["reg_number"]] = n_variants.get(r["reg_number"], 0) + 1
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, all_ports = [], {}
    for r in rows_in:
        reg, variant = r["reg_number"], int(r["variant"] or 1)
        sheet = ext.find_sheet(sheets, reg)
        if sheet is None:
            raise RuntimeError(f"REG# {reg} is not on any BK-2020 sheet")
        svg, ports, (_, _, vw, vh), gap = export_one(
            sheet, reg, variant, r["port_override"], dpi, pad_mm, r.get("placement", "")
        )
        name = template_name(reg, variant, n_variants[reg])
        (out_dir / f"{name}.svg").write_bytes(svg)
        detected = port_config(ports)
        all_ports[name] = ports
        rows.append(
            {
                "file": name,
                "reg_number": reg,
                "variant": variant,
                "group": r["group"],
                "fine_name": r["fine_name"],
                "in_vocab": r["in_vocab"],
                "scope": r.get("scope", ""),
                "expected": r["port_config"],
                "detected": detected,
                "sides": "".join(sorted(p.side for p in ports)),
                "gap_mm": f"{gap:.3f}",
                "ok": "yes"
                if config_matches(r["port_config"], detected) and gap <= _GAP_OK_MM
                else "no",
                "width_mm": f"{vw:.4f}",
                "height_mm": f"{vh:.4f}",
                "source": sheet.name,
            }
        )

    with (out_dir / "templates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "ports.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "reg_number",
                "variant",
                "fine_name",
                "port_id",
                "side",
                "x_mm",
                "y_mm",
                "u",
                "v",
                "source",
            ]
        )
        for row in rows:
            vw, vh = float(row["width_mm"]), float(row["height_mm"])
            for i, p in enumerate(all_ports[row["file"]]):
                w.writerow(
                    [
                        row["file"],
                        row["reg_number"],
                        row["variant"],
                        row["fine_name"],
                        i,
                        p.side,
                        f"{p.x:.4f}",
                        f"{p.y:.4f}",
                        f"{p.x / vw:.4f}",
                        f"{p.y / vh:.4f}",
                        p.source,
                    ]
                )
    return rows, all_ports


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--svg-dir", type=Path, default=Path("third_party/bk2020/DWG"))
    ap.add_argument("--taxonomy", type=Path, default=Path("docs/symbol_mapping/iso_taxonomy.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/template_iso_svg"))
    ap.add_argument("--dpi", type=int, default=150, help="raster DPI used to measure ink boxes")
    ap.add_argument("--pad-mm", type=float, default=0.5, help="margin around the symbol, mm")
    ap.add_argument("--preview", type=Path, default=None, help="contact sheet PNG with ports")
    args = ap.parse_args()

    rows, ports = extract(args.svg_dir, args.taxonomy, args.out, args.dpi, args.pad_mm)
    bad = [r for r in rows if r["ok"] != "yes"]
    print(f"wrote {len(rows)} templates -> {args.out}")
    for r in bad:
        print(
            f"  NOT OK {r['file']}: expected {r['expected']}, detected {r['detected']} "
            f"({r['sides']}), gap {r['gap_mm']} mm"
        )
    if args.preview:
        _preview(args.out, rows, ports, args.preview)
        print(f"preview -> {args.preview}")


if __name__ == "__main__":
    main()
