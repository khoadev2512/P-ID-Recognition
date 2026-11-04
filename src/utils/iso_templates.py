"""ISO 10628-2 vector templates + ports (V1 of PLAN_ISO_RENDER_PID2GRAPH): the loader.

Reads what `scripts/extract_iso_svg_templates.py` wrote under `data/raw/template_iso_svg/`
(`<file>.svg`, `ports.csv`, `templates.csv`) and joins it with `docs/symbol_mapping/
iso_taxonomy.csv`. V3 (symbol choice) and V5 (page drawing) read templates only through here.

Coordinates. A template's *local* frame is millimetres from the top-left corner of its viewBox,
x right, y down, box `[0, w_mm] x [0, h_mm]`; `ports.csv` is in this frame. The SVG body itself is
still in the BK-2020 sheet's coordinates, so `Template.matrix` (body -> local) starts with
`translate(-viewBox_x, -viewBox_y)`. A page drawing wraps `Template.body(...)` in a `<g>` whose
transform is `matrix` followed by the placement (scale, translate) on the page.

Orientation. `oriented(rot, flip)` mirrors left-right first, then turns `rot` degrees clockwise on
screen (SVG `rotate(+rot)`, y down), and moves the origin back to the new top-left corner. It
returns a new Template: ports (side, mm, u, v), size and `matrix` follow; the SVG is untouched.
Orientation is always taken from the canonical drawing, not stacked on an earlier call.
"""

from __future__ import annotations

import csv
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

_SVG_NS = "http://www.w3.org/2000/svg"
_NOISE_NS = (
    "http://www.inkscape.org/namespaces/inkscape",
    "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
)
# Top-level children that are not drawing: dropped, never serialised into a body.
_NON_DRAWING = {"defs", "metadata", "namedview", "title", "desc"}

ROTATIONS = (0, 90, 180, 270)
# Side after a 90 degree clockwise turn; flipping left-right swaps L and R.
_TURN_SIDE = {"L": "T", "T": "R", "R": "B", "B": "L"}
_FLIP_SIDE = {"L": "R", "R": "L", "T": "T", "B": "B"}

Matrix = tuple[float, float, float, float, float, float]  # SVG (a, b, c, d, e, f)

_STROKE_WIDTH = re.compile(r"(?<![\w-])stroke-width\s*:\s*([^;\"]+)")
_STROKE_NONE = re.compile(r"(?<![\w-])stroke\s*:\s*none")
_TRANSFORM_FN = re.compile(r"(translate|rotate|scale|matrix)\s*\(([^)]*)\)")


@dataclass(frozen=True)
class TemplatePort:
    """Where a pipe stops on the template. (x_mm, y_mm) is in the template's local frame."""

    port_id: int
    side: str  # "L" | "R" | "T" | "B"
    x_mm: float
    y_mm: float
    u: float  # x_mm / width_mm
    v: float  # y_mm / height_mm
    source: str  # "stub" (measured) | "override" (taxonomy `port_override`); a "+nearest"
    # suffix means the pipe axis missed the body and the port took the nearest ink (X411 L)


@dataclass(frozen=True)
class Template:
    file: str  # `reg`, or `reg_variant` when a REG# has several drawings
    reg_number: str
    variant: int
    group: str  # G1..G5
    fine_name: str
    in_vocab: bool
    scope: str  # "html" | "extra"
    port_config: str  # expected port layout from the taxonomy
    directional: bool
    placement: str  # pipe | pipe_or_end | end | attach
    w_mm: float
    h_mm: float
    ports: tuple[TemplatePort, ...]
    svg_path: Path
    matrix: Matrix  # SVG body coordinates -> local frame of this orientation
    rot: int = 0
    flip: bool = False
    base: Template | None = None  # the canonical (rot 0, no flip) template; None on itself

    @property
    def sides(self) -> tuple[str, ...]:
        """Sorted port sides, comparable with `pid2graph.PortConfig.sides`."""
        return tuple(sorted(p.side for p in self.ports))

    @property
    def transform(self) -> str:
        """`matrix` as an SVG transform attribute value."""
        return "matrix(" + " ".join(f"{v:.6g}" for v in self.matrix) + ")"

    def body(self, stroke: float | None = None) -> str:
        """The symbol's drawing as SVG markup (no <svg> wrapper), in sheet coordinates.

        Defs, metadata, ids and editor attributes are gone, so hundreds of copies can share one
        SVG without clashing. With `stroke=W` every stroke is rewritten to
        `width / w_base * W`, `w_base` being the template's most common stroke; zero strokes stay
        zero. W is in the body's own units (mm), i.e. *before* `matrix` and the page scale: to
        get a page stroke of `p` px under a scale of `s` px/mm, pass `p / s`.
        """
        canonical = _parse_body(str(self.svg_path))
        elements = [deepcopy(e) for e in canonical.elements]
        if stroke is not None and canonical.w_base is not None:
            for e in elements:
                _rescale_strokes(e, canonical.w_base, stroke, 1.0)
        return "".join(ET.tostring(e, encoding="unicode") for e in elements)

    def oriented(self, rot: int = 0, flip: bool = False) -> Template:
        """This drawing mirrored (if `flip`) and turned `rot` degrees clockwise."""
        if rot not in ROTATIONS:
            raise ValueError(f"rot must be one of {ROTATIONS}, got {rot}")
        base = self.base or self
        w, h = base.w_mm, base.h_mm
        # canonical local (x, y) -> mirrored -> turned, then the box is [0, w2] x [0, h2]
        mirror: Matrix = (-1.0, 0.0, 0.0, 1.0, w, 0.0) if flip else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        turn, w2, h2 = _turn_matrix(rot, w, h)
        local = _compose(turn, mirror)
        ports = []
        for p in base.ports:
            x, y = _apply(local, p.x_mm, p.y_mm)
            side = _FLIP_SIDE[p.side] if flip else p.side
            for _ in range(rot // 90):
                side = _TURN_SIDE[side]
            ports.append(replace(p, side=side, x_mm=x, y_mm=y, u=x / w2, v=y / h2))
        return replace(
            base,
            w_mm=w2,
            h_mm=h2,
            ports=tuple(ports),
            matrix=_compose(local, base.matrix),
            rot=rot,
            flip=flip,
            base=base,
        )

    def all_orientations(self) -> list[Template]:
        """All 8 orientations (4 turns, with and without the mirror), none merged: a symmetric
        drawing still comes out 8 times, which only skews how often a pose is drawn."""
        return [self.oriented(r, f) for f in (False, True) for r in ROTATIONS]


def _compose(outer: Matrix, inner: Matrix) -> Matrix:
    """The matrix that applies `inner` first, then `outer`."""
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply(m: Matrix, x: float, y: float) -> tuple[float, float]:
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def _turn_matrix(rot: int, w: float, h: float) -> tuple[Matrix, float, float]:
    """Clockwise turn of the box [0,w] x [0,h]; returns (matrix, new width, new height)."""
    if rot == 0:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), w, h
    if rot == 90:  # (x, y) -> (h - y, x)
        return (0.0, 1.0, -1.0, 0.0, h, 0.0), h, w
    if rot == 180:  # (x, y) -> (w - x, h - y)
        return (-1.0, 0.0, 0.0, -1.0, w, h), w, h
    return (0.0, -1.0, 1.0, 0.0, 0.0, w), h, w  # 270: (x, y) -> (y, w - x)


# --- SVG body ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Body:
    elements: tuple[ET.Element, ...]
    w_base: float | None  # most common drawn stroke width, None when nothing is stroked


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean(elem: ET.Element) -> ET.Element | None:
    """Copy of `elem` without namespaces, ids or editor attributes; None for non-drawing."""
    if elem.tag.startswith("{") and not elem.tag.startswith(f"{{{_SVG_NS}}}"):
        return None
    if _local(elem.tag) in _NON_DRAWING:
        return None
    out = ET.Element(_local(elem.tag))
    for key, value in elem.attrib.items():
        if key == "id" or any(key.startswith(f"{{{ns}}}") for ns in _NOISE_NS):
            continue
        out.set(key, value)
    out.text = elem.text if elem.text and elem.text.strip() else None
    out.tail = None
    for child in elem:
        kept = _clean(child)
        if kept is not None:
            kept.tail = child.tail
            out.append(kept)
    return out


def _own_stroke_width(elem: ET.Element) -> float | None:
    """Stroke width declared on `elem` itself (style or attribute), None if none or no stroke."""
    style = elem.get("style", "")
    if _STROKE_NONE.search(style) or elem.get("stroke") == "none":
        return None
    m = _STROKE_WIDTH.search(style)
    raw = m.group(1) if m else elem.get("stroke-width")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def _transform_scale(value: str) -> float:
    """Uniform stand-in for the scale a transform list applies (sqrt of |det|); rotate and
    translate do nothing to a stroke."""
    s = 1.0
    for name, args in _TRANSFORM_FN.findall(value):
        nums = [float(v) for v in re.split(r"[\s,]+", args.strip()) if v]
        if name == "scale" and nums:
            s *= math.sqrt(abs(nums[0] * (nums[1] if len(nums) > 1 else nums[0])))
        elif name == "matrix" and len(nums) >= 4:
            s *= math.sqrt(abs(nums[0] * nums[3] - nums[1] * nums[2]))
    return s


def _walk(elem: ET.Element):
    yield elem
    for child in elem:
        yield from _walk(child)


def _rescale_strokes(elem: ET.Element, w_base: float, stroke: float, acc: float) -> None:
    """Rewrite stroke widths in place to `width / w_base * stroke`, divided by the scale that
    the ancestors' transforms apply so the drawn width is the same."""
    acc *= _transform_scale(elem.get("transform", ""))
    width = _own_stroke_width(elem)
    if width is not None and width > 0 and acc > 0:
        new = f"{width / w_base * stroke / acc:.6g}"
        style = elem.get("style")
        if style is not None and _STROKE_WIDTH.search(style):
            elem.set("style", _STROKE_WIDTH.sub(f"stroke-width:{new}", style, count=1))
        else:
            elem.set("stroke-width", new)
    for child in elem:
        _rescale_strokes(child, w_base, stroke, acc)


@cache
def _parse_body(svg_path: str) -> _Body:
    root = ET.parse(svg_path).getroot()
    elements = tuple(e for e in (_clean(c) for c in root) if e is not None)
    widths: Counter[float] = Counter()
    for top in elements:
        for e in _walk(top):
            if e.tag in ("text", "tspan"):  # glyphs are filled; their stroke-width is inert
                continue
            w = _own_stroke_width(e)
            if w is not None and w > 0:
                widths[round(w, 6)] += 1
    if not widths:
        return _Body(elements, None)
    top_count = max(widths.values())
    # ties go to the thinner stroke, so the heavier marks of a mixed template stay heavier
    return _Body(elements, min(w for w, n in widths.items() if n == top_count))


def _viewbox(svg_path: Path) -> tuple[float, float, float, float]:
    root = ET.parse(svg_path).getroot()
    vx, vy, vw, vh = (float(v) for v in re.split(r"[\s,]+", root.attrib["viewBox"].strip()))
    for attr, size in (("width", vw), ("height", vh)):
        declared = root.attrib.get(attr, "")
        if not declared.endswith("mm") or abs(float(declared[:-2]) - size) > 1e-3:
            raise ValueError(f"{svg_path.name}: {attr}={declared!r} is not 1 viewBox unit = 1 mm")
    return vx, vy, vw, vh


# --- loading ----------------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_templates(
    template_dir: str | Path,
    taxonomy_csv: str | Path,
    scope: str = "html",
) -> dict[str, Template]:
    """Canonical-orientation templates keyed by `file`, in `templates.csv` order.

    `scope` is "html" (the 45 templates the plan draws) or "all" (all 62). Raises when a
    template is missing from the taxonomy or the SVG, when `templates.csv` flags it `ok != yes`,
    or when its ports disagree with `templates.csv` (`sides`) or the SVG (`viewBox`).
    """
    if scope not in ("html", "all"):
        raise ValueError(f"scope must be 'html' or 'all', got {scope!r}")
    template_dir = Path(template_dir)
    taxonomy = {(r["reg_number"], int(r["variant"])): r for r in _read_csv(Path(taxonomy_csv))}
    ports_by_file: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv(template_dir / "ports.csv"):
        ports_by_file.setdefault(row["file"], []).append(row)

    templates: dict[str, Template] = {}
    for row in _read_csv(template_dir / "templates.csv"):
        if scope == "html" and row["scope"] != "html":
            continue
        name = row["file"]
        if row["ok"] != "yes":
            raise ValueError(f"{name}: templates.csv says ok={row['ok']!r}")
        tax = taxonomy.get((row["reg_number"], int(row["variant"])))
        if tax is None:
            raise ValueError(f"{name}: not in the taxonomy")
        svg_path = template_dir / f"{name}.svg"
        vx, vy, vw, vh = _viewbox(svg_path)
        w_mm, h_mm = float(row["width_mm"]), float(row["height_mm"])
        if abs(vw - w_mm) > 1e-3 or abs(vh - h_mm) > 1e-3:
            raise ValueError(f"{name}: viewBox size differs from templates.csv")
        ports = tuple(
            TemplatePort(
                port_id=int(p["port_id"]),
                side=p["side"],
                x_mm=float(p["x_mm"]),
                y_mm=float(p["y_mm"]),
                u=float(p["x_mm"]) / w_mm,
                v=float(p["y_mm"]) / h_mm,
                source=p["source"],
            )
            for p in ports_by_file.get(name, [])
        )
        if "".join(sorted(p.side for p in ports)) != "".join(sorted(row["sides"])):
            raise ValueError(f"{name}: ports.csv sides differ from templates.csv {row['sides']!r}")
        templates[name] = Template(
            file=name,
            reg_number=row["reg_number"],
            variant=int(row["variant"]),
            group=row["group"],
            fine_name=row["fine_name"],
            in_vocab=row["in_vocab"] == "yes",
            scope=row["scope"],
            port_config=tax["port_config"],
            directional=tax["directional"] == "yes",
            placement=tax["placement"],
            w_mm=w_mm,
            h_mm=h_mm,
            ports=ports,
            svg_path=svg_path,
            matrix=(1.0, 0.0, 0.0, 1.0, -vx, -vy),
        )
    return templates
