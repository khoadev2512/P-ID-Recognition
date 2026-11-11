"""Text on a V5 page (V5c of PLAN_ISO_RENDER_PID2GRAPH): tag numbers in the ISA bubbles, line
numbers on the pipes and at the free end of leader lines, equipment tags beside pumps and vessels,
and a reference in the off-page flags. OPEN100 and the drawings Stage 1 / 2 meet have text;
without it a test page is easier than a real one (no false positives on text, empty bubbles).

Text is only ever put where the page is blank: `place_texts` gets the raster of the pipes and
symbols already drawn and keeps every label `margin` px clear of that ink and of the labels
placed before it. A label with no free place is left out (counted in the log). Nothing drawn
before moves, so the pipe and symbol layers, the COCO boxes and the G1 criteria on them are the
same with or without text; text is not a COCO class.

Glyphs are outlines (paths) from the DejaVu Sans font that ships with matplotlib, so a page looks
the same on every machine whatever fonts are installed. The strings are made up from the usual
P&ID conventions (ISA letter codes, size-service-number-spec line numbers, P- / V- / C- tags),
one plant unit number per page.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

Box = tuple[float, float, float, float]
Point = tuple[float, float]

# ISA-5.1 identification letters: the first letter is the loop's measured variable (F flow,
# L level, P pressure, T temperature, A analysis; H hand and X on-off for valves worked by hand or
# by a switch), the rest what the instrument does. Per variable: what sits on the line or the
# vessel (element, transmitter, local gauge) and what reads it out (controller, indicator, alarm,
# switch).
_SENSOR = {
    "F": ("FT", "FE", "FI"), "L": ("LT", "LG", "LI"), "P": ("PT", "PI", "PG"),
    "T": ("TT", "TE", "TI", "TW"), "A": ("AT", "AI"),
}  # fmt: skip
_READOUT = {
    "F": ("FIC", "FI", "FAL", "FSH"), "L": ("LIC", "LI", "LAH", "LAL", "LSH", "LSL"),
    "P": ("PIC", "PI", "PAH", "PSH", "PSL", "PDI"), "T": ("TIC", "TI", "TAH", "TSH"),
    "A": ("AIC", "AI", "AAH"),
}  # fmt: skip
# an on-off loop: the valve's own tag, then what else is on it (hand switch, position switches)
_ON_OFF = {"H": ("HV", ("HS",)), "X": ("XV", ("ZSO", "ZSC", "HS"))}
_LINE_SIZES = ('1/2"', '3/4"', '1"', '1-1/2"', '2"', '3"', '4"', '6"', '8"', '10"', '12"')
_SERVICES = ("P", "PL", "PW", "CW", "CWR", "ST", "LS", "MS", "N", "IA", "PA", "FW", "DR", "VT",
             "FG", "HO", "SW", "W")  # fmt: skip
_SPECS = ("A1", "A1A", "A2", "B1", "B2", "C1", "D1", "E1")
_INSULATION = ("", "", "", "-H", "-P", "-C")  # mostly none; heat / personnel / cold insulation


def neighbour_role(kind: str | None, fine_name: str | None) -> str:
    """What a bubble's line leads to, for its tag: bubble, safety (a relief device), control (a
    control valve), valve (any other valve: worked by hand or on-off), vessel, or pipe (a point on
    a line: a crossing, a fitting, anything else)."""
    name = fine_name or ""
    if kind == "bubble":
        return "bubble"
    if name.startswith("safety_valve") or name in ("rupture_disc", "breather_valve"):
        return "safety"
    if name == "valve_control":
        return "control"
    if "valve" in name:
        return "valve"
    if kind == "vessel" or name.startswith(("vessel", "tank", "column")):
        return "vessel"
    return "pipe"


_VALVES = ("safety", "control", "valve")


def _pick(rng: random.Random, options: Sequence[str], used: Sequence[str]) -> str:
    """One of `options`, one not in `used` when there is any (a TT and a TI, not two TI)."""
    fresh = [o for o in options if o not in used]
    return rng.choice(fresh or list(options))


def loop_tags(
    links: Mapping[str, Sequence[tuple[str, str]]], rng: random.Random, unit: int
) -> dict[str, tuple[str, str]]:
    """(letters, loop number) for every bubble of `links` (bubble -> [(the node at the other end
    of one of its lines, `neighbour_role` of that node)]).

    Bubbles joined to each other, or to one valve, are one loop: one measured variable and one
    loop number, different from every other loop of the page (`unit` is the page's plant unit).
    The variable follows the loop's valves and vessels (a relief valve: pressure; a valve worked
    by hand or on-off: H or X); a bubble's other letters follow what it is joined to: a control
    valve's own tag (FV, PCV), a relief valve (PSV), the valve of an on-off loop (HV, XV), an
    instrument on a line or vessel (FT, LG, TE), or, joined to nothing but bubbles, a readout
    (LIC with a control valve in the loop, else FI, PAH, ...). Letters already used in the loop
    are avoided where there is a choice; two bubbles left with the same letters (two relief
    valves) get suffixes A, B."""
    # loops: union of bubbles that share a line or a valve
    parent = {b: b for b in links}

    def root(b: str) -> str:
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    by_valve: dict[str, str] = {}
    for b in sorted(links):
        for other, role in links[b]:
            if role == "bubble" and other in parent:
                parent[root(other)] = root(b)
            elif role in _VALVES:
                if other in by_valve:
                    parent[root(by_valve[other])] = root(b)
                by_valve.setdefault(other, b)
    loops: dict[str, list[str]] = {}
    for b in sorted(links):
        loops.setdefault(root(b), []).append(b)

    numbers = rng.sample(range(1, 100), min(len(loops), 99))
    out: dict[str, tuple[str, str]] = {}
    for k, members in enumerate(loops.values()):
        roles = {r for b in members for _, r in links[b] if r != "bubble"}
        if "safety" in roles:
            var = "P"
        elif "control" in roles:
            var = rng.choice("LPT" if "vessel" in roles else "FLPT")
        elif "valve" in roles:
            var = rng.choice("HX")
        elif "vessel" in roles:
            var = rng.choice("LLPT")
        else:
            var = rng.choice("FFPPTTLA")
        number = f"{unit}{numbers[k % len(numbers)]:02d}"
        letters: list[str] = []

        for b in members:
            own = {r for _, r in links[b] if r != "bubble"}
            if var in _ON_OFF:
                valve_tag, others = _ON_OFF[var]
                tag = valve_tag if own & set(_VALVES) else _pick(rng, others, letters)
            elif "safety" in own:
                tag = f"{var}SV"
            elif "control" in own:
                tag = f"{var}V" if rng.random() < 0.7 else f"{var}CV"
            elif own:
                tag = _pick(rng, _SENSOR[var], letters)
            elif "control" in roles:
                tag = f"{var}IC"
            else:
                tag = _pick(rng, _READOUT[var], letters)
            letters.append(tag)
        for b, tag in zip(members, letters, strict=True):
            twins = [m for m, t in zip(members, letters, strict=True) if t == tag]
            suffix = "" if len(twins) == 1 else "AB"[twins.index(b) % 2]
            out[b] = (tag, number + suffix)
    return out


def line_number(rng: random.Random, unit: int) -> str:
    """A line number: size, service, number, piping class, insulation (2"-P-1203-A1A-H)."""
    return (
        f"{rng.choice(_LINE_SIZES)}-{rng.choice(_SERVICES)}-{unit}{rng.randint(0, 999):03d}-"
        f"{rng.choice(_SPECS)}{rng.choice(_INSULATION)}"
    )


def equipment_tag(rng: random.Random, prefix: str, unit: int) -> str:
    """An equipment number: prefix, unit, serial, a spare's letter now and then (P-301A)."""
    return f"{prefix}-{unit}{rng.randint(1, 30):02d}{rng.choice(('', '', 'A', 'B'))}"


def flag_text(rng: random.Random, unit: int) -> str:
    """What an off-page connector says: the drawing or unit the line comes from / goes to."""
    other = rng.choice([u for u in range(1, 10) if u != unit])
    return rng.choice(
        (f"{other}{rng.randint(0, 99):02d}", f"U-{other}00", f"PID-{other}0{rng.randint(1, 9)}")
    )


def equipment_prefix(fine_name: str | None, column: bool = False) -> str | None:
    """Tag prefix of an equipment symbol (None: not equipment)."""
    name = fine_name or ""
    if column or name == "column":
        return "C"
    if name.startswith("vessel") or name == "tank":
        return "V"
    if name.startswith("pump"):
        return "P"
    if name.startswith("compressor"):
        return "K"
    if name.startswith(("hx", "heat_exchanger")):
        return "E"
    return None


# --- glyphs ----------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _font():
    import matplotlib
    from matplotlib.font_manager import FontProperties

    return FontProperties(fname=str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"))


@lru_cache(maxsize=1)
def _cap_ratio() -> float:
    """Cap height of the font per unit of font size."""
    from matplotlib.textpath import TextPath

    e = TextPath((0, 0), "H", size=100, prop=_font()).get_extents()
    return e.height / 100


@lru_cache(maxsize=4096)
def _glyphs(text: str) -> tuple[np.ndarray, np.ndarray, Box]:
    """Outline of `text` at a cap height of 1, y down, its box's top-left corner at the origin:
    (vertices, codes, box). The box is the outline's (Bezier curves included), not the vertices':
    a CLOSEPOLY vertex carries no position."""
    from matplotlib.textpath import TextPath

    p = TextPath((0, 0), text, size=1 / _cap_ratio(), prop=_font())
    e = p.get_extents()
    v = np.array(p.vertices, float)
    v[:, 0] -= e.x0
    v[:, 1] = e.y1 - v[:, 1]  # y down, the top of the outline at 0
    return v, np.array(p.codes), (0.0, 0.0, float(e.width), float(e.height))


def text_size(text: str, cap: float, vertical: bool = False) -> tuple[float, float]:
    """(width, height) on the page of `text` at cap height `cap` (read bottom to top when
    `vertical`)."""
    _, _, (_, _, w, h) = _glyphs(text)
    return (h * cap, w * cap) if vertical else (w * cap, h * cap)


def text_path(text: str, cap: float, x: float, y: float, vertical: bool = False) -> str:
    """SVG path data of `text` at cap height `cap` whose box has its top-left corner at (x, y);
    `vertical` turns it a quarter left (read bottom to top, as along a vertical pipe)."""
    v, codes, (_, _, w, _) = _glyphs(text)
    v = v * cap
    if vertical:  # (u, t) -> (t, w*cap - u): the string's start at the bottom
        v = np.stack([v[:, 1], w * cap - v[:, 0]], axis=1)
    v = v + (x, y)
    parts: list[str] = []
    i = 0
    while i < len(codes):
        c = codes[i]
        if c == 1:  # MOVETO
            parts.append(f"M{v[i, 0]:.2f} {v[i, 1]:.2f}")
            i += 1
        elif c == 2:  # LINETO
            parts.append(f"L{v[i, 0]:.2f} {v[i, 1]:.2f}")
            i += 1
        elif c == 3:  # CURVE3: control, end
            parts.append(f"Q{v[i, 0]:.2f} {v[i, 1]:.2f} {v[i + 1, 0]:.2f} {v[i + 1, 1]:.2f}")
            i += 2
        elif c == 4:  # CURVE4: two controls, end
            parts.append(
                f"C{v[i, 0]:.2f} {v[i, 1]:.2f} {v[i + 1, 0]:.2f} {v[i + 1, 1]:.2f} "
                f"{v[i + 2, 0]:.2f} {v[i + 2, 1]:.2f}"
            )
            i += 3
        else:  # CLOSEPOLY (79) or STOP
            parts.append("Z")
            i += 1
    return "".join(parts)


# --- placement -------------------------------------------------------------------------------


@dataclass
class TextSettings:
    """The `text` block of render_iso.yaml (heights already in page px)."""

    cap: tuple[float, float]  # cap height range, page px
    gap: float  # x cap height between a label and its pipe / symbol
    margin: float  # x cap height of blank paper kept round every label
    bubbles: float  # share of bubbles given a tag
    lines: float  # share of pipe runs long enough that get a line number
    leaders: float  # share of leader lines given a line number at their free end
    equipment: float  # share of equipment symbols given a tag
    flags: float  # share of off-page flags given a reference
    min_cap: float  # page px: a label that only fits smaller than this is left out


def text_settings(cfg: Mapping[str, Any]) -> TextSettings | None:
    """Read `cfg["text"]` (`cfg` is the `render_iso` mapping); None when there is none."""
    t = cfg.get("text")
    if not t:
        return None
    unit = float(cfg["layout"]["scale"]) / 3.0  # heights are given for layout.scale 3
    s = TextSettings(
        cap=(float(t["cap_px"][0]) * unit, float(t["cap_px"][1]) * unit),
        gap=float(t["gap"]),
        margin=float(t["margin"]),
        bubbles=float(t["bubbles"]),
        lines=float(t["lines"]),
        leaders=float(t["leaders"]),
        equipment=float(t["equipment"]),
        flags=float(t["flags"]),
        min_cap=float(t["min_cap_px"]) * unit,
    )
    if not 0 < s.min_cap <= s.cap[0] <= s.cap[1]:
        raise ValueError(f"text: 0 < min_cap_px <= cap_px[0] <= cap_px[1] needed: {t}")
    return s


@dataclass
class _Free:
    """Where a label may go: blank paper on the raster `ink`, `margin` px from any ink and from
    the labels placed so far."""

    ink: np.ndarray  # bool, True on drawn ink
    margin: float
    taken: list[Box] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._sum = np.pad(self.ink.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def ok(self, box: Box) -> bool:
        h, w = self.ink.shape
        x0, y0, x1, y1 = box
        m = self.margin
        a, b = math.floor(x0 - m), math.floor(y0 - m)
        c, d = math.ceil(x1 + m), math.ceil(y1 + m)
        if a < 0 or b < 0 or c > w or d > h:
            return False
        s = self._sum
        if s[d, c] - s[b, c] - s[d, a] + s[b, a]:
            return False
        return all(x1 + m <= t[0] or t[2] + m <= x0 or y1 + m <= t[1] or t[3] + m <= y0
                   for t in self.taken)  # fmt: skip

    def take(self, box: Box) -> None:
        self.taken.append(box)


@dataclass
class Label:
    kind: str  # bubble | line | leader | equipment | flag
    owner: str | None  # the node it belongs to (None: a pipe run)
    text: str
    box: Box
    vertical: bool = False


def _two_lines(top: str, bottom: str, cap: float, cx: float, cy: float) -> tuple[list[tuple], Box]:
    """Two centred lines (`top`, `bottom`) around (cx, cy): ([(text, x, y)], their box)."""
    (wt, ht), (wb, hb) = text_size(top, cap), text_size(bottom, cap)
    lead = 0.45 * cap
    h = ht + lead + hb
    y0 = cy - h / 2
    rows = [(top, cx - wt / 2, y0), (bottom, cx - wb / 2, y0 + ht + lead)]
    return rows, (cx - max(wt, wb) / 2, y0, cx + max(wt, wb) / 2, y0 + h)


def place_texts(
    ink: np.ndarray,
    w_px: float,
    rng: random.Random,
    st: TextSettings,
    bubbles: Mapping[str, tuple[Point, float]],
    flags: Mapping[str, Box],
    equipment: Mapping[str, tuple[Box, str, bool]],
    pipes: Sequence[Sequence[Point]],
    leaders: Mapping[str, Point],
    bubble_links: Mapping[str, Sequence[tuple[str, str]]] | None = None,
) -> tuple[str, list[Label], dict[str, int]]:
    """Labels for one page. `ink`: the raster of what is drawn (True on ink); `bubbles`: node ->
    circle; `flags`: node -> box; `equipment`: node -> (ink box, tag prefix, may hold its tag
    inside: a vessel); `pipes`: drawn pipe polylines; `leaders`: node -> the free end of its
    line; `bubble_links`: bubble -> [(node at the other end of one of its lines, its
    `neighbour_role`)], for tags that agree with what each bubble is joined to (`loop_tags`).
    Returns (SVG fragment, labels, log: placed / skipped per kind)."""
    cap = round(rng.uniform(*st.cap), 2)
    unit = rng.randint(1, 9)
    links = {b: list((bubble_links or {}).get(b, ())) for b in bubbles}
    tags = loop_tags(links, rng, unit)
    free = _Free(ink, max(st.margin * cap, w_px))
    gap = st.gap * cap
    labels: list[Label] = []
    parts: list[str] = []
    log = {f"text_{k}_{s}": 0 for k in ("bubble", "line", "leader", "equipment", "flag")
           for s in ("placed", "skipped")}  # fmt: skip

    def put(kind: str, owner: str | None, rows: list[tuple], box: Box, vertical: bool = False,
            size: float = cap) -> None:  # fmt: skip
        free.take(box)
        for text, x, y in rows:
            parts.append(f'<path d="{text_path(text, size, x, y, vertical)}"/>')
        labels.append(Label(kind, owner, " ".join(r[0] for r in rows), box, vertical))
        log[f"text_{kind}_placed"] += 1

    # 1. bubbles: letters over the loop number, as large as fits inside the circle
    for nid in sorted(bubbles):
        if rng.random() >= st.bubbles:
            continue
        (cx, cy), r = bubbles[nid]
        top, bottom = tags[nid]
        size = min(cap, 0.42 * r)
        while size >= st.min_cap:
            rows, box = _two_lines(top, bottom, size, cx, cy)
            # inside the circle with room for the margin: all four corners
            inner = r - w_px / 2 - free.margin
            if all(math.hypot(x - cx, y - cy) <= inner for x in box[0::2] for y in box[1::2]) \
                    and free.ok(box):  # fmt: skip
                put("bubble", nid, rows, box, size=size)
                break
            size *= 0.9
        else:
            log["text_bubble_skipped"] += 1

    # 2. flags: one line in the box, shrunk to fit
    for nid in sorted(flags):
        if rng.random() >= st.flags:
            continue
        x0, y0, x1, y1 = flags[nid]
        text = flag_text(rng, unit)
        size = min(cap, 0.45 * (y1 - y0), 0.45 * (x1 - x0))
        vertical = (y1 - y0) > (x1 - x0)
        while size >= st.min_cap:
            w, h = text_size(text, size, vertical)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if free.ok(box):
                put("flag", nid, [(text, box[0], box[1])], box, vertical, size)
                break
            size *= 0.9
        else:
            log["text_flag_skipped"] += 1

    # 3. equipment: above, below, beside the symbol, or inside a vessel
    for nid in sorted(equipment):
        if rng.random() >= st.equipment:
            continue
        (x0, y0, x1, y1), prefix, inside = equipment[nid]
        text = equipment_tag(rng, prefix, unit)
        w, h = text_size(text, cap)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        spots = [
            (cx - w / 2, y0 - gap - h), (cx - w / 2, y1 + gap),
            (x1 + gap, cy - h / 2), (x0 - gap - w, cy - h / 2),
        ]  # fmt: skip
        rng.shuffle(spots)
        if inside:
            spots.insert(0, (cx - w / 2, cy - h / 2))
        for x, y in spots:
            box = (x, y, x + w, y + h)
            if free.ok(box):
                put("equipment", nid, [(text, x, y)], box)
                break
        else:
            log["text_equipment_skipped"] += 1

    # 4. leader lines: the line number at the free end, as OPEN100 has it
    for nid in sorted(leaders):
        if rng.random() >= st.leaders:
            continue
        fx, fy = leaders[nid]
        text = line_number(rng, unit)
        w, h = text_size(text, cap)
        # beside the end, then over / under it: centred, or starting / ending at the end (the
        # line often has a short level tail there that the text sits on)
        spots = [(fx + gap, fy - h / 2), (fx - gap - w, fy - h / 2),
                 (fx - w / 2, fy - gap - h), (fx - w / 2, fy + gap),
                 (fx, fy - gap - h), (fx - w, fy - gap - h),
                 (fx, fy + gap), (fx - w, fy + gap)]  # fmt: skip
        for x, y in spots:
            box = (x, y, x + w, y + h)
            if free.ok(box):
                put("leader", nid, [(text, x, y)], box)
                break
        else:
            log["text_leader_skipped"] += 1

    # 5. pipes: the line number along the longest straight stretch of a run, beside it
    for pl in pipes:
        segs = [(a, b) for a, b in zip(pl, pl[1:], strict=False) if math.dist(a, b) > 0]
        if not segs or rng.random() >= st.lines:
            continue
        a, b = max(segs, key=lambda s: math.dist(*s))
        text = line_number(rng, unit)
        vertical = abs(b[0] - a[0]) < abs(b[1] - a[1])
        w, h = text_size(text, cap, vertical)
        length = math.dist(a, b)
        if length < (h if vertical else w) + 4 * cap:
            continue
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        placed = False
        for shift in (0.0, -0.25, 0.25, -0.4, 0.4):
            along = shift * (length - (h if vertical else w))
            for side in rng.sample((-1, 1), 2):
                if vertical:  # to the left or right of the pipe, read bottom to top
                    x = mx - gap - w if side < 0 else mx + gap
                    y = my + along - h / 2
                else:  # above or below the pipe
                    x = mx + along - w / 2
                    y = my - gap - h if side < 0 else my + gap
                box = (x, y, x + w, y + h)
                if free.ok(box):
                    put("line", None, [(text, x, y)], box, vertical)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            log["text_line_skipped"] += 1

    svg = f'<g fill="black" stroke="none">{"".join(parts)}</g>' if parts else ""
    return svg, labels, log
