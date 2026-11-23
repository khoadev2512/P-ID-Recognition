"""Engineering sheet frame round a V5 page (PLAN_PHASE_C §2M item 7, OPEN100-ISO v2).

Real P&IDs (OPEN100's own sheets among them) and the DigitizePID pages Stage 1 learns from sit in
a drawing frame: a trim line, a band of zone references (numbers along the top and bottom,
letters down the sides), a heavy border, and a column of notes, a revision table and a title
block. A page without one is easier to test on than a real drawing. `add_frame` puts the drawn
page, unchanged, inside such a frame: the page grows, the drawing moves by a whole number of
pixels, and every coordinate the page records (COCO boxes, ports, signal lines, labels) moves
with it. Nothing of the frame lies over the drawing (the column sits beside it, the band round
it), and nothing of it is a COCO object.

The layout follows the usual conventions (ISO 5457 zones, a revision table over a title block);
every string is made up here from generic parts (no text of DigitizePID or OPEN100), one draw of
`rng` per page. Glyphs come from `utils.page_text` (outlines, the same font as the labels).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from utils.page_svg import Page
from utils.page_text import text_path, text_size
from utils.pid2graph import Node, PipeRun, Sheet

Box = tuple[float, float, float, float]

# --- settings --------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSettings:
    cap_px: tuple[float, float]  # cap height of the notes / table text, page px
    column: tuple[float, float]  # width of the notes column, share of the drawing's width
    notes: tuple[int, int]  # numbered notes (some of them "DELETED")
    revisions: tuple[int, int]  # rows of the revision table
    rows: tuple[int, ...]  # zone rows to choose from (letters)
    p_references: float  # chance of a reference-drawings list under the notes
    p_centring: float  # chance of centring marks on the four sides


def frame_settings(cfg: Mapping[str, Any]) -> FrameSettings | None:
    """The `frame` block of a render_iso config, or None when it is missing or not enabled."""
    f = cfg.get("frame")
    if not f or not f.get("enabled", False):
        return None
    return FrameSettings(
        cap_px=tuple(map(float, f["cap_px"])),
        column=tuple(map(float, f["column"])),
        notes=tuple(map(int, f["notes"])),
        revisions=tuple(map(int, f["revisions"])),
        rows=tuple(map(int, f["rows"])),
        p_references=float(f["p_references"]),
        p_centring=float(f["p_centring"]),
    )


# --- made-up content -------------------------------------------------------------------------

_SYLLABLES = (
    "AR",
    "BEL",
    "COR",
    "DAN",
    "EL",
    "FER",
    "GAL",
    "HOL",
    "IN",
    "KAR",
    "LUM",
    "MER",
    "NOR",
    "OR",
    "PAX",
    "QUIN",
    "RAV",
    "SOL",
    "TER",
    "UL",
    "VAN",
    "WEST",
    "XEN",
    "ZOR",
)
_TRADES = (
    "ENGINEERING",
    "PROCESS SYSTEMS",
    "CHEMICALS",
    "ENERGY",
    "TECHNOLOGIES",
    "INDUSTRIAL SERVICES",
    "PROJECTS",
    "PLANT DESIGN",
)
_FORMS = ("LTD", "CO.", "INC.", "JSC", "GROUP", "")
_SYSTEMS = (
    "FEED PREHEAT",
    "COOLING WATER",
    "CONDENSATE RECOVERY",
    "NITROGEN DISTRIBUTION",
    "PRODUCT TRANSFER",
    "FLARE KNOCK-OUT",
    "SOLVENT REGENERATION",
    "STEAM GENERATION",
    "CHEMICAL INJECTION",
    "FUEL GAS",
    "INSTRUMENT AIR",
    "WASTE WATER TREATMENT",
    "REACTOR LOOP",
    "STORAGE AND LOADING",
    "DRAIN AND VENT",
    "SEPARATION TRAIN",
    "HEAT RECOVERY",
    "DOSING PACKAGE",
    "COMPRESSOR SUCTION",
    "SAMPLE CONDITIONING",
)
_DEPARTMENTS = ("ENGINEERING DIVISION", "PROCESS DEPARTMENT", "DESIGN OFFICE")
_PROJECT_KINDS = ("UNIT", "PLANT", "EXPANSION", "REVAMP", "UPGRADE", "FACILITY")
_ISSUES = (
    "ISSUED FOR REVIEW",
    "ISSUED FOR COMMENTS",
    "ISSUED FOR APPROVAL",
    "ISSUED FOR HAZOP",
    "ISSUED FOR DESIGN",
    "ISSUED FOR CONSTRUCTION",
    "REVISED PER COMMENTS",
    "AS BUILT",
)
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_ZONE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I, no O (ISO 5457)
# notes: {d} a drawing number, {u} the unit, {n} a small number, {v} a valve size, {t} a tag
_NOTES = (
    "ALL LINE SIZES ARE IN INCHES UNLESS OTHERWISE NOTED.",
    "FOR SYMBOLS AND LEGEND SEE DRAWING {d}.",
    "VALVES ARE LINE SIZE UNLESS OTHERWISE NOTED.",
    "INSTRUMENT TAG NUMBERS ARE PREFIXED WITH UNIT NUMBER {u}.",
    "LINE TO BE SLOPED {n}:100 TOWARDS THE VESSEL, NO POCKETS.",
    "LOCATE DRAIN VALVES AT LOW POINTS OF THE LINE.",
    "HIGH POINT VENTS AND LOW POINT DRAINS ARE NOT SHOWN.",
    "VENDOR PACKAGE: SCOPE OF SUPPLY AS SHOWN BY THE DASHED BOUNDARY.",
    "CONTROL VALVE BYPASS TO BE {v} MINIMUM.",
    "MINIMUM STRAIGHT RUN UPSTREAM OF FLOW ELEMENT {t}: {n}0 D.",
    "HOLD: RELIEF VALVE SIZING TO BE CONFIRMED BY VENDOR.",
    "FOR INTERLOCK DETAILS SEE CAUSE AND EFFECT CHART {d}.",
    "TIE-IN POINTS ARE NUMBERED AS PER TIE-IN LIST {d}.",
    "ALL INSTRUMENT CONNECTIONS {v} UNLESS OTHERWISE NOTED.",
    "PUMP CASING DRAINS ROUTED TO CLOSED DRAIN SYSTEM.",
    "CAR SEAL OPEN VALVES SHALL BE LOCKED DURING NORMAL OPERATION.",
    "INSULATION AND TRACING AS PER LINE LIST.",
    "SPECTACLE BLINDS SHOWN IN NORMAL OPERATING POSITION.",
    "EQUIPMENT ELEVATIONS ARE TO BE CONFIRMED BY PLOT PLAN {d}.",
    "SAMPLE POINT TO BE ACCESSIBLE FROM GRADE.",
    "NITROGEN PURGE CONNECTION {v} WITH DOUBLE BLOCK VALVES.",
    "LEVEL GAUGE TO COVER THE FULL RANGE OF LEVEL TRANSMITTER {t}.",
)
_REFERENCE_TITLES = (
    "LEGEND AND SYMBOLS",
    "UTILITY DISTRIBUTION",
    "LINE LIST",
    "INSTRUMENT INDEX",
    "CAUSE AND EFFECT CHART",
    "PLOT PLAN",
    "PROCESS FLOW DIAGRAM",
    "TIE-IN LIST",
    "EQUIPMENT LIST",
)


def _name(rng: random.Random, parts: int) -> str:
    return "".join(rng.choice(_SYLLABLES) for _ in range(parts))


def _initials(rng: random.Random) -> str:
    return "".join(rng.choice("ABCDEGHKLMNPRSTVW") for _ in range(rng.choice((2, 2, 3))))


def _date(rng: random.Random, day: int, style: int) -> str:
    """Day `day` counted from 1 Jan 2015 (months of 30 days, a day past the 28th drawn as the
    28th: every date exists and later days never come out earlier)."""
    y, rest = 2015 + day // 360, day % 360
    m, d = rest // 30 + 1, min(rest % 30 + 1, 28)
    if style == 0:
        return f"{d:02d}/{m:02d}/{y % 100:02d}"
    if style == 1:
        return f"{d:02d}-{_MONTHS[m - 1]}-{y}"
    return f"{y}-{m:02d}-{d:02d}"


@dataclass
class Content:
    company: str
    company_line: str
    project: str
    title: tuple[str, ...]
    drawing: str
    unit: str
    scale: str
    sheet: str
    revisions: list[tuple[str, str, str, str, str, str]]  # rev, date, description, by, chk, app
    notes: list[str]
    references: list[tuple[str, str]]  # number, title
    signers: tuple[str, str, str]  # drawn, checked, approved


def make_content(st: FrameSettings, rng: random.Random) -> Content:
    unit = str(rng.randint(1, 9) * 100 + rng.randint(0, 99))
    code = _name(rng, 1)[:2]

    def drawing() -> str:
        return f"{code}-{unit}-PID-{rng.randint(1, 999):03d}"

    company = f"{_name(rng, rng.choice((2, 3)))} {rng.choice(_TRADES)} {rng.choice(_FORMS)}"
    n_rev = rng.randint(*st.revisions)
    numeric = rng.random() < 0.5
    style = rng.randrange(3)
    day = rng.randint(0, 360 * 8)
    revs = []
    issues = sorted(rng.sample(range(len(_ISSUES)), min(n_rev, len(_ISSUES))))
    for i in range(n_rev):
        rev = str(i) if numeric else _ZONE_LETTERS[i]
        desc = _ISSUES[issues[min(i, len(issues) - 1)]]
        revs.append((rev, _date(rng, day, style), desc, *(_initials(rng) for _ in range(3))))
        day += rng.randint(10, 120)
    revs.reverse()  # newest on top
    notes = []
    for i, tpl in enumerate(rng.sample(_NOTES, min(rng.randint(*st.notes), len(_NOTES)))):
        text = "DELETED." if rng.random() < 0.15 else tpl
        text = text.format(
            d=drawing(),
            u=unit,
            n=rng.randint(1, 5),
            v=rng.choice(('3/4"', '1"', '2"')),
            t=f"{rng.choice(('FE', 'LT', 'PT'))}-{unit}{rng.randint(1, 99):02d}",
        )
        notes.append(f"{i + 1}. {text}")
    refs = []
    if rng.random() < st.p_references:
        refs = [(drawing(), t) for t in rng.sample(_REFERENCE_TITLES, rng.randint(2, 5))]
    total = rng.randint(1, 6)
    return Content(
        company=company.strip(),
        company_line=rng.choice(_DEPARTMENTS),
        project=f"{_name(rng, 2)} {rng.choice(_PROJECT_KINDS)} - PROJECT {rng.randint(1, 99):05d}",
        title=(
            "PIPING AND INSTRUMENTATION DIAGRAM",
            f"{rng.choice(_SYSTEMS)} SYSTEM",
            f"UNIT {unit}",
        ),
        drawing=drawing(),
        unit=unit,
        scale=rng.choice(("NTS", "NONE", "N.T.S.")),
        sheet=f"{rng.randint(1, total)} OF {total}",
        revisions=revs,
        notes=notes,
        references=refs,
        signers=(_initials(rng), _initials(rng), _initials(rng)),
    )


# --- drawing ---------------------------------------------------------------------------------


class _Ink:
    """SVG fragments of the frame: lines (by stroke width) and text outlines."""

    def __init__(self) -> None:
        self.lines: dict[float, list[str]] = {}
        self.paths: list[str] = []
        self.bold: list[str] = []

    def line(self, x0: float, y0: float, x1: float, y1: float, w: float) -> None:
        self.lines.setdefault(round(w, 2), []).append(f"M{x0:.1f} {y0:.1f}L{x1:.1f} {y1:.1f}")

    def rect(self, b: Box, w: float) -> None:
        x0, y0, x1, y1 = b
        self.lines.setdefault(round(w, 2), []).append(
            f"M{x0:.1f} {y0:.1f}H{x1:.1f}V{y1:.1f}H{x0:.1f}Z"
        )

    def text(
        self,
        s: str,
        cap: float,
        x: float,
        y: float,
        w_max: float | None = None,
        align: str = "left",
        bold: bool = False,
    ) -> float:
        """`s` with its top-left corner at (x, y) (or centred / right-aligned on x), shrunk to fit
        `w_max`; returns the cap height used (0 when nothing is drawn)."""
        if not s:
            return 0.0
        w, _ = text_size(s, cap)
        if w_max is not None and w > w_max:
            cap *= w_max / w
            w = w_max
        if align == "centre":
            x -= w / 2
        elif align == "right":
            x -= w
        (self.bold if bold else self.paths).append(text_path(s, cap, x, y))
        return cap

    def svg(self) -> str:
        out = [
            f'<path d="{"".join(d)}" fill="none" stroke="black" stroke-width="{w:g}" '
            f'stroke-linecap="square"/>'
            for w, d in sorted(self.lines.items())
        ]
        if self.paths:
            out.append(f'<path d="{"".join(self.paths)}" fill="black"/>')
        if self.bold:
            out.append(
                f'<path d="{"".join(self.bold)}" fill="black" stroke="black" stroke-width="0.6"/>'
            )
        return "".join(out)


def _wrap(text: str, cap: float, width: float, indent: str = "") -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}" if cur else word
        if cur and text_size(trial, cap)[0] > width:
            lines.append(cur)
            cur = indent + word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


@dataclass(frozen=True)
class FrameLayout:
    """Where things are on the framed page (page px): `offset` moves the drawing, `content` is
    the drawing's own rectangle, `trim` the thin outer line, `border` the heavy one (the zone band
    lies between them), `column` the notes / tables column beside the drawing."""

    width: int
    height: int
    offset: tuple[int, int]
    content: Box
    trim: Box
    border: Box
    column: Box


def frame_layout(w: int, h: int, column_frac: float) -> FrameLayout:
    """Sizes from the drawing's height `h`: trim margin 1%, zone band 2.2%, gap round the
    drawing 1.5%; the column is `column_frac` of the drawing's width `w`."""
    m0 = round(0.010 * h)
    band = round(0.022 * h)
    pad = round(0.015 * h)
    cw = round(column_frac * w)
    ox = oy = m0 + band + pad
    width = ox + w + pad + cw + band + m0
    height = oy + h + pad + band + m0
    trim = (m0, m0, width - m0, height - m0)
    border = (m0 + band, m0 + band, width - m0 - band, height - m0 - band)
    column = (ox + w + pad, border[1], border[2], border[3])
    return FrameLayout(width, height, (ox, oy), (ox, oy, ox + w, oy + h), trim, border, column)


def _zones(
    ink: _Ink,
    lay: FrameLayout,
    rows: int,
    w: float,
    cap: float,
    rng: random.Random,
    centring: bool,
) -> None:
    """The zone band between the trim line and the border: ticks, numbers and letters."""
    x0, y0, x1, y1 = lay.border
    tx0, ty0, tx1, ty1 = lay.trim
    cols = max(2, 2 * round(rows * (x1 - x0) / (y1 - y0) / 2))
    numbers = list(range(1, cols + 1))
    if rng.random() < 0.4:
        numbers.reverse()  # some offices count from the right
    letters = list(_ZONE_LETTERS[:rows])
    if rng.random() < 0.5:
        letters.reverse()  # A at the bottom
    zw, zh = (x1 - x0) / cols, (y1 - y0) / rows
    for i in range(1, cols):
        x = x0 + i * zw
        ink.line(x, ty0, x, y0, w)
        ink.line(x, y1, x, ty1, w)
    for j in range(1, rows):
        y = y0 + j * zh
        ink.line(tx0, y, x0, y, w)
        ink.line(x1, y, tx1, y, w)
    for i, n in enumerate(numbers):
        cx = x0 + (i + 0.5) * zw
        for top, bottom in ((ty0, y0), (y1, ty1)):
            ink.text(str(n), cap, cx, (top + bottom - cap) / 2, align="centre")
    for j, letter in enumerate(letters):
        cy = y0 + (j + 0.5) * zh - cap / 2
        for left, right in ((tx0, x0), (x1, tx1)):
            ink.text(letter, cap, (left + right) / 2, cy, align="centre")
    if centring:  # heavy strokes at the middle of each side, across the band
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for a, b, c, e in (
            (cx, ty0, cx, y0),
            (cx, y1, cx, ty1),
            (tx0, cy, x0, cy),
            (x1, cy, tx1, cy),
        ):
            ink.line(a, b, c, e, 2.5 * w)


def _cell(
    ink: _Ink,
    box: Box,
    label: str,
    value: str,
    cap: float,
    w: float,
    align: str = "centre",
    bold: bool = False,
) -> None:
    """A table cell: its outline, a small label at the top left, the value under it."""
    x0, y0, x1, y1 = box
    ink.rect(box, w)
    pad = 0.4 * cap
    top = y0 + pad
    if label:
        ink.text(label, 0.65 * cap, x0 + pad, top, x1 - x0 - 2 * pad)
        top += 0.65 * cap + pad
    vy = (top + y1 - cap) / 2
    if align == "centre":
        ink.text(value, cap, (x0 + x1) / 2, vy, x1 - x0 - 2 * pad, "centre", bold)
    else:
        ink.text(value, cap, x0 + pad, vy, x1 - x0 - 2 * pad, bold=bold)


def _title_block(ink: _Ink, c: Content, box: Box, cap: float, w: float) -> float:
    """The title block at the bottom of the column (`box`: x0, -, x1, bottom); returns its top."""
    x0, _, x1, y = box
    cw = x1 - x0
    r = 3.4 * cap  # a two-line cell (label + value)

    def row(height: float) -> tuple[float, float]:
        nonlocal y
        y -= height
        return y, y + height

    a, b = row(r)
    for i, (lab, val) in enumerate(
        (("SCALE", c.scale), ("SHEET", c.sheet), ("REV", c.revisions[0][0] if c.revisions else "0"))
    ):
        _cell(ink, (x0 + i * cw / 3, a, x0 + (i + 1) * cw / 3, b), lab, val, cap, w)
    a, b = row(r)
    _cell(ink, (x0, a, x0 + 0.68 * cw, b), "DRAWING No.", c.drawing, 1.2 * cap, w, bold=True)
    _cell(
        ink, (x0 + 0.68 * cw, a, x1, b), "DATE", c.revisions[-1][1] if c.revisions else "", cap, w
    )
    a, b = row(r)
    for i, (lab, val) in enumerate(zip(("DRAWN", "CHECKED", "APPROVED"), c.signers, strict=True)):
        _cell(ink, (x0 + i * cw / 3, a, x0 + (i + 1) * cw / 3, b), lab, val, cap, w)
    a, b = row(0.65 * cap + 1.2 * cap + len(c.title) * 1.9 * cap)
    ink.rect((x0, a, x1, b), w)
    ink.text("TITLE", 0.65 * cap, x0 + 0.4 * cap, a + 0.4 * cap)
    ty = a + 0.65 * cap + 0.9 * cap
    for i, line in enumerate(c.title):
        ink.text(line, 1.2 * cap, (x0 + x1) / 2, ty + i * 1.9 * cap, cw - cap, "centre", i == 0)
    a, b = row(1.2 * r)
    _cell(ink, (x0, a, x1, b), "PROJECT", c.project, cap, w)
    a, b = row(2.4 * cap + 1.0 * cap + 2.2 * cap)
    ink.rect((x0, a, x1, b), w)
    ink.text(c.company, 2.0 * cap, (x0 + x1) / 2, a + 0.9 * cap, cw - 2 * cap, "centre", True)
    ink.text(c.company_line, 0.8 * cap, (x0 + x1) / 2, a + 3.6 * cap, cw - 2 * cap, "centre")
    return y


_REV_COLUMNS = (
    ("REV", 0.09),
    ("DATE", 0.19),
    ("DESCRIPTION", 0.42),
    ("BY", 0.10),
    ("CHK", 0.10),
    ("APP", 0.10),
)


def _revision_table(
    ink: _Ink, c: Content, x0: float, x1: float, bottom: float, cap: float, w: float
) -> float:
    """Revision rows (newest on top) over a header row, standing on `bottom`; returns its top."""
    rh = 2.0 * cap
    cw = x1 - x0
    rows = [*c.revisions, None]  # None: the header, at the bottom of the rows
    top = bottom - rh * len(rows)
    y = top
    for r in rows:
        x = x0
        for k, (head, frac) in enumerate(_REV_COLUMNS):
            box = (x, y, x + frac * cw, y + rh)
            ink.rect(box, w)
            text = head if r is None else r[k]
            ink.text(
                text,
                0.8 * cap if r is None else 0.85 * cap,
                x + frac * cw / 2,
                y + rh / 2 - 0.45 * cap,
                frac * cw - 0.6 * cap,
                "centre",
                r is None,
            )
            x += frac * cw
        y += rh
    return top


def _notes(ink: _Ink, c: Content, box: Box, cap: float) -> int:
    """NOTES header and numbered notes from the top of `box` down, as many as fit; returns how
    many were written."""
    x0, y0, x1, y1 = box
    pad = cap
    y = y0 + pad
    ink.text("NOTES", 1.4 * cap, (x0 + x1) / 2, y, align="centre", bold=True)
    y += 1.4 * cap + 1.2 * cap
    width = x1 - x0 - 2 * pad
    written = 0
    for note in c.notes:
        num, _, rest = note.partition(" ")
        indent = text_size(f"{len(c.notes)}.", cap)[0] + 0.7 * cap  # one column for all
        lines = _wrap(rest, cap, width - indent)
        need = len(lines) * 1.7 * cap
        if y + need > y1 - pad:
            break
        ink.text(num, cap, x0 + pad, y)
        for line in lines:
            ink.text(line, cap, x0 + pad + indent, y, width - indent)
            y += 1.7 * cap
        y += 0.6 * cap
        written += 1
    return written


def _references(
    ink: _Ink, c: Content, x0: float, x1: float, bottom: float, cap: float, w: float
) -> float:
    """Reference-drawings list standing on `bottom`; returns its top (`bottom` when none)."""
    if not c.references:
        return bottom
    rh = 1.9 * cap
    cw = x1 - x0
    top = bottom - rh * (len(c.references) + 1)
    ink.rect((x0, top, x1, bottom), w)
    ink.text(
        "REFERENCE DRAWINGS",
        0.9 * cap,
        (x0 + x1) / 2,
        top + rh / 2 - 0.45 * cap,
        cw - cap,
        "centre",
        True,
    )
    for i, (num, title) in enumerate(c.references):
        y = top + (i + 1) * rh
        ink.line(x0, y, x1, y, w)
        ink.text(num, 0.85 * cap, x0 + 0.5 * cap, y + rh / 2 - 0.42 * cap, 0.4 * cw - cap)
        ink.text(title, 0.85 * cap, x0 + 0.42 * cw, y + rh / 2 - 0.42 * cap, 0.58 * cw - cap)
    ink.line(x0 + 0.40 * cw, top + rh, x0 + 0.40 * cw, bottom, w)
    return top


@dataclass(frozen=True)
class Frame:
    layout: FrameLayout
    svg: str
    content: Content
    notes_written: int
    title_block: Box
    revision_table: Box


def draw_frame(w: int, h: int, stroke: float, st: FrameSettings, rng: random.Random) -> Frame:
    """The frame round a drawing of `w` x `h` px drawn with stroke `stroke` (one draw of `rng`)."""
    lay = frame_layout(w, h, rng.uniform(*st.column))
    c = make_content(st, rng)
    # the tables take about 60 cap heights at most: on a small page the text shrinks to fit
    cap = min(rng.uniform(*st.cap_px), (lay.column[3] - lay.column[1]) / 80)
    rows = rng.choice(st.rows)
    centring = rng.random() < st.p_centring
    ink = _Ink()
    ink.rect(lay.trim, stroke)
    ink.rect(lay.border, 2 * stroke)
    band = lay.border[0] - lay.trim[0]
    _zones(ink, lay, rows, stroke, min(0.45 * band, 2.2 * cap), rng, centring)
    x0, y0, x1, y1 = lay.column
    ink.line(x0, y0, x0, y1, 2 * stroke)
    tb_top = _title_block(ink, c, lay.column, cap, stroke)
    ink.line(x0, tb_top, x1, tb_top, 2 * stroke)
    rev_top = _revision_table(ink, c, x0, x1, tb_top, cap, stroke)
    ref_top = _references(ink, c, x0, x1, rev_top - 1.5 * cap, cap, stroke)
    written = _notes(ink, c, (x0, y0, x1, ref_top - cap), cap)
    return Frame(lay, ink.svg(), c, written, (x0, tb_top, x1, y1), (x0, rev_top, x1, tb_top))


# --- moving the drawing into the frame -------------------------------------------------------


def _shift_box(b: Sequence[float], dx: float, dy: float) -> list[float]:
    return [round(b[0] + dx, 2), round(b[1] + dy, 2), round(b[2] + dx, 2), round(b[3] + dy, 2)]


def _group(fragment: str, dx: int, dy: int) -> str:
    return f'<g transform="translate({dx} {dy})">{fragment}</g>' if fragment else ""


def add_frame(
    page: Page, stroke: float, st: FrameSettings, rng: random.Random
) -> tuple[Page, Frame]:
    """`page` inside a frame: a new Page of the framed size whose drawing (pipes, symbols, text)
    is moved by `frame.layout.offset`, as is every coordinate it records."""
    fr = draw_frame(page.width, page.height, stroke, st, rng)
    dx, dy = fr.layout.offset
    framed = replace(
        page,
        width=fr.layout.width,
        height=fr.layout.height,
        pipes=_group(page.pipes, dx, dy),
        symbols=_group(page.symbols, dx, dy),
        text=_group(page.text, dx, dy),
        frame=fr.svg,
        annotations=[{**a, "bbox": _shift_box(a["bbox"], dx, dy)} for a in page.annotations],
        ports=[{**p, "x": p["x"] + dx, "y": p["y"] + dy} for p in page.ports],
        stems=[{**s, "points": [(x + dx, y + dy) for x, y in s["points"]]} for s in page.stems],
        texts=[{**t, "box": _shift_box(t["box"], dx, dy)} for t in page.texts],
        log={**page.log, "frame_notes": fr.notes_written},
    )
    return framed, fr


def shift_sheet(sheet: Sheet, dx: float, dy: float, width: int, height: int) -> Sheet:
    nodes = {
        nid: Node(n.id, n.label, n.xmin + dx, n.ymin + dy, n.xmax + dx, n.ymax + dy)
        for nid, n in sheet.nodes.items()
    }
    return Sheet(sheet.name, width, height, nodes, list(sheet.edges))


def shift_runs(runs: Sequence[PipeRun], dx: float, dy: float) -> list[PipeRun]:
    return [replace(r, points=tuple((x + dx, y + dy) for x, y in r.points)) for r in runs]
