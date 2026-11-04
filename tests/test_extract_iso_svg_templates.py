"""Unit tests for scripts/extract_iso_svg_templates.py — SVG templates + measured ports.

A hand-written sheet in the BK-2020-04 layout stands in for the real repo. What the tests
pin: a stub half-covered by the symbol's own ink box still counts (X8087's left stub),
duplicated stub paths give one port, a REG# drawn twice exports the requested drawing,
stub-less symbols take their ports from `port_override`, and the Stage 2 template
directory is never written to.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import extract_iso_svg_templates as tpl  # noqa: E402

_STYLE = "fill:none;stroke:#000000;stroke-width:0.5"

# CV1: bow-tie + a diagonal that starts halfway along the left stub (X8087's layout) +
#      the right stub drawn twice. B1 twice: a bow-tie, then a box. G1: a bow-tie whose stubs
#      stop 2.5 mm short of it and 0.125 mm off each other, as BK-2020 draws them. D1: a disc
#      on a stem, no stubs at all. F1: two flange marks between two 5 mm pieces of pipe that
#      stop 2.5 mm short of them (511) — every path of it is stub-length.
_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg"
  xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
  width="200mm" height="100mm" viewBox="0 0 200 100">
  <g inkscape:label="REG#:CV1_symbol">
    <path style="{_STYLE}" d="m 20,10 v 5 l 10,-5 v 5 z"/>
    <path style="{_STYLE}" d="M 17.5,15 L 33,9"/>
    <path style="{_STYLE}" d="m 20,12.5 h -5"/>
    <path style="{_STYLE}" d="M 30,12.5 H 35"/>
    <path style="{_STYLE}" d="M 30,12.5 H 35"/>
  </g>
  <g inkscape:label="REG#:B1_symbol">
    <path style="{_STYLE}" d="m 60,10 v 5 l 10,-5 v 5 z"/>
  </g>
  <g inkscape:label="REG#:B1_symbol">
    <path style="{_STYLE}" d="m 60,40 h 10 v 5 h -10 z"/>
  </g>
  <g inkscape:label="REG#:G1_symbol">
    <path style="{_STYLE}" d="m 140,60 v 5 l 10,-5 v 5 z"/>
    <path style="{_STYLE}" d="M 137.5,62.5 H 132.5"/>
    <path style="{_STYLE}" d="M 152.5,62.625 H 157.5"/>
  </g>
  <g inkscape:label="REG#:D1_symbol">
    <path style="{_STYLE}" d="m 100,10 v 12"/>
    <path style="{_STYLE}" d="m 98,8 h 4 v 4 h -4 z"/>
  </g>
  <g inkscape:label="REG#:F1_symbol">
    <path style="{_STYLE}" d="m 165,80 v 5"/>
    <path style="{_STYLE}" d="m 167.5,80 v 5"/>
    <path style="{_STYLE}" d="m 170,82.5 h 5"/>
    <path style="{_STYLE}" d="m 157.5,82.5 h 5"/>
  </g>
  <g inkscape:label="REG#:D2_symbol">
    <path style="{_STYLE}" d="m 120,48 v 12"/>
    <path style="{_STYLE}" d="m 118,60 h 4 v 4 h -4 z"/>
  </g>
</svg>
"""

_TAXONOMY = [
    {
        "reg_number": "CV1",
        "variant": "1",
        "group": "G1",
        "fine_name": "valve_cv",
        "in_vocab": "yes",
        "scope": "html",
        "port_config": "2_straight",
        "port_override": "",
    },
    {
        "reg_number": "B1",
        "variant": "1",
        "group": "G1",
        "fine_name": "valve_b_bowtie",
        "in_vocab": "no",
        "scope": "extra",
        "port_config": "0",
        "port_override": "",
    },
    {
        "reg_number": "B1",
        "variant": "2",
        "group": "G1",
        "fine_name": "valve_b_box",
        "in_vocab": "no",
        "scope": "extra",
        "port_config": "0",
        "port_override": "",
    },
    {
        "reg_number": "G1",
        "variant": "1",
        "group": "G1",
        "fine_name": "valve_gapped",
        "in_vocab": "no",
        "scope": "extra",
        "port_config": "2_straight",
        "port_override": "",
    },
    {
        "reg_number": "D1",
        "variant": "1",
        "group": "G3",
        "fine_name": "disc",
        "in_vocab": "no",
        "scope": "extra",
        "port_config": "2_straight",
        "port_override": "L,R",
    },
    {
        "reg_number": "D2",
        "variant": "1",
        "group": "G3",
        "fine_name": "disc_up_stem",
        "in_vocab": "no",
        "scope": "extra",
        "port_config": "2_straight",
        "port_override": "L,R@tail",
    },
    {
        "reg_number": "F1",
        "variant": "1",
        "group": "G3",
        "fine_name": "flanges",
        "in_vocab": "yes",
        "scope": "extra",
        "port_config": "2_straight",
        "port_override": "L,R",
    },
]


@pytest.fixture()
def exported(tmp_path: Path):
    svg_dir = tmp_path / "DWG"
    svg_dir.mkdir()
    (svg_dir / "BK-2020-04-PID-1-SHT1.svg").write_text(_SVG, encoding="utf-8")
    taxonomy = tmp_path / "taxonomy.csv"
    with taxonomy.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_TAXONOMY[0]))
        w.writeheader()
        w.writerows(_TAXONOMY)
    out = tmp_path / "out"
    rows, ports = tpl.extract(svg_dir, taxonomy, out)
    return out, {r["file"]: r for r in rows}, ports


def test_half_covered_stub_is_a_port_and_duplicates_merge(exported):
    _, rows, ports = exported
    assert rows["CV1"]["detected"] == "2_straight"
    assert sorted(p.side for p in ports["CV1"]) == ["L", "R"]
    assert all(p.source == "stub" for p in ports["CV1"])


def test_ports_sit_where_the_stubs_met_the_body(exported):
    _, rows, ports = exported
    left = next(p for p in ports["CV1"] if p.side == "L")
    right = next(p for p in ports["CV1"] if p.side == "R")
    # inner stub ends at x=20 / x=30 in sheet mm (measured on ink, so within a stroke width)
    assert right.x - left.x == pytest.approx(10.0, abs=1.0)
    assert left.y == pytest.approx(right.y, abs=0.3)


def test_ports_slide_across_the_stub_gap_onto_the_body(exported):
    _, rows, ports = exported
    left = next(p for p in ports["G1"] if p.side == "L")
    right = next(p for p in ports["G1"] if p.side == "R")
    # body spans x 140-150 (stroke 0.5), stubs ended at 137.5 / 152.5: the gap is closed
    assert right.x - left.x == pytest.approx(10.5, abs=0.1)
    assert float(rows["G1"]["gap_mm"]) <= 0.15 and rows["G1"]["ok"] == "yes"
    assert left.y == pytest.approx(right.y, abs=1e-9)  # one axis, no sub-pixel jog in V5


def test_align_opposite_only_straightens_small_offsets():
    P = tpl.Port
    ports = [
        P("L", 0.0, 5.0, "stub"),
        P("R", 10.0, 5.125, "stub"),  # BK-2020 slip
        P("T", 3.0, 0.0, "stub"),
        P("B", 7.0, 9.0, "stub"),
    ]  # 4 mm apart: deliberate
    out = tpl.align_opposite(ports)
    assert out[0].y == out[1].y == pytest.approx(5.0625)
    assert (out[2].x, out[3].x) == (3.0, 7.0)
    assert (out[0].x, out[1].x) == (0.0, 10.0)  # only the cross-axis coordinate moves


def test_override_ports_land_on_the_stem_not_the_disc_box(exported):
    _, rows, ports = exported
    left = next(p for p in ports["D1"] if p.side == "L")
    right = next(p for p in ports["D1"] if p.side == "R")
    # disc box is 4 mm wide; the pipe crosses the 0.5 mm stem at x=100
    assert right.x - left.x == pytest.approx(0.5, abs=0.1)
    assert float(rows["D1"]["gap_mm"]) <= 0.15


def test_tail_override_puts_ports_on_the_stem_tip(exported):
    _, rows, ports = exported
    d1, d2 = ports["D1"], ports["D2"]
    # D2's stem tip is its top end (disc below); D1 keeps ports halfway up the box
    assert [p.side for p in d2] == ["L", "R"]
    assert all(p.y == pytest.approx(0.5, abs=0.1) for p in d2)  # the padding above the tip
    assert d2[0].y == d2[1].y
    assert d2[1].x - d2[0].x == pytest.approx(0.5, abs=0.1)  # both on the 0.5 mm stem
    assert d1[0].y == pytest.approx(float(rows["D1"]["height_mm"]) / 2, abs=0.1)
    assert rows["D2"]["ok"] == "yes"


def test_pipe_pieces_of_a_stubless_symbol_are_dropped(exported):
    out, rows, ports = exported
    svg = (out / "F1.svg").read_text(encoding="utf-8")
    assert "h 5" not in svg and svg.count("v 5") == 2  # the flange marks stay
    left, right = (next(p for p in ports["F1"] if p.side == s) for s in "LR")
    # on the outer faces of the two marks (2.5 mm apart, stroke 0.5), not at the pipe ends
    assert right.x - left.x == pytest.approx(3.0, abs=0.2)  # measured on ink
    assert float(rows["F1"]["width_mm"]) == pytest.approx(3.0 + 1.0, abs=0.2)  # + 0.5 mm pad
    assert rows["F1"]["ok"] == "yes"
    # a disc's stem is not a pipe piece: D1 keeps everything
    assert "v 12" in (out / "D1.svg").read_text(encoding="utf-8")


def test_unknown_override_position_is_rejected():
    with pytest.raises(ValueError):
        tpl.override_ports(None, (0, 0, 1, 1), "L,R@middle")
    with pytest.raises(ValueError):
        tpl.override_ports(None, (0, 0, 1, 1), "T@tail")


def test_each_drawing_of_a_shared_reg_is_exported_separately(exported):
    out, rows, _ = exported
    assert {"B1_1", "B1_2"} <= set(rows)
    bowtie = (out / "B1_1.svg").read_text(encoding="utf-8")
    box = (out / "B1_2.svg").read_text(encoding="utf-8")
    assert "l 10,-5" in bowtie and "l 10,-5" not in box
    assert "h 10 v 5" in box and "h 10 v 5" not in bowtie


def test_stubless_symbol_takes_override_ports(exported):
    _, rows, ports = exported
    assert [p.side for p in ports["D1"]] == ["L", "R"]
    assert all(p.source == "override" for p in ports["D1"])
    assert rows["D1"]["ok"] == "yes"


def test_templates_render_standalone_and_ports_are_normalised(exported):
    import pymupdf

    out, rows, _ = exported
    for name in rows:
        doc = pymupdf.open(stream=(out / f"{name}.svg").read_bytes(), filetype="svg")
        assert doc[0].get_pixmap(dpi=72).samples  # renders on its own
    with (out / "ports.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assert 0.0 <= float(row["u"]) <= 1.0 and 0.0 <= float(row["v"]) <= 1.0


def test_scope_is_carried_into_templates_csv(exported):
    out, _, _ = exported
    with (out / "templates.csv").open(newline="", encoding="utf-8") as f:
        scope = {r["file"]: r["scope"] for r in csv.DictReader(f)}
    assert scope == {
        "CV1": "html",
        "B1_1": "extra",
        "B1_2": "extra",
        "G1": "extra",
        "D1": "extra",
        "D2": "extra",
        "F1": "extra",
    }


def _marker_sheet(d: str) -> str:
    # triangle 8 x 4 marker units, tip at x=8; refX/refY at its centre; stroke 0.5 -> 4 x 2 mm
    line = "fill:none;stroke:#000000;stroke-width:0.5;marker-end:url(#m1)"
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
  width="40mm" height="40mm" viewBox="-20 -20 40 40">
  <defs><marker id="m1" orient="auto" refX="4" refY="2">
    <path id="tri" style="fill:#000000;stroke:none" d="M 0,0 0,4 8,2 Z"/></marker></defs>
  <g id="sym"><path style="{line}" d="{d}"/></g>
</svg>"""


def _ink_mm(svg: str) -> tuple[float, float, float, float]:
    """Ink box of a 40 x 40 mm sheet whose viewBox starts at (-20, -20), in user units."""
    import numpy as np
    import pymupdf

    pix = pymupdf.open(stream=svg.encode(), filetype="svg")[0].get_pixmap(dpi=254, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    ys, xs = np.nonzero(img.min(axis=2) < 128)
    mm = 40.0 / pix.w
    return xs.min() * mm - 20, ys.min() * mm - 20, xs.max() * mm - 20, ys.max() * mm - 20


@pytest.mark.parametrize(
    "d, tip",
    [
        ("M -10,0 L 0,0", ("x1", 2.0)),  # right-pointing: tip 2 mm past the end (4 units × 0.5)
        ("M 0,10 0,0", ("y0", -2.0)),  # implicit line-to after M, pointing up
        ("m 0,-10 v 10", ("y1", 2.0)),
    ],  # relative, pointing down
)
def test_markers_are_baked_into_shapes_mupdf_can_draw(d, tip):
    import xml.etree.ElementTree as ET

    root = ET.fromstring(_marker_sheet(d))
    target = next(e for e in root.iter() if e.get("id") == "sym")
    assert tpl.bake_markers(root, target) == 1
    baked = ET.tostring(root, encoding="unicode")
    assert "marker-end" not in baked and baked.count('id="tri"') == 1  # copy carries no id
    edge, want = tip
    x0, y0, x1, y1 = _ink_mm(baked)
    assert {"x1": x1, "y0": y0, "y1": y1}[edge] == pytest.approx(want, abs=0.3)


def test_an_overlay_keeps_only_its_marker():
    """241 (flow direction) is a line with an arrowhead marker; on a page the line is the pipe,
    so the template is the arrowhead alone."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(_marker_sheet("M -10,0 L 0,0"))
    target = next(e for e in root.iter() if e.get("id") == "sym")
    made: list = []
    assert tpl.bake_markers(root, target, made) == 1 and len(made) == 1
    assert tpl.keep_markers_only(target, made) == 1  # the line
    x0, y0, x1, y1 = _ink_mm(ET.tostring(root, encoding="unicode"))
    # the 4 x 2 mm arrowhead centred on the line's end (0, 0); nothing left of it
    assert (x0, x1) == (pytest.approx(-2.0, abs=0.3), pytest.approx(2.0, abs=0.3))
    assert (y0, y1) == (pytest.approx(-1.0, abs=0.3), pytest.approx(1.0, abs=0.3))


def test_unbaked_marker_is_invisible_to_mupdf():
    """Guards the reason bake_markers exists: if MuPDF ever draws markers, drop the baking."""
    _, _, x1, _ = _ink_mm(_marker_sheet("M -10,0 L 0,0"))
    assert x1 < 0.5


def test_refuses_to_write_into_stage2_templates(tmp_path: Path):
    with pytest.raises(ValueError, match="iso_symbols_clean"):
        tpl.extract(tmp_path, tmp_path / "t.csv", Path("data/raw/iso_symbols_clean"))


@pytest.mark.parametrize(
    "sides, expected",
    [("LR", "2_straight"), ("BT", "2_straight"), ("BL", "2_angle"), ("BLR", "3"), ("B", "1")],
)
def test_port_config(sides, expected):
    ports = [tpl.Port(s, 0.0, 0.0, "stub") for s in sides]
    assert tpl.port_config(ports) == expected
