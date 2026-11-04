"""Unit tests for utils/page_svg.py: one SVG per sheet from V3 assignments on a V4 layout (V5).

Small hand-built sheets pin the geometry (pipe ends, circles, flags, boxes); the tests
on a real OPEN100 sheet check what G1 asks for. Tests that need the ISO templates or the
OPEN100 data are skipped when `data/raw/template_iso_svg` / `data/raw/pid2graph/open100` is absent.
"""

from __future__ import annotations

import math
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from utils.iso_layout import transform_layout
from utils.iso_mapping import Assignment, assign_symbols, load_config
from utils.iso_templates import load_templates
from utils.page_svg import (
    Page,
    _arrow_box,
    _attach,
    _attach_end,
    _bubble_circles,
    _circle_hit,
    _fit_scale,
    _flag_polygon,
    _ray_polygon,
    _sheet_scale,
    ink_hit,
    oriented_ink_box,
    rasterize,
    render_page,
    touching_pairs,
)
from utils.pid2graph import Node, Port, Sheet, load_graphml, pipe_runs

ROOT = Path(__file__).resolve().parent.parent
_OPEN100 = ROOT / "data" / "raw" / "pid2graph" / "open100"
_TEMPLATES = ROOT / "data" / "raw" / "template_iso_svg"

needs_templates = pytest.mark.skipif(not _TEMPLATES.exists(), reason="template_iso_svg missing")
needs_open100 = pytest.mark.skipif(
    not (_OPEN100.exists() and _TEMPLATES.exists()), reason="OPEN100 data missing"
)

CFG = {
    "layout": {"scale": 3, "column_tol": 3, "gap_scale": [0.85, 1.15]},
    "page": {"stroke_px": 3, "junction_dot": False, "shrink_steps": [0.9, 0.8]},
}


def _node(nid, label, x0, y0, x1, y1):
    return Node(nid, label, x0, y0, x1, y1)


def _dot(nid, label, x, y):
    return _node(nid, label, x - 4, y - 4, x + 4, y + 4)


# the hand-built sheets are in output px already (no V4): symbols are drawn at scale 1
CFG1 = {**CFG, "layout": {**CFG["layout"], "scale": 1}}


def _render(sheet: Sheet, assignments: dict[str, Assignment], cfg=CFG) -> Page:
    runs = pipe_runs(sheet)
    return render_page(sheet, runs, assignments, cfg, sheet)


def _polylines(svg: str) -> list[list[tuple[float, float]]]:
    out = []
    for m in re.finditer(r'<polyline points="([^"]*)"', svg):
        pts = [tuple(map(float, p.split(","))) for p in m.group(1).split()]
        out.append(pts)
    return out


# --- pipe ends --------------------------------------------------------------------------------


def test_attach_on_target_only_moves_the_end():
    # pipe along x reaching an L port at (100, 50): the end slides to the port
    out = _attach([(0.0, 50.0), (90.0, 50.0)], (100.0, 50.0), "L", 6.0)
    assert out == [(0.0, 50.0), (100.0, 50.0)]


def test_attach_trims_when_the_port_is_short_of_the_end():
    out = _attach([(0.0, 50.0), (90.0, 50.0)], (70.0, 50.0), "L", 6.0)
    assert out == [(0.0, 50.0), (70.0, 50.0)]


def test_attach_offset_across_makes_a_jog_never_a_diagonal():
    out = _attach([(0.0, 50.0), (90.0, 50.0)], (100.0, 58.0), "L", 6.0)
    assert out[0] == (0.0, 50.0) and out[-1] == (100.0, 58.0)
    for p, q in zip(out, out[1:], strict=False):
        assert p[0] == q[0] or p[1] == q[1]
    assert max(p[0] for p in out) <= 100.0  # the jog stays before the port


def test_attach_never_trims_past_the_previous_vertex():
    pts = [(0.0, 50.0), (60.0, 50.0), (60.0, 90.0)]  # arrives vertically
    out = _attach(pts, (60.0, 70.0), "T", 6.0)
    assert out[:2] == pts[:2]
    assert out[-1] == (60.0, 70.0)


def test_attach_other_axis_gets_an_l():
    out = _attach([(0.0, 0.0), (40.0, 0.0)], (50.0, 30.0), "T", 6.0)
    assert out == [(0.0, 0.0), (50.0, 0.0), (50.0, 30.0)]


def test_attach_end_a_reverses_the_run():
    out = _attach_end([(90.0, 50.0), (0.0, 50.0)], "a", (100.0, 50.0), "L", 6.0)
    assert out == [(100.0, 50.0), (0.0, 50.0)]


def test_attach_end_runs_reach_into_the_symbol_at_either_end():
    b = _attach_end([(0.0, 50.0), (90.0, 50.0)], "b", (100.0, 50.0), "L", 6.0, reach=1.5)
    assert b == [(0.0, 50.0), (101.5, 50.0)]
    a = _attach_end([(90.0, 50.0), (0.0, 50.0)], "a", (100.0, 50.0), "L", 6.0, reach=1.5)
    assert a == [(101.5, 50.0), (0.0, 50.0)]
    v = _attach_end([(50.0, 0.0), (50.0, 90.0)], "b", (50.0, 100.0), "T", 6.0, reach=1.5)
    assert v == [(50.0, 0.0), (50.0, 101.5)]


# --- shapes -----------------------------------------------------------------------------------


def test_circle_hit_lies_on_the_circle():
    for side, start in (("L", (0.0, 55.0)), ("R", (200.0, 45.0)), ("T", (48.0, 0.0))):
        x, y = _circle_hit((100.0, 50.0), 20.0, start, side)
        assert (x - 100) ** 2 + (y - 50) ** 2 == pytest.approx(400.0, abs=1e-6)


def test_circle_hit_clamps_a_line_that_misses():
    x, y = _circle_hit((100.0, 50.0), 20.0, (0.0, 200.0), "L")
    assert (x - 100) ** 2 + (y - 50) ** 2 == pytest.approx(400.0, abs=1e-6)


def test_flag_polygon_points_the_right_way():
    box = (0.0, 0.0, 100.0, 30.0)
    fwd = _flag_polygon(box, "H", True)
    back = _flag_polygon(box, "H", False)
    assert max(p[0] for p in fwd) == 100.0 and fwd[2] == (100.0, 15.0)
    assert min(p[0] for p in back) == 0.0 and back[2] == (0.0, 15.0)
    down = _flag_polygon((0.0, 0.0, 30.0, 100.0), "V", True)
    assert down[2] == (15.0, 100.0)


def test_arrow_box_moves_onto_its_pipe():
    P = Port
    box = (100.0, 90.0, 120.0, 104.0)  # middle y = 97; the pipe runs at y = 100
    on = [P(100, 100, "L", "a", 0), P(120, 100, "R", "b", 1)]
    assert _arrow_box(box, "H", on) == (100.0, 93.0, 120.0, 107.0)
    # two lines (a real jog): the one nearest the box's middle
    jog = [P(100, 100, "L", "a", 0), P(120, 130, "R", "b", 1)]
    assert _arrow_box(box, "H", jog)[1] == 93.0
    # vertical arrows move along x; a port across the axis does not count
    vbox = (90.0, 100.0, 104.0, 120.0)
    assert _arrow_box(vbox, "V", [P(100, 100, "T", "a", 0)]) == (93.0, 100.0, 107.0, 120.0)
    assert _arrow_box(box, "H", [P(110, 90, "T", "a", 0)]) == box
    assert _arrow_box(box, "H", []) == box


def test_an_arrow_off_centre_in_its_box_is_drawn_on_the_pipe():
    nodes = {
        "X0": _dot("X0", "crossing", 40, 100),
        "c0": _dot("c0", "connector", 100, 100),
        "A": _node("A", "arrow", 100, 88, 120, 104),  # box middle y = 96, pipe at y = 100
        "c1": _dot("c1", "connector", 120, 100),
        "X1": _dot("X1", "crossing", 180, 100),
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    sheet = Sheet("arrow", 300, 200, nodes, [("X0", "c0"), ("c0", "A"), ("A", "c1"), ("c1", "X1")])
    page = _render(sheet, {"A": Assignment("arrow", "flow_direction", axis="H", forward=True)})
    (poly,) = re.findall(r'<polygon points="([^"]*)" fill="black"', page.symbols)
    ys = [float(q.split(",")[1]) for q in poly.split()]
    assert (min(ys) + max(ys)) / 2 == pytest.approx(100)  # tip and base centred on the pipe
    x0, y0, x1, y1 = page.annotations[0]["bbox"]
    assert (y0 + y1) / 2 == pytest.approx(100)  # the box moved with it
    for pl in _polylines(page.pipes):  # and the pipe runs straight through
        assert all(y == pytest.approx(100) for _, y in pl)


@needs_templates
def test_an_arrow_is_drawn_from_the_241_template_on_its_pipe(templates):
    arrow = next(t for t in templates.values() if t.placement == "overlay")
    left = next(
        o
        for o in (arrow.oriented(r) for r in (0, 90, 180, 270))
        if next(q.side for q in o.ports if q.port_id == 1) == "L"
    )
    nodes = {
        "X0": _dot("X0", "crossing", 40, 100),
        "c0": _dot("c0", "connector", 100, 100),
        "A": _node("A", "arrow", 100, 88, 124, 104),  # off the pipe like a PID2Graph box
        "c1": _dot("c1", "connector", 124, 100),
        "X1": _dot("X1", "crossing", 184, 100),
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    sheet = Sheet("arrow", 300, 200, nodes, [("X0", "c0"), ("c0", "A"), ("A", "c1"), ("c1", "X1")])
    asg = {"A": Assignment("arrow", "flow_direction", left, axis="H", forward=False)}
    page = _render(sheet, asg, CFG1)
    (ann,) = page.annotations
    assert (ann["label"], ann["fine_name"], ann["template"]) == ("flow_direction",) * 2 + ("241",)
    x0, y0, x1, y1 = ann["bbox"]
    assert (y0 + y1) / 2 == pytest.approx(100, abs=0.5)  # on the pipe
    assert (x1 - x0) == pytest.approx(2 * (y1 - y0), rel=0.25)  # 241's slender 2:1 head
    assert "<polygon" not in page.symbols  # the template, not the fallback triangle
    ink = rasterize(page.layer("symbols")) < 128
    # pointing left: the head is tall at its right (base) end, a point at its left (tip) end
    col = lambda x: int(ink[int(y0) : int(y1) + 1, int(x)].sum())  # noqa: E731
    assert col(x1 - 2) > 2 * col(x0 + 2)


@needs_templates
def test_an_arrow_against_a_bend_has_its_tip_on_the_bend(templates):
    arrow = next(t for t in templates.values() if t.placement == "overlay")
    right = arrow.oriented(0)  # BK-2020 draws it pointing right
    nodes = {
        "X0": _dot("X0", "crossing", 40, 100),
        "c0": _dot("c0", "connector", 100, 100),
        "A": _node("A", "arrow", 100, 92, 124, 108),
        "c1": _dot("c1", "connector", 124, 100),
        "B": _dot("B", "crossing", 124, 100),  # the bend, right at the head
        "c2": _dot("c2", "connector", 124, 160),
        "X1": _dot("X1", "crossing", 124, 160),
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    edges = [("X0", "c0"), ("c0", "A"), ("A", "c1"), ("c1", "B"), ("B", "c2"), ("c2", "X1")]
    sheet = Sheet("bend", 300, 200, nodes, edges)
    page = _render(
        sheet, {"A": Assignment("arrow", "flow_direction", right, axis="H", forward=True)}
    )
    x0, y0, x1, y1 = page.annotations[0]["bbox"]
    assert x1 == pytest.approx(124, abs=2)  # the tip is on the bend, not mid-box
    assert (y0 + y1) / 2 == pytest.approx(100, abs=0.5)


@needs_templates
def test_a_leader_points_at_the_pipe_from_its_line_and_draws_no_pipe(templates):
    arrow = next(t for t in templates.values() if t.placement == "overlay")
    nodes = {
        "X0": _dot("X0", "crossing", 40, 100),
        "c0": _dot("c0", "connector", 100, 100),
        "G": _node("G", "general", 100, 94, 112, 106),
        "c1": _dot("c1", "connector", 112, 100),
        "X1": _dot("X1", "crossing", 200, 100),
        "L": _node("L", "arrow", 124, 102, 136, 114),  # against the pipe, below it
        "c": _dot("c", "connector", 136, 114),
        "E": _dot("E", "crossing", 170, 160),  # the other end of the leader, down-right
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    edges = [("X0", "c0"), ("c0", "G"), ("G", "c1"), ("c1", "X1"), ("L", "c"), ("c", "E")]
    sheet = Sheet("leader", 300, 200, nodes, edges)
    asg = {
        "L": Assignment("leader", "leader_arrow", arrow.oriented(0)),
        "G": Assignment("generic", "other_generic"),
    }
    page = _render(sheet, asg, CFG1)
    assert page.log["leaders"] == 1
    ann = {a["id"]: a for a in page.annotations}
    assert (ann["L"]["label"], ann["L"]["fine_name"]) == ("leader_arrow", None)
    # no pipe for the leader's edge: every pipe is on the horizontal line
    for pl in _polylines(page.pipes):
        assert all(y == pytest.approx(100) for _, y in pl)
    # the head sits on its box and points up-left, away from the line's far end
    x0, y0, x1, y1 = ann["L"]["bbox"]
    assert 118 < (x0 + x1) / 2 < 136 and 98 < (y0 + y1) / 2 < 114
    ink = rasterize(page.layer("symbols")) < 128
    assert ink[int(y0) : int(y1) + 1, int(x0) : int(x1) + 1].any()
    # the line: thin, straight (at a slant) from the far end to the back of the head
    (line,) = re.findall(
        r'<polyline points="([^"]*)" fill="none" stroke="black" stroke-width="1.5"', page.symbols
    )
    (fx, fy), (bx, by) = (tuple(map(float, q.split(","))) for q in line.split())
    assert (fx, fy) == pytest.approx((170, 160)) and bx < fx and by < fy


def test_ray_polygon_stops_at_the_outline():
    poly = _flag_polygon((0.0, 0.0, 100.0, 30.0), "H", True)
    assert _ray_polygon(poly, (-50.0, 15.0), "L") == (0.0, 15.0)  # the flat back
    assert _ray_polygon(poly, (200.0, 15.0), "R") == (100.0, 15.0)  # the tip
    assert _ray_polygon(poly, (-50.0, 500.0), "L") is None


# --- a sheet without ISO templates ------------------------------------------------------------


def _plain_sheet() -> tuple[Sheet, dict[str, Assignment]]:
    """in-flag -> bubble -> box on one pipe, plus an arrow on the same pipe."""
    nodes = {
        "F": _node("F", "inlet/outlet", 20, 90, 70, 110),
        "c1": _dot("c1", "connector", 70, 100),
        "I": _node("I", "instrumentation", 100, 85, 130, 115),
        "c2": _dot("c2", "connector", 130, 100),
        "G": _node("G", "general", 200, 80, 240, 120),
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    edges = [("F", "c1"), ("c1", "I"), ("I", "c2"), ("c2", "G")]
    sheet = Sheet("plain", 300, 200, nodes, edges)
    asg = {
        "F": Assignment("flag", None, axis="H", forward=True),
        "I": Assignment("bubble", "other_isa"),
        "G": Assignment("generic", "other_generic"),
    }
    return sheet, asg


def test_plain_sheet_draws_every_symbol_once():
    sheet, asg = _plain_sheet()
    page = _render(sheet, asg)
    assert {a["id"]: a["kind"] for a in page.annotations} == {
        "F": "flag",
        "I": "bubble",
        "G": "generic",
    }
    assert [a["label"] for a in page.annotations] == [None, "other_isa", "other_generic"]
    root = ET.fromstring(page.svg)  # well-formed
    assert root.attrib["width"] == "300" and root.attrib["height"] == "200"


def test_plain_sheet_pipes_are_axis_aligned_and_meet_the_shapes():
    sheet, asg = _plain_sheet()
    page = _render(sheet, asg)
    for pl in _polylines(page.pipes):
        for p, q in zip(pl, pl[1:], strict=False):
            assert p[0] == q[0] or p[1] == q[1]
    pipes, symbols = rasterize(page.layer("pipes")) < 128, rasterize(page.layer("symbols")) < 128
    for port in page.ports:
        x, y = int(round(port["x"])), int(round(port["y"]))
        assert pipes[y - 2 : y + 3, x - 2 : x + 3].any()
        assert symbols[y - 2 : y + 3, x - 2 : x + 3].any()


def test_a_link_between_two_overlapping_symbols_is_not_drawn():
    nodes = {
        "A": _node("A", "instrumentation", 100, 80, 130, 110),
        "c": _dot("c", "connector", 128, 95),
        "B": _node("B", "instrumentation", 126, 80, 156, 110),  # overlaps A
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    sheet = Sheet("duo", 300, 200, nodes, [("A", "c"), ("c", "B")])
    asg = {"A": Assignment("bubble", "other_isa"), "B": Assignment("bubble", "other_isa")}
    page = _render(sheet, asg)
    assert page.ports == [] and page.log["zero_length_runs"] == 1 and "<polyline" not in page.pipes


def test_nested_symbols_are_not_drawn():
    sheet, asg = _plain_sheet()
    asg["G"] = Assignment("nested", None)
    page = _render(sheet, asg)
    assert {a["id"] for a in page.annotations} == {"F", "I"}


def test_rasterize_gives_one_pixel_per_svg_unit():
    img = rasterize(
        '<svg xmlns="http://www.w3.org/2000/svg" width="90" height="60" viewBox="0 0 90 60">'
        '<rect width="90" height="60" fill="white"/><rect x="10" y="20" width="30" height="10"/>'
        "</svg>"
    )
    assert img.shape == (60, 90) and img.dtype == np.uint8
    assert img[25, 20] == 0 and img[25, 50] == 255 and img[19, 20] == 255 and img[20, 20] == 0


# --- ISO templates ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def templates(cfg):
    return load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])


@needs_templates
def test_ink_box_sits_inside_the_template_box(templates):
    for t in templates.values():
        for o in t.all_orientations():
            x0, y0, x1, y1 = oriented_ink_box(o)
            assert -0.01 <= x0 < x1 <= o.w_mm + 0.01 and -0.01 <= y0 < y1 <= o.h_mm + 0.01


@needs_templates
def test_ink_box_follows_the_orientation(templates):
    t = next(iter(templates.values()))
    a = oriented_ink_box(t)
    b = oriented_ink_box(t.oriented(90))
    assert (b[2] - b[0], b[3] - b[1]) == pytest.approx((a[3] - a[1], a[2] - a[0]), abs=0.05)


@needs_templates
def test_template_ports_are_close_to_their_ink(templates):
    for t in templates.values():
        x0, y0, x1, y1 = oriented_ink_box(t)
        for p in t.ports:
            dx = max(x0 - p.x_mm, 0, p.x_mm - x1)
            dy = max(y0 - p.y_mm, 0, p.y_mm - y1)
            assert max(dx, dy) < 0.5, f"{t.file}: port {p.side} is {max(dx, dy):.2f} mm off the ink"


def _valve_sheet(gap: float = 60.0):
    """Two valves on a pipe, `gap` px apart (source px)."""
    nodes = {
        "A": _node("A", "valve", 100, 90, 124, 110),
        "c1": _dot("c1", "connector", 124, 100),
        "B": _node("B", "valve", 124 + gap, 90, 148 + gap, 110),
        "bg": _node("bg", "background", 0, 0, 400, 200),
    }
    return Sheet("valves", 400, 200, nodes, [("A", "c1"), ("c1", "B")])


@needs_templates
def test_a_valve_is_drawn_at_the_scaled_bbox_size_and_joined_to_the_pipe(templates):
    valve = next(t for t in templates.values() if t.fine_name == "valve_gate")
    sheet = _valve_sheet()
    runs = pipe_runs(sheet)
    s4, r4 = transform_layout(sheet, runs, random.Random(1), CFG)
    asg = {
        "A": Assignment("iso", "valve_gate", valve.oriented(0)),
        "B": Assignment("iso", "valve_gate", valve.oriented(0, True)),
    }
    page = render_page(s4, r4, asg, CFG, sheet)
    ann = {a["id"]: a for a in page.annotations}
    for nid in "AB":
        x0, y0, x1, y1 = ann[nid]["bbox"]
        assert ann[nid]["fine_name"] == "valve_gate"
        assert 0.8 * 72 <= max(x1 - x0, y1 - y0) <= 1.2 * 72 + 3  # about 24 px x 3
    pipes, symbols = rasterize(page.layer("pipes")) < 128, rasterize(page.layer("symbols")) < 128
    assert len(page.ports) == 2
    for port in page.ports:
        x, y = int(round(port["x"])), int(round(port["y"]))
        assert (
            pipes[y - 3 : y + 4, x - 3 : x + 4].any()
            and symbols[y - 3 : y + 4, x - 3 : x + 4].any()
        )


@needs_templates
def test_newly_overlapping_iso_symbols_are_shrunk_then_boxed(templates):
    valve = next(t for t in templates.values() if t.fine_name == "valve_gate")
    src = _valve_sheet(gap=60.0)  # the source boxes are 60 px apart ...
    close = Sheet(
        "close",
        src.width,
        src.height,
        {  # ... but the boxes handed to the renderer are 6 px apart, so ink sized from the
            # source boxes (x 3) overlaps
            "A": _node("A", "valve", 100, 90, 124, 110),
            "B": _node("B", "valve", 130, 90, 154, 110),
            "bg": src.nodes["bg"],
        },
        [],
    )
    asg = {
        "A": Assignment("iso", "valve_gate", valve.oriented(0)),
        "B": Assignment("iso", "valve_gate", valve.oriented(0)),
    }
    page = render_page(close, [], asg, CFG, src)
    assert page.log["shrunk"] + page.log["replaced"] >= 1
    boxes = {a["id"]: a["bbox"] for a in page.annotations}
    a, b = boxes["A"], boxes["B"]
    assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])  # resolved


@needs_templates
def test_overlap_that_existed_in_the_source_is_left_alone(templates):
    valve = next(t for t in templates.values() if t.fine_name == "valve_gate")
    src = Sheet(
        "overlap",
        200,
        100,
        {
            "A": _node("A", "valve", 50, 40, 90, 60),
            "B": _node("B", "valve", 80, 40, 120, 60),  # overlaps A already in the source
            "bg": _node("bg", "background", 0, 0, 200, 100),
        },
        [],
    )
    asg = {
        "A": Assignment("iso", "valve_gate", valve.oriented(0)),
        "B": Assignment("iso", "valve_gate", valve.oriented(0)),
    }
    page = render_page(src, [], asg, CFG, src)
    assert page.log["shrunk"] == 0 and page.log["replaced"] == 0


def test_touching_pairs_include_gaps_up_to_column_tol():
    sheet = Sheet(
        "gaps",
        400,
        100,
        {
            "A": _node("A", "valve", 0, 0, 20, 20),
            "B": _node("B", "valve", 22, 0, 42, 20),  # 2 px from A
            "C": _node("C", "valve", 45, 0, 65, 20),  # 3 px from B (still touching)
            "D": _node("D", "valve", 70, 0, 90, 20),  # 5 px from C
            "E": _node("E", "valve", 10, 21, 30, 40),  # 1 px under A, overlapping B in x
        },
        [],
    )
    pairs = touching_pairs(sheet, list(sheet.nodes), 3)
    assert frozenset("AB") in pairs and frozenset("BC") in pairs and frozenset("AE") in pairs
    assert frozenset("CD") not in pairs and frozenset("AC") not in pairs


@needs_templates
def test_a_pair_that_touched_in_the_source_is_not_shrunk(templates):
    valve = next(t for t in templates.values() if t.fine_name == "valve_gate")

    def nodes(gap: float):
        return {
            "A": _node("A", "valve", 100, 90, 124, 110),
            "B": _node("B", "valve", 124 + gap, 90, 148 + gap, 110),
            "bg": _node("bg", "background", 0, 0, 400, 200),
        }

    asg = {
        "A": Assignment("iso", "valve_gate", valve.oriented(0)),
        "B": Assignment("iso", "valve_gate", valve.oriented(0)),
    }
    # V4 closes both gaps to 0 and the symbols are sized from the source boxes (x 3)
    v4 = Sheet("v4", 400, 200, nodes(0.0), [])
    near = render_page(v4, [], asg, CFG, Sheet("src", 400, 200, nodes(2.0), []))
    far = render_page(v4, [], asg, CFG, Sheet("src", 400, 200, nodes(10.0), []))
    assert near.log["shrunk"] == near.log["replaced"] == 0
    assert far.log["shrunk"] + far.log["replaced"] >= 1


def _valve_with_actuator(bubble_dx: float = 0.0) -> Sheet:
    """A valve on a horizontal pipe with a bubble above it joined by a signal line (the valve
    has no port for that line: `symbol_ports` ignores `instrumentation`)."""
    nodes = {
        "X0": _dot("X0", "crossing", 40, 100),
        "c0": _dot("c0", "connector", 100, 100),
        "V": _node("V", "valve", 100, 90, 124, 110),
        "c1": _dot("c1", "connector", 124, 100),
        "X1": _dot("X1", "crossing", 184, 100),
        "I": _node("I", "instrumentation", 100 + bubble_dx, 30, 124 + bubble_dx, 54),
        "ca": _dot("ca", "connector", 112, 70),
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    edges = [("X0", "c0"), ("c0", "V"), ("V", "c1"), ("c1", "X1"), ("I", "ca"), ("ca", "V")]
    return Sheet("actuator", 300, 200, nodes, edges)


@needs_templates
def test_a_signal_line_to_an_iso_symbol_reaches_its_ink(templates):
    # a line from a plain box (not a bubble: no stem) still has to end on the valve's ink
    valve = next(t for t in templates.values() if t.fine_name == "valve_gate")
    sheet = _valve_with_actuator()
    asg = {
        "V": Assignment("iso", "valve_gate", valve.oriented(0)),
        "I": Assignment("generic", "other_generic"),
    }
    page = _render(sheet, asg)
    loose = [p for p in page.ports if p["loose"]]
    assert page.log["loose_ends"] == 1 and page.log["stems"] == 0
    loose = [p for p in loose if p["id"] == "V"]
    pipes, symbols = rasterize(page.layer("pipes")) < 128, rasterize(page.layer("symbols")) < 128
    x, y = int(round(loose[0]["x"])), int(round(loose[0]["y"]))
    assert pipes[y - 3 : y + 4, x - 3 : x + 4].any() and symbols[y - 3 : y + 4, x - 3 : x + 4].any()
    # the line stops on the valve, not above it: its end is inside the valve's ink box
    x0, y0, x1, y1 = next(a["bbox"] for a in page.annotations if a["id"] == "V")
    assert x0 <= loose[0]["x"] <= x1 and y0 <= loose[0]["y"] <= y1


@needs_templates
@pytest.mark.parametrize("dx", [0.0, 40.0])
def test_an_actuator_stem_runs_from_the_valve_centre_straight_to_the_bubble(templates, dx):
    valve = next(t for t in templates.values() if t.fine_name == "valve_gate")
    asg = {
        "V": Assignment("iso", "valve_gate", valve.oriented(0)),
        "I": Assignment("bubble", "other_isa"),
    }
    page = _render(_valve_with_actuator(dx), asg, CFG1)
    assert page.log["stems"] == 1 and page.log["loose_ends"] == 0
    x0, y0, x1, y1 = next(a["bbox"] for a in page.annotations if a["id"] == "V")
    centre = ((x0 + x1) / 2, (y0 + y1) / 2)
    (stem,) = [
        pl
        for pl in _polylines(page.pipes)
        if len(pl) == 2 and pl[0] == pytest.approx(centre, abs=0.01)
    ]
    bubble = (112 + dx, 42)  # centre of the bubble's box; its circle has radius 12
    r = math.hypot(stem[1][0] - bubble[0], stem[1][1] - bubble[1])
    assert r == pytest.approx(12, abs=0.1)  # on the circle (the middle of its stroke)
    # on the line between the two centres
    cross = (stem[1][0] - centre[0]) * (bubble[1] - centre[1]) - (stem[1][1] - centre[1]) * (
        bubble[0] - centre[0]
    )
    assert abs(cross) < 1.0
    # both ends meet the drawing
    ink = rasterize(page.layer("symbols")) < 128
    pipes = rasterize(page.layer("pipes")) < 128
    for x, y in stem:
        x, y = int(round(x)), int(round(y))
        assert ink[y - 2 : y + 3, x - 2 : x + 3].any() and pipes[y - 2 : y + 3, x - 2 : x + 3].any()


@needs_templates
@pytest.mark.parametrize("name", ["rupture_disc", "compensator", "reducer"])
def test_a_stem_to_a_hollow_fitting_starts_on_its_ink_nearest_the_bubble(templates, name):
    # a fitting that is not a valve nor a flange pair: the line starts on its drawing, at the ink
    # nearest the bubble, not from the middle of the symbol (which may be empty)
    tpl = next(t for t in templates.values() if t.fine_name == name)
    asg = {"V": Assignment("iso", name, tpl.oriented(0)), "I": Assignment("bubble", "other_isa")}
    page = _render(_valve_with_actuator(), asg, CFG1)
    assert page.log["stems"] == 1
    x0, y0, x1, y1 = next(a["bbox"] for a in page.annotations if a["id"] == "V")
    (stem,) = [pl for pl in _polylines(page.pipes) if len(pl) == 2 and pl[0][1] < 100]
    start = stem[0]
    ink = rasterize(page.layer("symbols")) < 128
    x, y = int(round(start[0])), int(round(start[1]))
    assert ink[y - 1 : y + 2, x - 1 : x + 2].any()  # on the drawing, not in a gap
    assert y0 - 1 <= start[1] <= y0 + (y1 - y0) / 2  # on the side facing the bubble above


@needs_templates
@pytest.mark.parametrize("dx", [0.0, 5.0, -5.0])
def test_a_flange_pair_tag_enters_the_gap_square_to_the_pipe(templates, dx):
    # OPEN100: the line of an FO / RO tag starts on the pipe axis between the two marks and runs
    # square to the pipe up to the tag; the gap has no ink, so the start is not on the symbol
    flange = next(t for t in templates.values() if t.fine_name == "flanged_connection")
    asg = {
        "V": Assignment("iso", "flanged_connection", flange.oriented(0)),
        "I": Assignment("bubble", "other_isa"),
    }
    page = _render(_valve_with_actuator(dx), asg, CFG1)
    assert page.log["stems"] == 1 and page.log["loose_ends"] == 0
    (port,) = [p for p in page.ports if p["id"] == "V" and p["loose"]]
    assert port["gap"]["axis"] == "H"
    ports = sorted(
        (p for p in page.ports if p["id"] == "V" and not p["loose"]), key=lambda p: p["x"]
    )
    assert len(ports) == 2
    assert port["x"] == pytest.approx((ports[0]["x"] + ports[1]["x"]) / 2, abs=0.5)  # mid-gap
    assert port["y"] == pytest.approx(ports[0]["y"], abs=0.5)  # on the pipe axis
    assert port["gap"]["half"] == pytest.approx((ports[1]["x"] - ports[0]["x"]) / 2, abs=0.5)
    (stem,) = [
        pl
        for pl in _polylines(page.pipes)
        if len(pl) == 2 and pl[0] == pytest.approx((port["x"], port["y"]), abs=0.01)
    ]
    assert stem[1][0] == pytest.approx(stem[0][0], abs=0.01)  # square to the pipe: vertical
    assert stem[1][1] < stem[0][1]  # towards the bubble above
    bubble = (112 + dx, 42)  # the circle has radius 12; its box is centred at x = 112 + dx
    assert math.hypot(stem[1][0] - bubble[0], stem[1][1] - bubble[1]) == pytest.approx(12, abs=0.1)


@needs_templates
@pytest.mark.parametrize("dx", [40.0, -40.0])
def test_a_flange_pair_tag_off_to_the_side_starts_where_the_pipe_meets_the_mark(templates, dx):
    # the square ray misses the circle: OPEN100 (RO 1603) runs the line diagonally from where the
    # pipe meets the mark on the bubble's side; it leaves on the mark's outer face, crossing no
    # mark, with no elbow; the start is on ink, so it is checked as a plain port (no `gap`)
    flange = next(t for t in templates.values() if t.fine_name == "flanged_connection")
    asg = {
        "V": Assignment("iso", "flanged_connection", flange.oriented(0)),
        "I": Assignment("bubble", "other_isa"),
    }
    page = _render(_valve_with_actuator(dx), asg, CFG1)
    assert page.log["stems"] == 1 and page.log["loose_ends"] == 0
    (port,) = [p for p in page.ports if p["id"] == "V" and p["loose"]]
    assert "gap" not in port
    pipe_ports = sorted(
        (p for p in page.ports if p["id"] == "V" and not p["loose"]), key=lambda p: p["x"]
    )
    near = pipe_ports[1] if dx > 0 else pipe_ports[0]  # the mark on the bubble's side
    # the template port; the pipe on that side stops within a fraction of a px of it
    assert (port["x"], port["y"]) == pytest.approx((near["x"], near["y"]), abs=0.5)
    (stem,) = [
        pl
        for pl in _polylines(page.pipes)
        if pl[0] == pytest.approx((port["x"], port["y"]), abs=0.01) and pl[-1][1] < port["y"] - 5
    ]
    assert len(stem) == 2  # one straight leg
    start, end = stem
    bubble = (112 + dx, 42)
    assert math.hypot(end[0] - bubble[0], end[1] - bubble[1]) == pytest.approx(12, abs=0.1)
    cross = (end[0] - start[0]) * (bubble[1] - start[1]) - (end[1] - start[1]) * (
        bubble[0] - start[0]
    )
    assert abs(cross) < 1.0  # aimed at the circle's centre
    # no ink of the symbol under the line after its first couple of strokes
    ink = rasterize(page.layer("symbols")) < 128
    n = int(math.dist(start, end) * 2)
    for k in range(n):
        t = k / n
        x, y = start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t
        if math.dist((x, y), end) > 3 and math.dist((x, y), start) > 3:
            assert not ink[int(round(y)), int(round(x))], (x, y)
    # it meets the drawing at the start: the mark and the pipe are both there
    pipes = rasterize(page.layer("pipes")) < 128
    x, y = int(round(start[0])), int(round(start[1]))
    assert ink[y - 1 : y + 2, x - 1 : x + 2].any() and pipes[y - 1 : y + 2, x - 1 : x + 2].any()


@needs_templates
def test_a_bubble_whose_box_overlaps_the_valve_still_gets_its_stem(templates):
    valve = next(t for t in templates.values() if t.fine_name == "valve_gate")
    nodes = {
        "X0": _dot("X0", "crossing", 40, 100),
        "c0": _dot("c0", "connector", 100, 100),
        "V": _node("V", "valve", 100, 90, 124, 110),
        "c1": _dot("c1", "connector", 124, 100),
        "X1": _dot("X1", "crossing", 184, 100),
        # the bubble's box overlaps the valve's; two links of no length join them
        "I": _node("I", "instrumentation", 104, 70, 128, 94),
        "ca": _dot("ca", "connector", 112, 92),
        "cb": _dot("cb", "connector", 118, 92),
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    edges = [("X0", "c0"), ("c0", "V"), ("V", "c1"), ("c1", "X1")]
    edges += [("I", "ca"), ("ca", "V"), ("I", "cb"), ("cb", "V")]
    sheet = Sheet("overlap", 300, 200, nodes, edges)
    asg = {
        "V": Assignment("iso", "valve_gate", valve.oriented(0)),
        "I": Assignment("bubble", "other_isa"),
    }
    page = _render(sheet, asg, CFG1)
    assert page.log["stems"] == 1  # one stem for the pair, not one per run
    ends = [pl[1] for pl in _polylines(page.pipes) if len(pl) == 2]
    assert any(abs(math.hypot(x - 116, y - 82) - 12) < 0.2 for x, y in ends)  # on the circle


@needs_templates
def test_a_bubble_whose_circle_touches_the_valve_gets_no_stem(templates):
    valve = next(t for t in templates.values() if t.fine_name == "valve_gate")
    sheet = _valve_with_actuator()
    nodes = dict(sheet.nodes)
    nodes["I"] = _node("I", "instrumentation", 100, 78, 124, 102)  # sits on the valve
    page = _render(
        Sheet("touch", 300, 200, nodes, sheet.edges),
        {"V": Assignment("iso", "valve_gate", valve.oriented(0)), "I": Assignment("bubble", "x")},
        CFG1,
    )
    assert page.log["stems"] == 0


@needs_templates
def test_a_disc_between_flanges_gets_two_boxes(templates):
    disc = next(t for t in templates.values() if t.fine_name == "blind_disc")
    o = next(o for o in disc.all_orientations() if o.sides == ("L", "R"))
    flange = next(t for t in templates.values() if t.fine_name == "flanged_connection")
    fo = next(f for f in flange.all_orientations() if set(f.sides) == {"L", "R"})
    nodes = {
        "X0": _dot("X0", "crossing", 40, 100),
        "c0": _dot("c0", "connector", 100, 100),
        "D": _node("D", "general", 100, 92, 108, 108),
        "c1": _dot("c1", "connector", 108, 100),
        "X1": _dot("X1", "crossing", 168, 100),
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    sheet = Sheet("disc", 300, 200, nodes, [("X0", "c0"), ("c0", "D"), ("D", "c1"), ("c1", "X1")])
    bare = _render(sheet, {"D": Assignment("iso", "blind_disc", o)}, CFG1)
    page = _render(sheet, {"D": Assignment("iso", "blind_disc", o, flanges=fo)}, CFG1)
    assert bare.log["flanges"] == 0 and page.log["flanges"] == 1
    ann = {a["id"]: a for a in page.annotations}
    assert set(ann) == {"D", "D/flanges"}
    # the disc keeps its own box; the flanges are a second symbol on the same node
    assert ann["D"]["bbox"] == next(a["bbox"] for a in bare.annotations)
    fl = ann["D/flanges"]
    assert (fl["node"], fl["label"], fl["template"]) == ("D", "flanged_connection", "511")
    x0, y0, x1, y1 = fl["bbox"]
    assert y0 < 100 < y1  # the marks cross the pipe
    w = CFG1["page"]["stroke_px"]
    assert x1 - x0 >= 4 * w  # each mark at least two strokes from the stem
    # one size for a flange pair, with or without a disc in it (at the real scale, S = 3: at
    # S = 1 the floor that keeps the marks off the stem decides instead)
    lone = _render(sheet, {"D": Assignment("iso", "flanged_connection", fo)}, CFG)
    discs = {**CFG, "flanges": {"p": 0.5, "classes": ["blind_disc"]}}
    held = _render(sheet, {"D": Assignment("iso", "blind_disc", o, flanges=fo)}, discs)

    def length(pg, aid):  # of the flange marks, across the pipe
        b = next(a["bbox"] for a in pg.annotations if a["id"] == aid)
        return b[3] - b[1]

    assert length(held, "D/flanges") == pytest.approx(length(lone, "D"), abs=1.0)
    # and the disc's head is as wide as the flange marks are long (5 mm each in BK-2020)
    disc_box = next(a["bbox"] for a in held.annotations if a["id"] == "D")
    assert disc_box[2] - disc_box[0] == pytest.approx(length(held, "D/flanges"), abs=3.0)
    # the pipes stop on the outer faces of the flanges, as on a lone flanged connection
    ends = sorted(p["x"] for p in page.ports if p["id"] == "D")
    assert len(ends) == 2
    assert ends[0] == pytest.approx(x0, abs=w) and ends[1] == pytest.approx(x1, abs=w)
    assert all(p["y"] == pytest.approx(100, abs=1) for p in page.ports if p["id"] == "D")
    ink = rasterize(page.layer("symbols")) < 128
    assert ink[int(y0) : int(y1) + 1, int(x0) : int(x0) + w + 1].any()  # a mark is drawn
    # and no pipe between the two marks: there the stem stands alone
    pipes = rasterize(page.layer("pipes")) < 128
    assert not pipes[95:106, int(x0) + w + 1 : int(x1) - w].any()
    assert pipes[100, int(x0) - 2 * w] and pipes[100, int(x1) + 2 * w]  # pipes outside


def _vessel_nozzle(right_pipe: bool) -> Sheet:
    """A vessel box with a small fitting against its right wall (a link of no length); with
    `right_pipe` the fitting is a long thin box (drawn shorter than it, so a gap is left to the
    wall) with a pipe leaving to the right."""
    nodes = {
        "T": _node("T", "tank", 100, 50, 200, 150),
        "c": _dot("c", "connector", 200, 100),
        "G": _node("G", "general", 200, 94, 212, 106),
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    edges = [("T", "c"), ("c", "G")]
    if right_pipe:
        nodes["G"] = _node("G", "general", 200, 97, 230, 103)
        nodes["c2"] = _dot("c2", "connector", 230, 100)
        nodes["X"] = _dot("X", "crossing", 290, 100)
        edges += [("G", "c2"), ("c2", "X")]
    return Sheet("nozzle", 300, 200, nodes, edges)


def _nozzle_cfg(frac: float):
    return {**CFG1, "page": {**CFG1["page"], "nozzle_stub": frac}}


@needs_templates
def test_a_device_against_a_vessel_moves_out_on_a_short_nozzle(templates):
    fa = next(t for t in templates.values() if t.fine_name == "flame_arrestor")
    o = next(o for o in fa.all_orientations() if "L" in o.sides)
    asg = {"T": Assignment("generic", "other_generic"), "G": Assignment("iso", "flame_arrestor", o)}
    flush = _render(_vessel_nozzle(False), asg, _nozzle_cfg(0.0))
    page = _render(_vessel_nozzle(False), asg, _nozzle_cfg(0.5))
    w = CFG1["page"]["stroke_px"]

    def box(pg, nid):
        return next(a["bbox"] for a in pg.annotations if a["id"] == nid)

    # without the nozzle the device hangs on the vessel's outline; with it, it stands off
    assert box(flush, "G")[0] < box(flush, "T")[2]
    assert box(page, "G")[0] >= box(page, "T")[2]
    assert page.log["nozzles"] == 1
    ((a, b),) = _polylines(page.pipes)  # the nozzle: from the vessel wall to the port
    assert a == pytest.approx((200, 100)) and b[1] == pytest.approx(100) and b[0] > 200 + w
    (port,) = page.ports
    pipes, symbols = rasterize(page.layer("pipes")) < 128, rasterize(page.layer("symbols")) < 128
    x, y = int(round(port["x"])), int(round(port["y"]))
    assert pipes[y - 1 : y + 2, x - 1 : x + 2].any() and symbols[y - 2 : y + 3, x - 2 : x + 3].any()


@needs_templates
def test_a_fitting_at_a_nozzle_is_joined_to_the_vessel_but_not_moved(templates):
    reducer = next(t for t in templates.values() if t.fine_name == "reducer")
    o = next(o for o in reducer.all_orientations() if set(o.sides) == {"L", "R"})
    asg = {"T": Assignment("generic", "other_generic"), "G": Assignment("iso", "reducer", o)}
    plain = _render(_vessel_nozzle(True), asg, _nozzle_cfg(0.0))
    page = _render(_vessel_nozzle(True), asg, _nozzle_cfg(0.5))
    g = [a["bbox"] for a in page.annotations if a["id"] == "G"]
    assert g == [a["bbox"] for a in plain.annotations if a["id"] == "G"]  # two ports: stays
    assert g[0][0] > 200 + 3  # drawn shorter than its box: off the wall ...
    assert page.log["nozzles"] == 1  # ... and the gap is piped
    assert any(pl[0] == pytest.approx((200, 100)) for pl in _polylines(page.pipes))
    assert {p["id"] for p in page.ports} == {"G"} and len(page.ports) == 2


@needs_templates
def test_a_device_standing_in_for_a_bubble_is_sized_like_a_fitting(templates):
    fa = next(t for t in templates.values() if t.fine_name == "flame_arrestor")
    o = next(o for o in fa.all_orientations() if "L" in o.sides)
    nodes = {
        "X": _dot("X", "crossing", 40, 100),
        "c": _dot("c", "connector", 100, 100),
        "I": _node("I", "instrumentation", 100, 80, 140, 120),  # a 40 px bubble
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    sheet = Sheet("bubble", 300, 200, nodes, [("X", "c"), ("c", "I")])
    asg = {"I": Assignment("iso", "flame_arrestor", o)}

    def size(page):
        x0, y0, x1, y1 = page.annotations[0]["bbox"]
        return max(x1 - x0, y1 - y0)

    big = _render(sheet, asg, CFG1)
    small = _render(sheet, asg, {**CFG1, "page": {**CFG1["page"], "replaced_size": 14}})
    w = CFG1["page"]["stroke_px"]
    assert size(big) > 30
    assert 10 < size(small) <= 14 + w and small.log["resized"] == 1
    assert len(small.ports) == 1  # still on its pipe


@needs_templates
def test_discs_and_flanges_take_the_scale_of_the_valves_on_the_sheet(templates):
    valve = next(t for t in templates.values() if t.fine_name == "valve_gate")
    disc = next(t for t in templates.values() if t.fine_name == "blind_disc")
    flange = next(t for t in templates.values() if t.fine_name == "flanged_connection")
    vo = valve.oriented(0)
    do = next(o for o in disc.all_orientations() if o.sides == ("L", "R"))
    fo = next(o for o in flange.all_orientations() if set(o.sides) == {"L", "R"})
    nodes = {
        "X0": _dot("X0", "crossing", 20, 100),
        "c0": _dot("c0", "connector", 40, 100),
        "V": _node("V", "valve", 40, 90, 64, 110),
        "c1": _dot("c1", "connector", 64, 100),
        "c2": _dot("c2", "connector", 120, 100),
        "D": _node("D", "general", 120, 94, 126, 106),  # a thin flange-joint box
        "c3": _dot("c3", "connector", 126, 100),
        "c4": _dot("c4", "connector", 200, 100),
        "F": _node("F", "general", 200, 94, 206, 106),
        "c5": _dot("c5", "connector", 206, 100),
        "X1": _dot("X1", "crossing", 260, 100),
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    chain = ["X0", "c0", "V", "c1", "c2", "D", "c3", "c4", "F", "c5", "X1"]
    sheet = Sheet("scale", 300, 200, nodes, list(zip(chain, chain[1:], strict=False)))
    asg = {
        "V": Assignment("iso", "valve_gate", vo),
        "D": Assignment("iso", "blind_disc", do, flanges=fo),
        "F": Assignment("iso", "flanged_connection", fo),
    }
    # at the real scale (S = 3): at S = 1 the floor that keeps flange marks off a disc's stem wins
    cfg = {**CFG, "flanges": {"p": 0.5, "classes": ["blind_disc"], "symbol": "flanged_connection"}}
    k_v = _sheet_scale(asg, sheet, 3.0)
    assert k_v == pytest.approx(_fit_scale(vo, (40, 90, 64, 110), 3.0))
    page = _render(sheet, asg, cfg)
    box = {a["id"]: a["bbox"] for a in page.annotations}
    w = CFG["page"]["stroke_px"]

    def mm(t, i):  # ink extent of a template along x (i = 0) or y (i = 1), mm
        b = oriented_ink_box(t)
        return b[2 + i] - b[i]

    # a lone flange pair and the one around the disc: 5 mm marks at the valves' scale
    for fid in ("F", "D/flanges"):
        x0, y0, x1, y1 = box[fid]
        assert y1 - y0 == pytest.approx(k_v * mm(fo, 1) + w, abs=1.5)
    # the disc too, not its own thin box: its head is as wide as the marks are long
    x0, y0, x1, y1 = box["D"]
    assert x1 - x0 == pytest.approx(k_v * mm(do, 0) + w, abs=1.5)
    # a sheet without valves keeps the old sizing
    assert _sheet_scale({"D": asg["D"]}, sheet, 3.0) is None


@needs_templates
def test_a_disc_turns_to_the_free_side_of_its_pipe_before_shrinking(templates):
    disc = next(t for t in templates.values() if t.fine_name == "blind_disc")
    turns = [o for o in disc.all_orientations() if o.sides == ("L", "R")]

    def stem_up(o):
        x0, y0, x1, y1 = oriented_ink_box(o)
        return (y0 + y1) / 2 < o.ports[0].y_mm

    up = next(o for o in turns if stem_up(o))
    nodes = {
        "X0": _dot("X0", "crossing", 40, 100),
        "c0": _dot("c0", "connector", 100, 100),
        "D": _node("D", "general", 100, 92, 108, 108),
        "c1": _dot("c1", "connector", 108, 100),
        "X1": _dot("X1", "crossing", 168, 100),
        "W": _node("W", "general", 90, 40, 120, 70),  # right where the stem would stand up
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    edges = [("X0", "c0"), ("c0", "D"), ("D", "c1"), ("c1", "X1")]
    sheet = Sheet("turn", 300, 200, nodes, edges)
    cfg = {**CFG1, "flanges": {"p": 0.5, "classes": ["blind_disc"]}}
    asg = {"D": Assignment("iso", "blind_disc", up), "W": Assignment("generic", "other_generic")}
    page = _render(sheet, asg, cfg)
    assert page.log["turned"] == 1 and page.log["shrunk"] == 0 and page.log["replaced"] == 0
    x0, y0, x1, y1 = next(a["bbox"] for a in page.annotations if a["id"] == "D")
    assert y0 > 95 and y1 > 120  # now below the pipe, at full size
    assert all(p["y"] == pytest.approx(100, abs=1) for p in page.ports)  # still on the pipe


@needs_templates
def test_a_pump_whose_box_is_wide_sits_under_its_pipes(templates):
    pump = next(t for t in templates.values() if t.fine_name == "pump_general")
    o = next(o for o in pump.all_orientations() if o.sides == ("L", "R"))
    nodes = {
        "P": _node("P", "pump", 100, 100, 225, 151),  # a wide box, two pipes in on top
        "ca": _dot("ca", "connector", 150, 100),
        "cb": _dot("cb", "connector", 190, 100),
        "Xa": _dot("Xa", "crossing", 150, 40),
        "Xb": _dot("Xb", "crossing", 190, 40),
        "bg": _node("bg", "background", 0, 0, 300, 200),
    }
    edges = [("P", "ca"), ("ca", "Xa"), ("P", "cb"), ("cb", "Xb")]
    sheet = Sheet("pump", 300, 200, nodes, edges)
    cfg = CFG1
    page = render_page(
        sheet, pipe_runs(sheet), {"P": Assignment("iso", "pump_general", o)}, cfg, sheet
    )
    x0, y0, x1, y1 = page.annotations[0]["bbox"]
    assert (x0 + x1) / 2 == pytest.approx(170, abs=2.0)  # centred on the two pipes
    assert y0 == pytest.approx(100, abs=3.0)  # right behind their ends
    assert x0 < 150 and 190 < x1  # both pipes land on the symbol
    for pl in _polylines(page.pipes):
        assert len(pl) == 2 and pl[0][0] == pl[1][0]  # straight down, no jog


@needs_templates
def test_ink_hit_lies_on_the_ink_inside_the_ink_box(templates):
    for t in list(templates.values())[:12]:
        o = t.oriented(0)
        x0, y0, x1, y1 = oriented_ink_box(o)
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        for side, cross in (("L", cy), ("R", cy), ("T", cx), ("B", cx)):
            x, y = ink_hit(o, side, cross)
            assert x0 - 0.1 <= x <= x1 + 0.1 and y0 - 0.1 <= y <= y1 + 0.1
        # entered from opposite sides along one line, the hits are ordered
        assert ink_hit(o, "L", cy)[0] <= ink_hit(o, "R", cy)[0]
        assert ink_hit(o, "T", cx)[1] <= ink_hit(o, "B", cx)[1]
        # a line that misses every stroke slides to the nearest one that hits
        assert y0 - 0.1 <= ink_hit(o, "L", -50.0)[1] <= y1 + 0.1


@needs_templates
def test_vessels_are_boxes_for_now(templates):
    vessel = next(t for t in templates.values() if t.port_config == "multi")
    sheet = Sheet(
        "vessel",
        200,
        200,
        {
            "T": _node("T", "tank", 40, 40, 140, 160),
            "bg": _node("bg", "background", 0, 0, 200, 200),
        },
        [],
    )
    page = render_page(sheet, [], {"T": Assignment("iso", "x", vessel)}, CFG, sheet)
    assert page.annotations[0]["kind"] == "generic"
    assert page.annotations[0]["label"] == "other_generic"
    assert page.log["multi_as_generic"] == 1


# --- a real sheet -----------------------------------------------------------------------------


def _real_page(cfg, templates, i: int = 3, seed: int = 0):
    from PIL import Image

    sheet = load_graphml(_OPEN100 / f"{i}.graphml", Image.open(_OPEN100 / f"{i}.png").size)
    runs = pipe_runs(sheet)
    rng = random.Random(seed)
    asg = assign_symbols(sheet, runs, templates, rng, cfg)
    s4, r4 = transform_layout(sheet, runs, rng, cfg)
    return sheet, asg, render_page(s4, r4, asg, cfg, sheet)


@needs_open100
def test_real_sheet_draws_every_symbol_but_the_nested(cfg, templates):
    sheet, asg, page = _real_page(cfg, templates)
    expected = {n for n, a in asg.items() if a.kind != "nested"}
    assert {a["node"] for a in page.annotations} == expected
    parts = [a for a in page.annotations if a["id"] != a["node"]]
    assert len(parts) == page.log["flanges"]  # the flanges around a disc: a second box
    assert page.width == pytest.approx(sheet.width * 3, rel=0.2)
    ET.fromstring(page.svg)


@needs_open100
def test_real_sheet_pipes_are_axis_aligned(cfg, templates):
    _, _, page = _real_page(cfg, templates)
    n = diagonal = 0
    for pl in _polylines(page.pipes):
        for p, q in zip(pl, pl[1:], strict=False):
            n += 1
            diagonal += not (abs(p[0] - q[0]) < 1e-6 or abs(p[1] - q[1]) < 1e-6)
    assert n > 100
    assert diagonal == page.log["stems"]  # an actuator stem is the one diagonal line


@needs_open100
def test_real_sheet_ports_are_joined(cfg, templates):
    _, _, page = _real_page(cfg, templates)
    pipes, symbols = rasterize(page.layer("pipes")) < 128, rasterize(page.layer("symbols")) < 128
    import cv2

    touch = cv2.dilate(symbols.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert len(page.ports) > 50
    for p in page.ports:
        x, y = int(round(p["x"])), int(round(p["y"]))
        zone = (slice(y - 3, y + 4), slice(x - 3, x + 4))
        assert (pipes[zone] & touch[zone]).any(), f"{p['id']} not joined"


@needs_open100
def test_real_sheet_is_deterministic(cfg, templates):
    _, _, a = _real_page(cfg, templates, seed=5)
    _, _, b = _real_page(cfg, templates, seed=5)
    assert a.svg == b.svg and a.annotations == b.annotations


@needs_open100
def test_real_sheet_has_no_new_iso_overlaps(cfg, templates):
    sheet, _, page = _real_page(cfg, templates)

    def hit(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    ann = page.annotations
    nodes = sorted({a["node"] for a in ann})
    touching = touching_pairs(sheet, nodes, cfg["layout"]["column_tol"])
    for i, a in enumerate(ann):
        for b in ann[i + 1 :]:
            if a["node"] == b["node"]:
                continue  # a disc and its flanges
            if "iso" in (a["kind"], b["kind"]) and hit(a["bbox"], b["bbox"]):
                pair = frozenset((a["node"], b["node"]))
                assert pair in touching, f"{a['id']} x {b['id']} is new"


@needs_open100
@pytest.mark.parametrize("i", range(12))
def test_every_open100_sheet_renders(cfg, templates, i):
    _, _, page = _real_page(cfg, templates, i=i)
    assert page.annotations and page.pipes
    assert page.log["zero_length_runs"] >= 0
    assert rasterize(page.svg).shape == (page.height, page.width)


# --- tags that stand side by side -------------------------------------------------------------


def _tags(boxes: dict[str, tuple[float, float, float, float]], joint: list[tuple[str, str]]):
    """A sheet of tags (bubbles) with a connector between each pair in `joint` (at the middle of
    the two boxes' overlap or gap, so the run between them has no length)."""
    nodes = {n: _node(n, "instrumentation", *b) for n, b in boxes.items()}
    edges = []
    for k, (a, b) in enumerate(joint):
        (ax0, ay0, ax1, ay1), (bx0, _, bx1, _) = boxes[a], boxes[b]
        x, y = (max(ax0, bx0) + min(ax1, bx1)) / 2, (ay0 + ay1) / 2
        nodes[f"c{k}"] = _dot(f"c{k}", "connector", x, y)
        edges += [(a, f"c{k}"), (f"c{k}", b)]
    sheet = Sheet("tags", 600, 300, nodes, edges)
    kind = {n: "bubble" for n in boxes}
    return sheet, pipe_runs(sheet), kind


def _gap(c, a, b):
    (ca, ra), (cb, rb) = c[a], c[b]
    return math.dist(ca, cb) - ra - rb


def test_overlapping_tags_end_up_touching_from_outside():
    sheet, runs, kind = _tags(
        {"A": (100, 100, 148, 148), "B": (140, 100, 188, 148)}, [("A", "B")]
    )  # the circles overlap by 8 px
    c = _bubble_circles(sheet, runs, kind, 3.0)
    assert _gap(c, "A", "B") == pytest.approx(0, abs=0.06)
    assert c["A"][1] == c["B"][1] == 24  # the radii stay
    # each takes half, 4 px, along the line between the centres (here horizontal)
    assert c["A"][0] == pytest.approx((120, 124), abs=0.06)  # from 124
    assert c["B"][0] == pytest.approx((168, 124), abs=0.06)  # from 164


def test_tags_joined_by_a_run_of_no_length_close_a_gap():
    # boxes wider than high: the circles are narrower than the boxes and leave a gap
    sheet, runs, kind = _tags({"A": (100, 100, 160, 148), "B": (160, 100, 220, 148)}, [("A", "B")])
    assert len({tuple(q) for q in runs[0].points}) <= 1
    c = _bubble_circles(sheet, runs, kind, 3.0)
    assert _gap(c, "A", "B") == pytest.approx(0, abs=0.06)
    assert c["A"][0][0] > 124 and c["B"][0][0] < 196  # moved towards each other from 130 / 190


def test_tags_that_are_not_joined_keep_their_gap_but_not_an_overlap():
    far = _tags({"A": (100, 100, 148, 148), "B": (153, 100, 201, 148)}, [])
    c = _bubble_circles(*far, 3.0)
    assert c["A"][0] == (124, 124) and c["B"][0] == (177, 124)  # 5 px apart, nothing joins them
    over = _tags({"A": (100, 100, 148, 148), "B": (140, 100, 188, 148)}, [])
    c = _bubble_circles(*over, 3.0)  # overlapping circles are separated even with no link
    assert _gap(c, "A", "B") == pytest.approx(0, abs=0.06)
    slight = _tags({"A": (100, 100, 148, 148), "B": (146, 100, 194, 148)}, [])
    c = _bubble_circles(*slight, 3.0)  # up to a stroke they count as touching
    assert c["A"][0] == (124, 124)


def test_a_row_of_three_tags_settles_without_overlaps():
    sheet, runs, kind = _tags(
        {"A": (100, 100, 148, 148), "B": (140, 100, 188, 148), "C": (180, 100, 228, 148)},
        [("A", "B"), ("B", "C")],
    )
    c = _bubble_circles(sheet, runs, kind, 3.0)
    assert _gap(c, "A", "B") == pytest.approx(0, abs=0.1)
    assert _gap(c, "B", "C") == pytest.approx(0, abs=0.1)
    assert _gap(c, "A", "C") > 0


def test_bubble_stats_count_moved_tags_and_pairs_left_unsettled():
    # a row of five overlapping tags: settling one pair upsets the next, so it takes several
    # passes (14 here); cut short, the pairs still to move are reported, not silently left
    boxes = {chr(65 + k): (100 + 40 * k, 100, 148 + 40 * k, 148) for k in range(5)}
    links = [(chr(65 + k), chr(66 + k)) for k in range(4)]
    stats: dict[str, int] = {}
    c = _bubble_circles(*_tags(boxes, links), 3.0, stats)
    assert stats == {"moved": 5, "unsettled": 0}
    assert all(_gap(c, a, b) == pytest.approx(0, abs=0.06) for a, b in links)
    stats = {}
    _bubble_circles(*_tags(boxes, links), 3.0, stats, rounds=1)
    assert stats["moved"] == 5 and stats["unsettled"] > 0
    # tags apart, with nothing joining them: none moves
    stats = {}
    _bubble_circles(*_tags({"A": (100, 100, 148, 148), "B": (160, 100, 208, 148)}, []), 3.0, stats)
    assert stats == {"moved": 0, "unsettled": 0}


def test_render_page_logs_the_moved_tags():
    nodes = {
        "I1": _node("I1", "instrumentation", 100, 100, 148, 148),
        "I2": _node("I2", "instrumentation", 140, 100, 188, 148),
        "c": _dot("c", "connector", 144, 124),
        "bg": _node("bg", "background", 0, 0, 300, 300),
    }
    sheet = Sheet("t", 300, 300, nodes, [("I1", "c"), ("c", "I2")])
    asg = {"I1": Assignment("bubble", "other_isa"), "I2": Assignment("bubble", "other_isa")}
    page = _render(sheet, asg, CFG1)
    assert page.log["bubbles_moved"] == 2 and page.log["bubbles_unsettled"] == 0


def test_a_drawn_tag_follows_its_moved_circle_and_its_pipe_stops_on_it():
    nodes = {
        "I1": _node("I1", "instrumentation", 100, 100, 148, 148),
        "I2": _node("I2", "instrumentation", 140, 100, 188, 148),
        "c": _dot("c", "connector", 144, 124),
        "ca": _dot("ca", "connector", 124, 148),
        "X": _dot("X", "crossing", 124, 220),
        "bg": _node("bg", "background", 0, 0, 300, 300),
    }
    sheet = Sheet("t", 300, 300, nodes, [("I1", "c"), ("c", "I2"), ("I1", "ca"), ("ca", "X")])
    asg = {"I1": Assignment("bubble", "other_isa"), "I2": Assignment("bubble", "other_isa")}
    page = _render(sheet, asg, CFG1)
    boxes = {a["id"]: a["bbox"] for a in page.annotations}
    circles = [
        (float(cx), float(cy), float(r))
        for cx, cy, r in re.findall(
            r'<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)"', page.symbols
        )
    ]
    assert len(circles) == 2
    (ax, ay, ar), (bx, by, br) = sorted(circles)
    assert math.hypot(bx - ax, by - ay) == pytest.approx(ar + br, abs=0.1)  # touching
    # the annotation boxes are the moved circles' squares (grown by half a stroke)
    assert boxes["I1"][2] == pytest.approx(ax + ar + 1.5, abs=0.1)
    # the pipe from the first tag down stops on its moved circle (the port), and its end is
    # inside the stroke: half a stroke short of that point
    (port,) = [p for p in page.ports if p["id"] == "I1"]
    assert math.hypot(port["x"] - ax, port["y"] - ay) == pytest.approx(ar, abs=0.1)
    (pipe,) = _polylines(page.pipes)
    assert pipe[0] == pytest.approx((port["x"], port["y"] - 1.5), abs=0.1)
