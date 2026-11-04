"""Tests for utils.pid2graph — pipe skeleton and symbol ports from PID2Graph GraphML.

Hand-written fixtures in the OPEN100 layout (connectors are 8x8 and sit flush on the
symbol's bbox edge) pin each rule: ports come from the connector at the bbox edge, near-axis
segments snap and genuine diagonals become an L, a port may slide along its edge but never
off it, actuator links do not count as ports. The last test runs the real 12 sheets.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from utils.pid2graph import (
    PortConfig,
    load_graphml,
    pipe_runs,
    port_config,
    symbol_ports,
)

OPEN100_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "pid2graph" / "open100"


def _graphml(
    nodes: dict[str, tuple[str, float, float, float, float]], edges: list[tuple[str, str]]
) -> str:
    """nodes: id -> (label, xmin, ymin, xmax, ymax). Key ids follow the real files."""
    out = [
        "<?xml version='1.0' encoding='utf-8'?>",
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '<key id="d0" for="node" attr.name="label" attr.type="string" />',
        '<key id="d1" for="node" attr.name="xmin" attr.type="double" />',
        '<key id="d2" for="node" attr.name="xmax" attr.type="double" />',
        '<key id="d3" for="node" attr.name="ymin" attr.type="double" />',
        '<key id="d4" for="node" attr.name="ymax" attr.type="double" />',
        '<key id="d9" for="edge" attr.name="edge_label" attr.type="string" />',
        '<graph edgedefault="undirected">',
    ]
    for nid, (label, x0, y0, x1, y1) in nodes.items():
        out.append(
            f'<node id="{nid}"><data key="d0">{label}</data><data key="d1">{x0}</data>'
            f'<data key="d2">{x1}</data><data key="d3">{y0}</data><data key="d4">{y1}</data></node>'
        )
    for u, v in edges:
        out.append(f'<edge source="{u}" target="{v}"><data key="d9">solid</data></edge>')
    out += ["</graph>", "</graphml>"]
    return "\n".join(out)


def _pt(
    label: str, cx: float, cy: float, size: float = 8.0
) -> tuple[str, float, float, float, float]:
    return (label, cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def _load(tmp_path: Path, nodes, edges):
    path = tmp_path / "s.graphml"
    path.write_text(_graphml(nodes, edges))
    return load_graphml(path)


def _port_sides(sheet, node_id: str, **kw) -> dict[str, tuple[float, float]]:
    return {p.side: (p.x, p.y) for p in symbol_ports(sheet, node_id, **kw)}


def _axis_aligned(points) -> bool:
    return all(
        abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6
        for a, b in zip(points, points[1:], strict=False)
    )


def _horizontal_valve(tmp_path: Path):
    # crossing - conn - [valve 100..120 x 90..110] - conn - crossing, on y = 100
    nodes = {
        "xa": _pt("crossing", 0, 100),
        "c1": _pt("connector", 100, 100),
        "v": ("valve", 100, 90, 120, 110),
        "c2": _pt("connector", 120, 100),
        "xb": _pt("crossing", 300, 100),
    }
    edges = [("xa", "c1"), ("c1", "v"), ("v", "c2"), ("c2", "xb")]
    return _load(tmp_path, nodes, edges)


def test_horizontal_valve_ports_sit_on_its_left_and_right_edges(tmp_path):
    sheet = _horizontal_valve(tmp_path)
    runs = pipe_runs(sheet)
    assert len(runs) == 2
    ports = _port_sides(sheet, "v", runs=runs)
    assert ports == {"L": (100.0, 100.0), "R": (120.0, 100.0)}
    assert port_config(symbol_ports(sheet, "v", runs)) == PortConfig("2_straight", "H", ("L", "R"))
    by_end = {r.a: r for r in runs}
    assert by_end["v"].points == ((120.0, 100.0), (300.0, 100.0))
    assert runs[0].points == ((0.0, 100.0), (100.0, 100.0))


def test_vertical_valve_is_2_straight_on_the_vertical_axis(tmp_path):
    nodes = {
        "xa": _pt("crossing", 110, 0),
        "c1": _pt("connector", 110, 90),
        "v": ("valve", 100, 90, 120, 110),
        "c2": _pt("connector", 110, 110),
        "xb": _pt("crossing", 110, 300),
    }
    sheet = _load(tmp_path, nodes, [("xa", "c1"), ("c1", "v"), ("v", "c2"), ("c2", "xb")])
    cfg = port_config(symbol_ports(sheet, "v"))
    assert (cfg.kind, cfg.axis, cfg.sides) == ("2_straight", "V", ("B", "T"))


def test_angle_valve_has_one_horizontal_and_one_vertical_port(tmp_path):
    nodes = {
        "xa": _pt("crossing", 0, 100),
        "c1": _pt("connector", 100, 100),
        "v": ("valve", 100, 90, 120, 110),
        "c2": _pt("connector", 110, 110),
        "xb": _pt("crossing", 110, 200),
    }
    sheet = _load(tmp_path, nodes, [("xa", "c1"), ("c1", "v"), ("v", "c2"), ("c2", "xb")])
    cfg = port_config(symbol_ports(sheet, "v"))
    assert (cfg.kind, cfg.axis) == ("2_angle", None)
    assert cfg.sides == ("B", "L")


def test_actuator_link_through_instrumentation_is_not_a_port(tmp_path):
    nodes = {
        "xa": _pt("crossing", 0, 100),
        "c1": _pt("connector", 100, 100),
        "v": ("valve", 100, 90, 120, 110),
        "c2": _pt("connector", 120, 100),
        "xb": _pt("crossing", 300, 100),
        "i": ("instrumentation", 100, 50, 120, 70),
    }
    sheet = _load(
        tmp_path, nodes, [("xa", "c1"), ("c1", "v"), ("v", "c2"), ("c2", "xb"), ("v", "i")]
    )
    assert port_config(symbol_ports(sheet, "v")).kind == "2_straight"
    with_all = symbol_ports(sheet, "v", ignore_labels=())
    assert port_config(with_all).kind == "3"
    assert sorted(p.side for p in with_all) == ["L", "R", "T"]


def test_t_junction_runs_share_the_crossing_centre(tmp_path):
    nodes = {
        "j": _pt("crossing", 50, 50),
        "a": _pt("crossing", 0, 50),
        "b": _pt("crossing", 100, 50),
        "c": _pt("crossing", 50, 0),
    }
    sheet = _load(tmp_path, nodes, [("j", "a"), ("j", "b"), ("j", "c")])
    runs = pipe_runs(sheet)
    assert len(runs) == 3
    for r in runs:
        assert r.connectors == ()
        assert (50.0, 50.0) in (r.points[0], r.points[-1])
        assert _axis_aligned(r.points)


def test_near_axis_connector_snaps_and_collinear_points_collapse(tmp_path):
    nodes = {
        "xa": _pt("crossing", 0, 100),
        "c1": _pt("connector", 200, 101.5),
        "m": _pt("connector", 100, 100.5),
        "xb": _pt("crossing", 200, 300),
    }
    sheet = _load(tmp_path, nodes, [("xa", "m"), ("m", "c1"), ("c1", "xb")])
    (run,) = pipe_runs(sheet)
    assert run.points == ((0.0, 100.0), (200.0, 100.0), (200.0, 300.0))
    assert run.elbows == 0 and run.warnings == ()
    assert run.connectors == ("m", "c1")


def test_diagonal_becomes_an_l_with_a_warning(tmp_path):
    sheet = _load(
        tmp_path,
        {"a": _pt("crossing", 0, 0), "b": _pt("crossing", 100, 40)},
        [("a", "b")],
    )
    (run,) = pipe_runs(sheet)
    assert run.elbows == 1
    assert len(run.warnings) == 1 and "off-axis" in run.warnings[0]
    assert len(run.points) == 3 and _axis_aligned(run.points)
    assert run.points[0] == (0.0, 0.0) and run.points[-1] == (100.0, 40.0)


def test_elbow_leaves_a_symbol_perpendicular_to_its_edge(tmp_path):
    # arrow's bottom edge -> a crossing far to the right and below: must go down first
    nodes = {
        "ar": ("arrow", 0, 0, 12, 12),
        "c": _pt("connector", 6, 12),
        "x": _pt("crossing", 100, 80),
    }
    sheet = _load(tmp_path, nodes, [("ar", "c"), ("c", "x")])
    (run,) = pipe_runs(sheet)
    assert run.points[0] == (6.0, 12.0)
    assert run.points[1][0] == 6.0 and run.points[1][1] > 12.0
    assert _axis_aligned(run.points)


def test_port_slides_along_its_edge_to_meet_a_pipe_one_pixel_off(tmp_path):
    nodes = {
        "v": ("valve", 100, 90, 120, 110),
        "c": _pt("connector", 120, 100),
        "x": _pt("crossing", 300, 101),
    }
    sheet = _load(tmp_path, nodes, [("v", "c"), ("c", "x")])
    (run,) = pipe_runs(sheet)
    assert run.elbows == 0
    assert run.points == ((120.0, 101.0), (300.0, 101.0))
    assert symbol_ports(sheet, "v")[0].x == 120.0  # stays on the edge


def test_interior_connector_port_is_where_the_pipe_crosses_the_bbox(tmp_path):
    # connector at (20, 50) is nearest the L edge, but the pipe runs down to (20, 200):
    # it leaves through the bottom edge at (20, 100), not through L (0, 50)
    nodes = {
        "t": ("tank", 0, 0, 100, 100),
        "c": _pt("connector", 20, 50),
        "x": _pt("crossing", 20, 200),
    }
    sheet = _load(tmp_path, nodes, [("t", "c"), ("c", "x")])
    runs = pipe_runs(sheet)
    (port,) = symbol_ports(sheet, "t", runs)
    assert (port.x, port.y, port.side) == (20.0, 100.0, "B")
    assert not runs[0].internal
    assert runs[0].points == ((20.0, 100.0), (20.0, 200.0))
    assert (runs[0].a_side, runs[0].b_side) == ("B", None)


def test_interior_connector_pipe_leaving_sideways_exits_through_that_side(tmp_path):
    nodes = {
        "t": ("tank", 0, 0, 100, 100),
        "c": _pt("connector", 50, 20),  # nearest edge is T, but the pipe goes right
        "x": _pt("crossing", 200, 20),
    }
    sheet = _load(tmp_path, nodes, [("t", "c"), ("c", "x")])
    (port,) = symbol_ports(sheet, "t")
    assert (port.x, port.y, port.side) == (100.0, 20.0, "R")


def test_pipe_to_a_symbol_nested_in_another_is_internal_and_has_no_ports(tmp_path):
    nodes = {
        "t": ("tank", 0, 0, 200, 200),
        "ar": ("arrow", 90, 90, 102, 102),
        "c": _pt("connector", 96, 102),
        "c2": _pt("connector", 96, 150),
    }
    # arrow's bottom connector, a bend, then the tank itself: every point is inside the tank
    sheet = _load(tmp_path, nodes, [("ar", "c"), ("c", "c2"), ("c2", "t")])
    (run,) = pipe_runs(sheet)
    assert run.internal and run.a_side is None and run.b_side is None
    assert symbol_ports(sheet, "t") == []
    assert symbol_ports(sheet, "ar") == []
    assert _axis_aligned(run.points)


def _nozzle_sheet(tmp_path: Path, flange_pipe: bool = True, flange: bool = True):
    # tank with a zero-length loop (two coincident connectors) 10 px inside its right edge;
    # the flange is a small `general` just on that edge, piped to a crossing on the right
    nodes = {
        "t": ("tank", 0, 0, 100, 200),
        "c1": _pt("connector", 90, 100),
        "c2": _pt("connector", 90, 100),
    }
    edges = [("t", "c1"), ("c1", "c2"), ("c2", "t")]
    if flange:
        nodes["f"] = ("general", 95, 95, 105, 105)
    if flange and flange_pipe:
        nodes["cf"] = _pt("connector", 105, 100)
        nodes["x"] = _pt("crossing", 300, 100)
        edges += [("f", "cf"), ("cf", "x")]
    return _load(tmp_path, nodes, edges)


def test_zero_length_tank_loop_becomes_a_nozzle_link_to_its_flange(tmp_path):
    sheet = _nozzle_sheet(tmp_path)
    runs = pipe_runs(sheet)
    nozzle = [r for r in runs if r.nozzle]
    assert len(nozzle) == 1 and len(runs) == 2
    assert (nozzle[0].a, nozzle[0].b) == ("t", "f")
    assert nozzle[0].points == ((100.0, 100.0), (100.0, 100.0))
    # one tank port, not two phantom ones at the same spot
    (tank_port,) = symbol_ports(sheet, "t", runs)
    assert (tank_port.x, tank_port.y, tank_port.side) == (100.0, 100.0, "R")
    # the flange is a 2-port fitting: tank side L, pipe side R
    cfg = port_config(symbol_ports(sheet, "f", runs))
    assert (cfg.kind, cfg.axis, cfg.sides) == ("2_straight", "H", ("L", "R"))


def test_flange_without_a_pipe_keeps_one_port_to_the_tank(tmp_path):
    sheet = _nozzle_sheet(tmp_path, flange_pipe=False)
    runs = pipe_runs(sheet)
    assert [p.side for p in symbol_ports(sheet, "f", runs)] == ["L"]
    assert [p.side for p in symbol_ports(sheet, "t", runs)] == ["R"]


def test_zero_length_loop_with_no_flange_is_dropped(tmp_path):
    sheet = _nozzle_sheet(tmp_path, flange=False)
    assert pipe_runs(sheet) == []
    assert symbol_ports(sheet, "t") == []


def test_a_flange_serves_only_one_loop(tmp_path):
    nodes = {
        "t": ("tank", 0, 0, 100, 200),
        "c1": _pt("connector", 90, 100),
        "c2": _pt("connector", 90, 100),
        "c3": _pt("connector", 90, 150),
        "c4": _pt("connector", 90, 150),
        "f": ("general", 95, 95, 105, 105),
    }
    edges = [("t", "c1"), ("c1", "c2"), ("c2", "t"), ("t", "c3"), ("c3", "c4"), ("c4", "t")]
    runs = pipe_runs(_load(tmp_path, nodes, edges))
    assert len(runs) == 1 and runs[0].b == "f"


def test_symbol_to_symbol_link_has_edge_ports_facing_each_other(tmp_path):
    nodes = {"g": ("general", 0, 0, 14, 14), "a": ("arrow", 30, 0, 42, 12)}
    sheet = _load(tmp_path, nodes, [("g", "a")])
    (run,) = pipe_runs(sheet)
    assert _port_sides(sheet, "g") == {"R": (14.0, run.points[0][1])}
    assert _port_sides(sheet, "a")["L"][0] == 30.0
    assert _axis_aligned(run.points)


def test_port_config_classes():
    from utils.pid2graph import Port

    def cfg(*sides):
        return port_config([Port(0, 0, s, "n", 0) for s in sides])

    assert cfg().kind == "0"
    assert cfg("L").kind == "1"
    assert (cfg("T", "B").kind, cfg("T", "B").axis) == ("2_straight", "V")
    assert cfg("L", "T").kind == "2_angle"
    assert cfg("T", "T") == PortConfig("2_same", None, ("T", "T"))
    assert cfg("L", "R", "B").kind == "3"
    assert cfg("L", "R", "T", "B").kind == "4"
    assert cfg("L", "L", "R", "T", "B").kind == "multi"


def test_load_graphml_image_size_overrides_inferred_size(tmp_path):
    path = tmp_path / "s.graphml"
    path.write_text(_graphml({"a": ("valve", 0, 0, 50, 40)}, []))
    assert (load_graphml(path).width, load_graphml(path).height) == (50, 40)
    sheet = load_graphml(path, image_size=(60, 45))
    assert (sheet.width, sheet.height) == (60, 45)


@pytest.mark.skipif(not OPEN100_DIR.is_dir(), reason="OPEN100 not present under data/raw/")
def test_all_open100_sheets_give_axis_aligned_runs_and_ports_on_bbox_edges():
    total_runs = 0
    for i in range(12):
        size = Image.open(OPEN100_DIR / f"{i}.png").size
        sheet = load_graphml(OPEN100_DIR / f"{i}.graphml", size)
        runs = pipe_runs(sheet)
        total_runs += len(runs)
        for r in runs:
            assert _axis_aligned(r.points), (i, r.a, r.b)
        for nid, node in sheet.nodes.items():
            if node.label in ("connector", "background", "crossing"):
                continue
            for p in symbol_ports(sheet, nid, runs, ignore_labels=()):
                if runs[p.run].nozzle and runs[p.run].b == nid:
                    continue  # the flange is seated on the tank's port, not where it was drawn
                near_x = min(abs(p.x - node.xmin), abs(p.x - node.xmax))
                near_y = min(abs(p.y - node.ymin), abs(p.y - node.ymax))
                inside = (
                    node.xmin - 2.5 <= p.x <= node.xmax + 2.5
                    and node.ymin - 2.5 <= p.y <= node.ymax + 2.5
                )
                assert inside and min(near_x, near_y) <= 2.5, (i, nid, p)
    assert total_runs > 2000
