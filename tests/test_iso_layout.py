"""Unit tests for utils/iso_layout.py — scale + axis remap that keeps the topology (V4).

A small hand-built sheet pins the mechanics (columns, rigid symbols, pipes, nozzles, seeds); the
tests on the 12 real OPEN100 sheets check what the plan asks for: same graph, axis-aligned pipes,
pipe ends on their symbol's edge, and no new overlapping pairs of symbols. Skipped when
`data/raw/pid2graph/open100` is absent.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from pathlib import Path

import pytest

from utils.iso_layout import AxisMap, build_axis_map, column_anchors, transform_layout
from utils.iso_mapping import load_config
from utils.pid2graph import (
    Node,
    PipeRun,
    Sheet,
    load_graphml,
    pipe_runs,
    port_config,
    symbol_ports,
)

_OPEN100 = Path(__file__).resolve().parent.parent / "data" / "raw" / "pid2graph" / "open100"

CFG = {"layout": {"scale": 3, "column_tol": 3, "gap_scale": [0.85, 1.15]}}
NOT_SYMBOLS = ("connector", "crossing", "background")


def _node(nid: str, label: str, x0: float, y0: float, x1: float, y1: float) -> Node:
    return Node(nid, label, x0, y0, x1, y1)


def _dot(nid: str, label: str, x: float, y: float) -> Node:
    return _node(nid, label, x - 4, y - 4, x + 4, y + 4)


def two_valves() -> Sheet:
    """valve A -> crossing X -> valve B on one horizontal pipe at y = 107.5."""
    nodes = {
        "A": _node("A", "valve", 100, 100, 120, 115),
        "c1": _dot("c1", "connector", 120, 107.5),
        "X": _dot("X", "crossing", 160, 107.5),
        "c2": _dot("c2", "connector", 200, 107.5),
        "B": _node("B", "valve", 200, 100, 220, 115),
        "bg": _node("bg", "background", 0, 0, 400, 300),
    }
    edges = [("A", "c1"), ("c1", "X"), ("X", "c2"), ("c2", "B")]
    return Sheet("two_valves", 400, 300, nodes, edges)


def apply(sheet: Sheet, seed: int = 0, cfg=CFG):
    runs = pipe_runs(sheet)
    return transform_layout(sheet, runs, random.Random(seed), cfg), runs


# --- columns and the axis map -----------------------------------------------------------------


def test_columns_do_not_chain():
    assert column_anchors([0, 2, 4, 6], 3) == [0, 4]  # a chain would make one column of 0..6
    assert column_anchors([5.0, 5.0, 5.5, 20], 3) == [5.0, 20]
    assert column_anchors([], 3) == []


def test_axis_map_is_constant_inside_a_column_and_increasing_across():
    rng = random.Random(0)
    values = [0, 1, 3, 10, 12, 30, 31, 100]
    axis = build_axis_map(values, [], 3.0, 3.0, (0.85, 1.15), rng)
    assert axis(0) == axis(1) == axis(3)
    assert axis(30) == axis(31)
    mapped = [axis(v) for v in sorted({0, 10, 30, 100})]
    assert mapped == sorted(set(mapped))


def test_gaps_are_scaled_by_s_times_a_factor_in_range_and_exactly_s_across_symbols():
    values = [0, 100, 200, 300, 400]
    axis = build_axis_map(values, [(100, 300)], 3.0, 3.0, (0.5, 2.0), random.Random(1))
    gaps = [axis(b) - axis(a) for a, b in itertools.pairwise(values)]
    assert 0.5 * 300 <= gaps[0] <= 2.0 * 300
    assert 0.5 * 300 <= gaps[3] <= 2.0 * 300
    assert gaps[1] == pytest.approx(300) and gaps[2] == pytest.approx(300)  # symbol 100..300
    assert axis(0) == 0


def test_unit_gap_scale_is_a_plain_scaling():
    values = [0, 7, 40, 41, 90]
    axis = build_axis_map(values, [], 3.0, 3.0, (1.0, 1.0), random.Random(0))
    assert [axis(v) for v in (0, 7, 40, 90)] == pytest.approx([0, 21, 120, 270])
    assert axis(41) == axis(40)  # same column


def test_value_below_the_first_column_takes_the_first_position():
    axis = build_axis_map([10.0, 50.0], [], 3.0, 3.0, (1.0, 1.0), random.Random(0))
    assert axis(-5.0) == axis(9.0) == axis(10.0) == 30.0  # not the last position


def test_axis_map_type_and_single_value():
    axis = build_axis_map([5.0], [], 3.0, 3.0, (0.85, 1.15), random.Random(0))
    assert isinstance(axis, AxisMap) and axis(5.0) == 15.0


def test_tied_values_share_a_position_and_the_map_stays_monotone():
    values = [0, 10, 100, 102, 200]  # 100 and 102 fall in two columns (tol 1)
    plain = build_axis_map(values, [], 3.0, 1.0, (1.0, 1.0), random.Random(0))
    tied = build_axis_map(values, [], 3.0, 1.0, (1.0, 1.0), random.Random(0), [(100, 102)])
    assert plain(100) != plain(102)
    assert tied(100) == tied(102)
    mapped = [tied(v) for v in sorted(values)]
    assert mapped == sorted(mapped)
    assert tied(0) == plain(0) and tied(10) == plain(10) and tied(100) == plain(100)


def test_a_tie_that_would_move_a_symbol_edge_is_not_made():
    values = [0, 100, 101, 102, 200]
    axis = build_axis_map(
        values, [(101, 150)], 3.0, 0.5, (1.0, 1.0), random.Random(0), [(100, 102)]
    )
    assert axis(100) != axis(102)  # the symbol's left edge (101) would have moved


# --- transform_layout on a small sheet --------------------------------------------------------


def test_graph_is_untouched():
    sheet = two_valves()
    (new, new_runs), runs = apply(sheet)
    assert list(new.nodes) == list(sheet.nodes)
    assert {k: n.label for k, n in new.nodes.items()} == {
        k: n.label for k, n in sheet.nodes.items()
    }
    assert Counter(map(frozenset, new.edges)) == Counter(map(frozenset, sheet.edges))
    assert [(r.a, r.b, r.connectors) for r in new_runs] == [(r.a, r.b, r.connectors) for r in runs]


def test_symbols_keep_their_size_and_dots_grow_by_s():
    sheet = two_valves()
    (new, _), _ = apply(sheet)
    assert new.nodes["A"].width == pytest.approx(60)
    assert new.nodes["A"].height == pytest.approx(45)
    assert new.nodes["c1"].width == pytest.approx(24)
    assert new.nodes["X"].height == pytest.approx(24)


def test_pipe_stays_horizontal_and_meets_the_valves():
    sheet = two_valves()
    (new, new_runs), _ = apply(sheet)
    for run in new_runs:
        assert len({p[1] for p in run.points}) == 1
    a_to_x = next(r for r in new_runs if r.a == "A")
    assert a_to_x.points[0][0] == pytest.approx(new.nodes["A"].xmax)
    assert a_to_x.points[-1][0] == pytest.approx(new.nodes["X"].center[0])
    b_run = next(r for r in new_runs if "B" in (r.a, r.b))
    assert any(abs(x - new.nodes["B"].xmin) < 1e-6 for x, _ in (b_run.points[0], b_run.points[-1]))


def test_canvas_follows_the_map_and_background_is_not_a_symbol():
    sheet = two_valves()
    (new, _), _ = apply(sheet)
    # the page frame covers everything: if it counted as a symbol no gap could be jittered
    assert new.nodes["bg"].xmin == 0 and new.nodes["bg"].ymin == 0
    assert new.width == round(new.nodes["bg"].xmax)
    assert new.height == round(new.nodes["bg"].ymax)
    widths = {apply(sheet, seed=s)[0][0].width for s in range(8)}
    assert len(widths) > 1  # gaps outside the symbols do vary


def test_free_gap_changes_with_the_seed_and_a_seed_repeats():
    sheet = two_valves()
    a1 = apply(sheet, seed=3)[0][0].nodes["X"].center
    a2 = apply(sheet, seed=3)[0][0].nodes["X"].center
    assert a1 == a2
    others = {apply(sheet, seed=s)[0][0].nodes["X"].center for s in range(6)}
    assert len(others) > 1


def test_collinear_points_collapse_after_the_map():
    # a 2 px jog between two crossings (an L the plan calls "lệch 1-3 px") disappears
    nodes = {
        "P": _dot("P", "crossing", 100, 100),
        "c": _dot("c", "connector", 100, 100),
        "Q": _dot("Q", "crossing", 200, 102),
        "d": _dot("d", "connector", 150, 102),
    }
    sheet = Sheet("jog", 300, 300, nodes, [("P", "c"), ("c", "d"), ("d", "Q")])
    runs = [PipeRun("P", "Q", ((100, 100), (150, 100), (150, 102), (200, 102)), ("c", "d"))]
    _, new_runs = transform_layout(sheet, runs, random.Random(0), CFG)
    assert len(new_runs[0].points) == 2
    assert new_runs[0].points[0][1] == new_runs[0].points[1][1]


def test_nozzle_run_keeps_its_two_points():
    nodes = {"T": _node("T", "tank", 0, 0, 100, 100), "F": _node("F", "general", 105, 40, 115, 50)}
    sheet = Sheet("nozzle", 200, 200, nodes, [])
    nozzle = PipeRun("T", "F", ((100.0, 45.0), (100.0, 45.0)), ("c",), nozzle=True)
    _, runs = transform_layout(sheet, [nozzle], random.Random(0), CFG)
    assert len(runs[0].points) == 2 and runs[0].points[0] == runs[0].points[1]
    assert runs[0].nozzle


# --- real OPEN100 sheets ----------------------------------------------------------------------

real = pytest.mark.skipif(not (_OPEN100 / "0.graphml").exists(), reason="no OPEN100")


def _sheets():
    from PIL import Image

    for i in range(12):
        size = Image.open(_OPEN100 / f"{i}.png").size
        sheet = load_graphml(_OPEN100 / f"{i}.graphml", size)
        yield sheet, pipe_runs(sheet)


def _overlap_depth(a: Node, b: Node) -> float:
    return min(min(a.xmax, b.xmax) - max(a.xmin, b.xmin), min(a.ymax, b.ymax) - max(a.ymin, b.ymin))


def _symbols(sheet: Sheet) -> list[str]:
    return [k for k, n in sheet.nodes.items() if n.label not in NOT_SYMBOLS]


@real
def test_real_straight_fittings_end_up_on_one_line():
    cfg = load_config()
    limit = cfg["layout"]["straight_max"]
    tied = left = 0
    for sheet, runs in _sheets():
        new_sheet, new_runs = transform_layout(sheet, runs, random.Random(0), cfg)
        for nid in _symbols(sheet):
            before = port_config(symbol_ports(sheet, nid, runs))
            if before.kind != "2_straight":
                continue
            a, b = symbol_ports(sheet, nid, runs)
            c, d = symbol_ports(new_sheet, nid, new_runs)
            k = "y" if before.axis == "H" else "x"
            if 0 < abs(getattr(a, k) - getattr(b, k)) <= limit:
                tied += 1
                left += abs(getattr(c, k) - getattr(d, k)) > 0.5
    assert tied > 100
    assert left <= 0.1 * tied  # the ties skipped for a symbol edge in the way are a few per cent


@real
def test_real_graph_is_identical_and_pipes_axis_aligned():
    cfg = load_config()
    for sheet, runs in _sheets():
        new, new_runs = transform_layout(sheet, runs, random.Random(7), cfg)
        assert {k: n.label for k, n in new.nodes.items()} == {
            k: n.label for k, n in sheet.nodes.items()
        }
        assert Counter(map(frozenset, new.edges)) == Counter(map(frozenset, sheet.edges))
        assert len(new_runs) == len(runs)
        for run in new_runs:
            for p, q in itertools.pairwise(run.points):
                assert abs(p[0] - q[0]) < 1e-9 or abs(p[1] - q[1]) < 1e-9, (sheet.name, run.a)
        assert new.width > sheet.width and new.height > sheet.height


@real
def test_real_pipe_ends_stay_on_their_symbols_edges_and_ports_keep_their_sides():
    cfg = load_config()
    for sheet, runs in _sheets():
        new, new_runs = transform_layout(sheet, runs, random.Random(11), cfg)
        for nid in _symbols(sheet):
            before = symbol_ports(sheet, nid, runs)
            after = symbol_ports(new, nid, new_runs)
            assert [p.side for p in before] == [p.side for p in after]
            assert port_config(before) == port_config(after)
            node = new.nodes[nid]
            for p in after:
                if p.run >= 0 and new_runs[p.run].nozzle:
                    continue  # a flange sits at the tank's port, not on its own bbox
                edge = {"L": node.xmin, "R": node.xmax, "T": node.ymin, "B": node.ymax}[p.side]
                on_axis = p.x if p.side in "LR" else p.y
                assert abs(on_axis - edge) < 1e-6, (sheet.name, nid, p.side)


@real
def test_real_symbol_sizes_follow_s_within_the_column_snap():
    cfg = load_config()
    s, tol = cfg["layout"]["scale"], cfg["layout"]["column_tol"]
    for sheet, runs in _sheets():
        new, _ = transform_layout(sheet, runs, random.Random(5), cfg)
        for nid in _symbols(sheet):
            old, now = sheet.nodes[nid], new.nodes[nid]
            assert abs(now.width - s * old.width) <= 2 * tol * s + 1e-6, (sheet.name, nid)
            assert abs(now.height - s * old.height) <= 2 * tol * s + 1e-6, (sheet.name, nid)


@real
def test_real_overlapping_pairs_are_never_added():
    """No new overlapping pair of symbols; an old one may go only if it was <= column_tol deep."""
    cfg = load_config()
    tol = cfg["layout"]["column_tol"]
    lost = kept = 0
    for seed in (1, 2):
        for sheet, runs in _sheets():
            new, _ = transform_layout(sheet, runs, random.Random(seed), cfg)
            for a, b in itertools.combinations(_symbols(sheet), 2):
                depth_old = _overlap_depth(sheet.nodes[a], sheet.nodes[b])
                depth_new = _overlap_depth(new.nodes[a], new.nodes[b])
                if depth_old <= 0:
                    assert depth_new <= 0, (sheet.name, a, b)
                elif depth_new <= 0:
                    assert depth_old <= tol, (sheet.name, a, b, depth_old)
                    lost += 1
                else:
                    kept += 1
    assert kept > 0  # the sheets do contain overlaps (actuators on valves, bubbles)
    assert lost > 0  # and shallow ones do vanish, so the `depth_old <= tol` check ran
