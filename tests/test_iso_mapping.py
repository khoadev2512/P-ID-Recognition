"""Unit tests for utils/iso_mapping.py — PID2Graph node -> ISO symbol rules (V3).

Synthetic sheets and templates pin the rules: the port-count principles, the special `multi` and
`2` (pump) templates, size classes, the rule list of configs/render_iso.yaml, nested symbols and
seeded reproducibility. The tests on the real OPEN100 sheets and templates are skipped when
`data/raw/` is absent.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from utils.iso_mapping import (
    Assignment,
    assign_symbols,
    load_config,
    orientations_for,
    size_class,
)
from utils.iso_templates import Template, TemplatePort, load_templates
from utils.pid2graph import Node, PortConfig, Sheet, load_graphml, pipe_runs

_ROOT = Path(__file__).resolve().parent.parent
_OPEN100 = _ROOT / "data" / "raw" / "pid2graph" / "open100"
_SVG_DIR = _ROOT / "data" / "raw" / "template_iso_svg"

_MID = {"L": (0.0, 0.5), "R": (1.0, 0.5), "T": (0.5, 0.0), "B": (0.5, 1.0)}


def make_template(
    file: str,
    group: str,
    port_config: str,
    sides: str,
    placement: str = "pipe",
    in_vocab: bool = True,
    fine_name: str | None = None,
) -> Template:
    ports = tuple(
        TemplatePort(i, s, _MID[s][0] * 10, _MID[s][1] * 10, _MID[s][0], _MID[s][1], "stub")
        for i, s in enumerate(sides)
    )
    return Template(
        file=file,
        reg_number=file,
        variant=1,
        group=group,
        fine_name=fine_name or f"name_{file}",
        in_vocab=in_vocab,
        scope="html",
        port_config=port_config,
        directional=False,
        placement=placement,
        w_mm=10.0,
        h_mm=10.0,
        ports=ports,
        svg_path=Path("unused.svg"),
        matrix=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    )


TEMPLATES = {
    t.file: t
    for t in (
        make_template("V2S", "G1", "2_straight", "LR"),
        make_template("VANG", "G2", "2_angle", "BL", in_vocab=False),
        make_template("V3W", "G2", "3", "BLR"),
        make_template("FIT", "G3", "2_straight", "LR"),
        make_template("PUMP", "G4", "2", "LR"),
        make_template(
            "VESSEL", "G4", "multi", "BL", in_vocab=False, fine_name="vessel_dished_ends"
        ),
        make_template("COLUMN", "G4", "multi", "BL", fine_name="column"),
        make_template("AG_A", "G4", "attach", "T", placement="attach", fine_name="agitator_a"),
        make_template("AG_B", "G4", "attach", "T", placement="attach", fine_name="agitator_b"),
        make_template("HX", "G4", "4", "TTBB"),
        make_template("VENT2", "G5", "2_straight", "LR", placement="pipe_or_end"),
        make_template("FUNNEL", "G5", "1", "B", placement="end"),
    )
}


class _Builder:
    """A sheet of symbol boxes, each with pipes leaving from the given sides: bbox edge middle
    -> connector -> a far crossing (the pipe's other end)."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[tuple[str, str]] = []
        self.n = 0

    def add(
        self,
        label: str,
        box: tuple[float, float, float, float],
        sides: str = "",
        far_label: str = "crossing",
    ) -> str:
        nid = f"n{self.n}"
        self.n += 1
        x0, y0, x1, y1 = box
        self.nodes[nid] = Node(nid, label, x0, y0, x1, y1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for i, side in enumerate(sides):
            slide = (sides[:i].count(side) - (sides.count(side) - 1) / 2) * 10  # parallel pipes
            edge = {
                "L": (x0, cy + slide),
                "R": (x1, cy + slide),
                "T": (cx + slide, y0),
                "B": (cx + slide, y1),
            }[side]
            far = {"L": (-40, 0), "R": (40, 0), "T": (0, -40), "B": (0, 40)}[side]
            conn, end = f"{nid}{i}c", f"{nid}{i}x"
            self.nodes[conn] = Node(
                conn, "connector", edge[0] - 4, edge[1] - 4, edge[0] + 4, edge[1] + 4
            )
            ex, ey = edge[0] + far[0], edge[1] + far[1]
            self.nodes[end] = Node(end, far_label, ex - 4, ey - 4, ex + 4, ey + 4)
            self.edges += [(nid, conn), (conn, end)]
        return nid

    def sheet(self) -> Sheet:
        return Sheet("synthetic", 1000, 1000, self.nodes, self.edges)


CFG = {
    "p_replace": 0.3,
    "size_class": {"small_max": 25, "large_min": 120},
    "rules": load_config()["rules"],
}


def run(builder: _Builder, seed: int = 0, **override) -> dict[str, Assignment]:
    sheet = builder.sheet()
    return assign_symbols(
        sheet, pipe_runs(sheet), TEMPLATES, random.Random(seed), {**CFG, **override}
    )


# --- size_class -----------------------------------------------------------------------------


def test_size_class_thresholds():
    node = lambda w, h: Node("n", "general", 0, 0, w, h)  # noqa: E731
    cfg3 = PortConfig("3")
    assert size_class(node(25, 10), cfg3, 25, 120) == "small"
    assert size_class(node(26, 10), cfg3, 25, 120) == "medium"
    assert size_class(node(119, 10), cfg3, 25, 120) == "medium"
    assert size_class(node(120, 10), cfg3, 25, 120) == "large"


def test_size_class_measures_a_straight_fitting_across_the_pipe():
    thin_long = Node("n", "general", 0, 0, 75, 14)
    assert size_class(thin_long, PortConfig("2_straight", "H"), 25, 120) == "small"
    # the same box with the pipe running vertically is measured by its width
    assert size_class(thin_long, PortConfig("2_straight", "V"), 25, 120) == "medium"
    assert size_class(thin_long, PortConfig("2_angle"), 25, 120) == "medium"


# --- port rules ---------------------------------------------------------------------------------


def _sides(ts: list[Template]) -> set[tuple[str, ...]]:
    return {t.sides for t in ts}


def test_exact_sides_after_turning_or_mirroring():
    straight = TEMPLATES["V2S"]
    assert _sides(orientations_for(straight, ["L", "R"])) == {("L", "R")}
    assert len(orientations_for(straight, ["L", "R"])) == 4  # rot 0/180, with and without flip
    assert _sides(orientations_for(straight, ["B", "T"])) == {("B", "T")}
    assert orientations_for(straight, ["L", "T"]) == []
    angle = TEMPLATES["VANG"]
    assert _sides(orientations_for(angle, ["T", "R"])) == {("R", "T")}
    assert orientations_for(angle, ["L", "R"]) == []


def test_node_with_more_ports_than_the_template_never_fits():
    assert orientations_for(TEMPLATES["V2S"], ["L", "R", "B"]) == []
    assert orientations_for(TEMPLATES["VENT2"], ["L", "R", "B"]) == []
    assert orientations_for(TEMPLATES["FUNNEL"], ["B", "T"]) == []


def test_node_with_fewer_ports_fits_only_when_a_port_may_stay_open():
    assert orientations_for(TEMPLATES["V2S"], ["L"]) == []  # placement "pipe"
    open_ones = orientations_for(TEMPLATES["VENT2"], ["L"])
    assert open_ones and all("L" in t.sides for t in open_ones)
    assert orientations_for(TEMPLATES["VENT2"], []) == []  # nothing to hang it on
    assert _sides(orientations_for(TEMPLATES["FUNNEL"], ["R"])) == {("R",)}


def test_multi_takes_any_node_with_a_port_in_a_fixed_pose():
    (only,) = orientations_for(TEMPLATES["VESSEL"], ["L", "T", "R", "B", "B"])
    assert (only.rot, only.flip) == (0, False)
    assert orientations_for(TEMPLATES["VESSEL"], []) == []


def test_pump_takes_two_port_nodes_by_best_side_agreement():
    pump = TEMPLATES["PUMP"]
    assert _sides(orientations_for(pump, ["L", "R"])) == {("L", "R")}
    # one top, one left: no pose has both, the best have one
    best = orientations_for(pump, ["T", "L"])
    assert best and all(t.sides.count("T") + t.sides.count("L") == 1 for t in best)
    assert orientations_for(pump, ["L"]) == []
    assert orientations_for(pump, ["L", "R", "B"]) == []


# --- assign_symbols -----------------------------------------------------------------------------


def test_valves_by_port_layout():
    b = _Builder()
    straight = b.add("valve", (100, 100, 120, 115), "LR")
    angle = b.add("valve", (300, 100, 320, 120), "BL")
    three = b.add("valve", (500, 100, 520, 120), "BLR")
    dead_end = b.add("valve", (700, 100, 720, 115), "L")
    out = run(b)
    assert out[straight].template.group == "G1" and out[straight].template.sides == ("L", "R")
    assert out[angle].template.group == "G2" and out[angle].template.sides == ("B", "L")
    assert out[angle].label == "other_iso"  # VANG is outside the vocabulary
    assert out[three].template.file == "V3W"
    assert out[dead_end] == Assignment("generic", "other_generic")


def test_labels_follow_the_vocabulary():
    b = _Builder()
    straight = b.add("valve", (100, 100, 120, 115), "LR")
    out = run(b)
    assert (out[straight].kind, out[straight].label) == ("iso", "name_V2S")


def test_non_symbol_nodes_get_no_assignment():
    b = _Builder()
    v = b.add("valve", (100, 100, 120, 115), "LR")
    out = run(b)
    assert list(out) == [v]


def test_general_fittings_by_size_and_ports():
    b = _Builder()
    small_straight = b.add("general", (100, 100, 114, 112), "LR")  # 12 px across the pipe
    medium = b.add("general", (300, 100, 340, 140), "LR")
    small_end = b.add("general", (500, 100, 514, 112), "L")
    large_any = b.add("general", (700, 100, 900, 300), "L")
    out = run(b)
    assert out[small_straight].template.group == "G3"
    assert out[medium].kind == "generic"
    assert out[small_end].template.group == "G5"  # no flange pair among these templates
    assert out[large_any].template.group == "G4" and out[large_any].template.port_config == "multi"


def test_a_small_fitting_at_a_pipe_end_is_a_flange_pair_not_a_g5_device():
    b = _Builder()
    end = b.add("general", (500, 100, 514, 112), "L")
    flange = make_template(
        "FLANGE", "G3", "2_straight", "LR", placement="pipe_or_end", fine_name="flanged_connection"
    )
    sheet = b.sheet()
    templates = {**TEMPLATES, "FLANGE": flange}
    out = assign_symbols(sheet, pipe_runs(sheet), templates, random.Random(0), CFG)
    assert out[end].template.file == "FLANGE" and out[end].label == "flanged_connection"


def test_pump_and_tank_use_their_own_templates():
    b = _Builder()
    pump = b.add("pump", (100, 100, 130, 130), "TT")
    tank = b.add("tank", (300, 100, 400, 300), "LR")
    empty_tank = b.add("tank", (600, 100, 700, 300))
    out = run(b)
    assert out[pump].template.file == "PUMP"
    assert out[tank].template.file == "VESSEL" and out[tank].label == "other_iso"
    assert out[empty_tank].template.file == "VESSEL"  # a tank may have no pipe, only tag lines


def test_a_tank_is_never_a_heat_exchanger_but_a_large_general_may_be():
    b = _Builder()
    for i in range(40):
        b.add("tank", (10 + i * 400, 100, 110 + i * 400, 300), "TTBB")
    tank_out = run(b)
    assert {a.template.file for a in tank_out.values()} == {"VESSEL"}
    g = _Builder()
    for i in range(40):
        g.add("general", (10 + i * 400, 100, 210 + i * 400, 300), "TTBB")
    assert {a.template.file for a in run(g).values()} == {"VESSEL", "HX"}


def test_classes_are_drawn_before_templates():
    """Two drawings of one class are as likely together as a class with a single drawing."""
    twin = make_template("V2S_B", "G1", "2_straight", "LR")
    twin = Template(**{**twin.__dict__, "fine_name": TEMPLATES["V2S"].fine_name})
    other = make_template("OTHER", "G1", "2_straight", "LR")
    pool = {"V2S": TEMPLATES["V2S"], "V2S_B": twin, "OTHER": other}
    b = _Builder()
    for i in range(600):
        b.add("valve", (10 + i * 100, 100, 30 + i * 100, 115), "LR")
    sheet = b.sheet()
    out = assign_symbols(sheet, pipe_runs(sheet), pool, random.Random(0), CFG)
    share = Counter(a.label for a in out.values())
    assert share["name_V2S"] == pytest.approx(300, abs=45)
    assert share["name_OTHER"] == pytest.approx(300, abs=45)


def test_arrows_and_flags_carry_an_axis_and_no_template():
    b = _Builder()
    arrow = b.add("arrow", (100, 100, 112, 112), "LR")
    arrow_v = b.add("arrow", (300, 100, 312, 112), "TB")
    flag = b.add("inlet/outlet", (500, 100, 520, 112), "L")
    lone = b.add("arrow", (700, 100, 730, 112))  # no port: follow the box
    out = run(b)
    assert (out[arrow].kind, out[arrow].label, out[arrow].axis) == ("arrow", "flow_direction", "H")
    assert out[arrow_v].axis == "V"
    assert (out[flag].kind, out[flag].label, out[flag].axis) == ("flag", None, "H")
    assert out[lone].axis == "H"
    assert all(a.template is None and a.forward in (True, False) for a in out.values())


def test_arrows_carry_the_arrow_template_pointing_their_way():
    arrow = make_template("ARW", "G0", "2_straight", "LR", placement="overlay")
    arrow = Template(**{**arrow.__dict__, "fine_name": "flow_direction"})
    pool = {**TEMPLATES, "ARW": arrow}
    b = _Builder()
    for i in range(40):
        b.add("arrow", (10 + i * 100, 100, 22 + i * 100, 112), "LR")
        b.add("arrow", (10 + i * 100, 500, 22 + i * 100, 512), "TB")
    sheet = b.sheet()
    out = assign_symbols(sheet, pipe_runs(sheet), pool, random.Random(0), CFG)
    tip = {("H", True): "R", ("H", False): "L", ("V", True): "B", ("V", False): "T"}
    assert {a.label for a in out.values()} == {"flow_direction"}
    for a in out.values():
        assert a.template is not None and a.template.fine_name == "flow_direction"
        tip_port = next(q for q in a.template.ports if q.port_id == 1)  # the base R port
        assert tip_port.side == tip[(a.axis, a.forward)]
    # the overlay is never drawn as a fitting of its own
    g = _Builder()
    g.add("general", (10, 100, 24, 112), "LR")
    sheet = g.sheet()
    fit = assign_symbols(sheet, pipe_runs(sheet), pool, random.Random(0), CFG)
    assert all(a.template is None or a.template.file != "ARW" for a in fit.values())


def _arrow_against(short_side: str) -> Sheet:
    """An arrow on a horizontal pipe with a bend 3 px off `short_side` (L or R), 60 px of
    pipe on the other side."""
    near, far = (112 + 3, 112 - 72) if short_side == "R" else (100 - 3, 100 + 72)
    nodes = {
        "A": Node("A", "arrow", 100, 94, 112, 106),
        "cL": Node("cL", "connector", 96, 96, 104, 104),
        "cR": Node("cR", "connector", 108, 96, 116, 104),
    }
    xl, xr = (far, near) if short_side == "R" else (near, far)
    nodes["XL"] = Node("XL", "crossing", xl - 2, 98, xl + 2, 102)
    nodes["XR"] = Node("XR", "crossing", xr - 2, 98, xr + 2, 102)
    edges = [("XL", "cL"), ("cL", "A"), ("A", "cR"), ("cR", "XR")]
    return Sheet("arrow", 1000, 1000, nodes, edges)


@pytest.mark.parametrize("short_side, forward", [("R", True), ("L", False)])
def test_an_arrow_against_a_bend_points_into_it(short_side, forward):
    sheet = _arrow_against(short_side)
    for seed in range(8):  # not a coin toss: every seed agrees
        a = assign_symbols(sheet, pipe_runs(sheet), TEMPLATES, random.Random(seed), CFG)["A"]
        assert (a.axis, a.forward) == ("H", forward)


def test_a_one_port_arrow_on_a_pipe_is_a_leader_and_on_a_symbol_a_flow_arrow():
    b = _Builder()
    leader = b.add("arrow", (100, 100, 112, 112), "B")  # head on a pipe (a crossing)
    flow = b.add("arrow", (300, 100, 312, 112), "R", far_label="valve")
    out = run(b)
    assert out[leader].kind == "leader" and out[leader].label == "leader_arrow"
    assert out[flow].kind == "arrow" and out[flow].label == "flow_direction"
    # 40 px of pipe to the valve: an arrow at the end of a pipe points away from it
    assert (out[flow].axis, out[flow].forward) == ("H", False)


def test_instrumentation_bubble_or_group5_device():
    b = _Builder()
    one_port = b.add("instrumentation", (100, 100, 130, 130), "L")
    two_ports = b.add("instrumentation", (300, 100, 330, 130), "LR")
    none = b.add("instrumentation", (500, 100, 530, 130))
    always = run(b, p_replace=1.0)
    assert always[one_port].kind == "iso" and always[one_port].template.group == "G5"
    assert always[two_ports] == Assignment("bubble", "other_isa")
    assert always[none] == Assignment("bubble", "other_isa")
    never = run(b, p_replace=0.0)
    assert never[one_port] == Assignment("bubble", "other_isa")


def test_pump_piped_from_one_side_gets_its_ports_across():
    pump = TEMPLATES["PUMP"]
    assert _sides(orientations_for(pump, ["T", "T"])) == {("L", "R")}
    assert _sides(orientations_for(pump, ["B", "B"])) == {("L", "R")}
    assert _sides(orientations_for(pump, ["L", "L"])) == {("B", "T")}
    assert _sides(orientations_for(pump, ["R", "R"])) == {("B", "T")}


def test_a_bubble_on_a_valve_is_never_a_group5_device():
    b = _Builder()
    on_valve = b.add("instrumentation", (100, 100, 130, 130), "L", far_label="valve")
    on_pipe = b.add("instrumentation", (300, 100, 330, 130), "L")
    out = run(b, p_replace=1.0)
    assert out[on_valve] == Assignment("bubble", "other_isa")
    assert out[on_pipe].kind == "iso" and out[on_pipe].template.group == "G5"


def test_a_tag_on_a_general_node_or_a_valve_is_never_a_group5_device():
    b = _Builder()
    on_general = b.add("instrumentation", (100, 100, 130, 130), "L", far_label="general")
    on_pipe = b.add("instrumentation", (300, 100, 330, 130), "L")
    out = run(b, p_replace=1.0)
    assert out[on_general] == Assignment("bubble", "other_isa")
    assert out[on_pipe].kind == "iso" and out[on_pipe].template.group == "G5"


def test_next_to_and_classes_narrow_a_rule():
    """A fitting next to a tank (a vessel nozzle) draws from its own class list."""
    pool = {**TEMPLATES, "DISC": make_template("DISC", "G3", "2_straight", "LR")}
    rules = [
        {"label": "general", "next_to": ["tank"], "groups": ["G3"], "classes": ["name_DISC"]},
        {"label": "general", "groups": ["G3"], "classes": ["name_FIT"]},
    ]
    b = _Builder()
    nozzles = [
        b.add("general", (10 + i * 100, 100, 24 + i * 100, 112), "LR", far_label="tank")
        for i in range(20)
    ]
    lines = [b.add("general", (10 + i * 100, 500, 24 + i * 100, 512), "LR") for i in range(20)]
    sheet = b.sheet()
    out = assign_symbols(sheet, pipe_runs(sheet), pool, random.Random(0), {**CFG, "rules": rules})
    assert {out[n].label for n in nozzles} == {"name_DISC"}
    assert {out[n].label for n in lines} == {"name_FIT"}


def test_tagged_filters_on_a_bubble_linked_to_the_node():
    """`tagged`: a run leads to an `instrumentation` node (a tag), whichever side it is on."""
    pool = {**TEMPLATES, "FLG": make_template("FLG", "G3", "2_straight", "LR")}
    rules = [
        {"label": "general", "tagged": True, "groups": ["G3"], "classes": ["name_FLG"]},
        {"label": "general", "groups": ["G3"], "classes": ["name_FIT"]},
    ]
    b = _Builder()
    # the pipe sides are L and R; a third pipe, to the tag, leaves from the top
    tagged, bare = [], []
    for i in range(20):
        t = b.add("general", (10 + i * 100, 100, 24 + i * 100, 112), "LRT", far_label="crossing")
        b.nodes[f"{t}2x"] = Node(
            f"{t}2x",
            "instrumentation",
            *(b.nodes[f"{t}2x"].center[k] + d for k, d in ((0, -12), (1, -12), (0, 12), (1, 12))),
        )
        tagged.append(t)
        bare.append(b.add("general", (10 + i * 100, 500, 24 + i * 100, 512), "LR"))
    sheet = b.sheet()
    out = assign_symbols(sheet, pipe_runs(sheet), pool, random.Random(0), {**CFG, "rules": rules})
    assert {out[n].label for n in tagged} == {"name_FLG"}
    assert {out[n].label for n in bare} == {"name_FIT"}
    # `tagged: false` is the other half
    rules[0]["tagged"] = False
    out = assign_symbols(sheet, pipe_runs(sheet), pool, random.Random(0), {**CFG, "rules": rules})
    assert {out[n].label for n in bare} == {"name_FLG"}


def test_flanges_are_rolled_only_for_their_classes():
    flange = make_template("FLG", "GX", "2_straight", "LR")  # outside every rule's groups
    disc = make_template("DISC", "G3", "2_straight", "LR")
    pool = {"FIT": TEMPLATES["FIT"], "DISC": disc, "FLG": flange}
    b = _Builder()
    for i in range(200):
        b.add("general", (10 + i * 100, 100, 24 + i * 100, 112), "LR")
        b.add("general", (10 + i * 100, 500, 22 + i * 100, 514), "TB")
    sheet = b.sheet()
    cfg = {
        **CFG,
        "rules": [{"label": "general", "groups": ["G3"]}],
        "flanges": {"p": 0.5, "classes": ["name_DISC"], "symbol": "name_FLG"},
    }
    out = assign_symbols(sheet, pipe_runs(sheet), pool, random.Random(0), cfg)
    discs = [a for a in out.values() if a.label == "name_DISC"]
    assert not any(a.flanges for a in out.values() if a.label == "name_FIT")
    with_flanges = [a for a in discs if a.flanges is not None]
    assert len(with_flanges) / len(discs) == pytest.approx(0.5, abs=0.1)
    for a in with_flanges:  # turned to the disc's pipe axis
        assert a.flanges.fine_name == "name_FLG"
        assert set(a.flanges.sides) == set(a.template.sides)
    # no `flanges` block, or no flange template in scope: never
    bare = assign_symbols(sheet, pipe_runs(sheet), pool, random.Random(0), CFG)
    assert all(a.flanges is None for a in bare.values())
    del pool["FLG"]
    missing = assign_symbols(sheet, pipe_runs(sheet), pool, random.Random(0), cfg)
    assert all(a.flanges is None for a in missing.values())


def test_replace_rate_follows_p_replace():
    b = _Builder()
    for i in range(400):
        b.add("instrumentation", (10 + i * 100, 100, 40 + i * 100, 130), "L")
    out = run(b, p_replace=0.3)
    share = sum(a.kind == "iso" for a in out.values()) / len(out)
    assert share == pytest.approx(0.3, abs=0.06)


def test_nested_symbol_is_dropped():
    b = _Builder()
    tank = b.add("tank", (100, 100, 500, 500))
    inner = b.add("arrow", (250, 250, 262, 262))
    b.nodes["link"] = Node("link", "connector", 300, 300, 308, 308)
    b.edges += [(inner, "link"), ("link", tank)]
    out = run(b)
    assert out[inner] == Assignment("nested", None)
    assert out[tank].kind != "nested"


def test_arrow_in_a_box_with_an_outside_pipe_is_not_nested():
    b = _Builder()
    b.add("tank", (100, 100, 500, 500))
    arrow = b.add("arrow", (250, 250, 262, 262), "LR")  # its pipes leave the tank
    assert run(b)[arrow].kind == "arrow"


def test_same_seed_same_result_and_other_seeds_differ():
    b = _Builder()
    for i in range(30):
        b.add("valve", (10 + i * 100, 100, 30 + i * 100, 115), "LR")
    first = run(b, seed=1)
    again = run(b, seed=1)
    assert [(a.template.file, a.template.rot, a.template.flip) for a in first.values()] == [
        (a.template.file, a.template.rot, a.template.flip) for a in again.values()
    ]
    other = run(b, seed=2)
    assert [a.template.oriented(0).file for a in first.values()] != [
        a.template.oriented(0).file for a in other.values()
    ] or [(a.template.rot, a.template.flip) for a in first.values()] != [
        (a.template.rot, a.template.flip) for a in other.values()
    ]


def test_bad_rules_are_rejected():
    b = _Builder()
    b.add("valve", (100, 100, 120, 115), "LR")
    with pytest.raises(ValueError, match="kind"):
        run(b, rules=[{"label": "valve", "kind": "circle"}])
    with pytest.raises(ValueError, match="groups"):
        run(b, rules=[{"label": "valve"}])


def test_shipped_config_is_consistent():
    cfg = load_config()
    assert cfg["scope"] == "html"
    assert 0 <= cfg["p_replace"] <= 1
    assert cfg["size_class"]["small_max"] < cfg["size_class"]["large_min"]
    assert 0 <= cfg["flanges"]["p"] <= 1
    import csv

    with open(_ROOT / cfg["taxonomy_csv"], newline="") as fh:
        fine_names = {row["fine_name"] for row in csv.DictReader(fh)}
    named = {*cfg["flanges"]["classes"], cfg["flanges"]["symbol"]}
    named |= {c for r in cfg["rules"] for c in r.get("classes", [])}
    assert named <= fine_names, named - fine_names
    groups = {g for r in cfg["rules"] for g in r.get("groups", [])}
    assert groups <= {"G1", "G2", "G3", "G4", "G5"}
    # every PID2Graph label that carries a symbol has a rule
    labels = {r["label"] for r in cfg["rules"]}
    assert {
        "arrow",
        "inlet/outlet",
        "valve",
        "pump",
        "tank",
        "general",
        "instrumentation",
    } <= labels


# --- real OPEN100 sheets + real templates -------------------------------------------------------

real = pytest.mark.skipif(
    not (_OPEN100 / "0.graphml").exists() or not (_SVG_DIR / "templates.csv").exists(),
    reason="no OPEN100 / template_iso_svg",
)


def _real_runs(seed: int = 0):
    from PIL import Image

    cfg = load_config()
    templates = load_templates(_SVG_DIR, _ROOT / cfg["taxonomy_csv"], cfg["scope"])
    rng = random.Random(seed)
    for i in range(12):
        size = Image.open(_OPEN100 / f"{i}.png").size
        sheet = load_graphml(_OPEN100 / f"{i}.graphml", size)
        runs = pipe_runs(sheet)
        yield sheet, runs, assign_symbols(sheet, runs, templates, rng, cfg)


@real
def test_real_sheets_every_symbol_gets_a_valid_assignment():
    from utils.pid2graph import symbol_ports

    for sheet, runs, out in _real_runs():
        symbols = {
            n
            for n, node in sheet.nodes.items()
            if node.label not in ("connector", "crossing", "background")
        }
        assert set(out) == symbols
        for nid, a in out.items():
            if a.kind != "iso":
                continue
            sides = Counter(p.side for p in symbol_ports(sheet, nid, runs))
            t = a.template
            assert t.port_config in ("multi", "2") or not (sides - Counter(t.sides)), (nid, t.file)
            assert a.label == (t.fine_name if t.in_vocab else "other_iso")


@real
def test_real_sheets_counts_match_the_plan():
    from utils.pid2graph import port_config, symbol_ports

    kinds: Counter[tuple[str, str]] = Counter()
    valves_without_symbol = 0
    replaceable = 0  # a one-port tag on neither a valve nor a `general` node: `replace` may hit it
    for sheet, runs, out in _real_runs():
        for nid, a in out.items():
            label = sheet.nodes[nid].label
            kinds[(label, a.kind)] += 1
            valves_without_symbol += label == "valve" and a.kind != "iso"
            if label == "instrumentation":
                ports = symbol_ports(sheet, nid, runs)
                near = {sheet.nodes[p.neighbor].label for p in ports}
                replaceable += port_config(ports).kind == "1" and not near & {"valve", "general"}
    assert valves_without_symbol == 18  # 1 valve with no port + 17 with one
    # 451 arrows: 72 leaders (one port, head on a pipe), the rest flow arrows (or nested)
    assert kinds[("arrow", "leader")] == 72
    assert kinds[("arrow", "arrow")] + kinds[("arrow", "nested")] == 451 - 72
    assert kinds[("general", "iso")] >= 190  # 138 + 57 small, plus large ones
    assert replaceable > 100
    assert kinds[("instrumentation", "iso")] == pytest.approx(0.3 * replaceable, abs=20)


@real
def test_real_valve_histogram_is_flat_over_classes():
    """Classes are drawn uniformly: over many seeds no G1 class (fine_name) is more than 2x
    another's count. X8075's two drawings share `valve_butterfly`, so it gets no extra share."""
    counts: Counter[str] = Counter()
    for seed in range(15):
        for sheet, _, out in _real_runs(seed):
            for nid, a in out.items():
                if (
                    a.kind == "iso"
                    and sheet.nodes[nid].label == "valve"
                    and a.template.group == "G1"
                ):
                    counts[a.template.fine_name] += 1
    assert len(counts) == 12  # 13 templates, X8075 twice
    assert max(counts.values()) <= 2 * min(counts.values())


@real
def test_real_vessel_nozzles_get_no_inline_only_fittings():
    """A fitting next to a tank stands at a vessel nozzle: never an orifice plate (it needs
    straight pipe both sides), a strainer or a sight glass."""
    from utils.pid2graph import symbol_ports

    line_only = {"orifice_plate", "strainer", "strainer_cone", "viewing_glass"}
    nozzles = 0
    for seed in range(3):
        for sheet, runs, out in _real_runs(seed):
            for nid, a in out.items():
                if a.kind != "iso" or a.template.group != "G3":
                    continue
                ports = symbol_ports(sheet, nid, runs)
                if any(sheet.nodes[p.neighbor].label == "tank" for p in ports):
                    nozzles += 1
                    assert a.template.fine_name not in line_only, (sheet.name, nid)
    assert nozzles > 50


@real
def test_real_tagged_fittings_are_all_flange_pairs():
    """OPEN100: a small two-port fitting with a tag (FO / RO) is a flange pair, never a disc, a
    reducer or another fitting (11 on the 12 sheets)."""
    from utils.pid2graph import port_config, symbol_ports

    cfg = load_config()
    seen = 0
    for seed in range(3):
        for sheet, runs, out in _real_runs(seed):
            for nid, node in sheet.nodes.items():
                if node.label != "general" or nid not in out:
                    continue
                tagged = any(
                    sheet.nodes[r.b if r.a == nid else r.a].label == "instrumentation"
                    for r in runs
                    if nid in (r.a, r.b) and r.a != r.b
                )
                pcfg = port_config(symbol_ports(sheet, nid, runs)).kind
                small = max(node.width, node.height) <= cfg["size_class"]["small_max"]
                if tagged and small and pcfg == "2_straight":
                    seen += 1
                    assert out[nid].label == "flanged_connection", (sheet.name, nid)
                    assert out[nid].flanges is None
    assert seen == 3 * 11


@real
def test_real_fittings_at_a_pipe_end_are_flange_pairs():
    """OPEN100: a small one-port `general` node is a flange pair at a vessel nozzle or a pipe end
    (53 of the 55 on the 12 sheets), never a flame arrestor or another G5 device. G5 devices
    still come from replaced bubbles."""
    from utils.pid2graph import port_config, symbol_ports

    seen, g5 = 0, 0
    for seed in range(3):
        for sheet, runs, out in _real_runs(seed):
            for nid, a in out.items():
                g5 += a.kind == "iso" and a.template.group == "G5"
                node = sheet.nodes[nid]
                if node.label != "general" or a.kind == "nested":
                    continue
                pcfg = port_config(symbol_ports(sheet, nid, runs))
                if pcfg.kind == "1" and size_class(node, pcfg, 25, 120) == "small":
                    seen += 1
                    assert a.label == "flanged_connection", (sheet.name, nid)
    assert seen == 3 * 55
    assert g5 > 0


# --- vessels: column / dished ends / lying, and agitators (V5b) -------------------------------

AGITATORS = {
    "p": 1.0,
    "host": "vessel_dished_ends",
    "labels": ["tank"],
    "aspect": [1.2, 3.0],
    "clearance": [0.25, 0.65],
    "classes": ["agitator_a", "agitator_b"],
}


def test_a_vessel_three_times_as_tall_as_wide_is_a_column():
    b = _Builder()
    tall = b.add("tank", (100, 100, 200, 400), "T")  # 3.0
    standing = b.add("tank", (400, 100, 500, 399), "T")  # 2.99
    big = b.add("general", (700, 100, 820, 500), "T")  # a large general, 3.33
    out = run(b)
    assert out[tall].template.file == "COLUMN" and out[tall].label == "column"
    assert out[standing].template.file == "VESSEL"
    assert out[big].template.file == "COLUMN"


def test_a_vessel_wider_than_tall_lies_down_and_a_standing_one_does_not():
    b = _Builder()
    lying = b.add("tank", (100, 100, 400, 200), "T")
    standing = b.add("tank", (600, 100, 700, 300), "T")
    out = run(b)
    assert out[lying].template.rot == 90 and out[standing].template.rot == 0


def test_a_column_is_never_drawn_for_a_squat_general():
    g = _Builder()
    for i in range(40):
        g.add("general", (10 + i * 400, 100, 210 + i * 400, 300), "TTBB")
    assert "COLUMN" not in {a.template.file for a in run(g).values()}


def test_agitators_go_into_standing_dished_vessels_only():
    b = _Builder()
    ok = b.add("tank", (100, 100, 200, 250), "T")  # 100 wide, 1.5
    narrow = b.add("tank", (300, 100, 360, 250), "T")  # 60 wide, 2.5: drawn at its own scale
    squat = b.add("tank", (500, 100, 700, 200), "T")  # lying
    column = b.add("tank", (800, 100, 900, 500), "T")  # 4.0
    out = run(b, agitators=AGITATORS)
    assert out[ok].agitator is not None and out[ok].agitator.placement == "attach"
    assert out[ok].agitator.fine_name in ("agitator_a", "agitator_b")
    assert out[narrow].agitator is not None
    assert out[squat].agitator is None and out[column].agitator is None


def test_agitators_go_into_tanks_only_not_into_a_large_general_drawn_as_a_vessel():
    b = _Builder()
    tank = b.add("tank", (100, 100, 400, 500), "T")
    big = b.add("general", (600, 100, 900, 500), "T")  # large, 1.33: an air cooler in OPEN100
    out = run(b, agitators=AGITATORS)
    assert out[big].template.file == "VESSEL"
    assert out[tank].agitator is not None and out[big].agitator is None
    assert run(b, agitators={**AGITATORS, "labels": ["tank", "general"]})[big].agitator


def test_agitators_follow_their_chance_and_use_every_class():
    b = _Builder()
    for i in range(60):
        b.add("tank", (10 + i * 400, 100, 110 + i * 400, 250), "T")
    out = run(b, agitators={**AGITATORS, "p": 0.3})
    placed = [a.agitator for a in out.values() if a.agitator is not None]
    assert 8 <= len(placed) <= 30  # 60 x 0.3 = 18
    assert {g.fine_name for g in placed} == {"agitator_a", "agitator_b"}
    gaps = [a.clearance for a in out.values() if a.agitator is not None]
    assert all(0.25 <= c <= 0.65 for c in gaps) and len(set(gaps)) == len(gaps)
    assert all(a.agitator is None for a in run(b).values())  # no `agitators` block: none


@real
def test_real_sheets_vessels_follow_the_box_shape_and_agitators_their_rule():
    cfg = load_config()["agitators"]
    eligible = 0
    kinds: Counter[str] = Counter()
    for sheet, _, out in _real_runs():
        for nid, a in out.items():
            node = sheet.nodes[nid]
            if a.kind != "iso" or a.template.port_config != "multi":
                continue
            aspect = node.height / node.width
            kinds[a.template.fine_name] += 1
            assert (a.template.fine_name == "column") == (aspect >= 3)
            assert (a.template.rot == 90) == (aspect < 1 and a.template.fine_name != "column")
            ok = (
                a.template.fine_name == "vessel_dished_ends"
                and node.label in cfg["labels"]
                and 1.2 <= aspect < 3
            )
            if a.agitator is not None:  # PID2Graph has no such node: only the rule can put it here
                assert ok, (sheet.name, nid)
            eligible += ok
    assert kinds["column"] >= 3 and kinds["vessel_dished_ends"] >= 25
    # 20 tanks qualify on the 12 sheets (plan, V5b), the 0-port tanks of sheet 9 included
    assert eligible == 20
