"""Unit tests for the OPEN100-ISO v2 renderer fixes (PLAN_PHASE_C §2M.2): pipe ties and their
bubble guard (utils.iso_layout), arrow chains (utils.iso_mapping), leaders stopped at a vessel
wall (utils.page_svg.clip_at_wall), and, on real OPEN100 sheets, that the v2 config passes the
G1 criteria where it once failed them (dropped pipes left their ports; ties shrank bubbles;
equipment placed by one side's ports)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from PIL import Image

from utils.iso_layout import _pipe_ties, build_axis_map
from utils.iso_mapping import Assignment, arrow_chains, load_config, unify_arrow_chains
from utils.page_svg import clip_at_wall
from utils.pid2graph import PipeRun

ROOT = Path(__file__).resolve().parent.parent
_OPEN100 = ROOT / "data" / "raw" / "pid2graph" / "open100"
_TEMPLATES = ROOT / "data" / "raw" / "template_iso_svg"
needs_open100 = pytest.mark.skipif(
    not (_OPEN100.exists() and _TEMPLATES.exists()), reason="OPEN100 data missing"
)


def _run(a, b, *pts, **kw):
    return PipeRun(a, b, tuple(pts), (), **kw)


# --- pipe ties (layout.straighten_pipes) -----------------------------------------------------


def test_pipe_ties_one_pipe_through_a_node():
    # arrow n: one pipe in from the left at y 100, one out to the right at y 103 -> tie y
    runs = [_run("p", "n", (0, 100), (50, 100)), _run("n", "q", (60, 103), (120, 103))]
    x, y = _pipe_ties(runs, 15)
    assert x == [] and [sorted(t) for t in y] == [[100, 103]]
    # an offset over the limit is a real jog of the pipe
    far = [_run("p", "n", (0, 100), (50, 100)), _run("n", "q", (60, 130), (120, 130))]
    assert _pipe_ties(far, 15) == ([], [])
    # two pipes in from the left (a vessel's nozzles): not one pipe through, no tie
    two = [*runs, _run("r", "n", (0, 90), (50, 90))]
    assert _pipe_ties(two, 15) == ([], [])


def test_pipe_ties_small_step_inside_one_run():
    run = _run("a", "b", (0, 100), (50, 100), (50, 104), (120, 104))
    x, y = _pipe_ties([run], 15)
    assert x == [] and sorted(y[0]) == [100, 104]
    back = _run("a", "b", (0, 100), (50, 100), (50, 104), (10, 104))  # a U-turn: not a step
    assert _pipe_ties([back], 15) == ([], [])


def test_free_ties_cross_symbol_edges_but_not_bubble_edges():
    values = [0.0, 100.0, 102.0, 104.0, 150.0, 200.0]
    spans = [(102.0, 150.0)]  # a symbol whose top edge lies between the tied values
    rng = lambda: random.Random(0)  # noqa: E731
    guarded = build_axis_map(values, spans, 1.0, 0.5, (1.0, 1.0), rng(), [(100.0, 104.0)])
    assert guarded(100.0) != guarded(104.0)  # a guarded tie never moves a symbol edge
    free = build_axis_map(values, spans, 1.0, 0.5, (1.0, 1.0), rng(), None, [(100.0, 104.0)])
    assert free(100.0) == free(104.0)
    hard = build_axis_map(
        values, spans, 1.0, 0.5, (1.0, 1.0), rng(), None, [(100.0, 104.0)], [(102.0, 150.0)]
    )
    assert hard(100.0) != hard(104.0)  # a bubble's edge: its circle would shrink
    assert hard(150.0) - hard(102.0) == 48.0


# --- arrow chains ----------------------------------------------------------------------------


def _arrow(forward):
    return Assignment("arrow", "flow_direction", None, "H", forward)


def test_arrow_chains_follow_straight_runs_on_the_arrows_axis():
    runs = [
        _run("a1", "a2", (0, 0), (10, 0)),
        _run("a2", "a3", (20, 0), (30, 0)),
        _run("a3", "a4", (30, 0), (30, 10)),  # turns: a4 is another chain
        _run("a4", "v", (40, 10), (50, 10)),
    ]
    asg = {n: _arrow(True) for n in ("a1", "a2", "a3", "a4")}
    asg["v"] = Assignment("iso", "valve_gate")
    assert arrow_chains(runs, asg) == [["a1", "a2", "a3"]]


def test_unify_takes_the_tipped_majority_else_the_first_arrow():
    runs = [_run("a1", "a2", (0, 0), (10, 0)), _run("a2", "a3", (20, 0), (30, 0))]
    asg = {"a1": _arrow(True), "a2": _arrow(False), "a3": _arrow(False)}
    out = unify_arrow_chains(runs, asg, {"a2", "a3"}, [])
    assert [out[n].forward for n in ("a1", "a2", "a3")] == [False, False, False]
    out = unify_arrow_chains(runs, asg, set(), [])  # nothing read from the drawing
    assert [out[n].forward for n in ("a1", "a2", "a3")] == [True, True, True]
    assert asg["a2"].forward is False  # the input is left as it was


# --- leaders stop at a vessel wall -----------------------------------------------------------


def test_clip_at_wall_moves_the_head_to_the_outline():
    wall = [(100, 100), (200, 100), (200, 200), (100, 200)]
    far, centre = (0, 150), (150, 150)
    head = clip_at_wall(far, centre, [wall], 20)
    assert head[0] < 100 + 1e-6 and head[1] == pytest.approx(150)  # outside or on the wall
    assert clip_at_wall(far, (50, 150), [wall], 20) == (50, 150)  # not inside: as it was
    assert clip_at_wall((120, 150), centre, [wall], 20) == centre  # starts inside: as it was


# --- G1 on real sheets with the v2 config ----------------------------------------------------


@needs_open100
@pytest.mark.parametrize(
    "sheet_id, index, why",
    [
        (5, 0, "a dropped hanging pipe left its port behind (criterion 1)"),
        (2, 0, "a tie shrank two touching bubbles apart (criterion 5)"),
        (4, 4, "an exchanger placed by its two top ports only (criterion 5)"),
    ],
)
def test_v2_config_passes_g1_where_it_once_failed(sheet_id, index, why):
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_g1

    from utils.iso_templates import load_templates
    from utils.iso_variants import plan_variant, render_variant
    from utils.pid2graph import load_graphml, pipe_runs

    cfg = load_config(ROOT / "configs" / "render_iso_v2.yaml")
    templates = load_templates(cfg["template_dir"], cfg["taxonomy_csv"], cfg["scope"])
    sheet = load_graphml(
        _OPEN100 / f"{sheet_id}.graphml", Image.open(_OPEN100 / f"{sheet_id}.png").size
    )
    runs = pipe_runs(sheet)
    r = render_variant(plan_variant(sheet, runs, templates, cfg, sheet_id, index), sheet, runs, cfg)
    passed = check_g1.evaluate(r.page, r.runs, r.source, r.cfg)[1]
    assert all(passed.values()), (why, passed)
