"""scripts/draw_e2e_sheet.py: per-box kinds follow the evaluator's e2e matching."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import draw_e2e_sheet as d  # noqa: E402

from pipeline.detection_eval import E2EPred, GtBox, IgnoreRegion  # noqa: E402
from utils.classmap import OTHER_CLASS_NAME  # noqa: E402


def _p(name, box, cos=0.9, score=0.9):
    return E2EPred("im", name, score, cos, 0.1, box)


def test_classify_matches_label_and_box_like_e2e_block():
    gts = [
        GtBox("im", "valve_gate", "valve", (0, 0, 10, 10)),  # right
        GtBox("im", "valve_ball", "valve", (20, 0, 30, 10)),  # wrong name
        GtBox("im", "valve_globe", "valve", (40, 0, 50, 10)),  # gated to Other: wrong name
        GtBox("im", "pump_general", "pump", (60, 0, 70, 10)),  # no prediction: missed
        GtBox("im", "other_isa", "other", (80, 0, 90, 10)),  # Other expected, predicted Other
    ]
    preds = [
        _p("valve_gate", (0, 0, 10, 10)),
        _p("valve_gate", (20, 0, 30, 10)),
        _p("valve_globe", (40, 0, 50, 10), cos=0.05),
        _p("tank", (200, 200, 220, 220)),  # nothing there: extra box
        _p("tank", (300, 300, 310, 310)),  # inside an ignore region: not drawn
        _p("valve_gate", (80, 0, 90, 10), cos=0.05),
    ]
    ignores = [IgnoreRegion("im", "leader_arrow", (300, 300, 310, 310))]
    expected = {g.name: g.name for g in gts[:4]} | {"other_isa": OTHER_CLASS_NAME}
    c = d.classify("im", preds, gts, ignores, expected, tau=0.2)
    assert c["pred_kind"] == [
        "correct",
        "wrong_name",
        "wrong_name",
        "extra_box",
        "ignored",
        "correct",
    ]
    assert c["pred_truth"][1] == "valve_ball" and c["answers"][2] == OTHER_CLASS_NAME
    assert c["gt_missed"].tolist() == [False, False, False, True, False]
    assert c["gt_kind"] == ["correct", "wrong_name", "wrong_name", "missed", "correct"]
    assert (c["tp"], c["fp"]) == (2, 3)


def test_densest_window_holds_the_cluster():
    gts = [GtBox("im", "x", "", (900 + i, 900, 910 + i, 910)) for i in range(5)]
    gts.append(GtBox("im", "x", "", (5, 5, 10, 10)))
    x0, y0, x1, y1 = d.densest_window(gts, 1000, 1000, frac=0.3)
    assert x0 <= 905 < x1 and y0 <= 905 < y1
