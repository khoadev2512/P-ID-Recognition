"""scripts/k4_colab_compare.py (PLAN_PHASE_C §2K.6): consistency check and the adoption rule."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from k4_colab_compare import decide, final_config, form_consistency  # noqa: E402


def _v(verdict):
    return {"verdict": verdict}


def _inputs(p4=None, p2=None, d_top1=0.0, d_cos=0.0, auroc4="không khác"):
    groups = {
        "S": "không khác",
        "V": "không khác",
        "H_chon": "không khác",
        "H_tu_nhien": "không khác",
    }
    groups.update(p4 or {})
    deg = {g: _v(v) for g, v in groups.items()} | {"auroc_cos": _v(auroc4)}
    p2d = {"S_top1": _v("không khác"), "V_top1": _v("không khác"), "auroc_cos": _v("không khác")}
    p2d.update({k: _v(v) for k, v in (p2 or {}).items()})
    return {"d_macro_top1": d_top1, "d_auroc_cos": d_cos}, p2d, {"degraded": deg}


def test_rule_ii_needs_a_better_group_and_no_worse_one():
    assert decide(*_inputs(p4={"V": "tốt hơn"}), need_better=True)["adopt"]
    assert not decide(*_inputs(), need_better=True)["adopt"]  # nothing better
    d = decide(*_inputs(p4={"V": "tốt hơn", "S": "kém hơn"}), need_better=True)
    assert not d["adopt"] and d["p4_worse"] == ["S"]


def test_rule_ii_guards_p2_p1_and_auroc():
    ok = {"V": "tốt hơn"}
    assert not decide(*_inputs(p4=ok, p2={"S_top1": "kém hơn"}), need_better=True)["adopt"]
    assert not decide(*_inputs(p4=ok, d_top1=-0.02), need_better=True)["adopt"]
    assert not decide(*_inputs(p4=ok, d_cos=-0.02), need_better=True)["adopt"]
    assert not decide(*_inputs(p4=ok, auroc4="kém hơn"), need_better=True)["adopt"]


def test_rule_i_does_not_need_a_gain_nor_check_p1_auroc():
    assert decide(*_inputs(d_cos=-0.5), need_better=False)["adopt"]
    assert not decide(*_inputs(p4={"H_chon": "kém hơn"}), need_better=False)["adopt"]


def test_final_config_prefers_the_combination_only_when_all_three_pass():
    assert final_config({"formB": True, "max": True, "formB_max": True}) == "formB_max"
    assert final_config({"formB": True, "max": True, "formB_max": False}) == "max"
    assert final_config({"formB": True, "max": False, "formB_max": True}) == "formB"
    assert final_config({"formB": False, "max": False, "formB_max": False}) == "base41"


def test_form_consistency_flags_changes_away_from_the_butterfly():
    ref = pd.DataFrame(
        {"top1": ["a", "b", "c"], "top2": ["x", "valve_butterfly", "y"], "top3": ["z"] * 3}
    )
    cand = ref.copy()
    cand.loc[1, "top1"] = "valve_butterfly"  # allowed: butterfly in ref's top-3
    assert form_consistency(ref, cand)["ok"]
    cand.loc[0, "top1"] = "q"  # not allowed
    c = form_consistency(ref, cand)
    assert not c["ok"] and c["n_changed_without_form"] == 1
