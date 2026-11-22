"""scripts/ft_colab_compare.py (PLAN_PHASE_C §2L): rule (ii) §2K.6 + guard (b), dry run on CSVs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ft_colab_compare as fc  # noqa: E402
from test_k4_colab_compare import _inputs  # noqa: E402

GOOD = {"valve_three_way": 1.0, "breather_valve": 0.82}


def test_guard_b_blocks_a_drop_beyond_ten_points_only():
    assert fc.guard_b(GOOD, {"valve_three_way": 0.91, "breather_valve": 0.82})["ok"]
    g = fc.guard_b(GOOD, {"valve_three_way": 1.0, "breather_valve": 0.70})
    assert not g["ok"] and g["delta"]["breather_valve"] == pytest.approx(-0.12)
    with pytest.raises(ValueError):
        fc.guard_b({"valve_three_way": 1.0}, GOOD)


def test_decision_is_rule_ii_and_guard_b():
    ok = fc.guard_b(GOOD, GOOD)
    bad = fc.guard_b(GOOD, {"valve_three_way": 0.5, "breather_valve": 0.82})
    assert fc.decide(*_inputs(p4={"V": "tốt hơn"}), ok)["adopt"]
    assert not fc.decide(*_inputs(), ok)["adopt"]  # rule (ii) needs a better group
    d = fc.decide(*_inputs(p4={"V": "tốt hơn"}), bad)
    assert not d["adopt"] and d["checks"]["guard_b"] is False


def test_e2e_summary_reads_the_degraded_block(tmp_path):
    assert fc.e2e_summary(tmp_path / "missing.json") is None
    block = {"precision": 0.5, "recall": 0.4, "box_recall": 0.9, "by_group": {"S": {"recall": 0.8}}}
    f = tmp_path / "e2e.json"
    f.write_text(json.dumps({"tau": 0.2, "degraded": block, "clean": block}))
    assert fc.e2e_summary(f) == {
        "tau": 0.2,
        "precision": 0.5,
        "recall": 0.4,
        "box_recall": 0.9,
        "by_group": {"S": 0.8},
    }


def test_per_class_degraded_ignores_clean_and_other_rows():
    df = pd.DataFrame(
        {
            "split": ["degraded", "degraded", "clean", "degraded"],
            "group": ["V", "V", "V", "other_x"],
            "expected": ["valve_three_way", "valve_three_way", "valve_three_way", "other"],
            "top1": ["valve_three_way", "valve_needle", "valve_needle", "x"],
        }
    )
    assert fc.per_class_degraded(df) == {"valve_three_way": 0.5}
