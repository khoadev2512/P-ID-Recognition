"""scripts/viso_probe.py (PLAN_PHASE_C §2K): per-class scores, pair rates, Spearman, R2 rule."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from viso_probe import decide_r2, macro, pair_rate, per_class_top1, spearman  # noqa: E402


def test_per_class_top1_macro_and_pair_rate():
    true = ["a", "a", "a", "a", "b", "b"]
    pred = ["a", "b", "b", "a", "b", "c"]
    per = per_class_top1(true, pred)
    assert per == {"a": 0.5, "b": 0.5}
    assert macro(per, ["a", "b", "zz"]) == 0.5
    assert pair_rate(true, pred, "a", "b") == 0.5
    assert pair_rate(true, pred, "b", "c") == 0.5


def test_spearman_uses_ranks_and_handles_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 1000]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)


def test_decide_r2_needs_close_gain_and_no_drop_elsewhere():
    base = {"close": 0.50, "all": 0.70, "p1val": 0.477}
    assert decide_r2(base, {"close": 0.56, "all": 0.695, "p1val": 0.470})["pass"]
    assert not decide_r2(base, {"close": 0.54, "all": 0.75, "p1val": 0.50})["pass"]
    assert not decide_r2(base, {"close": 0.60, "all": 0.68, "p1val": 0.50})["pass"]
    assert not decide_r2(base, {"close": 0.60, "all": 0.75, "p1val": 0.46})["pass"]
