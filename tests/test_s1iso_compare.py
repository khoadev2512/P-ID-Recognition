"""Unit tests for scripts/s1iso_compare.py (PLAN_PHASE_C §2N rule 6) on hand-made metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import s1iso_compare as sc  # noqa: E402


def _det(per: dict[str, tuple[int, int, int, int, int]]) -> dict:
    """per sheet: (tp, fp, n_gt, unseen tp, unseen n)."""

    sheets = {}
    for s, (tp, fp, n, ut, un) in per.items():
        g = {"other_iso": {"tp": ut, "n": un}, "S": {"tp": tp - ut, "n": n - un}}
        sheets[s] = {"tp": tp, "fp": fp, "n_gt": n, "recall_by_group": g}
    tp, fp, n, ut, un = (sum(v[i] for v in per.values()) for i in range(5))
    rg = {"other_iso": {"recall": ut / un}, "S": {"recall": (tp - ut) / (n - un)}}
    block = {"per_sheet": sheets, "tp": tp, "fp": fp, "n_gt": n, "n_predictions": tp + fp}
    block.update(precision=tp / (tp + fp), recall=tp / n, recall_by_group=rg)
    return {"degraded": block}


OLD = _det({f"iso{i}": (60, 20, 100, 5, 10) for i in range(100, 110)})


def test_adopts_a_clear_gain():
    new = _det({f"iso{i}": (80, 21, 100, 6, 10) for i in range(100, 110)})
    r = sc.decide(OLD, new, b=2000)
    assert r["delta_recall"]["delta"] == pytest.approx(0.2)
    assert r["delta_recall"]["ci95"][0] > 0 and r["adopt"]


def test_rejects_on_precision_or_unseen_recall():
    worse_p = _det({f"iso{i}": (80, 40, 100, 5, 10) for i in range(100, 110)})
    r = sc.decide(OLD, worse_p, b=500)
    assert not r["checks"]["b_precision"] and not r["adopt"]
    worse_u = _det({f"iso{i}": (80, 21, 100, 4, 10) for i in range(100, 110)})
    r = sc.decide(OLD, worse_u, b=500)
    assert r["delta_unseen_recall"] == pytest.approx(-0.1) and not r["checks"]["c_unseen_recall"]


def test_noisy_gain_is_not_enough():
    # +1 TP on half the sheets, -1 on the others but one: tiny mean gain, CI spans 0
    per = {f"iso{i}": (61 if i % 2 else 59, 20, 100, 5, 10) for i in range(100, 110)}
    per["iso100"] = (62, 20, 100, 5, 10)
    r = sc.decide(OLD, _det(per), b=2000)
    assert r["delta_recall"]["delta"] > 0 and r["delta_recall"]["ci95"][0] <= 0
    assert not r["adopt"]


def test_needs_the_same_sheets_and_boxes():
    with pytest.raises(ValueError):
        sc.decide(OLD, _det({"iso100": (60, 20, 100, 5, 10)}), b=10)
    with pytest.raises(ValueError):
        sc.decide(OLD, _det({f"iso{i}": (60, 20, 90, 5, 10) for i in range(100, 110)}), b=10)
