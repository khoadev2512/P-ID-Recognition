"""scripts/nopipe_compare.py (PLAN_PHASE_C §2G): dropping `pipeline` from a 42-template CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nopipe_compare import consistency, drop_template, p2_stats  # noqa: E402


def _rows():
    # pipeline at top-1, top-2, top-3, absent
    return pd.DataFrame(
        {
            "class": ["A", "A", "B", "C"],
            "group": ["S", "S", "V", "OOV_seen"],
            "image_id": [1, 2, 3, 4],
            "expected": ["a", "a", "b", "Other"],
            "top1": ["pipeline", "a", "b", "x"],
            "cos1": [0.9, 0.8, 0.7, 0.6],
            "top2": ["a", "pipeline", "x", "y"],
            "cos2": [0.5, 0.4, 0.3, 0.2],
            "top3": ["x", "x", "pipeline", "z"],
            "cos3": [0.3, 0.2, 0.1, 0.1],
            "best_cos": [0.9, 0.8, 0.7, 0.6],
            "second_cos": [0.5, 0.4, 0.3, 0.2],
        }
    )


def test_drop_template_moves_the_rest_up_and_leaves_other_rows_alone():
    df = _rows()
    out = drop_template(df)
    assert list(out.top1) == ["a", "a", "b", "x"]
    assert np.allclose(out.best_cos, [0.5, 0.8, 0.7, 0.6])
    assert np.allclose(out.second_cos, [0.3, 0.2, 0.3, 0.2])
    assert out.top3.iloc[3] == "z" and out.iloc[3].equals(df.iloc[3])
    assert (out[["top1", "top2", "top3"]] != "pipeline").all().all()


def test_consistency_accepts_the_exact_drop_and_rejects_a_wrong_run():
    df = _rows()
    good = drop_template(df).fillna({"top3": "q", "cos3": 0.0})
    c = consistency(df, good, "p2")
    assert c["n_changed"] == 1 and c["right_to_wrong"] == 0 and c["n_had_pipeline_top2"] == 2
    bad = good.copy()
    bad.loc[1, "top1"], bad.loc[1, "cos1"] = "x", 0.85  # a crop that was right turns wrong
    with pytest.raises(AssertionError):
        consistency(df, bad, "p2")
    with pytest.raises(AssertionError):  # the 41 run still names pipeline
        consistency(df, df, "p2")


def test_p2_stats_uses_s_and_v_as_known_and_oov_seen_as_unknown():
    s = p2_stats(drop_template(_rows()))
    assert s["S_top1"] == 1.0 and s["V_top1"] == 1.0
    assert s["auroc_cos"] == pytest.approx(2 / 3)  # known 0.5 / 0.8 / 0.7 vs unknown 0.6
