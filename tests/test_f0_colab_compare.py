"""scripts/f0_colab_compare.py (PLAN_PHASE_C §2E.3): the locked decision on synthetic CSV frames."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import f0_colab_compare as cmp  # noqa: E402


def _p1(seed: int, known_boost: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cls, group in [(f"S{i}", "H_chon") for i in range(6)] + [("S9", "OOV_heldout")]:
        for img in range(5):
            for _ in range(4):
                cos = rng.uniform(0.2, 0.6) + (known_boost if group == "H_chon" else 0.0)
                rows.append(
                    {
                        "split": "train",
                        "path": f"{cls}/{img}/{len(rows)}.png",
                        "image_id": img,
                        "class": cls,
                        "group": group,
                        "expected": cls,
                        "top1": cls if rng.random() < 0.5 else "x",
                        "top2": "y",
                        "top3": "z",
                        "best_cos": cos,
                        "second_cos": cos - 0.05,
                    }
                )
    return pd.DataFrame(rows)


def test_a_known_class_cosine_lift_counts_as_better_separation_only():
    base = _p1(0)
    cand = base.copy()
    cand.loc[cand.group == "H_chon", "best_cos"] += 0.3
    cand.loc[cand.group == "H_chon", "second_cos"] += 0.3
    r = cmp.p1_compare(base, cand)
    assert r["d_auroc_cos"] > 0.05 and r["d_macro_top1"] == 0
    assert r["separation_improved"] and not r["naming_improved"]
    assert r["ci_by_image"]["auroc_cos"][0] > 0


def test_a_top1_drop_beyond_the_tolerance_blocks_separation():
    base = _p1(0)
    cand = base.copy()
    cand.loc[cand.group == "H_chon", "best_cos"] += 0.3
    cand.loc[cand["class"] == "S0", "top1"] = "x"
    r = cmp.p1_compare(base, cand)
    assert r["d_macro_top1"] < -0.01 and not r["separation_improved"]


def test_adoption_needs_p1_separation_and_no_worse_p4():
    ok = {"verdict": "không khác"}
    p4 = {"degraded": {g: dict(ok) for g in (*cmp.GROUPS, "auroc_cos", "auroc_margin")}}
    p1 = {"separation_improved": True}
    assert cmp.decide(p1, p4)["adopt"]
    p4["degraded"]["V"] = {"verdict": "kém hơn"}
    d = cmp.decide(p1, p4)
    assert not d["adopt"] and d["p4_worse_groups"] == ["V"]
    p4["degraded"]["V"] = dict(ok)
    p4["degraded"]["auroc_cos"] = {"verdict": "kém hơn"}
    assert not cmp.decide(p1, p4)["adopt"]
    assert not cmp.decide({"separation_improved": False}, {"degraded": {
        g: dict(ok) for g in (*cmp.GROUPS, "auroc_cos")}})["adopt"]  # fmt: skip
