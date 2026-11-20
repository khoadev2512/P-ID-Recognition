"""scripts/k4_probe.py (PLAN_PHASE_C §2K.4): view aggregation, multi-template classes, bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from k4_probe import boot_delta, class_of, class_scores, macro_top1  # noqa: E402

E = torch.eye(3)


def test_class_of_strips_the_form_suffix():
    assert class_of("valve_butterfly@B") == "valve_butterfly" and class_of("x") == "x"


def test_mean_matches_probe_rank_order():
    import oneshot_probe as probe

    g = torch.Generator().manual_seed(0)
    v = F.normalize(torch.randn(4, 8, 5, generator=g), dim=-1)
    p = F.normalize(torch.randn(6, 5, generator=g), dim=-1)
    c = torch.randn(5, generator=g) * 0.1
    s = class_scores(v, p, c, "mean", list("abcdef"), list(range(6)))
    idx, val = probe.rank(v, p, c, k=3)
    assert torch.equal(s.topk(3, dim=1).indices, idx)
    assert torch.allclose(s.topk(3, dim=1).values, val, atol=1e-6)


def test_vote_beats_mean_when_one_view_is_decisive_and_max_takes_best_view():
    # 3 views: two weakly toward class 0's direction mixed, one exactly class 1
    v = F.normalize(torch.tensor([[[1.0, 0.9, 0.0], [1.0, 0.8, 0.0], [0.0, 1.0, 0.0]]]), dim=-1)
    p = E[:2]
    vote = class_scores(v, p, None, "vote", ["a", "b"], [0, 1])
    mx = class_scores(v, p, None, "max", ["a", "b"], [0, 1])
    assert vote.argmax().item() == 0  # two votes for a
    assert mx.argmax().item() == 1  # one view is exactly b
    assert mx[0, 1].item() == pytest.approx(1.0)


def test_two_prototypes_of_one_class_count_as_that_class():
    v = F.normalize(torch.tensor([[[0.0, 0.0, 1.0]]]), dim=-1)
    p = torch.stack([E[0], E[1], E[2]])  # prototype 2 is a second form of class 0
    for agg in ("mean", "max", "vote"):
        s = class_scores(v, p, None, agg, ["a", "b"], [0, 1, 0])
        assert s.shape == (1, 2) and s.argmax().item() == 0


def test_macro_and_bootstrap_are_zero_for_identical_answers():
    df = pd.DataFrame(
        {"sheet": [1, 1, 2, 2], "class": ["a", "b", "a", "b"], "top1": ["a", "a", "a", "b"]}
    )
    assert macro_top1(df, ["a", "b"]) == pytest.approx(0.75)
    lo, hi = boot_delta(df, df, ["a", "b"], b=50)
    assert lo == hi == 0.0


def test_confidence_reports_each_aggregations_own_cosine():
    from k4_probe import confidence

    v = F.normalize(torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), dim=-1)
    p = E[:2]
    assert confidence(v, p, None, "max").item() == pytest.approx(1.0)
    assert confidence(v, p, None, "vote").item() == pytest.approx(1.0)  # 2 votes for row 0, cos 1
    mean = F.normalize(v.mean(1), dim=-1)
    assert confidence(v, p, None, "mean").item() == pytest.approx((mean @ p.T).max().item())
