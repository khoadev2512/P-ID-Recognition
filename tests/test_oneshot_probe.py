"""scripts/oneshot_probe.py scoring (PLAN_PHASE_C Bước 1): rank = the oneshot_eval mean path
(+ CL2N), summary metrics, the §1.2 decision rule, HOG-lite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import oneshot_probe as probe  # noqa: E402

from pipeline.oneshot_eval import rank_names  # noqa: E402
from utils.oneshot import PrototypeDirectory, center_embeddings  # noqa: E402


def _unit(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(*shape, generator=g), dim=-1)


@pytest.mark.parametrize("centered", [False, True])
def test_rank_matches_the_oneshot_eval_mean_path(centered):
    views, protos = _unit(5, 8, 16, seed=1), _unit(7, 16, seed=2)
    names = [f"p{i}" for i in range(7)]
    center = _unit(16, seed=3) * 0.4 if centered else None
    idx, val = probe.rank(views, protos, center)
    d = PrototypeDirectory(names, protos)
    for k in range(5):
        q = views[k]
        directory = d
        if centered:
            q, directory = center_embeddings(q, center), d.centered(center)
        top = rank_names(directory, q, "mean", 3)
        assert [names[i] for i in idx[k].tolist()] == [n for n, _ in top]
        assert np.allclose(val[k].tolist(), [c for _, c in top], atol=1e-5)


def _row(cls, known, expected, top, best, second):
    return {"class": cls, "known": known, "expected": expected, "top": top, "best": best,
            "second": second}  # fmt: skip


def test_summarize_is_macro_over_known_classes_with_auroc_and_hub():
    rows = [
        _row("A", True, "a", ["a", "b", "c"], 0.9, 0.5),
        _row("A", True, "a", ["b", "a", "c"], 0.8, 0.7),
        _row("B", True, "b", ["c", "d", "e"], 0.7, 0.6),
        _row("U", False, "", ["b", "a", "c"], 0.6, 0.4),
    ]
    s = probe.summarize(rows)
    assert s["per_class"]["A"]["top1"] == 0.5 and s["per_class"]["A"]["top3"] == 1.0
    assert s["per_class"]["B"]["top1"] == 0.0 and s["per_class"]["B"]["top3"] == 0.0
    assert s["macro_top1"] == pytest.approx(0.25) and s["macro_top3"] == pytest.approx(0.5)
    assert s["auroc_cos"] == 1.0  # every known best cos above the unknown one
    assert s["auroc_margin"] == pytest.approx(1 / 3)  # margins 0.4, 0.1, 0.1 vs 0.2
    assert s["hub"] == "b" and s["hub_share"] == 0.5
    assert (s["n_known"], s["n_unknown"]) == (3, 1)


def _summary(per_top1, top1, auc, margin):
    return {"per_class": {c: {"top1": v} for c, v in per_top1.items()}, "macro_top1": top1,
            "auroc_cos": auc, "auroc_margin": margin}  # fmt: skip


def test_decide_needs_six_classes_up_and_a_five_point_macro_gain():
    base = _summary({f"c{i}": 0.1 for i in range(10)} | {"z": 0.0}, 0.10, 0.6, 0.6)
    six_up = {f"c{i}": (0.2 if i < 6 else 0.1) for i in range(10)} | {"z": 0.0}
    assert probe.decide(base, _summary(six_up, 0.16, 0.6, 0.6))["pass_top1"]
    assert not probe.decide(base, _summary(six_up, 0.14, 0.6, 0.6))["pass"]  # gain < 0.05
    five_up = {f"c{i}": (0.9 if i < 5 else 0.1) for i in range(10)} | {"z": 0.0}
    assert not probe.decide(base, _summary(five_up, 0.50, 0.6, 0.6))["pass"]
    d = probe.decide(base, _summary(six_up, 0.16, 0.6, 0.6))
    assert "z" not in d["classes_up"]  # 0 -> 0 is not a rise


def test_decide_by_auroc_allows_at_most_a_one_point_top1_drop():
    base = _summary({"c": 0.2}, 0.20, 0.60, 0.70)
    assert probe.decide(base, _summary({"c": 0.2}, 0.195, 0.66, 0.70))["pass_auroc"]
    assert probe.decide(base, _summary({"c": 0.2}, 0.20, 0.60, 0.75))["pass_auroc"]  # margin
    assert not probe.decide(base, _summary({"c": 0.1}, 0.18, 0.70, 0.70))["pass"]


def test_hog_lite_is_a_unit_vector_and_sees_orientation():
    img = np.full((40, 40), 255, dtype=np.uint8)
    img[:, 18:22] = 0  # a vertical bar
    v = probe.hog_lite(img)
    n = (probe.HOG_SIZE // probe.HOG_CELL) ** 2 * probe.HOG_BINS
    assert v.shape == (n,) and np.isclose(np.linalg.norm(v), 1.0)
    h = probe.hog_lite(img.T.copy())  # the same bar, horizontal
    assert float(v @ h) < 0.5 and np.isclose(float(v @ probe.hog_lite(img)), 1.0)
