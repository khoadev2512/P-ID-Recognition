"""scripts/probe_2j.py (PLAN_PHASE_C §2J): one-way Chamfer, ink-patch mean, CLS concat."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from patch_probe import chamfer  # noqa: E402
from probe_2j import VARIANTS, concat_cls, directional, masked_mean, pick_best  # noqa: E402


def test_directional_halves_match_hand_example_and_rebuild_chamfer():
    e = torch.eye(4)
    q = e[[0, 1, 2, 3]][None]
    qmask = torch.tensor([[True, True, False, False]])
    t = torch.stack([e[[0, 1, 3, 3]], e[[0, 2, 2, 2]]])
    tmask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    q2t, t2q = directional(q, qmask, t, tmask)
    assert torch.allclose(q2t, torch.tensor([[1.0, 0.5]]))
    assert torch.allclose(t2q, torch.tensor([[1.0, 0.25]]))
    assert torch.allclose(0.5 * (q2t + t2q), chamfer(q, qmask, t, tmask))


def test_directional_rebuilds_chamfer_on_random_input():
    g = torch.Generator().manual_seed(0)
    q = F.normalize(torch.randn(3, 6, 5, generator=g), dim=-1)
    t = F.normalize(torch.randn(4, 7, 5, generator=g), dim=-1)
    qm, tm = torch.rand(3, 6, generator=g) > 0.3, torch.rand(4, 7, generator=g) > 0.3
    qm[:, 0] = tm[:, 0] = True
    a, b = directional(q, qm, t, tm)
    assert torch.allclose(0.5 * (a + b), chamfer(q, qm, t, tm), atol=1e-6)


def test_masked_mean_uses_only_masked_tokens():
    tok = torch.tensor([[[1.0, 0.0], [0.0, 5.0], [3.0, 0.0]]])
    out = masked_mean(tok, torch.tensor([[True, False, True]]))
    assert torch.allclose(out, torch.tensor([[1.0, 0.0]]))


def test_concat_cls_is_unit_and_keeps_block_order():
    cls = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    out = concat_cls(cls)
    assert torch.allclose(out, torch.tensor([[1.0, 0.0, 0.0, 1.0]]) / 2**0.5)


def test_pick_best_ties_by_order():
    s = {k: {"macro_top1": 0.2} for k in VARIANTS}
    s["cat4/raw"]["macro_top1"] = s["inkmean/cl2n"]["macro_top1"] = 0.3
    assert pick_best(s) == "inkmean/cl2n"
