"""scripts/patch_probe.py (PLAN_PHASE_C §2H): ink grid, masked Chamfer, variant choice."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from patch_probe import chamfer, ink_grid, pick_best, variant_name  # noqa: E402


def test_ink_grid_keeps_a_one_pixel_line_of_a_large_image():
    a = np.full((448, 448), 255, np.uint8)
    a[100, :] = 0  # 1 px line: it would fade under a 224 resize, must still count
    m = ink_grid(Image.fromarray(a)).view(16, 16)
    assert m[3].all() and m.sum() == 16
    blank = ink_grid(Image.fromarray(np.full((50, 50), 255, np.uint8)))
    assert blank.all()


def test_chamfer_is_symmetric_mean_of_best_matches_over_ink_patches_only():
    d = 4
    e = torch.eye(d)
    q = e[[0, 1, 2, 3]][None]  # 1 view, 4 patches
    qmask = torch.tensor([[True, True, False, False]])  # only patches 0, 1 are ink
    t = torch.stack([e[[0, 1, 3, 3]], e[[0, 2, 2, 2]]])  # 2 templates
    tmask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    s = chamfer(q, qmask, t, tmask)
    # template 0: q->t: patch0 1, patch1 1 -> 1; t->q: 1, 1 -> 1  => 1
    # template 1: q->t: patch0 1, patch1 0 -> .5; t->q: e0 1, e2 x3 -> 0 (q patch 2 masked) -> .25
    assert torch.allclose(s, torch.tensor([[1.0, 0.375]]))


def test_chamfer_matches_a_brute_force_loop():
    g = torch.Generator().manual_seed(0)
    q = F.normalize(torch.randn(3, 6, 5, generator=g), dim=-1)
    t = F.normalize(torch.randn(4, 7, 5, generator=g), dim=-1)
    qm = torch.rand(3, 6, generator=g) > 0.3
    tm = torch.rand(4, 7, generator=g) > 0.3
    qm[:, 0] = tm[:, 0] = True
    s = chamfer(q, qm, t, tm)
    for v in range(3):
        for k in range(4):
            sim = q[v][qm[v]] @ t[k][tm[k]].T
            want = 0.5 * (sim.max(1).values.mean() + sim.max(0).values.mean())
            assert torch.isclose(s[v, k], want, atol=1e-6)


def test_pick_best_takes_the_highest_top1_and_breaks_ties_by_order():
    s = {variant_name(9, False): 0.5, variant_name(9, True): 0.6, variant_name(12, False): 0.6}
    s[variant_name(12, True)] = 0.1
    assert pick_best({k: {"macro_top1": v} for k, v in s.items()}) == "L9/center"
