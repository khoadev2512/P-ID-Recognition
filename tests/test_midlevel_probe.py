"""scripts/midlevel_probe.py (PLAN_PHASE_C §2I): flat-map scores, transform, variant choice."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from midlevel_probe import COMPETING, flat_scores, pick_best, to_tensor  # noqa: E402


def test_flat_scores_euclid_is_negative_squared_distance():
    g = torch.Generator().manual_seed(0)
    q, t = torch.randn(3, 50, generator=g) * 20, torch.randn(4, 50, generator=g) * 20
    s = flat_scores(q, t, "E")
    want = -torch.cdist(q.double(), t.double()).pow(2).float()
    assert torch.allclose(s, want, rtol=1e-5, atol=1e-2)
    assert torch.allclose(flat_scores(t, t, "E").diagonal(), torch.zeros(4), atol=1e-2)


def test_flat_scores_cosine_ignores_scale():
    g = torch.Generator().manual_seed(1)
    q, t = torch.randn(2, 30, generator=g), torch.randn(5, 30, generator=g)
    s = flat_scores(q * 7.0, t, "C")
    assert torch.allclose(s, F.normalize(q, dim=1) @ F.normalize(t, dim=1).T, atol=1e-6)


def test_to_tensor_resizes_without_crop_and_normalizes():
    a = np.zeros((100, 300, 3), np.uint8)
    a[:, :10] = 255  # a white band at the far left must survive (no centre crop)
    x = to_tensor(Image.fromarray(a))
    assert x.shape == (3, 224, 224)
    assert torch.allclose(x[0, 0, 0], torch.tensor((1 - 0.485) / 0.229), atol=1e-4)
    assert torch.allclose(x[0, 0, -1], torch.tensor(-0.485 / 0.229), atol=1e-4)


def test_pick_best_takes_highest_top1_ties_by_order():
    s = {k: {"macro_top1": 0.1} for k in COMPETING}
    s["densenet121/mid/C"]["macro_top1"] = 0.4
    s["resnet34/mid/E"]["macro_top1"] = 0.4
    assert pick_best(s) == "densenet121/mid/C"
