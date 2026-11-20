"""utils.linear_align (PLAN_PHASE_C §2E.1, F0)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from utils.linear_align import LAMBDAS, ResidualRidge, choose_lambda, domain_center


def _unit(n, d, seed):
    return F.normalize(torch.randn(n, d, generator=torch.Generator().manual_seed(seed)), dim=-1)


def test_domain_center_removes_the_template_mean():
    t = _unit(6, 16, 0) + 3.0  # a big common offset
    out = domain_center(t)
    assert torch.allclose(out.norm(dim=-1), torch.ones(6), atol=1e-6)
    expect = F.normalize(t - t.mean(0), dim=-1)
    assert torch.allclose(out, expect)


def test_inf_lambda_is_a_pure_shift():
    x, y = _unit(5, 32, 1), _unit(5, 32, 2)
    m = ResidualRidge.fit(x, y, math.inf)
    z = _unit(3, 32, 3)
    b = (y - x).mean(0)
    assert torch.allclose(m(z), F.normalize(z + b, dim=-1), atol=1e-6)


def test_small_lambda_interpolates_the_training_pairs():
    x, y = _unit(6, 64, 4), _unit(6, 64, 5)
    m = ResidualRidge.fit(x, y, 1e-8)
    assert torch.allclose(m(x), y, atol=1e-4)


def test_large_lambda_tends_to_the_shift():
    x, y = _unit(6, 64, 6), _unit(6, 64, 7)
    z = _unit(4, 64, 8)
    big, inf = ResidualRidge.fit(x, y, 1e9), ResidualRidge.fit(x, y, math.inf)
    assert torch.allclose(big(z), inf(z), atol=1e-6)


def test_fit_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        ResidualRidge.fit(_unit(3, 8, 0), _unit(4, 8, 1), 1.0)


def test_choose_lambda_picks_from_the_grid_and_prefers_change_when_it_generalizes():
    # a common rotation-free map: y = normalize(x + fixed offset) -> the shift alone is exact
    x = _unit(8, 32, 9)
    off = _unit(1, 32, 10)[0]
    y = F.normalize(x + off, dim=-1)
    lam = choose_lambda(x, y)
    assert lam in LAMBDAS
    # a linear map shared by all pairs: x -> x @ M (plus normalization); ridge beats the shift
    mat = torch.eye(32) + 0.5 * _unit(32, 32, 11)
    x2 = _unit(12, 32, 12)
    y2 = F.normalize(x2 @ mat, dim=-1)
    assert not math.isinf(choose_lambda(x2, y2))


def test_pair_shift_is_the_inf_ridge_shift_on_cl2n_pairs(tmp_path):
    from utils.linear_align import (
        align_templates,
        load_pairs,
        pair_shift,
        pair_targets,
        save_pairs,
    )

    g = torch.Generator().manual_seed(3)
    t = _unit(4, 16, 13)
    views = [torch.randn(5, 3, 16, generator=g) for _ in range(4)]
    c = 0.1 * _unit(1, 16, 14)[0]
    path = save_pairs(tmp_path / "p.pt", ["a", "b", "c", "d"], t, views, {"input_norm": None})
    p = load_pairs(path)
    assert p["names"] == ["a", "b", "c", "d"] and len(p["crop_views"]) == 4
    b = pair_shift(p["templates"], p["crop_views"], c)
    x = F.normalize(t - c, dim=-1)
    ref = ResidualRidge.fit(x, pair_targets(views, c), math.inf)
    assert torch.allclose(b, ref.b.float(), atol=1e-6)
    out = align_templates(t, c, "pair_shift", b)
    assert torch.allclose(out, ref(x), atol=1e-6)
    assert torch.allclose(align_templates(t, c, "domain_center"), domain_center(t))
    with pytest.raises(ValueError):
        align_templates(t, c, "rotate")
    with pytest.raises(ValueError):
        align_templates(t, c, "pair_shift")
