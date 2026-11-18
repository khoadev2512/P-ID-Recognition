"""utils.template_variants (PLAN_PHASE_C §2D.1 H2 / PLAN_STAGE2_ONESHOT B9)."""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from utils.template_variants import (
    PipeAxis,
    contact,
    dirty_view,
    ink_box,
    pick_template,
    template_views,
)


def _valve(w=120, h=60) -> Image.Image:
    """A bow-tie-ish box: a hollow rectangle of 4 px strokes, tight to the image."""
    g = np.full((h, w), 255, dtype=np.uint8)
    g[:4, :] = g[-4:, :] = 0
    g[:, :4] = g[:, -4:] = 0
    return Image.fromarray(g).convert("RGB")


def _ink(img):
    return np.asarray(img.convert("L")) < 128


def test_k_zero_is_the_clean_template_only():
    img = _valve()
    views = template_views(img, [PipeAxis("L", 0.5, 0.0)], k=0)
    assert len(views) == 1 and (np.asarray(views[0]) == np.asarray(img)).all()
    with pytest.raises(ValueError):
        template_views(img, [], k=-1)


def test_same_seed_same_views_and_name_changes_them():
    img, axes = _valve(), [PipeAxis("L", 0.5, 0.0), PipeAxis("R", 0.5, 1.0)]
    a = template_views(img, axes, k=3, seed=0, name="valve")
    b = template_views(img, axes, k=3, seed=0, name="valve")
    c = template_views(img, axes, k=3, seed=0, name="other")
    assert all((np.asarray(x) == np.asarray(y)).all() for x, y in zip(a, b, strict=True))
    assert any(
        x.size != y.size or not (np.asarray(x) == np.asarray(y)).all()
        for x, y in zip(a[1:], c[1:], strict=True)
    )


@pytest.mark.parametrize("side", ["L", "R", "T", "B"])
def test_a_pipe_leaves_the_ink_box_on_its_port_side_only(side):
    img = _valve()
    out = dirty_view(img, [PipeAxis(side, 0.5, 0.0 if side in "LT" else 1.0)], random.Random(1))
    ink = _ink(out)
    # the template sits on a white margin; ink beyond its box can only be the pipe
    h, w = ink.shape
    my, mx = (h - 60) // 2, (w - 120) // 2
    beyond = {
        "L": ink[:, :mx].any(),
        "R": ink[:, mx + 120 :].any(),
        "T": ink[:my, :].any(),
        "B": ink[my + 60 :, :].any(),
    }
    assert beyond[side]
    assert not any(v for s, v in beyond.items() if s != side)


def test_contact_is_the_first_ink_along_the_axis():
    m = np.zeros((20, 40), dtype=bool)
    m[10, 15:25] = True  # a bar in the middle, nothing at the box edge on row 10
    m[0, 0] = m[19, 39] = True  # corners set the box to the whole image
    box = ink_box(m)
    assert contact(m, box, PipeAxis("L", 0.5, 0.0)) == (15, 10)
    assert contact(m, box, PipeAxis("R", 0.5, 1.0)) == (24, 10)
    # an axis that meets no ink falls back to the port's own position
    assert contact(m, box, PipeAxis("T", 0.25, 0.3)) == (10, 6)


def test_a_template_without_ports_gets_no_pipe():
    img = _valve()
    out = dirty_view(img, [], random.Random(0))
    ink = _ink(out)
    ys, xs = np.nonzero(ink)
    h, w = ink.shape
    my, mx = (h - 60) // 2, (w - 120) // 2
    assert (
        xs.min() >= mx - 1 and xs.max() <= mx + 120 and ys.min() >= my - 1 and ys.max() <= my + 60
    )


def test_pick_template_takes_the_closest_aspect():
    class T:
        def __init__(self, w, h):
            self.w_mm, self.h_mm = w, h

    a, b = T(11.3, 6.1), T(11.3, 6.2)
    assert pick_template([a, b], 11.3 / 6.2) is b
    assert pick_template([], 1.0) is None
