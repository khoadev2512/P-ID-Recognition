"""utils.input_norm (PLAN_PHASE_C 1b, BACKUP_PLAN A4): Otsu ink, central symbol box, letterbox."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from utils.input_norm import MARGIN_FRAC, ink_letterbox, ink_mask, input_norm_fn, symbol_box


def _canvas(w=100, h=100, bg=230):
    return np.full((h, w), bg, dtype=np.uint8)


def _rgb(gray):
    return Image.fromarray(gray).convert("RGB")


def test_ink_mask_is_otsu_and_blank_for_a_flat_image():
    g = _canvas()
    assert not ink_mask(g).any()
    g[40:60, 40:60] = 20
    m = ink_mask(g)
    assert m[50, 50] and not m[5, 5] and m.sum() == 400
    with pytest.raises(ValueError, match="2-D"):
        ink_mask(np.zeros((4, 4, 3), dtype=np.uint8))


def test_symbol_box_ignores_rim_components_that_miss_the_center():
    g = _canvas()
    g[40:60, 35:65] = 0  # the symbol, in the centre
    g[2:4, 0:100] = 0  # text / pipe along the top rim, not touching the centre
    assert symbol_box(ink_mask(g)) == (35, 40, 65, 60)


def test_symbol_box_keeps_a_pipe_attached_to_the_symbol():
    g = _canvas()
    g[40:60, 40:60] = 0
    g[49:51, 0:100] = 0  # a pipe through the symbol: one component, the box spans it
    assert symbol_box(ink_mask(g)) == (0, 40, 100, 60)


def test_symbol_box_falls_back_to_every_component_when_none_is_central():
    g = _canvas()
    g[0:10, 0:10] = 0
    g[90:100, 90:100] = 0
    assert symbol_box(ink_mask(g)) == (0, 0, 100, 100)


def test_ink_letterbox_keeps_the_aspect_ratio_and_binarizes():
    g = _canvas(120, 80)
    g[30:40, 20:100] = 10  # an 80 x 10 bar
    out = np.asarray(ink_letterbox(_rgb(g)).convert("L"))
    side = round(80 * (1 + 2 * MARGIN_FRAC))
    assert out.shape == (side, side)
    assert set(np.unique(out)) <= {0, 255}
    ys, xs = np.nonzero(out == 0)
    assert xs.max() - xs.min() + 1 == 80 and ys.max() - ys.min() + 1 == 10  # not stretched
    assert abs((xs.min() + xs.max()) / 2 - (side - 1) / 2) <= 1  # centred


def test_ink_letterbox_of_a_blank_image_is_a_white_square():
    out = np.asarray(ink_letterbox(_rgb(_canvas(30, 50))).convert("L"))
    assert out.shape == (50, 50) and (out == 255).all()


def test_ink_letterbox_is_the_same_for_a_crop_and_its_tight_template():
    """The point of A4: a crop with extra white margin and grey tone, and the tight template of
    the same shape, come out identical."""
    tight = np.full((20, 40), 255, dtype=np.uint8)
    tight[5:15, 5:35] = 0
    loose = np.full((60, 80), 200, dtype=np.uint8)
    loose[20:40, 20:60] = np.where(tight == 0, 40, 200)
    a = np.asarray(ink_letterbox(_rgb(tight)))
    b = np.asarray(ink_letterbox(_rgb(loose)))
    assert a.shape == b.shape and (a == b).all()


def test_input_norm_fn_maps_the_config_value():
    assert input_norm_fn(None) is None and input_norm_fn("") is None
    assert input_norm_fn("ink_letterbox") is ink_letterbox
    with pytest.raises(ValueError, match="ink_letterbox"):
        input_norm_fn("otsu")
