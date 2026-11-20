"""Unit tests for the per-subpart augment (utils.augment.SubpartWarp) and its wiring
into the FGC train transform (utils.dataset._build_transform).

Synthetic symbols only — two dark squares on a grey page stand in for a symbol with
two separable parts, a bow-tie outline for a one-stroke valve. Whether the augment
looks right on REAL crops is a visual check (docs/PLAN_STAGE2_ONESHOT.md B1), not
something a unit test can decide.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

pytest.importorskip("torchvision")

from utils.augment import SubpartWarp  # noqa: E402
from utils.dataset import _build_transform, build_eval_transform  # noqa: E402

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
_BG = 200  # DigitizePID pages are grey, not white


def _two_parts() -> Image.Image:
    a = np.full((80, 100, 3), _BG, np.uint8)
    a[20:40, 10:30] = 0
    a[40:60, 65:85] = 0
    return Image.fromarray(a)


def _bow_tie() -> Image.Image:
    a = np.full((64, 96, 3), _BG, np.uint8)
    pts = np.array([[10, 12], [48, 32], [10, 52], [86, 12], [48, 32], [86, 52]], np.int32)
    cv2.polylines(a, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 0), thickness=3)
    return Image.fromarray(a)


def _ink_components(img: Image.Image) -> int:
    gray = np.asarray(img.convert("L"))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= 20).sum())


def _ink_pixels(img: Image.Image) -> int:
    return int((np.asarray(img.convert("L")) < _BG // 2).sum())


# --------------------------------------------------------------------------- #
# SubpartWarp
# --------------------------------------------------------------------------- #


def test_p_zero_returns_the_input_unchanged():
    img = _bow_tie()
    out = SubpartWarp(p=0.0, seed=0)(img)
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_p_one_changes_the_symbol():
    img = _bow_tie()
    out = SubpartWarp(p=1.0, seed=0)(img)
    assert not np.array_equal(np.asarray(out), np.asarray(img))


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_size_and_mode_preserved(mode):
    img = _bow_tie().convert(mode)
    out = SubpartWarp(p=1.0, seed=1)(img)
    assert out.size == img.size
    assert out.mode == mode


def test_same_seed_same_output():
    img = _two_parts()
    a = SubpartWarp(p=1.0, seed=7)(img)
    b = SubpartWarp(p=1.0, seed=7)(img)
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("value", [255, 0, _BG])
def test_blank_or_solid_image_does_not_crash(value):
    img = Image.fromarray(np.full((40, 40, 3), value, np.uint8))
    out = SubpartWarp(p=1.0, seed=0)(img)
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_tiny_image_passes_through():
    img = Image.fromarray(np.zeros((4, 4, 3), np.uint8))
    assert SubpartWarp(p=1.0, seed=0)(img) is img


def test_separate_parts_stay_separate():
    # Two well-separated parts move independently but must neither merge nor vanish.
    aug = SubpartWarp(p=1.0, p_thin=0.0, p_thick=0.0, seed=0)
    for _ in range(20):
        assert _ink_components(aug(_two_parts())) == 2


def test_one_stroke_symbol_stays_one_stroke():
    # The smooth warp moves regions of a connected stroke without tearing it — the
    # reason it was chosen over splitting strokes into clusters.
    aug = SubpartWarp(p=1.0, p_thin=0.0, p_thick=0.0, seed=0)
    for _ in range(20):
        assert _ink_components(aug(_bow_tie())) == 1


def test_background_level_and_ink_mass_preserved():
    img = _bow_tie()
    before = _ink_pixels(img)
    aug = SubpartWarp(p=1.0, seed=3)
    for _ in range(20):
        out = np.asarray(aug(img).convert("L"))
        assert np.median(out) == pytest.approx(_BG, abs=2)  # page stays grey, not white
        assert 0.4 * before <= _ink_pixels(Image.fromarray(out)) <= 2.0 * before


def test_thinning_reduces_ink_and_thickening_adds_it():
    img = _bow_tie()
    base = SubpartWarp(p=1.0, p_thin=0.0, p_thick=0.0, seed=5)(img)
    thin = SubpartWarp(p=1.0, p_thin=1.0, p_thick=0.0, seed=5)(img)
    thick = SubpartWarp(p=1.0, p_thin=0.0, p_thick=1.0, seed=5)(img)
    assert _ink_pixels(thin) < _ink_pixels(base) < _ink_pixels(thick)


@pytest.mark.parametrize(
    "kwargs",
    [{"p": 1.5}, {"p": -0.1}, {"p_thin": 0.8, "p_thick": 0.3}, {"warp_grid": 1}],
)
def test_invalid_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        SubpartWarp(**kwargs)


# --------------------------------------------------------------------------- #
# Wiring into the FGC transforms
# --------------------------------------------------------------------------- #


def _cfg(p: float | None = None):
    cfg = OmegaConf.load(_CONFIG)
    if p is not None:
        cfg.fgc.aug.subpart.p = p
    return cfg


def test_from_cfg_reads_the_config_node():
    aug = SubpartWarp.from_cfg(_cfg().fgc.aug.subpart)
    node = _cfg().fgc.aug.subpart
    assert aug.p == node.p
    assert aug.scale == (node.scale[0], node.scale[1])
    assert aug.p_thin == node.p_thin


def test_train_transform_applies_subpart_before_resize():
    ops = _build_transform(_cfg(), "train").transforms
    assert isinstance(ops[0], SubpartWarp)
    assert type(ops[1]).__name__ == "Resize"


def test_subpart_p_zero_leaves_it_out_of_train_transform():
    ops = _build_transform(_cfg(p=0.0), "train").transforms
    assert not any(isinstance(op, SubpartWarp) for op in ops)


@pytest.mark.parametrize("split", ["val", "test"])
def test_eval_transforms_never_augment(split):
    for compose in (_build_transform(_cfg(p=1.0), split), build_eval_transform(_cfg(p=1.0))):
        assert not any(isinstance(op, SubpartWarp) for op in compose.transforms)
