"""Unit tests for utils/scan_degrade.py: the stroke-width measure, the online degradation (no
geometry, thin lines kept), and how the two stages switch it on (off by default)."""

from __future__ import annotations

from functools import partial

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from utils.iso_degrade import MIN_PEAK
from utils.scan_degrade import DEFAULTS, ScanDegrade, stroke_width

ROOT_CFG = "configs/config.yaml"


def _page(w: int, paper: int = 204, size=(300, 400)) -> np.ndarray:
    """Gray paper with a horizontal and a vertical line, and a square outline, all `w` px wide."""
    img = np.full(size, paper, np.uint8)
    img[40 : 40 + w, 20:380] = 0
    img[60:280, 50 : 50 + w] = 0
    img[120:220, 150 : 150 + w] = 0
    img[120:220, 250 : 250 + w] = 0
    img[120 : 120 + w, 150:250] = 0
    img[220 - w : 220, 150:250] = 0
    return img


@pytest.mark.parametrize("w", [2, 3, 4, 6])
def test_stroke_width_is_the_width_of_the_lines(w):
    assert stroke_width(_page(w)) == pytest.approx(w)
    assert stroke_width(_page(w, paper=250)) == pytest.approx(w)  # any shade of paper


def test_stroke_width_takes_the_thin_end_and_gives_up_on_blank_images():
    mixed = _page(6)
    mixed[250:252, 100:380] = 0  # a long 2 px line among 6 px ones
    mixed[270:272, 100:380] = 0
    assert stroke_width(mixed) == pytest.approx(2)
    assert stroke_width(np.full((100, 100), 230, np.uint8)) is None  # no ink
    faint = _page(3)
    faint[faint == 0] = 180  # 24 levels of contrast: nothing to measure
    assert stroke_width(faint) is None


def test_config_switches_it_on_only_with_p_above_zero():
    assert ScanDegrade.from_cfg(None) is None
    assert ScanDegrade.from_cfg({"p": 0.0}) is None
    d = ScanDegrade.from_cfg({"p": 0.5, "noise_sigma": [0, 6]})
    assert d is not None and d.p == 0.5
    assert d.ranges["noise_sigma"] == (0.0, 6.0) and d.ranges["morph"] == DEFAULTS["morph"]
    with pytest.raises(ValueError, match="unknown"):
        ScanDegrade(0.5, {"rotate_deg": [-1, 1]})  # geometry is not offered
    with pytest.raises(ValueError, match="morph"):
        ScanDegrade(0.5, {"morph": [-2, 0]})
    with pytest.raises(ValueError, match="p must"):
        ScanDegrade(1.5)


def _ink_box(img):
    ys, xs = np.nonzero(img < 128)
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_degrading_changes_pixels_not_geometry_and_keeps_thin_lines():
    w = 2
    page = _page(w)
    aug = ScanDegrade(1.0, seed=0)
    for _ in range(20):
        out = aug.array(page)
        assert out.shape == page.shape and out.dtype == np.uint8
        # no rotation / shear: the ink stays where it was, give or take the stroke change
        assert np.allclose(_ink_box(out), _ink_box(page), atol=w / 2 + 1)
        # the 2 px line keeps its darkness (V7's MIN_PEAK, less the noise and JPEG)
        line = 1 - out[38:44, 40:360].min(axis=0).astype(float) / 204
        assert np.median(line) >= MIN_PEAK - 0.15
    assert not np.array_equal(aug.array(page), page)


def test_p_decides_how_often_and_colour_images_come_back_with_their_channels():
    page = _page(3)
    aug = ScanDegrade(0.5, seed=1)
    changed = sum(not np.array_equal(aug.array(page), page) for _ in range(200))
    assert 70 < changed < 130
    bgr = np.repeat(page[:, :, None], 3, axis=2)
    out = ScanDegrade(1.0, seed=2).array(bgr)
    assert out.shape == bgr.shape and (out[:, :, 0] == out[:, :, 2]).all()
    rgb = Image.fromarray(bgr)
    res = ScanDegrade(1.0, seed=3)(rgb)
    assert res.mode == "RGB" and res.size == rgb.size
    blank = Image.new("RGB", (40, 40), "white")
    assert ScanDegrade(1.0, seed=4)(blank) is blank  # nothing to measure: left as it is


# --- Stage 2: the crop transform ---------------------------------------------------------------


def _fgc_cfg(degrade=None):
    cfg = OmegaConf.load(ROOT_CFG)
    if degrade is not None:
        cfg.fgc.aug.degrade = degrade
    return cfg


def test_stage2_degrades_train_crops_only_and_only_when_asked():
    from utils.dataset import _build_transform

    def has(cfg, split):
        return any(isinstance(t, ScanDegrade) for t in _build_transform(cfg, split).transforms)

    assert not has(_fgc_cfg(), "train")  # default config: off
    cfg = _fgc_cfg({"p": 0.5})
    assert has(cfg, "train") and not has(cfg, "val") and not has(cfg, "test")
    ops = [type(t).__name__ for t in _build_transform(cfg, "train").transforms]
    assert ops.index("ScanDegrade") < ops.index("Resize")  # on the native crop


# --- Stage 1: the YOLO train dataset -----------------------------------------------------------


def _detector_pipe(tmp_path, degrade=None):
    from pipeline.detect_train import DetectTrainPipeline

    cfg = OmegaConf.load(ROOT_CFG)
    cfg.data_root = str(tmp_path / "data")
    cfg.run_dir = str(tmp_path / "runs")
    if degrade is not None:
        cfg.detector.aug.degrade = degrade
    return DetectTrainPipeline(cfg)


def test_stage1_uses_the_default_trainer_unless_degradation_is_asked_for(tmp_path):
    from pipeline.detect_train import ScanDegradeTrainer

    assert _detector_pipe(tmp_path)._trainer() is None
    assert _detector_pipe(tmp_path, {"p": 0.0})._trainer() is None
    t = _detector_pipe(tmp_path, {"p": 0.5})._trainer()
    assert isinstance(t, partial) and t.func is ScanDegradeTrainer
    assert t.keywords["scan_degrade"].p == 0.5


def test_stage1_train_images_are_degraded_on_load_and_labels_kept(tmp_path):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset

    from pipeline.detect_train import ScanDegradeDataset

    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    page = _page(3, size=(256, 256))
    cv2.imwrite(str(images / "a.png"), cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))
    (labels / "a.txt").write_text("0 0.5 0.665 0.4 0.4\n")
    off = ("mosaic", "fliplr", "hsv_v", "hsv_s", "hsv_h", "scale", "translate", "erasing")
    hyp = get_cfg(overrides=dict.fromkeys(off, 0.0))
    ds = YOLODataset(
        img_path=str(images), imgsz=256, augment=True, hyp=hyp,
        data={"names": {0: "a"}, "nc": 1, "channels": 3},
    )  # fmt: skip
    clean = ds.load_image(0)[0].copy()
    ds.__class__ = ScanDegradeDataset  # what ScanDegradeTrainer.build_dataset does
    ds.scan_degrade = ScanDegrade(1.0, seed=0)
    im, hw0, hw = ds.load_image(0)
    assert im.shape == clean.shape and hw0 == hw == (256, 256)
    assert not np.array_equal(im, clean)
    assert not np.array_equal(ds.load_image(0)[0], im)  # a new draw on every load
    assert ds.stroke_widths == {(0, clean.shape): 3.0}  # measured once, on the clean image
    sample = ds[0]  # the full train transform runs on it
    assert sample["img"].shape[-2:] == (256, 256) and len(sample["cls"]) == 1
