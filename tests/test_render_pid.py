"""Unit tests for scripts/render_pid.py + scripts/extract_bg_patches.py.

Exercises the pure logic (transparency, overlap, category build, ink ratio, empty-patch
check) and a tiny end-to-end render into tmp_path with a 2-symbol library and a fake
background, verifying the COCO output shape. No dependency on the real 1.3G dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import extract_bg_patches as bg  # noqa: E402
import render_pid as rp  # noqa: E402

# ------------------------------------------------------------------ #
# render_pid pure helpers
# ------------------------------------------------------------------ #


def test_white_to_transparent_clears_white_keeps_ink():
    arr = np.full((10, 10, 3), 255, dtype=np.uint8)
    arr[4:6, 4:6] = 0  # a black ink block
    out = rp._white_to_transparent(Image.fromarray(arr))
    a = np.array(out)
    assert a.shape[2] == 4
    assert a[0, 0, 3] == 0  # white corner -> transparent
    assert a[4, 4, 3] == 255  # ink -> opaque


def test_overlaps_detects_and_clears():
    placed = [rp.Placed("valve", (10, 10, 50, 50))]
    assert rp._overlaps((40, 40, 80, 80), placed) is True  # overlapping
    assert rp._overlaps((100, 100, 140, 140), placed) is False  # far away


def test_build_categories_fine_and_coarse():
    templates = [
        rp.SymbolTemplate("X8074", "valve_gate", "valve", True, Image.new("RGBA", (4, 4))),
        rp.SymbolTemplate("X8068", "valve_globe", "valve", True, Image.new("RGBA", (4, 4))),
        rp.SymbolTemplate("2301", "pump_general", "pump", False, Image.new("RGBA", (4, 4))),
    ]
    cats, name_to_id = rp.build_categories(templates)
    assert len(cats) == 3
    assert cats[0] == {"id": 1, "name": "valve_gate", "supercategory": "valve", "fgc": True}
    assert name_to_id["pump_general"] == 3
    # two coarse: valve, pump
    assert {c["supercategory"] for c in cats} == {"valve", "pump"}
    # fgc flag: valve is an FGC family (is_fgc=True), pump is not
    fgc_by_coarse = {c["supercategory"]: c["fgc"] for c in cats}
    assert fgc_by_coarse["valve"] is True
    assert fgc_by_coarse["pump"] is False


def test_sample_template_respects_fgc_boost():
    rng = __import__("random").Random(0)
    fgc = rp.SymbolTemplate("A", "a", "valve", True, Image.new("RGBA", (4, 4)))
    plain = rp.SymbolTemplate("B", "b", "pump", False, Image.new("RGBA", (4, 4)))
    cfg = rp.RenderConfig(fgc_boost=5.0)
    picks = [rp._sample_template([fgc, plain], cfg, rng).fine_name for _ in range(400)]
    # with 5x weight the FGC symbol should dominate (well above 50%)
    assert picks.count("a") > picks.count("b")


# ------------------------------------------------------------------ #
# extract_bg_patches pure helpers
# ------------------------------------------------------------------ #


def test_ink_ratio():
    white = np.full((10, 10), 255, dtype=np.uint8)
    assert bg._ink_ratio(white) == pytest.approx(0.0)
    half = np.full((10, 10), 255, dtype=np.uint8)
    half[:5] = 0
    assert bg._ink_ratio(half) == pytest.approx(0.5)


def test_patch_has_symbol():
    boxes = [(100.0, 100.0, 40.0, 40.0)]  # cx, cy, w, h -> box 80..120
    assert bg._patch_has_symbol(60, 60, 60, boxes) is True  # patch 60..120 overlaps
    assert bg._patch_has_symbol(200, 200, 50, boxes) is False  # far patch


def test_read_yolo_boxes_scales_to_pixels(tmp_path):
    p = tmp_path / "0.txt"
    p.write_text("3 0.5 0.5 0.2 0.1\n")
    boxes = bg._read_yolo_boxes(p, 100, 200)
    assert boxes == [(50.0, 100.0, 20.0, 20.0)]  # cx,cy,w,h in px


# ------------------------------------------------------------------ #
# tiny end-to-end render
# ------------------------------------------------------------------ #


def _tiny_library():
    def sym(reg, fine, coarse, fgc):
        a = np.full((20, 20, 3), 255, dtype=np.uint8)
        a[6:14, 6:14] = 0  # ink block
        img = rp._white_to_transparent(Image.fromarray(a))
        return rp.SymbolTemplate(reg, fine, coarse, fgc, img)

    return [
        sym("X8074", "valve_gate", "valve", True),
        sym("2301", "pump_general", "pump", False),
    ]


def test_render_dataset_writes_coco(tmp_path):
    bg_dir = tmp_path / "bg"
    bg_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (256, 256), "white").save(bg_dir / f"bg_{i}.png")

    out = tmp_path / "canonical"
    cfg = rp.RenderConfig(symbols_min=3, symbols_max=5, do_degrade=False, do_pipes=False,
                          do_tags=False, margin=8, seed=1)
    rp.render_dataset(bg_dir, _tiny_library(), out, n_images=4, cfg=cfg, val_ratio=0.25)

    coco = json.loads((out / "annotations.coco.json").read_text())
    assert len(coco["images"]) == 4
    assert len(coco["categories"]) == 2  # valve_gate, pump_general
    assert all(im["file_name"].startswith("synthetic/") for im in coco["images"])
    assert len(coco["annotations"]) > 0
    # every annotation references a real category + image
    cat_ids = {c["id"] for c in coco["categories"]}
    img_ids = {im["id"] for im in coco["images"]}
    for a in coco["annotations"]:
        assert a["category_id"] in cat_ids
        assert a["image_id"] in img_ids
        assert a["bbox"][2] > 0 and a["bbox"][3] > 0

    # split map + images on disk
    splits = json.loads((out / "digitizepid_splits.json").read_text())
    assert set(splits.values()) <= {"train", "val"}
    assert (out / "images" / "iso_00000.jpg").exists()
