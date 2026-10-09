"""Unit tests for scripts/visualize_detections.py.

Tests the pure parsing/mapping/drawing helpers on tiny synthetic inputs — no real
1.3G dataset, no full-image rendering. cv2/numpy are the only heavy deps (already
pipeline deps), guarded via importorskip to match the other script tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("yaml")

import numpy as np  # noqa: E402

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import visualize_detections as viz  # noqa: E402


def test_color_for_cycles_palette():
    # same id -> same color; ids wrap around the palette
    assert viz._color_for(0) == viz._color_for(len(viz._PALETTE))
    assert viz._color_for(0) != viz._color_for(1)


def test_read_pred_txt_parses_6col_and_converts_to_pixels(tmp_path):
    # one box centered at (0.5,0.5), wh (0.2,0.2) in a 100x80 image -> px (40,32)-(60,48)
    p = tmp_path / "0.txt"
    p.write_text("3 0.5 0.5 0.2 0.2 0.91\n")
    rows = viz._read_pred_txt(p, img_w=100, img_h=80)
    assert len(rows) == 1
    cid, (x1, y1, x2, y2), score = rows[0]
    assert cid == 3
    assert (x1, y1, x2, y2) == (40, 32, 60, 48)
    assert score == pytest.approx(0.91)


def test_read_pred_txt_handles_7col_ignoring_fine_score(tmp_path):
    # det_plus_fgc rows have a trailing fine_score column — first 6 cols still parse.
    p = tmp_path / "0.txt"
    p.write_text("1 0.5 0.5 0.4 0.4 0.8 0.77\n")
    rows = viz._read_pred_txt(p, 100, 100)
    assert len(rows) == 1 and rows[0][0] == 1
    assert rows[0][2] == pytest.approx(0.8)  # score, not fine_score


def test_read_pred_txt_missing_file_is_empty(tmp_path):
    assert viz._read_pred_txt(tmp_path / "nope.txt", 100, 100) == []


def test_load_class_names(tmp_path):
    import yaml

    (tmp_path / "classes.yaml").write_text(
        yaml.safe_dump({"coarse": ["valve", "instrument", "flange"], "fine": {}, "fgc_groups": []})
    )
    assert viz._load_class_names(tmp_path / "classes.yaml") == ["valve", "instrument", "flange"]


def test_read_gt_coco_maps_category_to_coarse_id():
    coco = {
        "images": [{"id": 7, "file_name": "synthetic/0.jpg"}],
        "annotations": [
            {"image_id": 7, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"image_id": 7, "category_id": 2, "bbox": [5, 5, 10, 10]},
            {"image_id": 99, "category_id": 1, "bbox": [0, 0, 1, 1]},  # other image
        ],
        "categories": [
            {"id": 1, "name": "Symbol_1", "supercategory": "valve"},
            {"id": 2, "name": "Symbol_2", "supercategory": "instrument"},
        ],
    }
    cat_to_coarse = {1: 0, 2: 1}  # valve=0, instrument=1
    rows = viz._read_gt_coco(coco, "0", cat_to_coarse)
    assert len(rows) == 2  # only image "0"'s two anns
    # bbox xywh -> xyxy
    assert rows[0] == (0, (10, 20, 40, 60))
    assert rows[1] == (1, (5, 5, 15, 15))


def test_draw_boxes_does_not_crash_and_marks_pixels():
    img = np.full((80, 100, 3), 255, dtype=np.uint8)  # white
    boxes = [(0, (10, 10, 50, 50), 0.9)]
    viz._draw_boxes(img, boxes, names=["valve"])
    # something was drawn (not all white anymore)
    assert (img != 255).any()


def test_save_full_downscales_wide_image(tmp_path):
    img = np.zeros((500, 4000, 3), dtype=np.uint8)
    out = tmp_path / "x_full.jpg"
    viz._save_full(img, out, max_width=1000)
    assert out.exists()
    import cv2

    saved = cv2.imread(str(out))
    assert saved.shape[1] == 1000  # downscaled to max_width


def test_save_crops_writes_windows(tmp_path):
    img = np.zeros((2000, 2000, 3), dtype=np.uint8)
    boxes = [(0, (100, 100, 150, 150), 0.9), (1, (1800, 1800, 1850, 1850), 0.8)]
    n = viz._save_crops(img, boxes, tmp_path, "img0", n_crops=4, crop_size=512)
    assert n == 2
    assert (tmp_path / "img0_crop0.jpg").exists()
    assert (tmp_path / "img0_crop1.jpg").exists()


# ------------------------------------------------------------------ #
# legend + fill + no-text
# ------------------------------------------------------------------ #


def test_make_legend_has_one_row_per_class():
    names = ["valve", "instrument", "flange"]
    panel = viz._make_legend(names, height=600, compare=False)
    # panel is a fixed-width white sidebar of the requested height
    assert panel.shape[0] == 600
    assert panel.shape[1] == 360
    # not blank: swatches + text drawn
    assert (panel != 255).any()


def test_make_legend_compare_mode_shows_gt_pred():
    # compare mode doesn't use per-class colors, so it documents GT/pred instead —
    # should contain the GT green and pred red swatches somewhere.
    panel = viz._make_legend(["valve"], height=400, compare=True)
    flat = panel.reshape(-1, 3)
    colors = {tuple(int(v) for v in c) for c in flat}
    assert viz._GT_COLOR in colors
    assert viz._PRED_COLOR in colors


def test_attach_legend_widens_the_image():
    img = np.zeros((500, 800, 3), dtype=np.uint8)
    out = viz._attach_legend(img, ["valve", "instrument"], compare=False)
    assert out.shape[0] == 500          # same height
    assert out.shape[1] == 800 + 360    # image + legend panel


def test_draw_boxes_fill_blends_inside():
    # fill=True should tint the box interior (not just the border), so more pixels change
    img_border = np.full((100, 100, 3), 255, dtype=np.uint8)
    img_fill = img_border.copy()
    boxes = [(0, (20, 20, 80, 80), 0.9)]
    viz._draw_boxes(img_border, boxes, ["valve"], fill=False)
    viz._draw_boxes(img_fill, boxes, ["valve"], fill=True)
    changed_border = (img_border != 255).any(axis=2).sum()
    changed_fill = (img_fill != 255).any(axis=2).sum()
    assert changed_fill > changed_border  # fill touches the interior too


def test_draw_boxes_no_text_still_draws_border():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    viz._draw_boxes(img, [(0, (20, 20, 80, 80), 0.9)], ["valve"], with_text=False)
    assert (img != 255).any()  # border drawn even without text
