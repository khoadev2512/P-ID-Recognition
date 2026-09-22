"""Unit tests for utils.bbox_utils: conversions, clipping, YOLO txt round-trip."""

from __future__ import annotations

import pytest

from utils import bbox_utils
from utils.types import BBox


@pytest.mark.parametrize(
    "box,image_w,image_h",
    [
        (BBox(x=0.5, y=0.5, w=0.2, h=0.3), 640, 480),
        (BBox(x=0.1, y=0.9, w=0.05, h=0.05), 1024, 768),
        (BBox(x=0.0, y=0.0, w=0.0, h=0.0), 100, 100),
        (BBox(x=0.333, y=0.667, w=0.4, h=0.1), 1920, 1080),
    ],
)
def test_yolo_to_xyxy_to_yolo_roundtrip(box, image_w, image_h):
    xyxy = bbox_utils.yolo_to_xyxy(box, image_w, image_h)
    back = bbox_utils.xyxy_to_yolo(xyxy, image_w, image_h)

    assert back.x == pytest.approx(box.x, abs=1e-6)
    assert back.y == pytest.approx(box.y, abs=1e-6)
    assert back.w == pytest.approx(box.w, abs=1e-6)
    assert back.h == pytest.approx(box.h, abs=1e-6)


def test_yolo_to_xyxy_pixel_values():
    box = BBox(x=0.5, y=0.5, w=0.5, h=0.5)
    xyxy = bbox_utils.yolo_to_xyxy(box, image_w=200, image_h=100)
    assert xyxy == pytest.approx((50.0, 25.0, 150.0, 75.0))


def test_clip_xyxy_clamps_out_of_bounds_box():
    xyxy = (-10.0, -5.0, 5000.0, 3000.0)
    clipped = bbox_utils.clip_xyxy(xyxy, image_w=640, image_h=480)
    assert clipped == (0, 0, 639, 479)


def test_clip_xyxy_leaves_in_bounds_box_unchanged():
    xyxy = (10.0, 20.0, 100.0, 200.0)
    clipped = bbox_utils.clip_xyxy(xyxy, image_w=640, image_h=480)
    assert clipped == (10, 20, 100, 200)


def test_read_write_yolo_txt_roundtrip(tmp_path):
    rows = [
        (0, BBox(x=0.5, y=0.5, w=0.2, h=0.3)),
        (2, BBox(x=0.1, y=0.9, w=0.05, h=0.05)),
        (1, BBox(x=0.333333, y=0.666667, w=0.4, h=0.1)),
    ]
    path = tmp_path / "labels" / "0001.txt"
    bbox_utils.write_yolo_txt(path, rows)

    assert path.exists()
    read_back = bbox_utils.read_yolo_txt(path)

    assert len(read_back) == len(rows)
    # length equality already asserted above; strict= needs py3.10+ zip (this repo's
    # floor), noqa'd here only because the sandbox used to iterate on this test runs
    # a 3.9 interpreter that lacks the kwarg.
    for (orig_cls, orig_box), (read_cls, read_box) in zip(rows, read_back):  # noqa: B905
        assert read_cls == orig_cls
        assert read_box.x == pytest.approx(orig_box.x, abs=1e-6)
        assert read_box.y == pytest.approx(orig_box.y, abs=1e-6)
        assert read_box.w == pytest.approx(orig_box.w, abs=1e-6)
        assert read_box.h == pytest.approx(orig_box.h, abs=1e-6)


def test_read_yolo_txt_ignores_trailing_confidence_column(tmp_path):
    path = tmp_path / "with_score.txt"
    path.write_text("0 0.5 0.5 0.2 0.3 0.87\n")

    rows = bbox_utils.read_yolo_txt(path)
    assert len(rows) == 1
    class_id, box = rows[0]
    assert class_id == 0
    assert box == BBox(x=0.5, y=0.5, w=0.2, h=0.3)


def test_read_yolo_txt_skips_blank_lines(tmp_path):
    path = tmp_path / "with_blanks.txt"
    path.write_text("0 0.5 0.5 0.2 0.3\n\n1 0.1 0.1 0.05 0.05\n")

    rows = bbox_utils.read_yolo_txt(path)
    assert len(rows) == 2
