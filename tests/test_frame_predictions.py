"""Unit tests for scripts/frame_predictions.py (predictions in the sheet frame, §2M item 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import frame_predictions as fp  # noqa: E402

COCO = {
    "images": [
        {"file_name": "synthetic/iso00_v000_deg.jpg", "width": 1000, "height": 500,
         "degraded": True, "drawing_box": [100, 50, 800, 450]},
        {"file_name": "synthetic/iso00_v000.png", "width": 1000, "height": 500,
         "degraded": False, "drawing_box": [100, 50, 800, 450]},
    ]
}  # fmt: skip


def test_stage1_files_count_centres_outside_the_drawing(tmp_path):
    (tmp_path / "iso00_v000_deg.txt").write_text(
        "0 0.5 0.5 0.1 0.1 0.9\n"  # inside
        "0 0.9 0.5 0.05 0.05 0.8\n"  # x 900 > 800: column
        "0 0.5 0.95 0.05 0.05 0.7\n"  # y 475 > 450: zone band
        "0 0.9 0.5 0.05 0.05 0.1\n"  # below score_thr: not counted
    )
    r = fp.count(COCO, tmp_path)
    assert (r["n_images"], r["n_predictions"], r["in_frame"]) == (1, 3, 2)
    assert "symbol_in_frame" not in r and r["per_image_in_frame"] == {"iso00_v000_deg": 2}


def test_oneshot_files_split_symbol_labels_from_other(tmp_path):
    row = "{} {} {} 0.5 0.05 0.05 0.9 0.3 0.2 0\n"
    (tmp_path / "iso00_v000_deg.txt").write_text(
        row.format("valve_gate", "valve_gate", 0.9)
        + row.format("__other__", "pump", 0.95)
        + row.format("vessel_spherical", "vessel_spherical", 0.95)
        + row.format("pump", "pump", 0.5)  # inside
    )
    r = fp.count(COCO, tmp_path)
    assert (r["n_predictions"], r["in_frame"], r["symbol_in_frame"]) == (4, 3, 2)
    assert r["labels_in_frame"] == {"valve_gate": 1, "vessel_spherical": 1}


def test_needs_a_framed_export_and_every_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        fp.count(COCO, tmp_path)
    bare = {"images": [{**COCO["images"][0]}]}
    del bare["images"][0]["drawing_box"]
    with pytest.raises(ValueError):
        fp.count(bare, tmp_path)
