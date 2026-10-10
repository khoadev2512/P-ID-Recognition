"""Unit tests for scripts/convert_yolo_to_coco.py (DigitizePID YOLO -> canonical COCO)
and for build_manifest's split-map preservation.

The converter lives in scripts/ (not an installed package), so we add it to sys.path.
Image-touching paths (convert()) are exercised on a tiny synthetic YOLO tree written to
tmp_path with real (small) JPEGs, so no dependency on the 1.3G dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import convert_yolo_to_coco as conv  # noqa: E402

# ------------------------------------------------------------------ #
# pure mapping / bbox helpers
# ------------------------------------------------------------------ #


def test_coarse_map_covers_all_32_classes_exactly():
    assert set(conv.COARSE_MAP.keys()) == set(range(32))
    assert conv.NUM_CLASSES == 32


def test_coarse_map_matches_agreed_grouping():
    # The 8-coarse table agreed with the user (valve c0-15, blind_disc c16-18, reducer
    # c19, flange c20, heat_exchanger c21-22, flow_direction c23, safety_valve c24,
    # instrument c25-31).
    assert all(conv.COARSE_MAP[c] == "valve" for c in range(0, 16))
    assert all(conv.COARSE_MAP[c] == "blind_disc" for c in (16, 17, 18))
    assert conv.COARSE_MAP[19] == "reducer"
    assert conv.COARSE_MAP[20] == "flange"
    assert all(conv.COARSE_MAP[c] == "heat_exchanger" for c in (21, 22))
    assert conv.COARSE_MAP[23] == "flow_direction"
    assert conv.COARSE_MAP[24] == "safety_valve"
    assert all(conv.COARSE_MAP[c] == "instrument" for c in range(25, 32))


def test_exactly_four_fgc_groups_have_more_than_one_fine():
    # fgc_groups = coarse classes with >1 fine child (build_classes' rule). The agreed
    # answer is exactly these four.
    from collections import Counter

    counts = Counter(conv.COARSE_MAP.values())
    fgc = {name for name, n in counts.items() if n > 1}
    assert fgc == {"valve", "blind_disc", "heat_exchanger", "instrument"}


def test_fine_name_is_one_indexed():
    assert conv.fine_name(0) == "Symbol_1"
    assert conv.fine_name(31) == "Symbol_32"


def test_yolo_to_coco_bbox_center_to_topleft():
    # A box centered at (0.5, 0.5) with w=h=0.5 in a 100x200 image -> pixel wh 50x100,
    # top-left at (25, 50).
    bbox = conv._yolo_to_coco_bbox(0.5, 0.5, 0.5, 0.5, 100, 200)
    assert bbox == pytest.approx([25.0, 50.0, 50.0, 100.0])


def test_build_categories_ids_and_supercategories():
    cats, yolo_to_cat = conv.build_categories()
    assert len(cats) == 32
    # 1-indexed COCO category ids, ordered by class.
    assert cats[0] == {"id": 1, "name": "Symbol_1", "supercategory": "valve"}
    assert yolo_to_cat[0] == 1 and yolo_to_cat[31] == 32
    # every supercategory is one of the 8 agreed coarse names
    assert {c["supercategory"] for c in cats} == {
        "valve",
        "blind_disc",
        "heat_exchanger",
        "instrument",
        "reducer",
        "flange",
        "safety_valve",
        "flow_direction",
    }


# ------------------------------------------------------------------ #
# convert() on a tiny synthetic YOLO tree
# ------------------------------------------------------------------ #


def _make_tiny_yolo(root: Path):
    for split, ids in (("train", ["0", "1"]), ("val", ["2"])):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in ids:
            Image.new("RGB", (100, 80), "white").save(root / "images" / split / f"{i}.jpg")
    # one valve box (class 0) and one instrument box (class 25) in image 0
    (root / "labels" / "train" / "0.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n25 0.25 0.25 0.1 0.1\n"
    )
    (root / "labels" / "train" / "1.txt").write_text("19 0.5 0.5 0.4 0.4\n")  # reducer
    (root / "labels" / "val" / "2.txt").write_text("")  # image with no annotations


def test_convert_builds_expected_coco_and_split_map(tmp_path):
    src = tmp_path / "yolo"
    out = tmp_path / "canonical"
    _make_tiny_yolo(src)

    coco, split_map = conv.convert(src, out)

    # 3 images, file_names prefixed synthetic/, sizes read from the real JPEGs.
    assert len(coco["images"]) == 3
    assert all(im["file_name"].startswith("synthetic/") for im in coco["images"])
    assert all((im["width"], im["height"]) == (100, 80) for im in coco["images"])

    # 3 annotations total (2 in image 0, 1 in image 1, 0 in image 2).
    assert len(coco["annotations"]) == 3

    # split map preserves the original train/val partition.
    assert split_map == {"0": "train", "1": "train", "2": "val"}

    # images actually copied into canonical/images/.
    assert (out / "images" / "0.jpg").exists()
    assert (out / "images" / "2.jpg").exists()

    # a class-0 annotation maps to category_id 1 (valve).
    cat_by_id = {c["id"]: c for c in coco["categories"]}
    ann0 = next(a for a in coco["annotations"] if a["image_id"] == 1)  # coco image id 1 == "0"
    assert cat_by_id[ann0["category_id"]]["supercategory"] == "valve"


def test_convert_bbox_is_topleft_pixels(tmp_path):
    src = tmp_path / "yolo"
    out = tmp_path / "canonical"
    _make_tiny_yolo(src)
    coco, _ = conv.convert(src, out)
    # image 0's valve box: center (0.5,0.5) wh (0.2,0.2) in 100x80 -> wh (20,16), tl (40,32)
    valve = next(
        a
        for a in coco["annotations"]
        if a["image_id"] == 1 and a["category_id"] == 1
    )
    assert valve["bbox"] == pytest.approx([40.0, 32.0, 20.0, 16.0])


# ------------------------------------------------------------------ #
# build_manifest split-map preservation
# ------------------------------------------------------------------ #


def test_build_manifest_honors_split_map(tmp_path):
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    from pipeline.build_manifest import BuildManifestPipeline

    canonical = tmp_path / "data" / "canonical"
    canonical.mkdir(parents=True)

    # minimal classes.yaml + COCO with 3 synthetic images
    import yaml

    (canonical / "classes.yaml").write_text(
        yaml.safe_dump(
            {"coarse": ["valve"], "fine": {"Symbol_1": "valve"}, "fgc_groups": []}
        )
    )
    coco = {
        "images": [
            {"id": 1, "file_name": "synthetic/0.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "synthetic/1.jpg", "width": 100, "height": 80},
            {"id": 3, "file_name": "synthetic/2.jpg", "width": 100, "height": 80},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "Symbol_1", "supercategory": "valve"}],
    }
    (canonical / "annotations.coco.json").write_text(json.dumps(coco))
    # split map forces a partition the ratio-based splitter would never produce
    # (all three to val), so a pass means the map was honored, not the ratios.
    (canonical / "digitizepid_splits.json").write_text(
        json.dumps({"0": "val", "1": "val", "2": "val"})
    )

    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "data": {"split": {"ratios": {"train": 0.8, "val": 0.2}, "seed": 42}},
        }
    )
    BuildManifestPipeline(cfg).run()

    import pandas as pd

    m = pd.read_csv(canonical / "manifest.csv", dtype={"id": str})
    assert set(m["split"]) == {"val"}  # all honored from the split map


def test_build_manifest_falls_back_to_ratio_without_split_map(tmp_path):
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    from pipeline.build_manifest import BuildManifestPipeline

    canonical = tmp_path / "data" / "canonical"
    canonical.mkdir(parents=True)
    import yaml

    (canonical / "classes.yaml").write_text(
        yaml.safe_dump({"coarse": ["valve"], "fine": {"Symbol_1": "valve"}, "fgc_groups": []})
    )
    coco = {
        "images": [
            {"id": i, "file_name": f"synthetic/{i}.jpg", "width": 10, "height": 10}
            for i in range(1, 11)
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "Symbol_1", "supercategory": "valve"}],
    }
    (canonical / "annotations.coco.json").write_text(json.dumps(coco))
    # NO split map -> ratio-based split (train 0.8) should apply.

    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "data": {"split": {"ratios": {"train": 0.8, "val": 0.2}, "seed": 42}},
        }
    )
    BuildManifestPipeline(cfg).run()

    import pandas as pd

    m = pd.read_csv(canonical / "manifest.csv", dtype={"id": str})
    counts = m["split"].value_counts().to_dict()
    assert counts.get("train") == 8 and counts.get("val") == 2
