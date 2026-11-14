"""Ignore regions (`iscrowd: 1`, category IGNORE_CATEGORY; PLAN B6d.2) never become training
data: build_classes, build_manifest, coco_to_yolo and extract_crops give byte for byte what they
give without them, so a DigitizePID COCO (no ignore region) is read exactly as before."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf
from PIL import Image

from pipeline.build_classes import BuildClassesPipeline
from pipeline.build_manifest import BuildManifestPipeline
from pipeline.coco_to_yolo import CocoToYoloPipeline
from pipeline.extract_crops import ExtractCropsPipeline
from utils.classmap import IGNORE_CATEGORY

# DigitizePID-shaped: tier-0 prefix, `iscrowd: 0` on every annotation (convert_yolo_to_coco)
_COCO = {
    "images": [
        {"id": 1, "file_name": "synthetic/a.png", "width": 200, "height": 100},
        {"id": 2, "file_name": "synthetic/b.png", "width": 200, "height": 100},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400,
         "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 3, "bbox": [60, 20, 30, 40], "area": 1200,
         "iscrowd": 0},
        {"id": 3, "image_id": 2, "category_id": 2, "bbox": [100, 30, 25, 25], "area": 625,
         "iscrowd": 0},
    ],
    "categories": [
        {"id": 1, "name": "gate_valve", "supercategory": "valve"},
        {"id": 2, "name": "ball_valve", "supercategory": "valve"},
        {"id": 3, "name": "pump", "supercategory": "pump"},
    ],
}  # fmt: skip


def _with_ignore(coco: dict) -> dict:
    """`coco` plus the ISO export's ignore category and two ignore regions, one of them over a
    symbol (a leader arrow's head on a valve)."""
    out = copy.deepcopy(coco)
    out["categories"].append(
        {"id": 4, "name": IGNORE_CATEGORY, "supercategory": IGNORE_CATEGORY, "fgc": False}
    )
    out["annotations"] += [
        {"id": 4, "image_id": 1, "category_id": 4, "bbox": [12, 12, 10, 10], "area": 100,
         "iscrowd": 1, "ignore_kind": "leader_arrow"},
        {"id": 5, "image_id": 2, "category_id": 4, "bbox": [150, 60, 20, 10], "area": 200,
         "iscrowd": 1, "ignore_kind": "flag"},
    ]  # fmt: skip
    return out


def _run(root: Path, coco: dict) -> dict[str, bytes]:
    """Every file the four stages write under `root`, by relative path (inputs left out)."""
    canonical = root / "canonical"
    (canonical / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for stem in ("a", "b"):
        pixels = rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(canonical / "images" / f"{stem}.png")
    (canonical / "annotations.coco.json").write_text(json.dumps(coco))
    cfg = OmegaConf.create(
        {
            "data_root": str(root), "run_dir": str(root / "runs"),
            "data": {"label_level": "coarse", "split": {"ratios": {"train": 0.5, "val": 0.5},
                                                        "seed": 0}},
            "fgc": {"global_encoder": True},
        }
    )  # fmt: skip
    BuildClassesPipeline(cfg).run()
    BuildManifestPipeline(cfg).run()
    CocoToYoloPipeline(cfg).run()
    CocoToYoloPipeline(OmegaConf.merge(cfg, {"data": {"label_level": "fine"}})).run()
    ExtractCropsPipeline(cfg).run()
    inputs = {"canonical/annotations.coco.json", "canonical/images/a.png", "canonical/images/b.png"}
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and str(p.relative_to(root)) not in inputs
    }


def test_ignore_regions_change_no_output_of_the_data_stages(tmp_path):
    plain = _run(tmp_path / "plain", _COCO)
    ignored = _run(tmp_path / "ignore", _with_ignore(_COCO))
    assert plain.keys() == ignored.keys()
    for name in plain:
        assert plain[name] == ignored[name], name
    # and the outputs are the real thing, not empty: 3 symbols, 3 classes, no ignore class
    classes = yaml.safe_load(plain["canonical/classes.yaml"])
    assert IGNORE_CATEGORY not in classes["fine"] and IGNORE_CATEGORY not in classes["coarse"]
    assert classes["fine"] == {"gate_valve": "valve", "ball_valve": "valve", "pump": "pump"}
    assert plain["derived/yolo_coarse/labels/a.txt"].decode().count("\n") == 2
    assert sum(1 for n in plain if n.startswith("derived/crops_fgc/") and n.endswith(".png")) == 3
    manifest = plain["canonical/manifest.csv"].decode().splitlines()
    assert [r.split(",")[3] for r in manifest[1:]] == ["2", "1"]  # n_symbols of a, b


def test_coco_without_iscrowd_reads_like_iscrowd_zero(tmp_path):
    """A COCO with no `iscrowd` key at all (e.g. a hand-made one) is read as before too."""
    bare = copy.deepcopy(_COCO)
    for a in bare["annotations"]:
        del a["iscrowd"]
    assert _run(tmp_path / "bare", bare) == _run(tmp_path / "zero", _COCO)


def test_manifest_still_refuses_an_unknown_category(tmp_path):
    """Only IGNORE_CATEGORY is let through; any other name outside classes.yaml still fails."""
    root = tmp_path / "r"
    _run(root, _COCO)
    coco = _with_ignore(_COCO)
    coco["categories"].append({"id": 5, "name": "mystery", "supercategory": "valve"})
    (root / "canonical" / "annotations.coco.json").write_text(json.dumps(coco))
    cfg = OmegaConf.create(
        {"data_root": str(root), "run_dir": "x",
         "data": {"split": {"ratios": {"train": 0.5, "val": 0.5}, "seed": 0}}}
    )  # fmt: skip
    with pytest.raises(ValueError, match="mystery"):
        BuildManifestPipeline(cfg).run()
