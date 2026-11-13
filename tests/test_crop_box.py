"""Unit tests for utils/crop_box.py: one classification view everywhere (stored context ->
the same pixels as a cut at inference), the box jitter, and how crop extraction, the FGC
dataset and fgc_infer use them."""

from __future__ import annotations

import json
import random

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from utils.classmap import GLOBAL_GROUP
from utils.crop_box import (
    BOXES_CSV,
    CROP_CONTEXT_FRAC,
    CROP_MARGIN_FRAC,
    BoxJitter,
    StoredCrop,
    context_region,
    cut,
    grow,
    read_boxes,
    view,
    write_boxes,
)


def _noise(h=300, w=400, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _stored(image: np.ndarray, box) -> tuple[np.ndarray, StoredCrop]:
    """What pid-prep-crops keeps for `box`: the context crop and where it sits."""
    x1, y1, x2, y2 = context_region(box, image.shape[1], image.shape[0])
    return image[y1:y2, x1:x2], StoredCrop(box, (x1, y1))


def _random_boxes(n, w=400, h=300, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        bw, bh = rng.uniform(3, 80), rng.uniform(3, 80)
        x1, y1 = rng.uniform(-10, w - 5), rng.uniform(-10, h - 5)  # some cross the edges
        yield (x1, y1, x1 + bw, y1 + bh)


def test_the_view_of_a_stored_crop_is_the_cut_at_inference():
    """The point of the module: val / test crops and inference crops are the same pixels,
    including boxes clipped by the image edge."""
    image = _noise()
    for box in _random_boxes(300):
        stored, where = _stored(image, box)
        expected = cut(image, box)
        assert np.array_equal(view(stored, where), expected)
        got = view(Image.fromarray(stored), where)  # the dataset path works on PIL
        assert np.array_equal(np.asarray(got), expected)


def test_cut_adds_the_margin_and_clips_to_the_image():
    image = _noise()
    c = cut(image, (100.0, 50.0, 200.0, 150.0))  # 100 x 100 box, 10 px margin per side
    assert c.shape[:2] == (120, 120)
    assert np.array_equal(c, image[40:160, 90:210])
    assert cut(image, (-20.0, -20.0, 20.0, 20.0)).shape[:2] == (24, 24)  # clipped at 0
    assert cut(image, (390.0, 290.0, 450.0, 350.0)).shape[:2] == (16, 16)  # 384..400: at W, H


def test_boxes_csv_round_trips(tmp_path):
    rows = [("a_0.png", StoredCrop((1.25, 2.5, 30.125, 40.0), (0, 0))),
            ("a_1.png", StoredCrop((100.1, 7.3, 140.9, 60.7), (84, 0)))]  # fmt: skip
    write_boxes(tmp_path, rows)
    assert read_boxes(tmp_path) == dict(rows)
    assert read_boxes(tmp_path / "missing") == {}


# --- box jitter ------------------------------------------------------------------------------


def test_jitter_config_and_validation():
    assert BoxJitter.from_cfg(None) is None
    assert BoxJitter.from_cfg({"p": 0}) is None
    j = BoxJitter.from_cfg({"p": 0.5, "shift": 0.05})
    assert j.p == 0.5 and j.shift == 0.05 and j.scale == BoxJitter.DEFAULTS["scale"]
    with pytest.raises(ValueError, match="unknown"):
        BoxJitter(0.5, {"rotate": 5})
    with pytest.raises(ValueError, match="p must"):
        BoxJitter(1.5)
    with pytest.raises(ValueError, match="scale"):
        BoxJitter(0.5, {"scale": [1.2, 0.9]})
    with pytest.raises(ValueError, match="stored context"):
        BoxJitter(0.5, {"scale": [1.0, 1.6]})  # would need pixels pid-prep-crops never kept


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


def test_jittered_boxes_stay_near_the_symbol_and_inside_the_context():
    box = (100.0, 100.0, 160.0, 130.0)
    j = BoxJitter(1.0, seed=0)
    reach = grow(box, CROP_CONTEXT_FRAC)
    ious = []
    for _ in range(500):
        b = j(box)
        v = grow(b, CROP_MARGIN_FRAC)  # the view actually cut
        assert reach[0] <= v[0] and reach[1] <= v[1] and v[2] <= reach[2] and v[3] <= reach[3]
        ious.append(_iou(b, box))
    assert min(ious) > 0.6  # still a plausible detector box for this symbol
    assert np.mean(ious) < 0.95  # and not just the ground truth again


def test_p_decides_how_often_a_box_moves():
    box = (10.0, 10.0, 50.0, 40.0)
    assert all(BoxJitter(1e-9, seed=1)(box) == box for _ in range(50))
    j = BoxJitter(0.5, seed=3)
    moved = sum(j(box) != box for _ in range(400))
    assert 140 < moved < 260


# --- the pipeline: extract -> dataset -> inference ---------------------------------------------

_CLASSES_YAML = """
coarse: [valve, flange]
fine:
  valve_gate: valve
  valve_ball: valve
  flange: flange
fgc_groups: [valve]
"""


def _data_root(tmp_path, image: np.ndarray, boxes: list[tuple[str, tuple]]):
    canonical = tmp_path / "data" / "canonical"
    (canonical / "images").mkdir(parents=True)
    (canonical / "classes.yaml").write_text(_CLASSES_YAML)
    cv2.imwrite(str(canonical / "images" / "img0.png"), image)
    (canonical / "manifest.csv").write_text("id,source,split\nimg0,synthetic,train\n")
    names = ["valve_gate", "valve_ball", "flange"]
    coco = {
        "images": [{"id": 1, "file_name": "synthetic/img0.png", "width": 400, "height": 300}],
        "categories": [
            {"id": i, "name": n, "supercategory": "flange" if n == "flange" else "valve"}
            for i, n in enumerate(names)
        ],
        "annotations": [
            {
                "id": k,
                "image_id": 1,
                "category_id": names.index(c),
                "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
            }
            for k, (c, b) in enumerate(boxes)
        ],  # fmt: skip
    }
    (canonical / "annotations.coco.json").write_text(json.dumps(coco))
    return OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "run_dir": str(tmp_path / "runs"),
            "device": "cpu",
            "fgc": {
                "global_encoder": True,
                "imgsz": 32,
                "exclude_classes": [],
                "aug": {"hflip": 0.0, "rotate": 0.0, "jitter": 0.0, "subpart": {"p": 0.0}},
            },
        }
    )


_BOXES = [
    ("valve_gate", (40.3, 30.7, 90.2, 61.9)),
    ("valve_gate", (0.0, 120.0, 25.5, 160.0)),  # on the image edge
    ("valve_ball", (200.6, 100.1, 251.0, 150.4)),
    ("flange", (370.2, 260.4, 399.0, 299.0)),  # in the corner
]


def test_prep_crops_store_context_and_every_split_cuts_the_inference_view(tmp_path):
    from pipeline.extract_crops import ExtractCropsPipeline
    from utils.dataset import build_dataset

    image = _noise()
    cfg = _data_root(tmp_path, image, _BOXES)
    ExtractCropsPipeline(cfg).run()

    crops = tmp_path / "data" / "derived" / "crops_fgc" / "train"
    for fine in ("valve_gate", "valve_ball", "flange"):
        assert (crops / fine / BOXES_CSV).exists()
    gate = read_boxes(crops / "valve_gate")
    assert set(gate) == {"img0_0.png", "img0_1.png"}

    ds = build_dataset(cfg, "train", GLOBAL_GROUP)
    assert len(ds) == 4 and len(ds.stored) == 4
    ds.transform = lambda img: img  # compare the raw view, before resize / tensor
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, (_c, box) in enumerate(_sorted(_BOXES)):
        got = np.asarray(ds[i][0])
        assert np.array_equal(got, cut(rgb, box)), ds.samples[i][0].name


def _sorted(boxes):
    """_BOXES in the dataset's sample order (class order, then file name)."""
    order = ["valve_gate", "valve_ball", "flange"]
    named = [(c, b, f"img0_{k}.png") for k, (c, b) in enumerate(boxes)]
    named.sort(key=lambda t: (order.index(t[0]), t[2]))
    return [(c, b) for c, b, _ in named]


def test_train_split_jitters_val_does_not_and_old_crops_refuse_jitter(tmp_path):
    from pipeline.extract_crops import ExtractCropsPipeline
    from utils.dataset import build_dataset

    cfg = _data_root(tmp_path, _noise(), _BOXES)
    ExtractCropsPipeline(cfg).run()
    cfg.fgc.aug.box_jitter = {"p": 1.0}
    ds = build_dataset(cfg, "train", GLOBAL_GROUP)
    assert ds.jitter is not None and ds.jitter.p == 1.0
    sizes = {ds.jitter(ds.stored[ds.samples[0][0]].box) for _ in range(5)}
    assert len(sizes) > 1  # a new box per draw

    # The same crops read as val: no jitter (cfg.fgc.aug.box_jitter applies to train only).
    val_root = tmp_path / "data" / "derived" / "crops_fgc" / "val"
    (tmp_path / "data" / "derived" / "crops_fgc" / "train").rename(val_root)
    assert build_dataset(cfg, "val", GLOBAL_GROUP).jitter is None

    # Crops cut before the stored-context format: usable as they are, but not jittered.
    (val_root / "valve_gate" / BOXES_CSV).unlink()
    val_root.rename(tmp_path / "data" / "derived" / "crops_fgc" / "train")
    with pytest.raises(ValueError, match="re-run pid-prep-crops"):
        build_dataset(cfg, "train", GLOBAL_GROUP)
    cfg.fgc.aug.box_jitter = {"p": 0.0}
    assert len(build_dataset(cfg, "train", GLOBAL_GROUP)) == 4


def test_fgc_infer_classifies_the_box_with_the_margin():
    """Before crop_box, inference cut the bare detector box while training used a 10%
    margin; now both cut `crop_box.cut`."""
    from pipeline.fgc_infer import FGCInferPipeline
    from utils.types import BBox

    seen = {}
    pipe = FGCInferPipeline.__new__(FGCInferPipeline)
    pipe.cfg = OmegaConf.create({"fgc": {"route": {"min_score": 0.25}}})
    pipe._global = False

    class _Map:
        def needs_fgc(self, coarse):
            return True

    pipe._classmap = _Map()

    def classify(group, crop):
        seen["crop"] = crop
        return "x", 1.0

    pipe._classify_crop = classify
    image = _noise()
    box = BBox.from_xyxy(100 / 400, 50 / 300, 200 / 400, 150 / 300)
    pipe._resolve_fine_label("valve", box, 0.9, image, 400, 300)
    assert seen["crop"].shape[:2] == (120, 120)
    assert np.array_equal(seen["crop"], image[40:160, 90:210])


def test_augments_pickle_for_spawned_dataloader_workers():
    """`++fgc.workers=N` sends the dataset to worker processes; with spawn / forkserver
    (macOS, Python 3.14+) every augment must pickle. The global `random` module does not,
    so it is dropped and the worker's own (torch-reseeded) global `random` is used."""
    import pickle

    from utils.augment import SubpartWarp
    from utils.scan_degrade import ScanDegrade

    for aug in (BoxJitter(0.5), ScanDegrade(0.5), SubpartWarp(p=0.5)):
        back = pickle.loads(pickle.dumps(aug))
        assert back._rng is random and back.p == aug.p
    seeded = pickle.loads(pickle.dumps(BoxJitter(1.0, seed=7)))
    box = (0.0, 0.0, 10.0, 10.0)
    assert seeded(box) == BoxJitter(1.0, seed=7)(box)  # a seeded generator keeps its state
