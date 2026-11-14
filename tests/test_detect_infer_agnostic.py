"""detector.agnostic_merge (PLAN B6e): two overlapping boxes of different labels on one symbol
become one box, labelled by its highest-scoring member; off (the default) the merge is per
label, exactly the library WBF the pipeline used before."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

pytest.importorskip("ensemble_boxes")
pytest.importorskip("ultralytics")
pytest.importorskip("sahi")
from ensemble_boxes import weighted_boxes_fusion  # noqa: E402

from pipeline.detect_infer import DetectInferPipeline, fuse_boxes  # noqa: E402
from utils.cli import CONFIG_DIR  # noqa: E402

A = [0.40, 0.40, 0.50, 0.50]
B = [0.405, 0.40, 0.505, 0.50]  # IoU with A ~0.9: one symbol seen twice
FAR = [0.80, 0.80, 0.90, 0.90]


def test_two_overlapping_labels_merge_only_when_agnostic():
    boxes, scores, labels = [A, B], [0.6, 0.9], [0.0, 3.0]
    off = fuse_boxes(boxes, scores, labels, 0.55, 0.001)
    on = fuse_boxes(boxes, scores, labels, 0.55, 0.001, agnostic=True)
    assert len(off[0]) == 2 and sorted(off[2].tolist()) == [0.0, 3.0]
    assert len(on[0]) == 1
    assert on[2][0] == 3.0  # the label of the 0.9 box, not of the 0.6 one
    assert on[1][0] == pytest.approx(0.75)  # WBF `avg` score of the two


def test_label_follows_the_highest_scoring_member_whatever_the_order():
    on = fuse_boxes([B, A, FAR], [0.3, 0.8, 0.5], [5.0, 1.0, 2.0], 0.55, 0.001, agnostic=True)
    assert len(on[0]) == 2
    assert on[2].tolist() == [1.0, 2.0]  # best score first: the merged pair (0.55), then FAR


def test_default_is_the_library_wbf_unchanged():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 0.9, (40, 2))
    wh = rng.uniform(0.02, 0.1, (40, 2))
    boxes = np.clip(np.hstack([xy, xy + wh]), 0, 1).tolist()
    scores = rng.uniform(0.01, 1, 40).tolist()
    labels = rng.integers(0, 4, 40).astype(float).tolist()
    want = weighted_boxes_fusion([boxes], [scores], [labels], iou_thr=0.3, skip_box_thr=0.05)
    got = fuse_boxes(boxes, scores, labels, 0.3, 0.05)
    for w, g in zip(want, got, strict=True):
        np.testing.assert_array_equal(w, g)


def test_agnostic_on_one_label_is_exactly_the_library_wbf():
    """The agnostic loop repeats the library's: with a single label it must give the same
    boxes and scores, in the same order (and skip_box_thr / empty input the same way)."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 0.9, (60, 2))
    boxes = np.hstack([xy, xy + rng.uniform(0.02, 0.1, (60, 2))]).tolist()
    scores = rng.uniform(0.01, 1, 60).tolist()
    labels = [2.0] * 60
    want = weighted_boxes_fusion([boxes], [scores], [labels], iou_thr=0.3, skip_box_thr=0.05)
    got = fuse_boxes(boxes, scores, labels, 0.3, 0.05, agnostic=True)
    assert len(got[0]) < 60  # something merged
    for w, g in zip(want, got, strict=True):
        np.testing.assert_allclose(w, g, rtol=0, atol=1e-6)
    empty = fuse_boxes([], [], [], 0.55, 0.001, agnostic=True)
    assert [len(x) for x in empty] == [0, 0, 0]


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _L(xyxy), _L(conf), _L(cls)


class _L(list):
    def tolist(self):
        return list(self)


class _FakeYolo:
    """One tile's prediction: two overlapping boxes of different classes."""

    def __init__(self):
        self.calls = []

    def predict(self, image, **kw):
        self.calls.append(kw)
        res = type("R", (), {})()
        res.boxes = _Boxes([[40, 40, 50, 50], [40.5, 40, 50.5, 50]], [0.6, 0.9], [0.0, 1.0])
        return [res]


def _pipe(tmp_path, **detector):
    canonical = tmp_path / "canonical"
    canonical.mkdir(parents=True)
    (canonical / "classes.yaml").write_text(
        "coarse: [valve, pump]\nfine: {gate_valve: valve, pump_c: pump}\nfgc_groups: []\n"
    )
    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path), "run_dir": str(tmp_path / "runs"),
            "detector": {"sahi": {"slice_size": 100, "overlap_ratio": 0.0},
                         "wbf": {"iou_thr": 0.55, "skip_box_thr": 0.001}, **detector},
        }
    )  # fmt: skip
    return DetectInferPipeline(cfg)


@pytest.mark.parametrize(
    ("detector", "agnostic", "n", "classes"),
    [({}, False, 2, ["pump", "valve"]), ({"agnostic_merge": True}, True, 1, ["pump"])],
)
def test_detect_image_passes_the_switch_to_nms_and_wbf(tmp_path, detector, agnostic, n, classes):
    pipe = _pipe(tmp_path, **detector)
    model = _FakeYolo()
    image = np.zeros((100, 100, 3), np.uint8)  # one tile
    dets = pipe._detect_image(model, image, 100, 100)
    assert [c["agnostic_nms"] for c in model.calls] == [agnostic]
    assert len(dets) == n and sorted(d.coarse_class for d in dets) == classes


def test_agnostic_merge_is_off_and_not_in_config_yaml(tmp_path):
    """Off unless asked for with `++detector.agnostic_merge=true`: config.yaml is untouched."""
    with open(Path(CONFIG_DIR) / "config.yaml") as fh:
        assert "agnostic_merge" not in yaml.safe_load(fh)["detector"]
    assert _pipe(tmp_path)._agnostic_merge() is False
