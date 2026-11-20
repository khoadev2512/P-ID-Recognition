"""Closed-set use of the global encoder `__global__.pt` (protocol C, PLAN B6c).

pid-fgc-infer   one head classifies every routed detection over the checkpoint's own
                fine_classes; fgc.route.by_coarse restricts it to the Stage-1 family;
                no "Other" gate (that is the one-shot path's job)
pid-evaluate    eval.mode=fgc_only scores the head on GT crops without detector output:
                top-k, macro-F1, per-class P/R/F1, confusion, per-family breakdown;
                det_only scores a fine-label (one-stage, 32-class) detector per fine
                class; predictions without a vocabulary label are left out of mAP
pid-detect-infer  a fine-label checkpoint writes fine ids, and its class names must be
                the label set detector.label_level names
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from pipeline.evaluation import EvaluationPipeline  # noqa: E402
from pipeline.fgc_infer import FGCInferPipeline  # noqa: E402
from utils import fgc_metrics  # noqa: E402
from utils.classmap import GLOBAL_GROUP, OTHER_CLASS_NAME  # noqa: E402
from utils.model import build_model  # noqa: E402
from utils.types import BBox  # noqa: E402

_CLASSES_YAML = """
coarse: [valve, instrument, flange]
fine:
  valve_gate: valve
  valve_ball: valve
  bubble_field: instrument
  bubble_logic: instrument
  flange: flange
fgc_groups: [valve, instrument]
"""
_FINE = ["valve_gate", "valve_ball", "bubble_field", "bubble_logic", "flange"]
_COARSE_OF = {
    "valve_gate": "valve",
    "valve_ball": "valve",
    "bubble_field": "instrument",
    "bubble_logic": "instrument",
    "flange": "flange",
}


def _cfg(tmp_path, **over):
    canonical = tmp_path / "data" / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    (canonical / "classes.yaml").write_text(_CLASSES_YAML)
    cfg = OmegaConf.create(
        {
            "data_root": str(tmp_path / "data"),
            "run_dir": str(tmp_path / "runs"),
            "device": "cpu",
            "fgc": {
                "global_encoder": True,
                "metric_learning": True,
                "ckpt_dir": str(tmp_path / "ckpt"),
                "exclude_classes": [],
                "route": {"min_score": 0.25, "other_min_cosine": 0.35},
                "arcface": {"m": 0.5, "s": 30.0, "embed_dim": 32},
                "backbone": "resnet18",
                "pretrained": False,
                "imgsz": 32,
                "aug": {"subpart": {"p": 0.0}},
            },
            "detector": {"label_level": "coarse"},
            "eval": {
                "mode": "fgc_only",
                "split": "val",
                "topk": [1, 3],
                "iou_thrs": [0.5, 0.75],
                "score_thr": 0.25,
                "per_class_ap": True,
            },
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(over))


def _save_global(cfg, tmp_path, fine_classes=_FINE, excluded=()):
    """A real (untrained) ArcFace global checkpoint, saved the way pid-fgc-train does."""
    torch.manual_seed(0)
    model = build_model(cfg, GLOBAL_GROUP, len(fine_classes))
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "fine_classes": list(fine_classes),
            "excluded_classes": list(excluded),
            "epoch": 0,
            "val_acc": 0.0,
        },
        ckpt_dir / f"{GLOBAL_GROUP}.pt",
    )
    return model


class _FixedLogits(torch.nn.Module):
    """A head that always scores the classes in the given order."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)


def _infer_with_fixed_head(tmp_path, logits, **over):
    pipe = FGCInferPipeline(_cfg(tmp_path, **over))
    pipe._models = {GLOBAL_GROUP: _FixedLogits(logits)}
    pipe._fine_classes_per_group[GLOBAL_GROUP] = list(_FINE)
    return pipe


_IMG = np.full((100, 100, 3), 255, dtype=np.uint8)
_BOX = BBox(x=0.5, y=0.5, w=0.2, h=0.2)


# ------------------------------------------------------------------ #
# fgc_infer
# ------------------------------------------------------------------ #


def test_infer_accepts_a_global_encoder_and_needs_its_checkpoint(tmp_path):
    cfg = _cfg(tmp_path)
    det = tmp_path / "data" / "derived" / "detections" / "val"
    det.mkdir(parents=True)
    (det / "a.txt").write_text("")
    (tmp_path / "ckpt").mkdir()
    with pytest.raises(FileNotFoundError, match="__global__.pt"):
        FGCInferPipeline(cfg).validate()
    _save_global(cfg, tmp_path)
    assert FGCInferPipeline(cfg).validate() is True


def test_infer_refuses_detections_from_a_fine_label_detector(tmp_path):
    pipe = FGCInferPipeline(_cfg(tmp_path, detector={"label_level": "fine"}))
    with pytest.raises(ValueError, match="det_only"):
        pipe.validate()


def test_infer_takes_the_columns_from_the_checkpoint(tmp_path):
    """A leave-out checkpoint has fewer columns than classes.yaml; the head is rebuilt
    with its own list, not one recomputed from this command's fgc.exclude_classes."""
    cfg = _cfg(tmp_path)
    kept = ["valve_gate", "bubble_field", "flange"]
    _save_global(cfg, tmp_path, kept, excluded=["valve_ball", "bubble_logic"])
    pipe = FGCInferPipeline(cfg)
    assert list(pipe.models) == [GLOBAL_GROUP]
    assert pipe._fine_classes_per_group[GLOBAL_GROUP] == kept


def test_global_head_classifies_every_coarse_class_over_the_whole_vocabulary(tmp_path):
    """flange is 1:1 and not in fgc_groups: per-group mode never classifies it; the
    global head does, and may answer with a class of another family."""
    pipe = _infer_with_fixed_head(tmp_path, [0.0, 5.0, 1.0, 0.0, 0.0])
    label, _ = pipe._resolve_fine_label("flange", _BOX, 0.9, _IMG, 100, 100)
    assert label == "valve_ball"


def test_by_coarse_restricts_the_global_head_to_the_stage1_family(tmp_path):
    pipe = _infer_with_fixed_head(
        tmp_path, [0.0, 5.0, 1.0, 0.0, 0.0], fgc={"route": {"by_coarse": True}}
    )
    label, score = pipe._resolve_fine_label("instrument", _BOX, 0.9, _IMG, 100, 100)
    assert label == "bubble_field"
    assert score == pytest.approx(1 / (1 + np.exp(-1.0)))  # softmax over the 2 bubbles


def test_below_min_score_the_global_head_is_not_used(tmp_path):
    pipe = _infer_with_fixed_head(tmp_path, [0.0, 5.0, 1.0, 0.0, 0.0])
    assert pipe._resolve_fine_label("flange", _BOX, 0.1, _IMG, 100, 100) == ("flange", 0.1)
    assert pipe._resolve_fine_label("valve", _BOX, 0.1, _IMG, 100, 100)[0] == "valve"


def test_global_head_has_no_other_gate(tmp_path):
    """Closed-set: a threshold no cosine can reach rejects nothing."""
    cfg = _cfg(tmp_path, fgc={"route": {"other_min_cosine": 1.5}})
    _save_global(cfg, tmp_path)
    pipe = FGCInferPipeline(cfg)
    label, _ = pipe._resolve_fine_label("valve", _BOX, 0.9, _IMG, 100, 100)
    assert label in _FINE and label != OTHER_CLASS_NAME


def test_infer_run_writes_global_fine_ids(tmp_path):
    cfg = _cfg(tmp_path)
    data = tmp_path / "data"
    (data / "canonical" / "images").mkdir(parents=True)
    Image.fromarray(_IMG).save(data / "canonical" / "images" / "a.png")
    (data / "canonical" / "manifest.csv").write_text("id,split\na,val\n")
    det = data / "derived" / "detections" / "val"
    det.mkdir(parents=True)
    (det / "a.txt").write_text("2 0.5 0.5 0.2 0.2 0.9\n0 0.3 0.3 0.1 0.1 0.1\n")
    _save_global(cfg, tmp_path)  # validate() wants the file; the head is replaced below
    pipe = _infer_with_fixed_head(tmp_path, [0.0, 0.0, 0.0, 9.0, 0.0])
    pipe.run()
    rows = (data / "derived" / "detections_fgc" / "val" / "a.txt").read_text().split("\n")
    assert rows[0].split()[0] == str(_FINE.index("bubble_logic"))  # flange box, refined
    assert rows[1].split()[0] == "-1"  # valve box below min_score: un-refined


# ------------------------------------------------------------------ #
# evaluation, fgc_only
# ------------------------------------------------------------------ #


def _crop_tree(tmp_path, per_class=3):
    rng = np.random.default_rng(0)
    for name in _FINE:
        d = tmp_path / "data" / "derived" / "crops_fgc" / "val" / name
        d.mkdir(parents=True)
        for i in range(per_class):
            arr = rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")


def _eval(tmp_path, cfg):
    (tmp_path / "data" / "canonical" / "manifest.csv").write_text("id,split\na,val\n")
    (tmp_path / "data" / "canonical" / "annotations.coco.json").write_text("{}")
    return EvaluationPipeline(cfg)  # writes to cfg.run_dir outside Hydra


def test_fgc_only_scores_the_global_head_without_detector_output(tmp_path):
    cfg = _cfg(tmp_path)
    _save_global(cfg, tmp_path)
    _crop_tree(tmp_path)
    pipe = _eval(tmp_path, cfg)
    pipe.run()
    out = json.loads((tmp_path / "runs" / "metrics.json").read_text())
    assert "detection" not in out
    fgc = out["fgc"]
    assert fgc["encoder"] == GLOBAL_GROUP and fgc["n_classes"] == 5 and fgc["n_samples"] == 15
    matrix = np.array(fgc["confusion"]["matrix"])
    assert fgc["confusion"]["labels"] == _FINE and matrix.shape == (5, 5)
    assert matrix.sum(axis=1).tolist() == [3] * 5
    assert fgc["top1"] == pytest.approx(np.trace(matrix) / 15)
    assert fgc["top3"] >= fgc["top1"]
    assert {c["support"] for c in fgc["per_class"].values()} == {3}
    assert set(fgc["families"]) == {"valve", "instrument", "flange"}
    assert 0.0 <= fgc["cross_family_error_rate"] <= 1.0
    assert "mean_max_cosine" in fgc["other_gate"]
    assert (tmp_path / "runs" / f"confusion_{GLOBAL_GROUP}.png").exists()


def test_fgc_only_follows_a_leave_out_checkpoint(tmp_path):
    """The held-out classes' crops exist (P1 needs them) but have no column: they are
    not scored, whatever fgc.exclude_classes the eval command carries."""
    cfg = _cfg(tmp_path)
    kept = ["valve_gate", "bubble_field", "flange"]
    _save_global(cfg, tmp_path, kept, excluded=["valve_ball", "bubble_logic"])
    _crop_tree(tmp_path)
    fgc = _eval(tmp_path, cfg).compute_fgc_metrics()
    assert fgc["n_classes"] == 3 and fgc["n_samples"] == 9
    assert fgc["confusion"]["labels"] == kept


def test_fgc_only_needs_a_checkpoint_dir(tmp_path):
    pipe = _eval(tmp_path, _cfg(tmp_path, fgc={"ckpt_dir": None}))
    with pytest.raises(ValueError, match="fgc.ckpt_dir"):
        pipe.validate()


def test_missing_global_checkpoint_is_an_error_not_a_skip(tmp_path):
    (tmp_path / "ckpt").mkdir()
    pipe = _eval(tmp_path, _cfg(tmp_path))
    with pytest.raises(FileNotFoundError, match="__global__.pt"):
        pipe.compute_fgc_metrics()


# ------------------------------------------------------------------ #
# evaluation, det_only on a fine-label detector; sentinel predictions
# ------------------------------------------------------------------ #


def _det_tree(tmp_path, mode, pred_rows):
    data = tmp_path / "data"
    coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        "categories": [
            {"id": 1, "name": "valve_ball", "supercategory": "valve"},
            {"id": 2, "name": "flange", "supercategory": "flange"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [60, 60, 20, 20]},
        ],
    }
    (data / "canonical" / "annotations.coco.json").write_text(json.dumps(coco))
    (data / "canonical" / "manifest.csv").write_text("id,split\na,val\n")
    sub = "detections" if mode == "det_only" else "detections_fgc"
    det = data / "derived" / sub / "val"
    det.mkdir(parents=True)
    (det / "a.txt").write_text("\n".join(pred_rows) + "\n")


def test_det_only_scores_a_fine_label_detector_per_fine_class(tmp_path):
    cfg = _cfg(tmp_path, eval={"mode": "det_only"}, detector={"label_level": "fine"})
    _det_tree(tmp_path, "det_only", ["1 0.2 0.2 0.2 0.2 0.9", "4 0.7 0.7 0.2 0.2 0.8"])
    pipe = EvaluationPipeline(cfg)
    result = pipe.compute_detection_metrics()
    assert result["map_50"] == pytest.approx(1.0)
    assert set(result["per_class_ap_50"]) == {_FINE.index("valve_ball"), _FINE.index("flange")}


def test_unlabelled_predictions_are_left_out_of_map(tmp_path):
    """An Other (-2) or un-refined (-1) box predicts no class; scoring it as a class of
    its own added an AP of 0 to the mean."""
    cfg = _cfg(tmp_path, eval={"mode": "det_plus_fgc"})
    rows = ["1 0.2 0.2 0.2 0.2 0.9", "4 0.7 0.7 0.2 0.2 0.8", "-2 0.5 0.5 0.1 0.1 0.9"]
    _det_tree(tmp_path, "det_plus_fgc", rows + ["-1 0.4 0.4 0.1 0.1 0.9"])
    result = EvaluationPipeline(cfg).compute_detection_metrics()
    assert result["map_50"] == pytest.approx(1.0)
    assert all(k >= 0 for k in result["per_class_ap_50"])


# ------------------------------------------------------------------ #
# detect_infer, fine label level
# ------------------------------------------------------------------ #


def test_detect_infer_checks_the_checkpoint_class_names(tmp_path):
    pytest.importorskip("ultralytics")
    pytest.importorskip("sahi")
    from pipeline.detect_infer import DetectInferPipeline

    class _Yolo:
        def __init__(self, names):
            self.names = dict(enumerate(names))

    pipe = DetectInferPipeline(_cfg(tmp_path, detector={"label_level": "fine"}))
    pipe._check_model_names(_Yolo(_FINE))
    with pytest.raises(ValueError, match="label_level=fine"):
        pipe._check_model_names(_Yolo(["valve", "instrument", "flange"]))
    det = pipe._detection(BBox.from_xyxy(0, 0, 10, 10), 1, 0.9)
    assert (det.fine_class, det.coarse_class) == ("valve_ball", "valve")


def test_detect_infer_writes_fine_ids_for_a_fine_detector(tmp_path):
    pytest.importorskip("ultralytics")
    pytest.importorskip("sahi")
    from pipeline.detect_infer import DetectInferPipeline, _write_detections_txt

    pipe = DetectInferPipeline(_cfg(tmp_path, detector={"label_level": "fine"}))
    det = pipe._detection(BBox.from_xyxy(10, 10, 30, 30), 4, 0.9)
    out = tmp_path / "a.txt"
    _write_detections_txt(out, [det], pipe.classmap, 100, 100, fine=True)
    assert out.read_text().split()[0] == "4"


# ------------------------------------------------------------------ #
# fgc_metrics
# ------------------------------------------------------------------ #


def test_family_breakdown_counts_errors_that_leave_the_family():
    y_true = ["valve_gate", "valve_gate", "valve_ball", "flange"]
    y_pred = ["valve_gate", "valve_ball", "bubble_field", "flange"]
    fam = fgc_metrics.family_breakdown(y_true, y_pred, _COARSE_OF)
    assert fam["valve"] == {"n_samples": 3, "accuracy": 1 / 3, "cross_family": 1 / 3}
    assert fam["flange"] == {"n_samples": 1, "accuracy": 1.0, "cross_family": 0.0}


def test_macro_f1_weighs_every_class_equally():
    y_true = ["a"] * 9 + ["b"]
    y_pred = ["a"] * 10
    assert fgc_metrics.macro_f1(y_true, y_pred, ["a", "b"]) == pytest.approx((18 / 19) / 2)
    prf = fgc_metrics.per_class_prf(y_true, y_pred, ["a", "b"])
    assert prf["b"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
