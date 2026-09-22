"""Smoke test for pipeline.evaluation.EvaluationPipeline's wiring end-to-end (det_only mode).

Kept separate from the dependency-light test files: constructs a DictConfig by hand
(no @hydra.main invocation) over a synthetic data/ tree written under tmp_path, and
exercises validate() + compute_detection_metrics() + run() for real. det_only mode
never touches fgc.* (that import is lazy, inside compute_fgc_metrics), so this
does not require torch/timm/ultralytics/sahi to be installed — but we still guard on
hydra/omegaconf import so this file can be skipped wholesale in an environment that
lacks even those (e.g. a maximally minimal test run).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

from omegaconf import OmegaConf  # noqa: E402

from pipeline.evaluation import EvaluationPipeline  # noqa: E402


def _build_synthetic_data_root(tmp_path):
    data_root = tmp_path / "data"
    canonical = data_root / "canonical"
    canonical.mkdir(parents=True)

    classes_yaml = {
        "coarse": ["valve"],
        "fine": {"valve": "valve"},
        "fgc_groups": [],
        "rare_fine": [],
    }
    import yaml

    with (canonical / "classes.yaml").open("w") as f:
        yaml.safe_dump(classes_yaml, f, sort_keys=False)

    coco = {
        "images": [{"id": 1, "file_name": "synthetic/0001.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "valve", "supercategory": "valve"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
        ],
    }
    with (canonical / "annotations.coco.json").open("w") as f:
        json.dump(coco, f)

    import pandas as pd

    manifest = pd.DataFrame(
        [
            {
                "id": "0001",
                "source": "synthetic",
                "split": "test",
                "n_symbols": 1,
                "classes_present": "valve",
            }
        ]
    )
    manifest.to_csv(canonical / "manifest.csv", index=False)

    det_dir = data_root / "derived" / "detections" / "test"
    det_dir.mkdir(parents=True)
    # normalized YOLO box matching the GT bbox [10,10,20,20] in a 100x100 image
    # exactly: xyxy=(10,10,30,30) -> center=(20,20)/100=(0.2,0.2), w=h=20/100=0.2.
    (det_dir / "0001.txt").write_text("0 0.2 0.2 0.2 0.2 0.9\n")

    return data_root


def _build_cfg(data_root, run_dir):
    return OmegaConf.create(
        {
            "data_root": str(data_root),
            "run_dir": str(run_dir),
            "eval": {
                "mode": "det_only",
                "iou_thrs": [0.5, 0.75],
                "score_thr": 0.25,
                "per_class_ap": True,
                "topk": [1, 3],
                "split": "test",
            },
            "fgc": {"ckpt_dir": None},
        }
    )


def test_validate_passes_on_well_formed_synthetic_tree(tmp_path):
    data_root = _build_synthetic_data_root(tmp_path)
    cfg = _build_cfg(data_root, tmp_path / "run_out")

    pipeline = EvaluationPipeline(cfg)
    assert pipeline.validate() is True


def test_validate_raises_when_predictions_missing(tmp_path):
    data_root = _build_synthetic_data_root(tmp_path)
    # Remove the predictions so validate() must catch it with a clear message.
    det_txt = data_root / "derived" / "detections" / "test" / "0001.txt"
    det_txt.unlink()

    cfg = _build_cfg(data_root, tmp_path / "run_out")
    pipeline = EvaluationPipeline(cfg)

    with pytest.raises(FileNotFoundError, match="pid-detect-infer"):
        pipeline.validate()


def test_compute_detection_metrics_perfect_match(tmp_path):
    data_root = _build_synthetic_data_root(tmp_path)
    cfg = _build_cfg(data_root, tmp_path / "run_out")

    pipeline = EvaluationPipeline(cfg)
    result = pipeline.compute_detection_metrics()

    # Single GT, single matching prediction with IoU=1.0 -> perfect AP at every thr.
    assert result["map_50"] == pytest.approx(1.0)
    assert result["map_50_95"] == pytest.approx(1.0)
    valve_id = pipeline.classmap.coarse_id("valve")
    assert result["per_class_ap_50"][valve_id] == pytest.approx(1.0)


def test_run_writes_metrics_json(tmp_path):
    data_root = _build_synthetic_data_root(tmp_path)
    run_dir = tmp_path / "run_out"
    cfg = _build_cfg(data_root, run_dir)

    pipeline = EvaluationPipeline(cfg)
    pipeline.run()

    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()

    with metrics_path.open("r") as f:
        written = json.load(f)

    assert written["mode"] == "det_only"
    assert written["split"] == "test"
    assert written["detection"]["map_50"] == pytest.approx(1.0)
    assert "fgc" not in written  # det_only mode never runs compute_fgc_metrics
