"""Unit tests for interrupted-training resume wiring (Colab split sessions).

Covers the parts that don't need a real training run:
  - detector: resume checkpoint path resolution + validate() error messaging, without
    invoking ultralytics YOLO (imported lazily / not at all here).
  - FGC: FGCTrainPipeline._maybe_resume restores start_epoch / best_val_acc from a saved
    checkpoint, and no-ops correctly when resume_from is unset or the file is missing.

torch is needed for the FGC checkpoint round-trip; guarded like the other heavy tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")

from omegaconf import OmegaConf  # noqa: E402

# ------------------------------------------------------------------ #
# detector resume path wiring (no YOLO / no torch needed)
# ------------------------------------------------------------------ #


def _detector_cfg(tmp_path, **detector_overrides):
    cfg = {
        "data_root": str(tmp_path / "data"),
        "device": "cpu",
        "run_dir": str(tmp_path / "runs"),
        "data": {
            "coco_json": "x",
            "classes_yaml": str(tmp_path / "classes.yaml"),
            "manifest_csv": "m",
        },
        "detector": {
            "weights": "models/base_yolo_models/yolov8n.pt",
            "epochs": 100,
            "batch": 16,
            "imgsz": 1024,
            "resume": False,
            "resume_dir": None,
            "aug": {"mosaic": 1.0, "degrees": 0.0, "hsv_v": 0.4, "scale": 0.5},
        },
    }
    cfg["detector"].update(detector_overrides)
    # A minimal classes.yaml so BasePipeline.classmap (lazy) never blocks construction.
    import yaml

    (tmp_path / "classes.yaml").write_text(
        yaml.safe_dump({"coarse": ["valve"], "fine": {"valve": "valve"}, "fgc_groups": []})
    )
    return OmegaConf.create(cfg)


def test_resume_ckpt_points_at_last_pt(tmp_path):
    from pipeline.detect_train import DetectTrainPipeline

    run_dir = tmp_path / "runs" / "2026-01-01_10-00-00"
    cfg = _detector_cfg(tmp_path, resume=True, resume_dir=str(run_dir))
    pipe = DetectTrainPipeline(cfg)
    assert pipe._resume_ckpt() == run_dir / "train" / "weights" / "last.pt"


def test_resume_without_resume_dir_raises(tmp_path):
    from pipeline.detect_train import DetectTrainPipeline

    cfg = _detector_cfg(tmp_path, resume=True, resume_dir=None)
    pipe = DetectTrainPipeline(cfg)
    with pytest.raises(ValueError, match="resume_dir"):
        pipe._resume_ckpt()


def test_validate_resume_missing_last_pt_raises(tmp_path):
    from pipeline.detect_train import DetectTrainPipeline

    run_dir = tmp_path / "runs" / "empty_run"
    cfg = _detector_cfg(tmp_path, resume=True, resume_dir=str(run_dir))
    pipe = DetectTrainPipeline(cfg)
    with pytest.raises(FileNotFoundError, match="no interrupted checkpoint"):
        pipe.validate()


def test_validate_resume_ok_when_last_pt_present(tmp_path):
    from pipeline.detect_train import DetectTrainPipeline

    run_dir = tmp_path / "runs" / "run1"
    weights = run_dir / "train" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"stub")
    cfg = _detector_cfg(tmp_path, resume=True, resume_dir=str(run_dir))
    pipe = DetectTrainPipeline(cfg)
    assert pipe.validate() is True


# ------------------------------------------------------------------ #
# FGC resume (torch checkpoint round-trip)
# ------------------------------------------------------------------ #

torch = pytest.importorskip("torch")


class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):  # pragma: no cover - not exercised here
        return self.fc(x)


def _fgc_pipeline(tmp_path, resume_from=None):
    """Build FGCTrainPipeline without BasePipeline.__init__ side effects."""
    from pipeline.fgc_train import FGCTrainPipeline

    pipe = object.__new__(FGCTrainPipeline)
    pipe.cfg = OmegaConf.create({"fgc": {"resume_from": resume_from}})
    return pipe


def _save_ckpt(path, epoch, val_acc, model, optimizer):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
        },
        path,
    )


def test_maybe_resume_noop_when_resume_from_unset(tmp_path):
    pipe = _fgc_pipeline(tmp_path, resume_from=None)
    model = _TinyNet()
    opt = torch.optim.Adam(model.parameters())
    start_epoch, best = pipe._maybe_resume("valve", model, opt, torch.device("cpu"), -1.0)
    assert start_epoch == 0
    assert best == pytest.approx(-1.0)


def test_maybe_resume_noop_when_group_ckpt_missing(tmp_path):
    ckpt_dir = tmp_path / "fgc_checkpoints"
    ckpt_dir.mkdir()
    pipe = _fgc_pipeline(tmp_path, resume_from=str(ckpt_dir))
    model = _TinyNet()
    opt = torch.optim.Adam(model.parameters())
    # No valve.pt in the dir -> start fresh.
    start_epoch, best = pipe._maybe_resume("valve", model, opt, torch.device("cpu"), -1.0)
    assert start_epoch == 0


def test_maybe_resume_restores_epoch_and_best(tmp_path):
    ckpt_dir = tmp_path / "fgc_checkpoints"
    ckpt_dir.mkdir()
    src = _TinyNet()
    src_opt = torch.optim.Adam(src.parameters())
    _save_ckpt(ckpt_dir / "valve.pt", epoch=29, val_acc=0.87, model=src, optimizer=src_opt)

    pipe = _fgc_pipeline(tmp_path, resume_from=str(ckpt_dir))
    model = _TinyNet()
    opt = torch.optim.Adam(model.parameters())
    start_epoch, best = pipe._maybe_resume("valve", model, opt, torch.device("cpu"), -1.0)

    # Saved at 0-based epoch 29 -> resume at 30; best carried forward.
    assert start_epoch == 30
    assert best == pytest.approx(0.87)
    # Weights actually loaded (match the source net).
    for p_loaded, p_src in zip(model.parameters(), src.parameters(), strict=True):
        assert torch.allclose(p_loaded, p_src)
