"""Train the YOLOv8 detector on the tiled dataset.

Label set is chosen by cfg.detector.label_level (coarse default; fine for the ablation
in §4.2.2) via whichever labels pipeline.tiling baked into derived/tiles/data.yaml.
Thin wrapper over ultralytics.YOLO(...).train — this module owns config plumbing and
result-path bookkeeping (mirrors PID_Symbol_Detection's YOLOTrainer), not a hand-rolled
training loop.

Scan-like degradation (`utils.scan_degrade`, off unless `detector.aug.degrade.p > 0`) has no
Ultralytics hyperparameter, so it comes in through `ScanDegradeTrainer`: the train dataset
degrades each image as it is loaded, before mosaic and the geometric augment (no box moves,
and the mosaic's grey padding is never taken for ink). Validation images stay clean.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

from pipeline.base import BasePipeline
from utils.cli import CONFIG_DIR
from utils.scan_degrade import ScanDegrade, gray_of, stroke_width

logger = logging.getLogger(__name__)


class ScanDegradeDataset(YOLODataset):
    """A YOLODataset whose images are degraded as they are loaded. Only the image changes, so
    the labels hold. The clean image is what Ultralytics caches / buffers; each load draws a
    new degradation. The stroke width of an image is measured once (per worker) and kept."""

    scan_degrade: ScanDegrade

    def load_image(self, i, rect_mode=True, resize_short=False):
        im, hw0, hw = super().load_image(i, rect_mode, resize_short)
        if not self.scan_degrade.fires():
            return im, hw0, hw
        widths = self.__dict__.setdefault("stroke_widths", {})
        if (i, im.shape) not in widths:
            widths[i, im.shape] = stroke_width(gray_of(im))
        return self.scan_degrade.apply(im, widths[i, im.shape]), hw0, hw


class ScanDegradeTrainer(DetectionTrainer):
    """DetectionTrainer whose train dataset is a `ScanDegradeDataset`. Pass it to
    `YOLO.train(trainer=partial(ScanDegradeTrainer, scan_degrade=...))`."""

    def __init__(self, *args, scan_degrade: ScanDegrade, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scan_degrade = scan_degrade

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            if type(dataset) is not YOLODataset:
                raise TypeError(f"scan degradation expects a YOLODataset, got {type(dataset)}")
            dataset.__class__ = ScanDegradeDataset  # built (and cached) clean, degraded on load
            dataset.scan_degrade = self.scan_degrade
        return dataset


class DetectTrainPipeline(BasePipeline):
    """Fine-tunes a YOLOv8 detector on the tiled dataset (derived/tiles/data.yaml)."""

    def validate(self) -> bool:
        if bool(self.cfg.detector.get("resume", False)):
            # Resume doesn't read data.yaml (Ultralytics restores it from the original
            # run's args.yaml); it needs the earlier run's last.pt instead.
            last = self._resume_ckpt()
            if not last.exists():
                raise FileNotFoundError(
                    f"detector.resume=true but no interrupted checkpoint at {last}. "
                    "Set detector.resume_dir to a previous run dir containing "
                    "train/weights/last.pt."
                )
            return True

        data_yaml = self.paths.tiles / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(
                f"{data_yaml} not found. Run `pid-prep-tiles` first to build the tiled "
                "YOLO dataset before training the detector."
            )
        return True

    def _resume_ckpt(self) -> Path:
        """`last.pt` of the run pointed at by detector.resume_dir (the interrupted run)."""
        resume_dir = self.cfg.detector.get("resume_dir", None)
        if resume_dir is None:
            raise ValueError(
                "detector.resume=true requires detector.resume_dir=<previous run dir>, "
                "e.g. detector.resume_dir=runs/2026-01-01_10-00-00"
            )
        return Path(str(resume_dir)) / "train" / "weights" / "last.pt"

    def _trainer(self):
        """The Ultralytics trainer to use: None (the default one) unless scan degradation is on.
        Resuming does not restore it: pass the same `detector.aug.degrade` override again."""
        degrade = ScanDegrade.from_cfg(self.cfg.detector.aug.get("degrade"))
        if degrade is None:
            return None
        logger.info("Scan degradation on for training images: %r", degrade)
        return partial(ScanDegradeTrainer, scan_degrade=degrade)

    def run(self) -> None:
        self.validate()
        trainer = self._trainer()

        if bool(self.cfg.detector.get("resume", False)):
            # True Ultralytics resume: reload last.pt AND its optimizer/LR/epoch state,
            # then continue the SAME run (Ultralytics reads epochs/data/aug back from the
            # original run's args.yaml, so we must NOT re-pass them here). This is what
            # makes split "train 6h, continue later" sessions pick up mid-schedule rather
            # than restarting the LR curve from scratch.
            last = self._resume_ckpt()
            logger.info("Resuming detector training from %s", last)
            model = YOLO(str(last))
            try:
                model.train(trainer=trainer, resume=True)
            except Exception:
                logger.exception("Detector training (resume) failed")
                raise
            self._log_best(model)
            return

        data_yaml = self.paths.tiles / "data.yaml"
        model = YOLO(str(self.cfg.detector.weights))
        try:
            model.train(
                trainer=trainer,
                data=str(data_yaml),
                epochs=self.cfg.detector.epochs,
                batch=self.cfg.detector.batch,
                imgsz=self.cfg.detector.imgsz,
                device=("0" if self.cfg.device == "cuda" else "cpu"),
                project=str(self.output_dir),
                mosaic=self.cfg.detector.aug.mosaic,
                degrees=self.cfg.detector.aug.degrees,
                hsv_v=self.cfg.detector.aug.hsv_v,
                scale=self.cfg.detector.aug.scale,
            )
        except Exception:
            logger.exception("Detector training failed")
            raise
        self._log_best(model)

    def _log_best(self, model: YOLO) -> None:
        """Log the best-weights path for the downstream infer/eval `detector.ckpt=` pointer."""
        best = getattr(getattr(model, "trainer", None), "best", None)
        if best is None:
            # Best-effort fallback when the trainer didn't expose `best` (e.g. a resume
            # that finished immediately): the weights live under the run dir either way —
            # the original run dir on resume, or this fresh Hydra run dir on a first run.
            run_dir = self.cfg.detector.get("resume_dir", None) or self.output_dir
            best = Path(str(run_dir)) / "train" / "weights" / "best.pt"
        logger.info(
            "Training complete. Best weights: %s -- pass this as detector.ckpt=... "
            "for pid-detect-infer / pid-eval.",
            best,
        )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    DetectTrainPipeline(cfg).run()


if __name__ == "__main__":
    main()
