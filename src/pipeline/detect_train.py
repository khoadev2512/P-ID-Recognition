"""Train the YOLOv8 detector on the tiled dataset.

Label set is chosen by cfg.detector.label_level (coarse default; fine for the ablation
in §4.2.2) via whichever labels pipeline.tiling baked into derived/tiles/data.yaml.
Thin wrapper over ultralytics.YOLO(...).train — this module owns config plumbing and
result-path bookkeeping (mirrors PID_Symbol_Detection's YOLOTrainer), not a hand-rolled
training loop.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

from pipeline.base import BasePipeline
from utils.cli import CONFIG_DIR

logger = logging.getLogger(__name__)


class DetectTrainPipeline(BasePipeline):
    """Fine-tunes a YOLOv8 detector on the tiled dataset (derived/tiles/data.yaml)."""

    def validate(self) -> bool:
        data_yaml = self.paths.tiles / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(
                f"{data_yaml} not found. Run `pid-prep-tiles` first to build the tiled "
                "YOLO dataset before training the detector."
            )
        return True

    def run(self) -> None:
        self.validate()
        data_yaml = self.paths.tiles / "data.yaml"

        model = YOLO(str(self.cfg.detector.weights))
        try:
            model.train(
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

        best = getattr(getattr(model, "trainer", None), "best", None)
        if best is None:
            # Best-effort fallback: ultralytics' default run name is "train" and
            # project=self.output_dir is a fresh Hydra run dir, so this is the first run.
            best = self.output_dir / "train" / "weights" / "best.pt"
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
