"""Train the YOLOv8 detector on the tiled dataset.

Label set is chosen by cfg.detector.label_level (coarse default; fine for the ablation
in §4.2.2). Thin wrapper over ultralytics.YOLO(...).train — this module owns config
plumbing and result-path bookkeeping, not the training loop.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pid.common.cli import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   1. resolve data.yaml (from data.coco_to_yolo) for the chosen label level
    #   2. YOLO(cfg.detector.weights).train(data=..., imgsz=cfg.detector.imgsz,
    #        epochs=..., augment params from cfg.detector.aug, ...)
    #   3. record run dir for downstream infer/eval
    raise NotImplementedError


if __name__ == "__main__":
    main()
