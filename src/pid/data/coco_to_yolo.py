"""Convert canonical COCO annotations into YOLO txt label sets.

Emits BOTH label sets from the same source, selected by cfg.data.label_level:
    coarse -> derived/yolo_coarse/  (Stage 1 detector training; report §4.2.2)
    fine   -> derived/yolo_fine/    (full-fine-label ablation; §4.2.2)

Class ids come from common.classmap (coarse_id / fine_id) so ids stay stable across
runs. Boxes are normalised to YOLO center-xywh in [0, 1].
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pid.common.cli import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   1. load classmap + COCO
    #   2. for each image, write <stem>.txt with lines: <class_id> <cx> <cy> <w> <h>
    #      using coarse_id or fine_id per cfg.data.label_level
    #   3. write the Ultralytics data.yaml (train/val paths, names) for this label set
    raise NotImplementedError


if __name__ == "__main__":
    main()
