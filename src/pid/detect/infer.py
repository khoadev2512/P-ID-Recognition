"""Run the detector over full-resolution diagrams with SAHI tiling + WBF merge.

Pipeline (report §4.2.2):
    full image -> SAHI sliced prediction (overlap-preserving, ref [40])
               -> per-patch YOLOv8 detections
               -> Weighted Boxes Fusion across overlapping patches (ref [41])
               -> diagram-level list[Detection] with coarse_class filled.

Output is the coarse symbol inventory consumed by fgc.infer.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pid.common.cli import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   1. build SAHI detection model from trained YOLOv8 weights
    #   2. get_sliced_prediction(image, slice size/overlap from cfg.detector.sahi)
    #   3. WBF-merge patch boxes (cfg.detector.wbf.{iou_thr, skip_box_thr})
    #   4. emit list[Detection] (coarse_class, score) per image
    raise NotImplementedError


if __name__ == "__main__":
    main()
