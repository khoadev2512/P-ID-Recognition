"""Overlap-preserving tiling of full-resolution diagrams (SAHI strategy, ref [40]).

P&ID sheets exceed 10,000 x 7,000 px (§2.2.1); naive tiling truncates symbols at
patch boundaries and drops recall (Gap 4). This step slices each canonical image
into fixed-size patches with a configurable overlap margin so every symbol is fully
visible in at least one patch, and rewrites the YOLO labels into patch-local
coordinates. WBF re-merges patch detections at inference time (see detect.infer).

Config: cfg.data.tiling.{size, overlap}. Output: derived/tiles/.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pid.common.cli import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   1. for each canonical image, generate patch grid (size, overlap)
    #   2. crop patch image -> derived/tiles/images/
    #   3. clip + reproject boxes into patch coords, keep boxes above visibility thresh
    #   4. write per-patch YOLO txt -> derived/tiles/labels/
    raise NotImplementedError


if __name__ == "__main__":
    main()
