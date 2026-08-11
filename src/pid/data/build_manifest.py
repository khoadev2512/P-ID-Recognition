"""Build canonical/manifest.csv — one row per image.

Columns (per data/ layout spec):
    id, source, split, n_symbols, classes_present

source is synthetic|real (tier-0 origin); split is train|val|test. The real-world
held-out set (Phase 3) is marked split=test and source=real, and is never used for
training (report §4.2.4). This file is the authoritative train/val/test partition
that every other stage reads.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pid.common.cli import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   1. scan COCO images + annotations
    #   2. assign split per cfg.data.split (seed, ratios; keep real -> test)
    #   3. write manifest.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()
