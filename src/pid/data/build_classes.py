"""Derive / validate canonical/classes.yaml from the COCO categories.

COCO categories in annotations.coco.json carry both fine names and (by convention)
a supercategory field for the coarse level. This step materialises classes.yaml —
the fine<->coarse map, fgc_groups, and rare_fine list — so every downstream stage
reads one authoritative file (see common.classmap).

Hydra entrypoint. Config group: configs/data/.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pid.common.cli import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   1. read cfg.data.coco_json
    #   2. collect (fine name, supercategory) pairs from categories[]
    #   3. flag rare fine classes by training-instance frequency (§2.2.2 long tail)
    #   4. write classes.yaml matching common.classmap schema
    raise NotImplementedError


if __name__ == "__main__":
    main()
