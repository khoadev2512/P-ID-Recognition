"""Extract per-symbol crops for FGC training (report §4.2.3).

Training the fine-grained stage on cropped symbol regions — not full diagrams — is
the mechanism that makes rare-class balancing practical (§2.2.2 long tail). This step
cuts one crop per ground-truth symbol from canonical images and writes them into a
per-fine-class folder tree that torchvision ImageFolder can read directly:

    derived/crops_fgc/{train,val}/<fine_class>/<image_id>_<inst>.png

Only symbols in fgc_groups (valve / instrument_bubble / fitting) are extracted by
default; controlled by cfg.fgc.groups.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pid.common.cli import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   1. load classmap + COCO + manifest split (train/val)
    #   2. for each symbol whose coarse class needs FGC, crop bbox (+ margin)
    #   3. save under crops_fgc/<split>/<fine_class>/
    raise NotImplementedError


if __name__ == "__main__":
    main()
