"""Apply the FGC head(s) to detector output, producing final fine labels.

Routing policy (report §4.2.3, an implementation decision): a detection is passed to
the fine-grained head only when its coarse_class is in fgc_groups AND its score meets
cfg.fgc.route.min_score; otherwise fine_class := coarse_class unchanged. This is where
the two stages compose into the final list[Detection].
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pid.common.cli import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   1. load per-group FGC checkpoints
    #   2. for each detection from detect.infer:
    #        if classmap.needs_fgc(det.coarse_class) and det.score >= min_score:
    #            crop -> group head -> det.fine_class, det.fine_score
    #        else: det.fine_class = det.coarse_class
    raise NotImplementedError


if __name__ == "__main__":
    main()
