"""Evaluation driver — runs both metric families and writes a report.

Reproduces the two-family protocol (Table 1) and supports the ablation interface of
§4.2.1: evaluate detector-only vs. detector+FGC, and coarse-trained vs. fine-trained
detector, on the same inputs. Also drives the Phase-3 synthetic-vs-real comparison
by running the same protocol on split=test (real) and the synthetic test set.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pid.common.cli import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   1. load predictions (detector-only or +FGC per cfg.eval.mode) and GT
    #   2. detection.mean_ap(...) -> mAP@50, mAP@50:95, per-class AP
    #   3. per fgc_group: within_group_confusion/accuracy; topk on GT crops;
    #      rare_macro_f1
    #   4. write metrics json + plots (confusion matrices, PR curves)
    raise NotImplementedError


if __name__ == "__main__":
    main()
