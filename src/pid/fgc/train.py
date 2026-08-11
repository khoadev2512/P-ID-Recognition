"""Train the fine-grained classifier(s) on symbol crops.

Trains one head per fgc_group over derived/crops_fgc/train, validates on
crops_fgc/val. Owns the loop (optimizer, scheduler, checkpoint) — kept minimal and
config-driven for the ablations in §4.2 / roadmap Phase 3.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pid.common.cli import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO:
    #   for each group in classmap.fgc_groups:
    #     ds = build_dataset(cfg, "train"); sampler = build_balanced_sampler(...)
    #     model = build_model(cfg, group, num_fine); loss = build_loss(...)
    #     train/val loop; save best checkpoint per group
    raise NotImplementedError


if __name__ == "__main__":
    main()
