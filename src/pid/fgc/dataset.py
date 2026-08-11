"""Crop dataset + balanced sampling for the FGC stage.

Reads derived/crops_fgc/{split}/<fine_class>/ (torchvision ImageFolder layout).
Rare-class balancing (§2.2.3): a WeightedRandomSampler that can build a batch with
roughly one patch per fine class regardless of natural frequency — impractical at
full-image scale, which is the point of the separable stage (§4.2.3).

One classifier head PER fgc_group is the default (a valve-family head, a bubble-family
head, ...), since fine classes only compete within their group.
"""

from __future__ import annotations

from omegaconf import DictConfig


def build_dataset(cfg: DictConfig, split: str):
    """TODO: ImageFolder over crops_fgc/<split>, augment per cfg.fgc.aug."""
    raise NotImplementedError


def build_balanced_sampler(dataset, cfg: DictConfig):
    """TODO: WeightedRandomSampler with per-fine-class inverse-frequency weights."""
    raise NotImplementedError
