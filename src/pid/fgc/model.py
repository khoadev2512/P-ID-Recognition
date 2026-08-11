"""ResNet-34 fine-grained classifier (report roadmap Phase 3).

Backbone from timm; one linear head per fgc_group sized to that group's fine-class
count. cfg.fgc.loss selects cross-entropy vs. Focal Loss (ref [1]) for rare classes;
cfg.fgc.metric_learning optionally swaps the head for an ArcFace margin head (ref [43],
listed as an implementation-level investigation in §4.2.3).
"""

from __future__ import annotations

from omegaconf import DictConfig


def build_model(cfg: DictConfig, group: str, num_fine: int):
    """TODO: timm.create_model('resnet34', pretrained=..., num_classes=num_fine)."""
    raise NotImplementedError


def build_loss(cfg: DictConfig, class_counts):
    """TODO: CE or Focal Loss with optional class-frequency reweighting."""
    raise NotImplementedError
