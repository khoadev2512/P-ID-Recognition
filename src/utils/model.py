"""ResNet-34 fine-grained classifier (report roadmap Phase 3).

Backbone from timm; one linear head per fgc_group sized to that group's fine-class
count. cfg.fgc.loss selects cross-entropy vs. Focal Loss (ref [1]) for rare classes;
cfg.fgc.metric_learning optionally swaps the head for an ArcFace margin head (ref [43],
listed as an implementation-level investigation in §4.2.3) — out of scope for now, this
module just needs to not crash when it's false.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def build_model(cfg: DictConfig, group: str, num_fine: int) -> nn.Module:
    """timm.create_model(cfg.fgc.backbone, pretrained=cfg.fgc.pretrained, num_classes=num_fine).

    `group` is not consumed by timm itself — it exists so the caller (one model
    instance per fgc_group) documents intent and call sites can log/name checkpoints
    by group without threading a second variable around.
    """
    del group  # unused here; kept in the signature for caller-side clarity (see docstring)
    return timm.create_model(
        str(cfg.fgc.backbone), pretrained=bool(cfg.fgc.pretrained), num_classes=num_fine
    )


class FocalLoss(nn.Module):
    """Standard multi-class focal loss (ref [1]).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Computed via F.cross_entropy(..., reduction="none") to get per-sample -log(p_t),
    then p_t = exp(-ce_loss), then the (1 - p_t)^gamma focusing term down-weights
    already-easy (high p_t) samples so training doesn't get swamped by the frequent
    classes. `weight` (per-class alpha_t, e.g. inverse frequency) is deliberately
    applied as a per-sample multiplier on the finished focal term rather than passed
    into the F.cross_entropy(...) call: nn.CrossEntropyLoss/F.cross_entropy's own
    `weight=` kwarg multiplies the per-sample loss BEFORE we'd extract p_t, which
    would make p_t = exp(-ce_loss) recover p(y)^weight[y] instead of the true softmax
    probability and silently corrupt the (1 - p_t)^gamma focusing term.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        # Registered as a buffer (not a plain attribute) so criterion.to(device) moves
        # it along with the module, same as nn.CrossEntropyLoss does internally.
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, target, reduction="none")  # -log(p_t), unweighted
        p_t = torch.exp(-ce_loss)
        focal = (1.0 - p_t) ** self.gamma * ce_loss
        if self.weight is not None:
            alpha_t = self.weight[target]  # per-sample class weight, i.e. alpha_t
            focal = alpha_t * focal
        return focal.mean()


def _class_weights(class_counts: list[int]) -> torch.Tensor:
    """Inverse-frequency per-class weights, normalized to average 1 across classes
    (so the overall loss scale doesn't shift with the number of fine classes in a
    group). Counts are clamped to >=1 to avoid a divide-by-zero for a fine class with
    no crops in this split.
    """
    counts = torch.tensor(class_counts, dtype=torch.float32).clamp(min=1.0)
    inv_freq = 1.0 / counts
    return inv_freq / inv_freq.sum() * len(counts)


def build_loss(cfg: DictConfig, class_counts: list[int]) -> nn.Module:
    """CE (nn.CrossEntropyLoss) or Focal Loss (cfg.fgc.loss), weighted by inverse
    class_counts frequency for either loss.
    """
    weight = _class_weights(class_counts)
    loss_name = str(cfg.fgc.loss)
    if loss_name == "focal":
        return FocalLoss(weight=weight, gamma=2.0)
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unknown cfg.fgc.loss={loss_name!r}; expected 'ce' or 'focal'")
